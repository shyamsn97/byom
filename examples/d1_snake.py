"""LiquidAI's d1-2B decision model playing snake from frames on a Modal GPU.

    pip install modal pillow && modal setup
    modal run examples/d1_snake.py [--steps 300] [--size 10] [--seed 0]

Every step renders the board as a picture and asks d1 one question about it: which way the head
should move. d1 answers typed questions instead of generating text, so it runs in Transformers
with its own code rather than in SGLang. If its top pick would crash, the game takes its likeliest
safe move instead and counts that as an override. The frames are saved to d1_snake.gif.
"""

import io
import os
import random
import time

import modal

MODEL = "LiquidAI/d1-2B"
REVISION = "a6bce774e6dde338dcc3036ceb5d91f78bf4c19e"

app = modal.App("byom-d1-snake")
cache = modal.Volume.from_name("byom-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("transformers>=5.19", "torch", "torchvision", "pillow", "numpy")
    .env({"HF_HOME": "/cache"})
)


@app.cls(image=image, gpu="L4", volumes={"/cache": cache}, scaledown_window=300)
class D1:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True, dtype=torch.bfloat16
        ).to("cuda")
        cache.commit()

    @modal.method()
    def decide(self, frame_png: bytes, questions: dict) -> dict:
        from PIL import Image

        frame = Image.open(io.BytesIO(frame_png)).convert("RGB")
        start = time.perf_counter()
        out = self.model.system_one(None, questions, media=[frame])
        return {**out, "ms": (time.perf_counter() - start) * 1000}


MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
QUESTION = {"move": {
    "type": "choice",
    "instructions": "This is a frame of a snake game. The dark blue square is the snake's head, "
                    "the light blue squares are its body, the red square is the food, and the "
                    "gray border is a wall. Which way should the head move next to get to the "
                    "food without hitting the wall or the body?",
    "criteria": {
        "up": "Toward the top of the picture",
        "down": "Toward the bottom of the picture",
        "left": "Toward the left side of the picture",
        "right": "Toward the right side of the picture",
    },
}}
CELL, BORDER = 64, 16


def render(snake: list[tuple[int, int]], food: tuple[int, int], size: int):  # -> PIL.Image
    from PIL import Image, ImageDraw

    side = size * CELL + 2 * BORDER
    img = Image.new("RGB", (side, side), "#9e9e9e")
    draw = ImageDraw.Draw(img)
    draw.rectangle([BORDER, BORDER, side - BORDER - 1, side - BORDER - 1], fill="white")

    def cell(x: int, y: int, color: str) -> None:
        x0, y0 = BORDER + x * CELL, BORDER + y * CELL
        draw.rectangle([x0 + 4, y0 + 4, x0 + CELL - 5, y0 + CELL - 5], fill=color)

    cell(*food, "#e53935")
    for x, y in snake[1:]:
        cell(x, y, "#64b5f6")
    cell(*snake[0], "#0d47a1")
    return img


def png(img) -> bytes:  # PIL.Image -> bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def text_board(snake: list[tuple[int, int]], food: tuple[int, int], size: int) -> str:
    rows = [["."] * size for _ in range(size)]
    for x, y in snake[1:]:
        rows[y][x] = "o"
    rows[food[1]][food[0]] = "F"
    rows[snake[0][1]][snake[0][0]] = "H"
    return "\n".join(" ".join(r) for r in rows)


def step(snake: list[tuple[int, int]], move: str) -> tuple[int, int]:
    dx, dy = MOVES[move]
    return snake[0][0] + dx, snake[0][1] + dy


def safe(snake: list[tuple[int, int]], move: str, size: int) -> bool:
    x, y = step(snake, move)
    # The tail moves out of the way on the same step, so moving onto it is fine.
    return 0 <= x < size and 0 <= y < size and (x, y) not in snake[:-1]


def place_food(snake: list[tuple[int, int]], size: int, rng: random.Random) -> tuple[int, int]:
    free = [(x, y) for x in range(size) for y in range(size) if (x, y) not in snake]
    return rng.choice(free)


@app.local_entrypoint()
def main(steps: int = 300, size: int = 10, seed: int = 0, gif: str = "d1_snake.gif") -> None:
    rng = random.Random(seed)
    d1 = D1()
    snake = [(size // 2, size // 2)]
    food = place_food(snake, size, rng)
    score, overrides, frames = 0, 0, []
    result = f"stopped after {steps} steps"
    print("starting d1-2B on Modal (the first call loads the model)...")

    for n in range(1, steps + 1):
        frame = render(snake, food, size)
        frames.append(frame)
        out = d1.decide.remote(png(frame), QUESTION)
        answer = out["answers"]["move"]
        probs = answer["probabilities"]

        options = [m for m in MOVES if safe(snake, m, size)]
        if not options:
            result = f"trapped after {n} steps"
            break
        move = max(options, key=probs.get)
        overrides += move != answer["choice"]

        head = step(snake, move)
        snake.insert(0, head)
        if head == food:
            score += 1
            if len(snake) == size * size:
                result = "filled the board"
                break
            food = place_food(snake, size, rng)
        else:
            snake.pop()

        bars = "  ".join(f"{m} {probs[m]:.2f}" for m in MOVES)
        print("\033[H\033[J" + text_board(snake, food, size))
        print(f"\nstep {n}  score {score}  move {move}  overrides {overrides}")
        print(f"{bars}  ({out['ms']:.0f} ms on the GPU)")

    frames.append(render(snake, food, size))
    path = os.path.abspath(gif)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"\n{result}, score {score}, overrides {overrides}")
    print(f"recording ({len(frames)} frames): {path}")
