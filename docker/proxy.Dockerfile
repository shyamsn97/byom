FROM python:3.12-slim

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY byom ./byom
RUN pip install --no-cache-dir . && rm -rf /src

WORKDIR /
EXPOSE 8080
CMD ["byom", "--help"]
