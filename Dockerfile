FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ruff pytest tabulate

COPY . .

# Default: run the smoke test (override with `docker run ... make train`)
CMD ["make", "smoke"]