FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --fix-missing \
    python3.11 python3.11-dev python3-pip \
    git wget curl apt-transport-https ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK (includes gsutil)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
    https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY sl/ ./sl/
COPY scripts/ ./scripts/
COPY run_phase_5.sh ./
COPY run_phase_5_with_logging.sh ./
COPY qwen2.5-7b-instruct/ ./models/qwen2.5-7b-instruct
COPY run_phase_5_prime.sh ./
COPY run_phase_5_prime_with_logging.sh ./
# COPY run_phase_4_prime.sh ./
# COPY run_phase_4_prime_with_logging.sh ./

RUN uv pip install --system --no-cache -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["bash", "-c"]