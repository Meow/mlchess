# NightyBot RL on an NVIDIA GPU: PyTorch with CUDA, the Rust search built in,
# and the repo mounted at /app so checkpoints land in it like they do outside.
#
#   docker compose build
#   docker compose run --rm train            # python3 train_rl.py, resumes from rl_checkpoint.pt
#   docker compose run --rm train --wandb --max-minutes 480
#   docker compose run --rm eval --opponent greedy:2
#   docker compose run --rm engine           # UCI on stdin
#
# Needs the NVIDIA driver and the NVIDIA Container Toolkit on the host
# (Fedora: `sudo dnf install nvidia-container-toolkit`, then
# `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).

ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# --- build the Rust search as a wheel ----------------------------------------
FROM ${BASE} AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
COPY rust /src/rust
# pip fetches maturin (the crate's build backend) itself.
RUN pip wheel --no-deps -w /wheels /src/rust

# --- the image everything runs in --------------------------------------------
FROM ${BASE}
RUN pip install --no-cache-dir chess numpy safetensors wandb
COPY --from=build /wheels/nighty_rs-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/nighty_rs-*.whl && rm /tmp/nighty_rs-*.whl

WORKDIR /app
# The compose file mounts the checkout over this, so the copy only matters
# when the image is run on its own.
COPY *.py run.sh run_rl.sh ./
COPY chess5.model chess_from.model ./

# Writable home for wandb and torch caches, whatever uid the container runs as.
ENV HOME=/tmp/home WANDB_DIR=/app/wandb PYTHONUNBUFFERED=1
RUN mkdir -p /tmp/home && chmod 777 /tmp/home

ENTRYPOINT ["python3"]
CMD ["train_rl.py"]
