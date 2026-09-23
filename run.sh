#!/usr/bin/env bash
# lichess-bot launches this as the engine binary. Resolve everything relative to
# the checkout so the model files load no matter where the bot was started from.
cd "$(dirname "$0")" || exit 1

# The Rust build of the engine (engine/, `cargo build --release` there) plays
# the same nets from the .safetensors exports, an order of magnitude faster.
# Fall back to run.py when it has not been built.
if [ -x engine/target/release/nightybot ]; then
  exec engine/target/release/nightybot "$@"
fi

python=python3
[ -x .venv/bin/python3 ] && python=.venv/bin/python3

exec "$python" run.py
