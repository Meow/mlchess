#!/usr/bin/env bash
# lichess-bot launches this as the engine binary. Resolve everything relative to
# the checkout so the .model files load no matter where the bot was started from.
cd "$(dirname "$0")" || exit 1

python=python3
[ -x .venv/bin/python3 ] && python=.venv/bin/python3

exec "$python" run.py
