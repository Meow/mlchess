#!/usr/bin/env bash
# lichess-bot entry point for the RL engine (run_rl.py); run.sh is the
# imitation-learned one. Resolves everything relative to the checkout, like
# run.sh, so nighty_rl.pt loads no matter where the bot was started from.
cd "$(dirname "$0")" || exit 1

python=python3
[ -x .venv/bin/python3 ] && python=.venv/bin/python3

exec "$python" run_rl.py "$@"
