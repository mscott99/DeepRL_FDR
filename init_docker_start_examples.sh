#!/usr/bin/env bash

rm -f /tmp/.X99-lock || true
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
xvfb=$!

export DISPLAY=:99
python3 run_model.py
process=$!
wait process
