#!/usr/bin/env bash
set -e

systemd-inhibit \
    --what=sleep:idle \
    --who="DiT Training" \
    --why="ML training in progress" \
    --mode=block \
    /home/hido-pinto/PycharmProjects/HomeMadeDiffusion/.venv/bin/python train.py "$@"
