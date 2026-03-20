#!/usr/bin/env bash
gnome-session-inhibit \
  --inhibit suspend:idle \
  --reason "ML training in progress" \
  systemd-inhibit \
    --what=sleep:idle \
    --who="DiT Training" \
    --why="ML training in progress" \
    /home/hido-pinto/PycharmProjects/Diffusion/.venv/bin/python train.py "$@"
