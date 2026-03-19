#!/usr/bin/env bash
systemd-inhibit \
  --what=sleep:idle \
  --who="DiT Training" \
  --why="ML training in progress" \
  python train.py "$@"
