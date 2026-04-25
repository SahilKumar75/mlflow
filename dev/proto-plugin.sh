#!/usr/bin/env bash

export PYTHONPATH="$PWD:$PYTHONPATH"
if command -v python &> /dev/null; then
  python dev/proto_plugin.py
else
  python3 dev/proto_plugin.py
fi
