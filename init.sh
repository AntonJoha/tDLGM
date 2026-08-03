#!/usr/bin/env bash


python -m venv .venv
source .venv/bin/activate
pip install -e "."
python -m tdlgm.main --help
