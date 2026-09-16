#!/usr/bin/env python3
"""Regenerate models/generated/ from config/arm.toml. Run via `just models`."""

from arm.models import write_models

if __name__ == "__main__":
    for path in write_models():
        print(f"wrote {path}")
