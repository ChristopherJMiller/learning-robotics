# Canonical commands. If it is not here, it is not a supported workflow.
set shell := ["bash", "-uc"]

default:
    @just --list

# Run the full test suite.
test *ARGS:
    pytest -q tests {{ARGS}}

# Lint and format check.
lint:
    ruff check src tests scripts
    ruff format --check src tests scripts

# Apply formatting.
fmt:
    ruff format src tests scripts
    ruff check --fix src tests scripts

# Everything CI would run, hermetically.
check:
    nix flake check

# Regenerate URDF + MJCF from config/arm.toml.
models:
    python scripts/gen_models.py

# Drive the arm through a trajectory and log it to Rerun.
viz:
    python scripts/viz_fk.py

# Build the CAD dependency chain (slow the first time).
cad-build:
    nix build .#build123d --print-build-logs

# Enter the CAD shell.
cad:
    nix develop .#cad
