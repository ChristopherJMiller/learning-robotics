# Canonical commands. If it is not here, it is not a supported workflow.
set shell := ["bash", "-uc"]

default:
    @just --list

# Run the full test suite.
test *ARGS:
    pytest -q tests {{ARGS}}

# Lint and format check.
lint:
    ruff check src tests scripts cad
    ruff format --check src tests scripts cad

# Apply formatting.
fmt:
    ruff format src tests scripts cad
    ruff check --fix src tests scripts cad

# Everything CI would run, hermetically.
check:
    nix flake check

# Regenerate URDF + MJCF from config/arm.toml.
models:
    python scripts/gen_models.py

# Track a trajectory; compares PD, gravity comp, feedforward, computed torque.
track *ARGS:
    python scripts/sim_track.py {{ARGS}}

# Encoder quantisation, velocity estimation, and friction dead zones.
sensing *ARGS:
    python scripts/sim_sensing.py {{ARGS}}

# Watch the physics live in MuJoCo's own viewer (shows contacts and the floor).
watch CONTROLLER="feedforward_pd":
    python scripts/sim_track.py --viewer --only {{CONTROLLER}} --seconds 20

# Cartesian impedance: push the arm and watch it yield by exactly F/K.
impedance *ARGS:
    python scripts/sim_impedance.py {{ARGS}}

# Hold a pose against gravity and compare PD with and without gravity comp.
hold *ARGS:
    python scripts/sim_hold.py {{ARGS}}

# Drive the arm through a trajectory; writes runs/*.rrd + *.parquet.
viz *ARGS:
    python scripts/viz_fk.py {{ARGS}}

# Same, but open the Rerun viewer live.
viz-live:
    python scripts/viz_fk.py --spawn

# Open a previously recorded run in the viewer.
view RUN="fk_sweep":
    rerun runs/{{RUN}}.rrd

# Recompute link mass properties from the CAD (dry run).
cad-props:
    nix develop .#cad --command python scripts/cad_update_params.py

# Recompute and write them into config/arm.toml, then regenerate the models.
cad-sync:
    nix develop .#cad --command python scripts/cad_update_params.py --write
    just models

# Export STEP and STL for every link.
cad-export:
    nix develop .#cad --command python scripts/cad_export.py

# Build the CAD dependency chain (slow the first time).
cad-build:
    nix build .#build123d --print-build-logs

# Enter the CAD shell.
cad:
    nix develop .#cad
