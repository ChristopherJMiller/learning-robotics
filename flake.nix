{
  description = "learn-robotics — a 3-DOF arm, built up from first principles";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);

      pkgsFor = system: import nixpkgs {
        inherit system;
        overlays = [ (import ./nix/overlay.nix) ];
      };
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = pkgsFor system;
          python = pkgs.python313;

          # Everything needed for kinematics, simulation, visualisation and
          # tests. All of this is packaged upstream in nixpkgs.
          corePython = ps: with ps; [
            numpy
            scipy
            pydantic
            mujoco
            rerun-sdk
            pinocchio
            pytest
            pytest-cov
            hypothesis
          ];

          # CAD adds the build123d chain, which lives in ./nix/pkgs.
          cadPython = ps: corePython ps ++ [ ps.build123d ];

          commonTools = with pkgs; [ just ruff git ];

          # MuJoCo and Rerun both open GL windows; make the loaders findable.
          graphicsLibs = with pkgs; [
            libGL
            libglvnd
            wayland
            libxkbcommon
            libx11
            libxcursor
            libxrandr
            libxinerama
            libxi
          ];

          mkShell = { name, pythonPkgs, extraTools ? [ ] }:
            pkgs.mkShell {
              inherit name;
              packages =
                [ pkgs.rerun-cli (python.withPackages pythonPkgs) ]
                ++ commonTools
                ++ extraTools;
              env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath graphicsLibs;
              shellHook = ''
                export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
                echo "[${name}] python ${python.pythonVersion} — run 'just' for available commands"
              '';
            };
        in
        {
          # Fast, pure, no heavy dependencies. This is where the learning happens.
          default = mkShell {
            name = "learn-robotics";
            pythonPkgs = corePython;
          };

          # Only needed once you are designing physical geometry.
          cad = mkShell {
            name = "learn-robotics-cad";
            pythonPkgs = cadPython;
          };

          # FreeCAD is used only to look at geometry build123d produced. It is
          # kept out of the cad shell because its closure includes TeX Live.
          viewer = pkgs.mkShell {
            name = "learn-robotics-viewer";
            packages = [ pkgs.freecad ];
          };

          # Electronics, much later.
          electronics = pkgs.mkShell {
            name = "learn-robotics-electronics";
            packages = [ pkgs.kicad ];
          };
        });

      # `nix flake check` runs the real test suite hermetically — possible
      # because the default shell has no impure dependencies.
      checks = forAllSystems (system:
        let
          pkgs = pkgsFor system;
          python = pkgs.python313;
        in
        {
          tests = pkgs.stdenv.mkDerivation {
            name = "learn-robotics-tests";
            src = ./.;
            nativeBuildInputs = [
              (python.withPackages (ps: with ps; [
                numpy scipy pydantic mujoco rerun-sdk pinocchio pytest hypothesis
              ]))
            ];
            buildPhase = ''
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              export HOME=$TMPDIR
              pytest -q tests
            '';
            installPhase = "touch $out";
          };

          lint = pkgs.runCommand "learn-robotics-lint" { nativeBuildInputs = [ pkgs.ruff ]; } ''
            cd ${./.}
            ruff check src tests scripts
            ruff format --check src tests scripts
            touch $out
          '';
        });

      # Individual CAD packages, exposed so they can be built and cached on
      # their own: `nix build .#cadquery-ocp-novtk`
      packages = forAllSystems (system:
        let
          pkgs = pkgsFor system;
          pyPkgs = pkgs.python313Packages;
        in {
          inherit (pkgs) rerun-cli;
          inherit (pyPkgs) cadquery-ocp-proxy cadquery-ocp-novtk ocpsvg ocp-gordon trianglesolver lib3mf build123d;
        });
    };
}
