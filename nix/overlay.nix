# Adds the build123d dependency chain to every Python package set in nixpkgs.
# None of these exist upstream; each is a normal derivation (binary wheels are
# handled with autoPatchelfHook, never an FHS environment).
final: prev: {
  # Version-matched Rerun viewer; see nix/pkgs/rerun-cli.nix for why nixpkgs'
  # own `rerun` cannot be used here.
  rerun-cli = final.callPackage ./pkgs/rerun-cli.nix { };

  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (pyfinal: pyprev: {
      cadquery-ocp-proxy = pyfinal.callPackage ./pkgs/cadquery-ocp-proxy.nix { };
      cadquery-ocp-novtk = pyfinal.callPackage ./pkgs/cadquery-ocp-novtk.nix { };
      trianglesolver     = pyfinal.callPackage ./pkgs/trianglesolver.nix { };
      ocpsvg             = pyfinal.callPackage ./pkgs/ocpsvg.nix { };
      ocp-gordon         = pyfinal.callPackage ./pkgs/ocp-gordon.nix { };
      lib3mf             = pyfinal.callPackage ./pkgs/lib3mf.nix { };
      webcolors_24       = pyfinal.callPackage ./pkgs/webcolors-24.nix { };
      build123d          = pyfinal.callPackage ./pkgs/build123d.nix {
        webcolors = pyfinal.webcolors_24;
      };
    })
  ];
}
