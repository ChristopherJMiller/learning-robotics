# The Rerun viewer, version-matched to the SDK in the devShell.
#
# nixpkgs packages `rerun` (the viewer) and `python3Packages.rerun-sdk`
# independently, and at the pinned nixpkgs revision they are ten minor versions
# apart -- viewer 0.27.2 against SDK 0.37.2. Recording format changes between
# releases, and more immediately the SDK's own `rerun` shim is broken: it
# imports `rerun_cli`, which the nixpkgs build does not install. The result is
# that `rr.spawn()` reports success, never binds port 9876, and fails
# asynchronously with a gRPC timeout several seconds later.
#
# Upstream ships the viewer binary inside the official rerun-sdk wheel under
# `rerun_cli/`. Extracting it gives an exactly version-matched viewer with no
# Rust toolchain and no multi-hour compile -- the same autoPatchelfHook
# approach already used for the OpenCASCADE bindings.
{ lib
, stdenv
, fetchurl
, unzip
, autoPatchelfHook
, makeWrapper
, libGL
, libglvnd
, libxkbcommon
, wayland
, vulkan-loader
, libx11
, libxcursor
, libxrandr
, libxi
, fontconfig
, freetype
, openssl
, zlib
, systemdLibs
}:

let
  # Runtime-only libraries: wgpu and winit dlopen these, so they never appear
  # in the ELF headers and autoPatchelf cannot discover them.
  runtimeLibs = [
    vulkan-loader
    libGL
    libglvnd
    libxkbcommon
    wayland
    libx11
    libxcursor
    libxrandr
    libxi
  ];
in
stdenv.mkDerivation rec {
  pname = "rerun-cli";
  version = "0.37.2";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/6c/f2/0ade77961a06e9563614f29ea7dede98efe7691b36948f78ae0a48bb6820/rerun_sdk-${version}-cp310-abi3-manylinux_2_28_x86_64.whl";
    hash = "sha256-WM3mBcLaGFL2uYe3YM7sNG3Ol/glAQFdIk4vax7VK+E=";
  };

  nativeBuildInputs = [ unzip autoPatchelfHook makeWrapper ];

  buildInputs = [
    stdenv.cc.cc.lib
    fontconfig
    freetype
    openssl
    systemdLibs
    zlib
  ] ++ runtimeLibs;

  inherit runtimeLibs;

  unpackPhase = ''
    runHook preUnpack
    unzip -q $src -d wheel
    runHook postUnpack
  '';

  installPhase = ''
    runHook preInstall

    if [ ! -f wheel/rerun_sdk/rerun_cli/rerun ]; then
      echo "ERROR: the wheel does not contain rerun_cli/rerun. Contents:"
      find wheel -maxdepth 2 -type d
      exit 1
    fi

    install -Dm755 wheel/rerun_sdk/rerun_cli/rerun $out/bin/.rerun-unwrapped
    makeWrapper $out/bin/.rerun-unwrapped $out/bin/rerun \
      --prefix LD_LIBRARY_PATH : "${lib.makeLibraryPath runtimeLibs}"

    runHook postInstall
  '';

  dontStrip = true;

  meta = with lib; {
    description = "Rerun viewer, extracted from the upstream SDK wheel to match it exactly";
    homepage = "https://rerun.io";
    license = with licenses; [ mit asl20 ];
    mainProgram = "rerun";
    platforms = [ "x86_64-linux" ];
  };
}
