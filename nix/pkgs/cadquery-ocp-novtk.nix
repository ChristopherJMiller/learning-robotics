# OpenCASCADE Python bindings (no-VTK variant).
#
# Built from the upstream manylinux wheel rather than from source: building OCP
# from scratch requires the pywrap/clang binding generator and several hours of
# OCCT compilation. autoPatchelfHook rewrites the wheel's ELF interpreter and
# RPATHs against nixpkgs libraries, which gives a normal, hermetic store path --
# no FHS sandbox, no nix-ld.
{ lib
, stdenv
, buildPythonPackage
, python
, fetchurl
, autoPatchelfHook
, libGL
, libglvnd
, fontconfig
, freetype
, expat
, zlib
, tbb_2021
, xorg
}:

let
  pyTag = "cp${lib.replaceStrings [ "." ] [ "" ] python.pythonVersion}";
in
buildPythonPackage rec {
  pname = "cadquery-ocp-novtk";
  version = "7.9.3.1.1";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/f3/31/82baf17406c0a13f2eb98c1d46d09a640795fc7d6b373a69bc5f44344db3/cadquery_ocp_novtk-${version}-${pyTag}-${pyTag}-manylinux_2_31_x86_64.whl";
    hash = "sha256-/80E1O+gh9OqlCNgAgJloawOEkB717z+s35wtNHuct8=";
  };

  nativeBuildInputs = [ autoPatchelfHook ];

  buildInputs = [
    stdenv.cc.cc.lib
    libGL
    libglvnd
    fontconfig
    freetype
    expat
    zlib
    tbb_2021
    xorg.libX11
    xorg.libXext
    xorg.libXmu
    xorg.libXi
    xorg.libSM
    xorg.libICE
    xorg.libXt
  ];

  # The wheel ships its own bundled OCCT shared objects next to the extension
  # modules; they must resolve against each other as well as against nixpkgs.
  postFixup = ''
    find $out -name '*.so*' -exec patchelf --add-rpath "$out/${python.sitePackages}/OCP" {} + || true
  '';

  pythonImportsCheck = [ "OCP" "OCP.gp" "OCP.TopoDS" ];

  meta = with lib; {
    description = "OpenCASCADE Technology Python bindings used by build123d/CadQuery";
    homepage = "https://github.com/CadQuery/OCP";
    license = licenses.lgpl21Only;
    platforms = [ "x86_64-linux" ];
  };
}
