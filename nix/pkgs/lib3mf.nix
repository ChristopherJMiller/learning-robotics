{ lib, stdenv, buildPythonPackage, fetchurl, autoPatchelfHook, zlib }:

buildPythonPackage rec {
  pname = "lib3mf";
  version = "2.5.0";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/88/83/8b987ba95ac0ed9cc7e9c407a579bf43eff6349b1792b4a66c992ce4f76b/lib3mf-${version}-py3-none-manylinux2014_x86_64.whl";
    hash = "sha256-tMAAM8R8/qyTt9qgaftG6N6kOR1VIrec1un2r3XjMBM=";
  };

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [ stdenv.cc.cc.lib zlib ];

  pythonImportsCheck = [ "lib3mf" ];

  meta = with lib; {
    description = "Python bindings for the 3MF Consortium reference implementation";
    homepage = "https://github.com/3MFConsortium/lib3mf";
    license = licenses.bsd2;
    platforms = [ "x86_64-linux" ];
  };
}
