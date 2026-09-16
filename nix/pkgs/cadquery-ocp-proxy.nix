# A tiny pure-Python shim. Packages in the CAD chain depend on the *proxy*,
# which resolves the `OCP` namespace to whichever concrete OCP distribution is
# installed (the novtk variant, here). Splitting it this way is upstream's
# design, not ours.
{ lib, buildPythonPackage, fetchurl }:

buildPythonPackage rec {
  pname = "cadquery-ocp-proxy";
  version = "7.9.3.1.1";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/30/c0/04e9363a99fee892de2776820e3dcf04f8825b6edc9580efe3416c9465a7/cadquery_ocp_proxy-${version}-py3-none-any.whl";
    hash = "sha256-ykFk7EtUlW2fw+aMZ9VVtUhsuWPC9x4Y3wBboWuSHJE=";
  };

  # Importing OCP through the proxy requires the concrete implementation,
  # which depends on this package; checking it here would be circular.
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = with lib; {
    description = "Namespace proxy resolving OCP to an installed OpenCASCADE binding";
    homepage = "https://github.com/CadQuery/OCP";
    license = licenses.lgpl21Only;
  };
}
