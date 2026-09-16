# Pinned to 0.6.x deliberately: ocpsvg >=0.7 requires cadquery_ocp_proxy >=8.0,
# while build123d 0.11.1 requires cadquery-ocp-novtk <8.0. See docs/adr/0003.
{ lib, buildPythonPackage, fetchurl, cadquery-ocp-novtk, svgelements, svgpathtools }:

buildPythonPackage rec {
  pname = "ocpsvg";
  version = "0.6.0";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/a9/d7/27d2c9d5a2645fdda9e502a2a1a1cb5d4c9d137223ef76a43296eb7c152b/ocpsvg-${version}-py3-none-any.whl";
    hash = "sha256-XPxt6y9gLe2EVZZAnkH+j14bOJ4U6MxyfbCVXw6I3ns=";
  };

  propagatedBuildInputs = [ cadquery-ocp-novtk svgelements svgpathtools ];
  pythonImportsCheck = [ "ocpsvg" ];

  meta = with lib; {
    description = "SVG <-> OpenCASCADE geometry conversion";
    homepage = "https://github.com/snoyer/ocpsvg";
    license = licenses.mit;
  };
}
