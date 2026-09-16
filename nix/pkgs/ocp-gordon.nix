# Pinned to 0.2.x: 0.3+ requires cadquery_ocp_proxy >=8.0. See docs/adr/0003.
{ lib, buildPythonPackage, fetchurl, cadquery-ocp-novtk, numpy, scipy }:

buildPythonPackage rec {
  pname = "ocp-gordon";
  version = "0.2.2";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/6c/d1/ee535cdbc502790dda6555e439ea8bcacdd38325859151769d109d307a94/ocp_gordon-${version}-py3-none-any.whl";
    hash = "sha256-ZBTpAaEQZaVug5yBtW9BgQC00Lgetaojux+Pcvl/8Jk=";
  };

  propagatedBuildInputs = [ cadquery-ocp-novtk numpy scipy ];
  pythonImportsCheck = [ "ocp_gordon" ];

  meta = with lib; {
    description = "Gordon surface construction for OpenCASCADE";
    license = licenses.asl20;
  };
}
