{ lib, buildPythonPackage, fetchurl, numpy }:

buildPythonPackage rec {
  pname = "trianglesolver";
  version = "1.2";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/ff/8e/43d45cf3e18e3f455e4b5ab333a7c27b8e38c4e535f7346b7148ce08eb65/trianglesolver-${version}-py3-none-any.whl";
    hash = "sha256-qgkDw3CLTitJbwbUkMrnLG/2J0sA0e3OQg/Po7K3ZoI=";
  };

  propagatedBuildInputs = [ numpy ];
  pythonImportsCheck = [ "trianglesolver" ];

  meta = with lib; {
    description = "Solve triangles given a subset of sides and angles";
    homepage = "https://github.com/dmishin/trianglesolver";
    license = licenses.mit;
  };
}
