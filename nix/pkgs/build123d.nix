{ lib, buildPythonPackage, fetchurl
, cadquery-ocp-novtk, ocpsvg, ocp-gordon, trianglesolver, lib3mf
, numpy, scipy, sympy, scikit-learn, svgpathtools, anytree, ezdxf
, ipython, webcolors, requests, typing-extensions
}:

buildPythonPackage rec {
  pname = "build123d";
  version = "0.11.1";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/e7/f2/c466dbd4cb3aa75a192ba39f1a49058f828fcc0eb6f9cf6936ed6078308b/build123d-${version}-py3-none-any.whl";
    hash = "sha256-TpX6fMvcg+YkMTvkkufE9fDrLqHfNhMOtxi7DCWonhA=";
  };

  propagatedBuildInputs = [
    cadquery-ocp-novtk ocpsvg ocp-gordon trianglesolver lib3mf
    numpy scipy sympy scikit-learn svgpathtools anytree ezdxf
    ipython webcolors requests typing-extensions
  ];

  # Import is heavy (pulls all of OCP); keep it as the single smoke test.
  pythonImportsCheck = [ "build123d" ];

  meta = with lib; {
    description = "Python code-CAD: a boundary-representation modeller built on OpenCASCADE";
    homepage = "https://github.com/gumyr/build123d";
    license = licenses.asl20;
  };
}
