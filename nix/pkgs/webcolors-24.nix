# build123d 0.11.1 pins webcolors ~=24.8.0 while nixpkgs carries 25.x, which
# changed the public API. Pinned as a scoped override rather than relaxed, so
# the constraint upstream declares is actually honoured. Applied only to
# build123d; nothing else in the package set sees this version.
{ lib, buildPythonPackage, fetchurl }:

buildPythonPackage rec {
  pname = "webcolors";
  version = "24.8.0";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/f0/33/12020ba99beaff91682b28dc0bbf0345bbc3244a4afbae7644e4fa348f23/webcolors-${version}-py3-none-any.whl";
    hash = "sha256-/Ew7WTWK2hZFUghKjr7mN8Ih5AWSZ9D4Mls7Vg9sfwo=";
  };

  pythonImportsCheck = [ "webcolors" ];

  meta = with lib; {
    description = "Working with color names and values formats defined by HTML and CSS";
    homepage = "https://github.com/ubernostrum/webcolors";
    license = licenses.bsd3;
  };
}
