let
  pkgs = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/04906337b6d53e8bb0eaf0470ae9f070a4d4eb55.tar.gz";
    sha256 = "1ixmpdkxv55py03w9gj4qn4p90jn494a5zzx3j90c703gaa3dhcl";
  }) { };
in
pkgs.mkShell {
  packages = [ pkgs.uv pkgs.python311 ];
  # Wheels bundle GDAL, but rely on these platform runtime libraries.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.expat
    pkgs.zlib
    pkgs.stdenv.cc.cc.lib
  ];
  UV_PYTHON = "${pkgs.python311}/bin/python3.11";
  UV_PYTHON_DOWNLOADS = "never";
}
