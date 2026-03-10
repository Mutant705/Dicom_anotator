{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";

    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    python = pkgs.python312.override {
      packageOverrides = pyself: pysuper: {
        cupy = pysuper.cupy.overridePythonAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [ pyself.cython ];
        });
      };
    };

  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        (python.withPackages (ps: [
          ps.numpy
          ps.matplotlib
          ps.pydicom
          ps.cython
          ps.cupy
        ]))
      ];
    };
  };
}
#after saving changes do nix develop to save changes ;) 

