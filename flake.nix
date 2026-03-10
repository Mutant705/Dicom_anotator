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

  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        (pkgs.python312.withPackages (ps: [
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
