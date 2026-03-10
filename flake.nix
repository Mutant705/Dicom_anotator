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
        (pkgs.python313.withPackages (ps: [
          ps.matplotlib
          ps.pydicom
          ps.numpy
          ps.cython
          ps.cupy
         
        ]))
      ];
    };
  };
}
#after saving changes do nix develop to save changes ;) 
