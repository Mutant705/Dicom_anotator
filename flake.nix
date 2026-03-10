{
  description = "Development environment for Label Annotator and Sorter";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        # CuPy often requires specific CUDA support enabled in nixpkgs
        config.cudaSupport = true;
      };

      python = pkgs.python313.override {
        self = python;
        packageOverrides = pySelf: pySuper: {
          cupy = pySuper.cupy.overridePythonAttrs (old: {
            # Move Cython to nativeBuildInputs so the PEP 517 builder sees it
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              pySelf.cython_3
              pySelf.setuptools
              pkgs.cudaPackages.cuda_nvcc
            ];
            
            # Ensure CUDA libraries are available during the build phase
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.cudaPackages.cudatoolkit
            ];
          });
        };
      };

      pythonEnv = python.withPackages (ps: [
        ps.numpy
        ps.matplotlib
        ps.pydicom
        ps.cython_3
        ps.cupy
      ]);

    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "label-annotator-env";

        packages = [
          pythonEnv
          pkgs.cudaPackages.cudatoolkit # Essential for runtime GPU access
        ];

        shellHook = ''
          echo "--- Label Annotator & Sorter Dev Environment ---"
          echo "Python version: $(python --version)"
          echo "CUDA Support: Enabled"
          
          # Helps CuPy find the CUDA driver libraries on NixOS
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.lib.makeLibraryPath [ pkgs.cudaPackages.cudatoolkit ]}:$LD_LIBRARY_PATH"
        '';
      };
    };
}
#after saving changes do nix develop to save changes ;) 

