{
  description = "Strata: run Qwen3.8-Flash-Next (125B MoE) locally on one NVIDIA GPU — NixOS package and service";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    strata = {
      url = "github:Niko1221/Strata/main";
      flake = false;
    };
    llama-pinned = {
      url = "github:ggml-org/llama.cpp/3cf03257f219afbe7334045ff7c6a06ac68c627d";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, strata, llama-pinned } @ inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      version = builtins.head (builtins.match
        ".*project\\(strata VERSION ([0-9.]+).*"
        (builtins.readFile (strata + "/CMakeLists.txt")));

      ggmlPin = builtins.head (builtins.match "([0-9a-f]+).*"
        (builtins.readFile (strata + "/third_party/ggml/VERSION.txt")));

      pinChecked =
        if (llama-pinned.rev or "") == ggmlPin then true
        else throw ''
          Strata's source pins llama.cpp/ggml at ${ggmlPin}, but this flake locks llama-pinned at
          ${llama-pinned.rev or "an unknown revision"}.
          Fix: set inputs.llama-pinned.url = "github:ggml-org/llama.cpp/${ggmlPin}" in flake.nix,
          then run: nix flake lock --update-input llama-pinned'';

      strataPkgs = self.packages.${system};
    in
    {
      overlays.default = final: prev: {
        strata-engine = strataPkgs.strata-engine;
        strata-serve = strataPkgs.strata-serve;
        strata = strataPkgs.strata;
      };

      packages.${system} = rec {
        strata-engine = pkgs.callPackage ./pkgs/engine.nix {
          inherit version ggmlPin pinChecked;
          src = inputs.strata;
          llamaSrc = llama-pinned;
        };

        strata-serve = pkgs.callPackage ./pkgs/serve.nix {
          inherit version pinChecked;
          strataSrc = inputs.strata;
          llamaSrc = llama-pinned;
          engine = strata-engine;
        };

        strata = strata-serve;
        default = strata;
      };

      nixosModules.default = import ./nix/module.nix {
        strataPkgs = strataPkgs;
      };

      nixosModules.strata = self.nixosModules.default;

      checks.${system} = {
        serve-smoke = pkgs.callPackage ./checks/serve-smoke.nix {
          strataPkg = strataPkgs.strata;
        };
        module-vm = pkgs.callPackage ./checks/vm.nix {
          strataModule = self.nixosModules.default;
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ nixpkgs-fmt ];
      };
    };
}
