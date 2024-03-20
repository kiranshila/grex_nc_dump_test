{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = { nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        nativeBuildInputs = with pkgs; [ rustPlatform.bindgenHook pkg-config ];
        buildInputs = with pkgs; [
          # Rust stuff, some stuff dev-only
          (rust-bin.nightly.latest.default.override {
            extensions = [ "rust-src" "rust-analyzer" ];
          })

          # The C-libraries needed to statically link
          netcdf
          hdf5

          # Linting support
          alejandra
        ];
      in with pkgs; {
        devShells.default = mkShell { inherit buildInputs nativeBuildInputs; };
      });
}
