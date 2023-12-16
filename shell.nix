# Run with `nix-shell cuda-shell.nix`
{ pkgs ? import <nixos> {
   config = {
      allowUnfree = true;
      cudaSupport = true;
   };
} }:
pkgs.mkShell {
   name = "cuda-env-shell";
   buildInputs = with pkgs; [
     git gitRepo gnupg autoconf curl
     procps gnumake util-linux m4 gperf unzip
     cudatoolkit linuxPackages_6_1.nvidia_x11
     libGLU libGL
     xorg.libXi xorg.libXmu freeglut
     xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
     ncurses5 stdenv.cc binutils
     python310
     python310Packages.torchWithCuda
     python310Packages.numbaWithCuda
     python310Packages.numpy
     python310Packages.safetensors
     python310Packages.wandb
     python310Packages.chess
   ];
   shellHook = ''
      export CUDA_PATH=${pkgs.cudatoolkit}
      export LD_LIBRARY_PATH=${pkgs.linuxPackages_6_1.nvidia_x11}/lib:${pkgs.ncurses5}/lib:$LD_LIBRARY_PATH
      export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages_6_1.nvidia_x11}/lib"
      export EXTRA_CCFLAGS="-I/usr/include"
   '';
}
