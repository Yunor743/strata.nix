{
  lib,
  cmake,
  ninja,
  patchelf,
  autoAddDriverRunpath,
  cudaPackages,
  src,
  llamaSrc,
  version,
  ggmlPin,
  pinChecked ? true,
  cudaArch ? "89",
}:

assert pinChecked;

cudaPackages.backendStdenv.mkDerivation {
  pname = "strata-vision";
  inherit version;
  strictDeps = true;

  src = llamaSrc;

  nativeBuildInputs = [
    cmake
    ninja
    patchelf
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ];

  buildInputs = [ cudaPackages.cudatoolkit ];

  # The pinned llama.cpp checkout is copied to a writable folder (some of its CMake glue wants to write next to
  # the sources); the build itself stays out of source.  No patch is applied: Strata's tools/vision builds
  # llama.cpp's mtmd at the same pinned commit the engine's ggml uses, as it is.
  # The CPU side keeps the portable AVX2 baseline (the CMakeLists' default): GGML_NATIVE=OFF with AVX2, FMA,
  # F16C and BMI2 on, so the binary runs on any x86-64 CPU and ggml's runtime dispatch picks AVX-512 where
  # the CPU has it.  STRATA_PORTABLE=OFF (the native build upstream's setup uses) would bake the build
  # machine's -march into the binary, which has no place in the nix store.
  # fixupPhase matters here: without it the CMake CUDA glue leaves the toolkit's stub `libcuda.so.1`
  # (cuda-merged/lib/stubs) in the binary's RUNPATH with nothing ahead of it, so ggml's dlopen("libcuda.so.1")
  # resolves the stub and CUDA init fails ("CUDA driver is a stub library").  postFixup then makes the real
  # driver's path the first RUNPATH entry for sure.
  phases = [ "unpackPhase" "patchPhase" "configurePhase" "buildPhase" "installPhase" "fixupPhase" ];

  postFixup = ''
    for f in $out/bin/*; do
      rpath=$(patchelf --print-rpath "$f" || true)
      case ":$rpath:" in
        *":/run/opengl-driver/lib:"*) ;;
        *) patchelf --set-rpath "/run/opengl-driver/lib''${rpath:+:$rpath}" "$f" ;;
      esac
    done
  '';

  unpackPhase = ''
    runHook preUnpack
    cp -rL ${llamaSrc} "$NIX_BUILD_TOP/llama"
    chmod -R u+w "$NIX_BUILD_TOP/llama"
    runHook postUnpack
  '';

  configurePhase = ''
    runHook preConfigure
    cd "$NIX_BUILD_TOP"
    cmake -S ${src}/tools/vision -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_DIR=$NIX_BUILD_TOP/llama \
      -DSTRATA_VISION_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${cudaArch} \
      -DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc
    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild
    ninja -j "$NIX_BUILD_CORES" -C "$NIX_BUILD_TOP/build" strata-vision
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 "$NIX_BUILD_TOP/build/bin/strata-vision" $out/bin/strata-vision
    runHook postInstall
  '';

  env = {
    PIN_ASSERT = toString pinChecked;
    GGML_PIN_ASSERT = ggmlPin;
    CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_INCLUDE_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_LIBRARY_ROOT = cudaPackages.cudatoolkit;
  };

  passthru = {
    inherit cudaArch;
  };

  meta = {
    description = "Strata image encoder (llama.cpp's mtmd with the Qwen3.8-Flash-Next vision tower, sm_${cudaArch})";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata-vision";
  };
}
