{
  lib,
  stdenv,
  cmake,
  ninja,
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
  pname = "strata-engine";
  inherit version;
  inherit src;
  strictDeps = true;

  nativeBuildInputs = [
    cmake
    ninja
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ];

  buildInputs = [ cudaPackages.cudatoolkit ];

  cmakeFlags = [
    "-DCMAKE_GENERATOR=Ninja"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DSTRATA_ENABLE_CUDA=ON"
    "-DSTRATA_BUILD_TESTS=OFF"
    "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
    "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
    "-DSTRATA_GGML_DIR=${llamaSrc}"
    "-DGGML_NATIVE=OFF"
    "-DGGML_AVX=ON"
    "-DGGML_AVX2=ON"
    "-DGGML_FMA=ON"
    "-DGGML_F16C=ON"
    "-DGGML_BMI2=ON"
    "-DGGML_AVX512=ON"
  ];

  env = {
    PIN_ASSERT = toString pinChecked;
    GGML_PIN_ASSERT = ggmlPin;
    CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_INCLUDE_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_LIBRARY_ROOT = cudaPackages.cudatoolkit;
  };

  buildFlags = [ "--target" "strata" "--target" "strata-device" ];

  installPhase = ''
    runHook preInstall
    bin=$(find "$NIX_BUILD_TOP" -maxdepth 4 -type f -name strata | head -1)
    dev=$(find "$NIX_BUILD_TOP" -maxdepth 4 -type f -name strata-device | head -1)
    if [[ -z "$bin" ]]; then
      echo "strata binary not found under $NIX_BUILD_TOP" >&2
      exit 1
    fi
    install -Dm755 "$bin" $out/bin/strata
    [[ -n "$dev" ]] && install -Dm755 "$dev" $out/bin/strata-device
    cat > $out/bin/BUILD.json <<EOF
{
  "version": "${version}",
  "archs": [${cudaArch}],
  "ptx": false,
  "source": "nix",
  "vision": "none"
}
EOF
    runHook postInstall
  '';

  passthru = {
    inherit cudaArch;
  };

  meta = {
    description = "Strata inference engine for Qwen3.8-Flash-Next (125B MoE) on a single NVIDIA GPU (sm_${cudaArch})";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata";
  };
}
