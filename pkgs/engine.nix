{
  lib,
  stdenv,
  cmake,
  ninja,
  autoAddDriverRunpath,
  cudaPackages,
  rocmPackages,
  src,
  llamaSrc,
  version,
  ggmlPin,
  pinChecked ? true,
  # "cuda" (NVIDIA) or "hip" (AMD RDNA, wave32)
  backend ? "cuda",
  cudaArch ? "89",
  hipArch ? "gfx1100",
}:

assert pinChecked;
assert lib.assertOneOf "backend" backend [ "cuda" "hip" ];

let
  hip = backend == "hip";
  arch = if hip then hipArch else cudaArch;
  hipblaslt = lib.getDev rocmPackages.hipblaslt;

  backendStdenv = if hip then stdenv else cudaPackages.backendStdenv;

  buildInfo = builtins.toJSON {
    inherit version backend;
    archs = [ (if hip then hipArch else lib.toInt cudaArch) ];
    ptx = false;
    source = "nix";
    vision = "none";
  };

  backendFlags =
    if hip then
      [
        "-DSTRATA_ENABLE_HIP=ON"
        "-DSTRATA_ENABLE_CUDA=OFF"
        "-DSTRATA_PREFILL_MMQ=ON"
        "-DSTRATA_MMQ_KQUANTS=ON"
        "-DCMAKE_HIP_ARCHITECTURES=${hipArch}"
        "-DCMAKE_HIP_COMPILER=${rocmPackages.clr.hipClangPath}/clang++"
      ]
    else
      [
        "-DSTRATA_ENABLE_CUDA=ON"
        "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
        "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
      ];
in
backendStdenv.mkDerivation {
  pname = "strata-engine" + lib.optionalString hip "-hip";
  inherit version;
  inherit src;
  strictDeps = true;

  nativeBuildInputs = [
    cmake
    ninja
  ]
  ++ lib.optionals (!hip) [
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ];

  buildInputs =
    if hip then
      with rocmPackages;
      [
        clr
        hipblas
        hipblas-common
        rocblas
        hipblaslt
      ]
    else
      [ cudaPackages.cudatoolkit ];

  cmakeFlags = [
    "-DCMAKE_GENERATOR=Ninja"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DSTRATA_BUILD_TESTS=OFF"
    "-DSTRATA_GGML_DIR=${llamaSrc}"
    "-DGGML_NATIVE=OFF"
    "-DGGML_AVX=ON"
    "-DGGML_AVX2=ON"
    "-DGGML_FMA=ON"
    "-DGGML_F16C=ON"
    "-DGGML_BMI2=ON"
    "-DGGML_AVX512=ON"
  ]
  ++ backendFlags;

  env = {
    PIN_ASSERT = toString pinChecked;
    GGML_PIN_ASSERT = ggmlPin;
  }
  // lib.optionalAttrs (!hip) {
    CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_INCLUDE_ROOT = cudaPackages.cudatoolkit;
    CUDAToolkit_LIBRARY_ROOT = cudaPackages.cudatoolkit;
  };

  # The engine accepts a hipBLASLt tuning table only for its exact card and library version; fail before the
  # long compile when upstream ships none for this pair instead of silently running the untuned prompt path.
  preConfigure = lib.optionalString hip ''
    header=${hipblaslt}/include/hipblaslt/hipblaslt-version.h
    [[ -f $header ]] || { echo "missing $header" >&2; exit 1; }
    ver() { sed -n "s/^#define HIPBLASLT_VERSION_$1 *\([0-9]*\).*/\1/p" "$header"; }
    hipblasltVersion=$(( $(ver MAJOR) * 100000 + $(ver MINOR) * 100 + $(ver PATCH) ))
    tuningTable=$PWD/tools/hip/${hipArch}-hipblaslt-$hipblasltVersion.txt
    if [[ ! -f $tuningTable ]]; then
      echo "no hipBLASLt tuning table for ${hipArch} with hipBLASLt $hipblasltVersion; Strata ships:" >&2
      ls tools/hip/${hipArch}-hipblaslt-*.txt >&2 || true
      exit 1
    fi
  '';

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
  ''
  + lib.optionalString hip ''
    install -Dm644 "$tuningTable" $out/share/strata/hipblaslt-tuning.txt
  ''
  + ''
    echo '${buildInfo}' > $out/bin/BUILD.json
    runHook postInstall
  '';

  passthru = {
    inherit backend arch;
  }
  // lib.optionalAttrs hip {
    hipblasltTuning = "share/strata/hipblaslt-tuning.txt";
  };

  meta = {
    description =
      "Strata inference engine for Qwen3.8-Flash-Next (125B MoE) on a single "
      + (if hip then "AMD GPU (${hipArch})" else "NVIDIA GPU (sm_${cudaArch})");
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata";
  };
}
