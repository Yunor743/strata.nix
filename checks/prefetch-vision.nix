{
  stdenv,
  strataPkg,
}:

# strata-prefetch's vision side, exercised through the real wrapper (the dry-run plan touches
# no network and downloads nothing): the mmproj per family, the refusals, and the auto encoder
# choice (no nvidia-smi in the sandbox: the CPU encoder).
stdenv.mkDerivation {
  name = "strata-prefetch-vision";

  nativeBuildInputs = [ strataPkg ];

  buildCommand = ''
    export HOME=$NIX_BUILD_TOP

    ${strataPkg}/bin/strata-prefetch --dry-run --family qwen --model IQ2_XS \
      --data-dir $NIX_BUILD_TOP/data --vision gpu --vision-exe /bin/strata-vision > plan-qwen.txt
    grep -q "vision gpu" plan-qwen.txt
    grep -q "picture reader mmproj-Qwen3.8-Flash-Next-BF16.gguf" plan-qwen.txt
    grep -q "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF" plan-qwen.txt
    grep -q "sha256 b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0" plan-qwen.txt

    ${strataPkg}/bin/strata-prefetch --dry-run --family swift --model IQ2_XS \
      --data-dir $NIX_BUILD_TOP/data --vision cpu --vision-exe /bin/strata-vision > plan-swift.txt
    grep -q "vision cpu" plan-swift.txt
    grep -q "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf" plan-swift.txt

    ${strataPkg}/bin/strata-prefetch --dry-run --family coder --model IQ1_M \
      --data-dir $NIX_BUILD_TOP/data --vision auto --vision-exe /bin/strata-vision > plan-coder.txt
    grep -q "vision cpu" plan-coder.txt
    grep -q "no nvidia-smi" plan-coder.txt
    grep -q "mmproj-Qwen3.8-Flash-Next-BF16.gguf" plan-coder.txt

    ${strataPkg}/bin/strata-prefetch --dry-run --family qwen --model IQ2_XS \
      --data-dir $NIX_BUILD_TOP/data --vision off > plan-off.txt
    if grep -q "picture reader" plan-off.txt; then echo "vision off must fetch no picture reader"; exit 1; fi

    if ${strataPkg}/bin/strata-prefetch --dry-run --family abliterated --model Q2_0 \
      --data-dir $NIX_BUILD_TOP/data --vision gpu --vision-exe /bin/strata-vision > abliterated.out; then
      echo "abliterated with vision should be refused"; exit 1
    fi
    grep -q "no picture reader file is published" abliterated.out

    if ${strataPkg}/bin/strata-prefetch --dry-run --family qwen --model IQ2_XS \
      --data-dir $NIX_BUILD_TOP/data --vision gpu > noexe.out; then
      echo "vision without --vision-exe should be refused"; exit 1
    fi
    grep -q "needs --vision-exe" noexe.out

    touch $out
  '';
}
