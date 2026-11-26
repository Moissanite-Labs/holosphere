# SPDX-License-Identifier: Apache-2.0
# Standalone Ampere sm_86 build & verification targets for the fused
# HoloSphere-V4 zero-HBM-input & zero-HBM-output-vocab megakernel (Refs #2).

NVCC   ?= nvcc
ARCH   ?= sm_86
PYTHON ?= python3
BUILD  ?= build

CUDA_FLAGS ?= -O3 -std=c++17 -arch=$(ARCH) --ptxas-options=-v --Werror all-warnings

.PHONY: help cuda-v4-monarch2 check-v4-monarch2-zero-spill verify-v4-monarch2

help:
	@echo "Targets:"
	@echo "  verify-v4-monarch2            Run 3x isolated -I CPU receipt, oracle, and R5 tamper suite"
	@echo "  cuda-v4-monarch2              Compile sm_86 shared library under --Werror all-warnings"
	@echo "  check-v4-monarch2-zero-spill  Verify 2/2 kernels have 0 stack/spill bytes via ptxas"

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/libholosphere_v4_monarch2_e2e_sm86.so: cuda/v4_monarch2/holosphere_v4_monarch2_e2e_sm86.cu | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) --shared -Xcompiler -fPIC,-Wall,-Wextra,-Werror $< -o $@

cuda-v4-monarch2: $(BUILD)/libholosphere_v4_monarch2_e2e_sm86.so

check-v4-monarch2-zero-spill: | $(BUILD)
	@set -eu; \
	LOG=$(BUILD)/v4_monarch2_ptxas_spill.log; \
	trap 'rm -f "$$LOG"' EXIT; \
	$(MAKE) -f cuda/v4_monarch2/holosphere_v4_monarch2.mk -B cuda-v4-monarch2 >"$$LOG" 2>&1 || { cat "$$LOG"; exit 1; }; \
	if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "$$LOG"; then \
		echo "ERROR: non-zero register/stack spill detected" >&2; \
		exit 1; \
	fi; \
	COUNT=$$(grep -c '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' "$$LOG"); \
	if [ "$$COUNT" -ne 2 ]; then \
		echo "ERROR: expected 2 zero-spill kernels, got $$COUNT" >&2; \
		exit 1; \
	fi; \
	echo "HOLOSPHERE_V4_MONARCH2_ZERO_SPILL_VERIFIED=1 ($$COUNT kernels)"

verify-v4-monarch2:
	@set -eu; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags cuda/v4_monarch2/verify_holosphere_v4_monarch2.py --check --require-verdict CONFIRMED --json; \
		CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags tests/test_holosphere_v4_monarch2_r5.py; \
	done
