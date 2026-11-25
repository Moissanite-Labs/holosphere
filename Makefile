PYTHON ?= python3
NVCC ?= nvcc
CUDA_ARCH ?= sm_86

.PHONY: help test test-3mode bench-cpu lean-verify lean-test cuda-build audit-hygiene clean

help:
	@echo "HoloSphere targets:"
	@echo "  make test          Run CPU unit test suite (CUDA_VISIBLE_DEVICES='')"
	@echo "  make test-3mode    Run 3-mode isolated -I parity (-I -B, -I -O -B, -I -B -W error)"
	@echo "  make bench-cpu     Run PREREG-S114 CPU preflight parity & gradcheck benchmark"
	@echo "  make lean-verify   Verify pure-Init Lean 4.36.0 certificate (36 theorems, 0 sorryAx)"
	@echo "  make lean-test     Run 3-mode isolated -I Lean R5 verification test suite"
	@echo "  make cuda-build    Compile bare-metal sm_86 CUDA kernel (verifying 0 B register spill)"
	@echo "  make audit-hygiene Verify 0 bare asserts, 0 /home/ leaks, 0 weights, 0 __pycache__"
	@echo "  make clean         Remove build artifacts and bytecode"

test:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -B -m unittest discover -s tests -p "test_*.py" -v

test-3mode:
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		echo "=== [$$flags] ==="; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags tests/run_all.py; \
	done

bench-cpu:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B holosphere/benchmarks/prereg_s114_suite.py --cpu-preflight --json

lean-verify:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B lean/verify_holosphere_lean.py --check --json

lean-test:
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		echo "=== [lean $$flags] ==="; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags lean/test_holosphere_lean_r5.py; \
	done

cuda-build:
	@if [ ! -f holosphere/kernels/csrc/su2_holonomy_scan_sm86.cu ]; then \
		echo "NOTE: su2_holonomy_scan_sm86.cu not yet staged in holosphere/kernels/csrc/"; \
		exit 0; \
	fi
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) --ptxas-options=-v -Xcompiler -fPIC -shared \
		holosphere/kernels/csrc/su2_holonomy_scan_sm86.cu \
		-o holosphere/kernels/csrc/libsu2_holonomy_sm86.so

audit-hygiene:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B tests/test_repo_hygiene.py -v

clean:
	rm -rf build dist *.egg-info holosphere/kernels/csrc/*.so holosphere/kernels/csrc/*.o
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
