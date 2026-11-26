PYTHON ?= python3
NVCC ?= nvcc
CUDA_ARCH ?= sm_86

.PHONY: help test test-3mode bench-cpu lean-verify lean-test cuda-build check-zero-spill audit-hygiene clean

help:
	@echo "HoloSphere targets:"
	@echo "  make test          Run CPU unit test suite (CUDA_VISIBLE_DEVICES='')"
	@echo "  make test-3mode    Run 3-mode isolated -I parity (-I -B, -I -O -B, -I -B -W error)"
	@echo "  make bench-cpu     Run PREREG-S114 CPU preflight parity & gradcheck benchmark"
	@echo "  make lean-verify   Verify pure-Init Lean 4.36.0 suite (5 modules, 51 theorems, 0 sorryAx)"
	@echo "  make lean-test     Run 3-mode isolated -I Lean R5 verification test suite"
	@echo "  make cuda-build       Compile all bare-metal sm_86 CUDA kernels (printing ptxas stats)"
	@echo "  make check-zero-spill Verify sm_86 ptxas zero-spill receipts and cu_sha256 source bindings"
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
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B lean/verify_holosphere_lean.py --verify-all-lean --json

lean-test:
	@set -e; \
	for flags in "-I -B" "-I -O -B" "-I -B -W error"; do \
		echo "=== [lean $$flags] ==="; \
		CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$flags lean/test_holosphere_lean_r5.py; \
	done

cuda-build:
	@set -e; \
	for pair in \
		"su2_holonomy_scan_sm86.cu:libsu2_holonomy_sm86.so" \
		"holosphere_h256_chunk32_sm86.cu:libholosphere_h256_chunk32_sm86.so" \
		"holosphere_v3_frontier_sm86.cu:libholosphere_v3_frontier_sm86.so"; do \
		cu="holosphere/kernels/csrc/$${pair%%:*}"; \
		so="holosphere/kernels/csrc/$${pair##*:}"; \
		if [ -f "$$cu" ]; then \
			$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) --ptxas-options=-v -Iinclude -Xcompiler -fPIC -shared "$$cu" -o "$$so"; \
		fi; \
	done

check-zero-spill:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -c "import hashlib, json, pathlib, sys; root = pathlib.Path('holosphere/kernels/csrc'); [sys.exit(f'FAIL {r}') if (d := json.loads((root / r).read_text('utf-8')))['nonzero_spill_kernels'] != 0 or hashlib.sha256((root / d['cu_filename']).read_bytes()).hexdigest() != d['cu_sha256'] else print('OK', r, d['total_kernels'], 'kernels') for r in ('ptxas_s117_zero_spill_receipt.json', 'ptxas_v3_frontier_zero_spill_receipt.json')]"

audit-hygiene:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B tests/test_repo_hygiene.py -v

clean:
	rm -rf build dist *.egg-info holosphere/kernels/csrc/*.so holosphere/kernels/csrc/*.o
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
