# ============================================================================
# Optimized Makefile for Reduced VGG FPGA Accelerator
# Complete flow: csim → csynth → ip → bitstream
# ============================================================================

VITIS_HLS = vitis_hls
VIVADO = vivado

PROJECT = reduced_vgg.prj
SOLUTION = solution1
TOP_FUNC = reduced_vgg_inference

# Directories
IP_DIR = $(PROJECT)/$(SOLUTION)/impl/ip
EXPORT_DIR = $(PROJECT)/$(SOLUTION)/impl/export
SYN_REPORT = $(PROJECT)/$(SOLUTION)/syn/report

# Vivado project
VIVADO_PRJ = vivado_project
BITSTREAM = $(VIVADO_PRJ)/$(VIVADO_PRJ).runs/impl_1/design_1_wrapper.bit

.PHONY: all clean help csim csynth ip vivado bitstream reports

# ============================================================================
# MAIN TARGETS
# ============================================================================

all: ip
	@echo "✅ IP package created successfully!"
	@echo "   Location: $(IP_DIR)"

# ============================================================================
# HLS FLOW
# ============================================================================

csim:
	@echo "=========================================="
	@echo "Running C Simulation..."
	@echo "=========================================="
	$(VITIS_HLS) -f vitis_hls.tcl
	@echo ""
	@echo "✅ C Simulation Complete!"

csynth:
	@echo "=========================================="
	@echo "Running C Synthesis..."
	@echo "=========================================="
	$(VITIS_HLS) -f vitis_hls.tcl
	@echo ""
	@echo "✅ Synthesis Complete!"
	@echo ""
	@$(MAKE) -s reports

ip: csynth
	@echo ""
	@echo "=========================================="
	@echo "IP Package Created!"
	@echo "=========================================="
	@echo "Location: $(IP_DIR)"
	@echo ""
	@echo "To use in Vivado:"
	@echo "  1. Tools → Settings → IP → Repository"
	@echo "  2. Add: $(shell pwd)/$(IP_DIR)"
	@echo ""

# ============================================================================
# SYNTHESIS REPORTS
# ============================================================================

reports:
	@echo ""
	@echo "=========================================="
	@echo "📊 SYNTHESIS REPORTS"
	@echo "=========================================="
	@if [ -f "$(SYN_REPORT)/$(TOP_FUNC)_csynth.rpt" ]; then \
		echo ""; \
		echo "--- TIMING ---"; \
		grep -A 5 "Timing" $(SYN_REPORT)/$(TOP_FUNC)_csynth.rpt | head -10; \
		echo ""; \
		echo "--- LATENCY ---"; \
		grep -A 8 "Latency" $(SYN_REPORT)/$(TOP_FUNC)_csynth.rpt | head -15; \
		echo ""; \
		echo "--- RESOURCES ---"; \
		grep -A 12 "Utilization Estimates" $(SYN_REPORT)/$(TOP_FUNC)_csynth.rpt | tail -10; \
		echo ""; \
		echo "Full report: $(SYN_REPORT)/$(TOP_FUNC)_csynth.rpt"; \
	else \
		echo "⚠️  Synthesis report not found. Run 'make csynth' first."; \
	fi
	@echo "=========================================="

# ============================================================================
# UTILITY TARGETS
# ============================================================================

clean:
	@echo "Cleaning generated files..."
	rm -rf $(PROJECT)
	rm -rf $(VIVADO_PRJ)
	rm -rf *.log *.jou .Xil
	rm -rf vivado*.str
	@echo "✅ Clean complete"

help:
	@echo "=========================================="
	@echo "Reduced VGG FPGA Accelerator Makefile"
	@echo "=========================================="
	@echo ""
	@echo "HLS Flow:"
	@echo "  make csim      - Run C simulation only"
	@echo "  make csynth    - Run C synthesis and show reports"
	@echo "  make ip        - Generate IP package for Vivado (default)"
	@echo "  make reports   - Show synthesis reports"
	@echo ""
	@echo "Vivado Flow:"
	@echo "  make vivado    - Instructions for Vivado integration"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean     - Remove all generated files"
	@echo "  make help      - Show this message"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make csynth  - Synthesize and check reports"
	@echo "  2. make ip      - Export IP for Vivado"

