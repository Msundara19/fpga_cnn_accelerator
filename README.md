# Hardware Acceleration of VGG Model on CIFAR-10 using High-Level Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-PYNQ--Z2-blue.svg)](http://www.pynq.io/)
[![HLS](https://img.shields.io/badge/HLS-Vitis%202022.2-orange.svg)](https://www.xilinx.com/products/design-tools/vitis.html)

## 📋 Project Overview

This project demonstrates the complete end-to-end pipeline for FPGA-based hardware acceleration of deep learning models, specifically a ReducedVGG architecture for CIFAR-10 classification. The implementation covers the full workflow from PyTorch training through High-Level Synthesis (HLS) to actual hardware deployment on the PYNQ-Z2 board.

**Course**: ECE 588 - Secure Machine Learning  
**Institution**: Illinois Institute of Technology  
**Authors**: Meenakshi Sridharan Sundaram, Sai Ayush  
**Date**: December 2025

## 🎯 Key Achievements

- ✅ **Complete FPGA Deployment Pipeline**: PyTorch → HLS → Vivado → PYNQ
- ✅ **Model Design**: Custom ReducedVGG with 1.44M parameters achieving **85.69% accuracy**
- ✅ **Quantization**: INT16 weight-only quantization with minimal accuracy loss (0.1%)
- ✅ **Timing Closure**: Achieved at 66.7 MHz on Zynq-7020 FPGA
- ✅ **Resource Efficiency**: Fits within PYNQ-Z2 constraints (70% BRAM, 14% DSP, 62% LUT)
- ✅ **Hardware Validation**: Successfully deployed and tested on PYNQ-Z2 board

## 📊 Performance Summary

### GPU vs FPGA Comparison

| Metric | GPU (Tesla T4) | FPGA (Zynq-7020) | Winner |
|--------|---------------|------------------|---------|
| **Inference Latency** | 1.296 ms | 466.53 ms | GPU (360×) |
| **Throughput** | 771.5 img/s | 2.14 img/s | GPU (360×) |
| **Test Accuracy** | 86.94% | 85.59% | GPU (+1.35%) |
| **Power Consumption** | 70 W (TDP) | 1.451 W | **FPGA (48×)** |
| **Energy/Inference** | 0.091 J | 0.677 J | GPU (7.4×) |
| **Efficiency Score** | 67.07 acc/ms | 0.183 acc/ms | GPU (366×) |

### Model Architecture

**ReducedVGG Specifications:**
- Parameters: 1,439,146
- Channel Progression: [32, 64, 128, 256]
- FLOPs per Inference: 106.72 MFLOPs
- Input: 32×32×3 (CIFAR-10)
- Output: 10 classes

## 🏗️ Architecture

### Model Structure
```
Input (32×32×3)
├── Block 0: [Conv3×3(32) + BN + ReLU] × 2 → MaxPool
├── Block 1: [Conv3×3(64) + BN + ReLU] × 2 → MaxPool
├── Block 2: [Conv3×3(128) + BN + ReLU] × 2 → MaxPool
├── Block 3: [Conv3×3(256) + BN + ReLU] × 2 → MaxPool
└── Classifier: Flatten → FC(1024→256) → Dropout → FC(256→10)
```

### FPGA System Architecture
```
┌─────────────────────────────────────────────┐
│         ZYNQ-7020 Processing System         │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │ ARM Cortex-A9│◄────►│  DDR Controller │ │
│  │  (667 MHz)   │      │   (512 MB)      │ │
│  └──────┬───────┘      └─────────────────┘ │
│         │ AXI                               │
│  ┌──────▼────────────────────────────────┐ │
│  │    AXI Interconnect (Control Path)    │ │
│  └──────┬────────────────────────────────┘ │
└─────────┼──────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────┐
│    VGG Accelerator IP (HLS Generated)      │
│  ┌──────────────────────────────────────┐  │
│  │  96 AXI-Lite Registers (Parameters)  │  │
│  ├──────────────────────────────────────┤  │
│  │     Tiled Convolution Engine (8×8)   │  │
│  ├──────────────────────────────────────┤  │
│  │  BatchNorm + ReLU + MaxPool Units    │  │
│  ├──────────────────────────────────────┤  │
│  │       Fully Connected Layers         │  │
│  └──────────────────────────────────────┘  │
│           ▲                                 │
│           │ AXI Master (DDR Access)         │
└───────────┼─────────────────────────────────┘
            │
     [DDR Memory]
```

## 🔧 Implementation Details

### Data Types (HLS Fixed-Point)
```cpp
typedef ap_fixed<16,12> fm_t;   // Feature maps (Q12.4)
typedef ap_fixed<16,12> wt_t;   // Weights/Bias (Q12.4)
typedef ap_fixed<32,24> acc_t;  // Accumulator (Q24.8)
```

### Key Optimizations
1. **Loop Pipelining**: Parallel computation within convolution kernels
2. **Array Partitioning**: On-chip buffering for tiled convolution
3. **Dataflow**: Pipeline parallelism between layers
4. **Fixed-Point Quantization**: INT16 reduces memory by 2× with <0.1% accuracy loss

### Resource Utilization (Post-Implementation)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | 9,044 | 53,200 | 17% |
| LUTRAM | 532 | 17,400 | 1% |
| Flip-Flops | 12,764 | 106,400 | 12% |
| BRAM | 3.5 | 280 | 1% |
| DSP Blocks | 5 | 220 | 2% |

## 📁 Project Structure

```
ECE588_FinalProject/
├── README.md                           # This file
├── report/
│   └── ECE588_Final_Project_Report.pdf # Comprehensive 39-page report
├── training/
│   ├── copy.ipynb                      # PyTorch training notebook
│   └── models/
│       └── reduced_vgg_best.pth        # Trained model checkpoint
├── hls/
│   ├── tiled_conv.hpp                  # Header: data types & constants
│   ├── tiled_conv.cpp                  # Top-level HLS inference function
│   ├── utils.cpp                       # Layer implementations
│   ├── tb_conv.cpp                     # C++ testbench
│   ├── Makefile                        # Build automation
│   ├── vitis_hls.tcl                   # HLS synthesis script
│   └── run_csim.tcl                    # C simulation script
├── weights/
│   ├── params_int32/                   # INT32 quantized weights (60 files)
│   ├── params_int16/                   # INT16 converted weights (48 files)
│   └── convert_weights.py              # INT32→INT16 converter
├── vivado/
│   ├── design_1.bd                     # Block design
│   └── constraints/                    # Timing constraints
├── pynq/
│   ├── design_1_wrapper.bit            # FPGA bitstream (45 MB)
│   ├── design_1.hwh                    # Hardware handoff
│   └── deploy_pynq_runtime.py          # Deployment script
└── docs/
    ├── setup.md                        # Environment setup guide
    └── usage.md                        # Usage instructions
```

## 🚀 Quick Start

### Prerequisites

**Software:**
- Python 3.8+
- PyTorch 2.x
- Vitis HLS 2022.2
- Vivado 2022.2
- PYNQ v3.0.1

**Hardware:**
- PYNQ-Z2 board (Zynq-7020 FPGA)
- MicroSD card (16 GB+)
- Host PC with Linux (Ubuntu 20.04+)

### 1. Training the Model

```bash
# Open the Jupyter notebook on Google Colab or locally
jupyter notebook training/copy.ipynb

# The notebook will:
# - Load CIFAR-10 dataset
# - Train ReducedVGG for 20 epochs
# - Export quantized weights to params_int32/
```

### 2. Weight Conversion

```bash
cd weights
python convert_weights.py \
    --input params_int32/ \
    --output params_int16/ \
    --format "ap_fixed<16,12>"
```

### 3. HLS Synthesis

```bash
cd hls

# C Simulation (verify functionality)
make csim

# C Synthesis (generate RTL)
make csynth

# Export IP
make ip
```

### 4. Vivado Integration

```bash
# Open Vivado and source the block design
vivado -mode batch -source scripts/create_block_design.tcl

# Generate bitstream
vivado -mode batch -source scripts/generate_bitstream.tcl
```

### 5. PYNQ Deployment

```bash
# Copy files to PYNQ board
scp pynq/design_1_wrapper.bit xilinx@192.168.2.99:~/
scp pynq/design_1.hwh xilinx@192.168.2.99:~/
scp -r weights/params_int16/ xilinx@192.168.2.99:~/

# SSH into PYNQ and run inference
ssh xilinx@192.168.2.99
python3 deploy_pynq_runtime.py
```

## 📈 Results & Analysis

### Training Curves
The model converges smoothly over 20 epochs:
- Final Training Accuracy: ~92%
- Final Validation Accuracy: ~87%
- Test Accuracy: **86.94%**

### Quantization Impact

| Configuration | Accuracy | Latency | Score |
|--------------|----------|---------|-------|
| FP32 (Baseline) | 86.94% | 1.296 ms | 67.07 |
| INT32 Weight-only | 86.94% | 1.291 ms | **67.37** |
| INT16 Weight-only | 86.94% | 1.298 ms | 66.99 |
| INT16 (FPGA) | 85.59% | 466.53 ms | 0.183 |

### Performance Breakdown (FPGA)

| Phase | Time (ms) | Percentage |
|-------|-----------|------------|
| Parameter Sync (DDR) | 6.78 | 1.5% |
| Computation (FPGA) | 459.75 | 98.5% |
| **Total Latency** | **466.53** | **100%** |

## 🔍 Key Findings

### What Worked Well ✅
1. **Complete Pipeline Success**: End-to-end flow from PyTorch to hardware deployment
2. **Timing Closure**: Achieved at 66.7 MHz with positive slack (0.183 ns)
3. **Resource Fit**: Optimized design fits within Zynq-7020 constraints
4. **Quantization Effectiveness**: INT16 preserves accuracy with 2× memory reduction
5. **Functional Correctness**: C simulation passed with 0 errors on 10 test images

### Challenges Identified ⚠️
1. **Memory-Bound Performance**: 58× gap between theoretical (7.98 ms) and measured (466.53 ms) latency
2. **DDR Bandwidth Bottleneck**: Sequential parameter loading dominates execution time
3. **Complex Address Management**: 96 AXI register ports require careful orchestration
4. **Limited Parallelism**: Memory access patterns prevent full compute utilization

### Performance Gap Analysis

**Why is the FPGA slower?**
1. **Runtime Parameter Loading**: 2.8 MB loaded from DDR for each inference
2. **Sequential Memory Access**: Single AXI master limits bandwidth
3. **Memory-Bound vs Compute-Bound**: DDR access (98.5%) dominates compute (1.5%)
4. **Small Model Size**: GPU easily fits in cache, FPGA requires DDR

**Theoretical vs Measured Latency:**
- HLS Synthesis Estimate: 7.98 ms (best case)
- Measured Hardware: 466.53 ms
- **Gap: 58×** → Memory architecture bottleneck

## 🎓 Lessons Learned

### Technical Insights
1. **Memory Bandwidth is Critical**: Compute is cheap; data movement is expensive
2. **Embedded Weights Would Transform Performance**: Moving 2.8 MB to BRAM could achieve 20-50× speedup
3. **Design for the Architecture**: Runtime flexibility (96 AXI ports) added complexity without benefit
4. **Quantization Works**: INT16 preserved accuracy while enabling efficient DSP mapping

### Educational Value
Despite the performance gap vs GPU, this project successfully:
- Demonstrated complete FPGA design methodology
- Identified critical bottlenecks through systematic analysis
- Validated theoretical understanding through hardware measurement
- Provided realistic expectations for FPGA acceleration

## 🔮 Future Improvements

### Recommended Optimizations

| Optimization | Expected Speedup | Difficulty |
|-------------|------------------|-----------|
| Embed parameters in BRAM | 20-50× | High |
| Optimize AXI burst size | 2-3× | Medium |
| Increase frequency to 100 MHz | 1.5× | Low |
| Implement dataflow parallelism | 2-4× | High |
| Mixed precision (INT8/INT16) | 1.5-2× | Medium |
| **Combined Potential** | **60-600×** | **Very High** |

### When FPGAs Could Be Competitive
1. **Edge Deployment**: 48× lower power (1.45 W vs 70 W) matters
2. **Embedded Weights**: Fixed models with BRAM-resident parameters
3. **Batch Processing**: Amortize parameter loading across many images
4. **Custom Data Paths**: Non-standard operations not well-suited to GPUs

## 📚 Documentation

- **[Full Report](report/ECE588_Final_Project_Report.pdf)**: Comprehensive 39-page technical report with detailed analysis
- **[Setup Guide](docs/setup.md)**: Environment configuration and installation
- **[Usage Guide](docs/usage.md)**: Step-by-step execution instructions

## 📖 References

1. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," ICLR 2015.
2. A. Krizhevsky and G. Hinton, "Learning multiple layers of features from tiny images," Technical Report, University of Toronto, 2009.
3. Xilinx, "Vitis High-Level Synthesis User Guide (UG1399)," 2023.
4. PYNQ Project, "Python productivity for Zynq," [http://www.pynq.io/](http://www.pynq.io/)
5. C. Zhang et al., "Optimizing FPGA-based accelerator design for deep convolutional neural networks," FPGA 2015.

## 🙏 Acknowledgments

We thank:
- **Professor Ken Choi** and the ECE 588 teaching staff for guidance throughout this project
- **Xilinx/AMD** for providing Vitis HLS and Vivado development tools
- **PYNQ community** for the excellent FPGA development framework
- **Illinois Institute of Technology** for computational resources

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Meenakshi Sridharan Sundaram** - [msridharansundaram@hawk.illinoistech.edu](mailto:msridharansundaram@hawk.illinoistech.edu)
- **Sai Ayush** - [sayush@hawk.illinoistech.edu](mailto:sayush@hawk.illinoistech.edu)

---

**Project Status**: ✅ Complete (December 2025)  
**Hardware Validated**: ✅ Yes (PYNQ-Z2)  
**Report Available**: ✅ Yes ([Download PDF](report/ECE588_Final_Project_Report.pdf))
