#!/usr/bin/env python3
"""
Create test input and golden output files for HLS testbench
"""

import struct
import random

# Create random test input (32x32x3)
input_size = 32 * 32 * 3
input_data = [random.random() for _ in range(input_size)]

# Create dummy golden output (10 classes)
output_size = 10
output_data = [random.random() for _ in range(output_size)]

# Write binary files
with open('params/input_fp32.bin', 'wb') as f:
    f.write(struct.pack(f'{input_size}f', *input_data))

with open('params/output_golden_fp32.bin', 'wb') as f:
    f.write(struct.pack(f'{output_size}f', *output_data))

print("Created test files:")
print("  params/input_fp32.bin")
print("  params/output_golden_fp32.bin")


