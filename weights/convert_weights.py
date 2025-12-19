#!/usr/bin/env python3
"""
Convert INT32 quantized weights to ap_fixed<16,12> format for HLS
Reads from params_int32/ and writes to params/
"""
import struct
import os
from pathlib import Path
import numpy as np

def read_int32_bin(filename):
    """Read INT32 binary file"""
    with open(filename, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.int32)

def write_fixed16_bin(filename, data):
    """Write 16-bit fixed-point binary file"""
    with open(filename, 'wb') as f:
        f.write(data.tobytes())

def convert_int32_to_fixed16_12(int32_data):
    """
    Convert INT32 quantized values to ap_fixed<16,12> format
    
    INT32 uses scale of 65536 (2^16)
    ap_fixed<16,12> needs scale of 16 (2^4 fractional bits)
    
    Conversion: int32_value / 65536.0 * 16 = fixed16_value
    Or: int32_value / 4096
    """
    # Convert to float
    fp32_data = int32_data / 65536.0
    
    # Scale to ap_fixed<16,12> range (multiply by 2^4 = 16)
    # This gives us 4 fractional bits
    fixed_data = np.round(fp32_data * 16.0)
    
    # Clamp to 16-bit signed range
    fixed_data = np.clip(fixed_data, -32768, 32767)
    
    # Convert to int16
    return fixed_data.astype(np.int16)

# Create output directory
os.makedirs("params", exist_ok=True)
print("=== Converting INT32 Quantized Weights to ap_fixed<16,12> ===\n")

# Get all .bin files from params_int32
params_int32_dir = Path("params_int32")
bin_files = sorted(params_int32_dir.glob("*.bin"))

converted_count = 0
total_overflow = 0

for bin_file in bin_files:
    input_path = str(bin_file)
    output_path = f"params/{bin_file.name}"
    
    # Read INT32 data
    int32_data = read_int32_bin(input_path)
    
    # Convert to ap_fixed<16,12>
    fixed16_data = convert_int32_to_fixed16_12(int32_data)
    
    # Check for potential overflow
    fp32_check = int32_data / 65536.0
    overflow_count = np.sum(np.abs(fp32_check) > 2048)
    if overflow_count > 0:
        print(f"  {bin_file.name}: {overflow_count} values may overflow ap_fixed<16,12> range")
        total_overflow += overflow_count
    
    # Write fixed16 data
    write_fixed16_bin(output_path, fixed16_data)
    
    converted_count += 1
    
    # Print progress every 10 files
    if converted_count % 10 == 0:
        print(f"Converted {converted_count} files...")

print(f"\nTotal files converted: {converted_count}")
print(f"Output directory: params/")
if total_overflow > 0:
    print(f"  Total overflow values: {total_overflow}")
    print(f"   (These were clamped to fit ap_fixed<16,12> range)")
else:
    print(f"No overflow detected - all values fit in ap_fixed<16,12>")
print("\nConversion complete!")
