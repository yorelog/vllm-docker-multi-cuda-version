#!/usr/bin/env python3
"""
Check if the vLLM wheel size is within the expected range.
This script is based on vLLM's wheel size checking mechanism.
"""

import os
import sys
import glob

def check_wheel_size(dist_dir, max_size_mb=500):
    """Check wheel size against the maximum allowed size."""
    wheel_files = glob.glob(os.path.join(dist_dir, "*.whl"))
    
    if not wheel_files:
        print("❌ No wheel files found in the dist directory!")
        return False
    
    total_size_mb = 0
    for wheel_file in wheel_files:
        if os.path.exists(wheel_file):
            size_bytes = os.path.getsize(wheel_file)
            size_mb = size_bytes / (1024 * 1024)
            total_size_mb += size_mb
            print(f"📦 {os.path.basename(wheel_file)}: {size_mb:.2f} MB")
    
    print(f"📊 Total wheel size: {total_size_mb:.2f} MB")
    print(f"📏 Maximum allowed size: {max_size_mb} MB")
    
    if total_size_mb > max_size_mb:
        print(f"❌ Wheel size ({total_size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb} MB)")
        return False
    else:
        print(f"✅ Wheel size is within the allowed range")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check-wheel-size.py <dist_directory>")
        sys.exit(1)
    
    dist_dir = sys.argv[1]
    max_size_mb = int(os.environ.get("VLLM_MAX_SIZE_MB", "500"))
    
    success = check_wheel_size(dist_dir, max_size_mb)
    sys.exit(0 if success else 1)
