"""
Run all examples from the examples/ directory using tecs.sh

Usage:
    python run_all_examples.py [BACKEND] [START]
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKENDS = ["SERIAL", "OPENMP", "THREADS", "HPX", "CUDA", "HIP", "SYCL", "OPENMPTARGET", "OPENACC"]

def main():
    backend = sys.argv[1] if len(sys.argv) > 1 else "SERIAL"
    if backend not in BACKENDS:
        print(f"Unknown backend '{backend}'. Choose from: {', '.join(BACKENDS)}", file=sys.stderr)
        sys.exit(1)

    start_from = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    examples = sorted(Path(SCRIPT_DIR / "examples").rglob("*.cpp"))

    if not examples:
        print("No examples found.", file=sys.stderr)
        sys.exit(1)
    
    if start_from < 1 or start_from > len(examples):
        print(f"Start number must be between 1 and {len(examples)}.", file=sys.stderr)
        sys.exit(1)

    print(f"Backend: {backend}")
    print(f"Examples: {len(examples)}")
    print(f"Starting from: {start_from}")

    for i, ex in enumerate(examples, start=1):
        if i < start_from:
            continue
        rel = ex.relative_to(SCRIPT_DIR / "examples")
        cmd = ["python3", str(SCRIPT_DIR / "tecs.py"), str(ex), "-B", backend, "-e"]
        print(f"\n[{i}/{len(examples)}] --- {rel} ---")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if result.returncode != 0:
            print(f"FAILED: {rel}")
            sys.exit(1)
    print("\nAll examples passed.")

if __name__ == "__main__":
    main()
