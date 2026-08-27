#!/usr/bin/env python3
"""
TECS test helper — cross-platform equivalent of test.sh

Configures, builds, and runs CTest tests for TECS with a configurable Kokkos backend.

Usage: test.py [-b build-dir] [-B backend] [-h help]
 -b build-dir    path to build directory (default: ./build/tests/[backend])
 -B backend      Kokkos backend to request (default: SERIAL)
 -h help         show this help
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure, build, and run TECS tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-b", "--build-dir",
        type=str,
        default=None,
        help="path to build directory (default: ./build/tests/[backend])",
    )
    parser.add_argument(
        "-B", "--backend",
        type=str,
        default="SERIAL",
        help="Kokkos backend to request (default: SERIAL)",
    )
    args = parser.parse_args()
    backend = args.backend

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir

    if args.build_dir is not None:
        build_dir = Path(args.build_dir)
    else:
        build_dir = Path(f"./build/tests/{backend}")

    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake = _find_cmake()

    print(f"Configuring tests in {build_dir} with backend {backend}...")
    subprocess.check_call([
        cmake, "-S", str(project_root), "-B", str(build_dir),
        "-DTECS_BUILD_TESTS=ON",
        f"-DKokkos_BACKEND={backend}",
    ])

    print("Building tests...")
    parallel_flag = _parallel_build_flag()
    subprocess.check_call([
        cmake, "--build", str(build_dir), "--", parallel_flag,
    ])

    print("Running tests...")
    subprocess.check_call(["ctest", "--output-on-failure"], cwd=str(build_dir))


def _find_cmake() -> str:
    """Locate cmake on the system."""
    cmake = shutil.which("cmake")
    if cmake is None:
        print("Error: cmake not found. Please install CMake and ensure it is on your PATH.", file=sys.stderr)
        sys.exit(1)
    return cmake


def _parallel_build_flag() -> str:
    """Return the parallel build flag for the current platform."""
    if platform.system() == "Windows":
        return "/maxcpucount"
    else:
        try:
            nprocs = os.cpu_count() or 1
        except Exception:
            nprocs = 1
        return f"-j{nprocs}"


if __name__ == "__main__":
    main()
