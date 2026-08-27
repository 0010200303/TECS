#!/usr/bin/env python3
"""
TECS build/run helper

Usage: tecs.py user-main [-b build-dir] [-o output] [-B backend] [-d] [-e] [-t] [-h]
"""

from __future__ import annotations
import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build (and optionally run) a TECS simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tecs.py examples/tribolium/main.cpp\n"
            "  tecs.py examples/tribolium/main.cpp -B OPENMP -e -t\n"
            "  tecs.py my_sim.cpp -b build/custom -o my_sim -d\n"
        ),
    )
    parser.add_argument(
        "user_main",
        metavar="user-main",
        type=str,
        help="path to user main source file (required)",
    )
    parser.add_argument(
        "-b", "--build-dir",
        type=str,
        default=None,
        help="path to build directory (default: ./build/<backend> or ./build/<backend>_DEBUG)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tecs",
        help="CMake target name to build (default: tecs)",
    )
    parser.add_argument(
        "-B", "--backend",
        type=str,
        default="SERIAL",
        help="Kokkos backend to request (default: SERIAL)",
    )
    parser.add_argument(
        "-G", "--generator",
        type=str,
        default=None,
        help="CMake generator to use (e.g. 'Unix Makefiles', 'Ninja'; default: cmake's default)",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        default=False,
        help="build in Debug mode (Kokkos bounds checking, -O0 -g3)",
    )
    parser.add_argument(
        "-e", "--execute",
        action="store_true",
        default=False,
        help="execute built target after successful build",
    )
    parser.add_argument(
        "-t", "--time",
        action="store_true",
        default=False,
        dest="time_execute",
        help="when executing, run the executable under 'time -p'",
    )

    args = parser.parse_args()

    user_main = args.user_main
    target_name = args.output
    backend = args.backend
    generator = args.generator
    do_debug = args.debug
    do_execute = args.execute
    do_time = args.time_execute

    # Resolve project root (directory containing tecs.py)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir

    if args.build_dir is not None:
        build_dir = Path(args.build_dir)
    else:
        if do_debug:
            build_dir = Path(f"./build/{backend}_DEBUG")
        else:
            build_dir = Path(f"./build/{backend}")

    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    # Build CMake options
    cmake_opts = [
        f"-DTECS_USER_MAIN={user_main}",
        f"-DTECS_USER_TARGET={target_name}",
        f"-DKokkos_BACKEND={backend}",
    ]

    # Config hash — excludes TECS_USER_MAIN so that switching examples
    # (user source) does NOT wipe the CMake cache and force a rebuild of
    # all dependencies (Kokkos, HighFive, etc.).  The user-source hash
    # below catches source changes and triggers a shallow reconfigure only.
    config_hash_opts = [
        f"-DTECS_USER_TARGET={target_name}",
        f"-DKokkos_BACKEND={backend}",
    ]
    if do_debug:
        cmake_opts.append("-DCMAKE_BUILD_TYPE=Debug")
        config_hash_opts.append("-DCMAKE_BUILD_TYPE=Debug")
        print("Building in DEBUG mode with Kokkos bounds checking enabled")

    cmake_cache_file = build_dir / "CMakeCache.txt"
    config_hash_file = build_dir / ".cmake_configure_hash"
    user_main_hash_file = build_dir / ".cmake_user_main_hash"

    # Compute config-level hash (CMakeLists.txt + cmake options; NOT user source).
    # TECS_USER_MAIN is intentionally excluded so that switching examples does
    # NOT trigger a deep reconfigure (cache wipe + full dependency rebuild).
    config_hasher = hashlib.sha256()
    try:
        config_hasher.update((project_root / "CMakeLists.txt").read_bytes())
    except FileNotFoundError:
        pass
    for opt in config_hash_opts:
        config_hasher.update(opt.encode())
    if generator:
        config_hasher.update(generator.encode())
    new_config_hash = config_hasher.hexdigest()

    # Compute user-source hash separately
    user_hasher = hashlib.sha256()
    try:
        user_hasher.update(Path(user_main).read_bytes())
    except FileNotFoundError:
        pass
    new_user_hash = user_hasher.hexdigest()

    needs_reconfigure = False
    needs_reconfigure_deep = False
    if not cmake_cache_file.exists():
        needs_reconfigure = True
        needs_reconfigure_deep = True
    else:
        old_config_hash = ""
        if config_hash_file.exists():
            old_config_hash = config_hash_file.read_text().strip()
        if new_config_hash != old_config_hash:
            needs_reconfigure = True
            needs_reconfigure_deep = True

        old_user_hash = ""
        if user_main_hash_file.exists():
            old_user_hash = user_main_hash_file.read_text().strip()
        if new_user_hash != old_user_hash:
            needs_reconfigure = True

    cmake = _find_cmake()

    if needs_reconfigure_deep:
        # CMakeLists.txt or options changed — need a clean slate
        if cmake_cache_file.exists():
            _remove_cmake_cache(cmake_cache_file)

        cmake_cmd = [cmake, "-S", str(project_root), "-B", str(build_dir)]
        if generator:
            cmake_cmd.extend(["-G", generator])
        cmake_cmd.extend(cmake_opts)
        print(f"Configuring: {' '.join(cmake_cmd)}")
        try:
            subprocess.check_call(cmake_cmd)
        except subprocess.CalledProcessError:
            print(f"\nConfiguration failed", file=sys.stderr)
            sys.exit(1)
        config_hash_file.write_text(new_config_hash)
        user_main_hash_file.write_text(new_user_hash)
    elif needs_reconfigure:
        # Only user source changed — reconfigure without clearing cache
        # so Kokkos and other dependencies are not rebuilt
        cmake_cmd = [cmake, "-S", str(project_root), "-B", str(build_dir)]
        if generator:
            cmake_cmd.extend(["-G", generator])
        cmake_cmd.extend(cmake_opts)
        print(f"Reconfiguring (user source changed): {' '.join(cmake_cmd)}")
        try:
            subprocess.check_call(cmake_cmd)
        except subprocess.CalledProcessError:
            print(f"\nReconfiguration failed", file=sys.stderr)
            sys.exit(1)
        user_main_hash_file.write_text(new_user_hash)

    # Build
    print(f"Building target '{target_name}' in {build_dir}")
    parallel_flag = _parallel_build_flag()
    try:
        subprocess.check_call([
            cmake,
            "--build",
            str(build_dir),
            "--target",
            target_name,
            "--parallel",
            str(os.cpu_count() or 1),
        ])
    except subprocess.CalledProcessError:
        print(f"\nBuild failed", file=sys.stderr)
        sys.exit(1)
    print(f"Built target {target_name} in {build_dir}")

    if do_execute:
        exe_path = _find_executable(build_dir, target_name)
        if exe_path is None or not os.access(exe_path, os.X_OK):
            print(
                f"Built, but could not find executable '{target_name}' to run in {build_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Executing {exe_path}")
        if do_time:
            _exec_with_timing(exe_path)
        else:
            try:
                subprocess.check_call([str(exe_path)])
            except subprocess.CalledProcessError as e:
                exit_code = e.returncode
                if exit_code < 0:
                    sig = -exit_code
                    print(f"\nExecution failed — killed by signal {sig} ({_signal_name(sig)})", file=sys.stderr)
                else:
                    print(f"\nExecution failed — exit code {exit_code}", file=sys.stderr)
                sys.exit(1)

def _remove_cmake_cache(cache_file: Path) -> None:
    """Remove CMakeCache.txt and the associated CMakeFiles directory to force
    a clean re-configure (e.g. when the generator changes)."""
    cache_file.unlink(missing_ok=True)
    cmake_files_dir = cache_file.parent / "CMakeFiles"
    if cmake_files_dir.exists():
        shutil.rmtree(cmake_files_dir)

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

def _find_executable(build_dir: Path, target_name: str) -> Path | None:
    """Find the built executable, which may have an extension on Windows."""
    candidates = [
        build_dir / target_name,
        build_dir / f"{target_name}.exe",
    ]
    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            return cand

    # Search up to 8 levels deep
    for root, dirs, files in os.walk(build_dir):
        depth = Path(root).relative_to(build_dir).parts
        if len(depth) > 8:
            dirs.clear()  # don't descend further
            continue
        for f in files:
            if f == target_name or f == f"{target_name}.exe":
                full = Path(root) / f
                if os.access(full, os.X_OK):
                    return full
    return None

def _signal_name(signum: int) -> str:
    """Return a human-readable name for a signal number."""
    import signal as _signal
    try:
        return _signal.Signals(signum).name
    except (ValueError, AttributeError):
        return f"SIGUNKNOWN({signum})"


def _exec_with_timing(exe_path: Path) -> None:
    """Run executable with timing output, cross-platform."""
    if platform.system() == "Windows":
        import time as time_mod
        start = time_mod.perf_counter()
        try:
            subprocess.check_call([str(exe_path)])
        except subprocess.CalledProcessError as e:
            exit_code = e.returncode
            if exit_code < 0:
                sig = -exit_code
                print(f"\nExecution failed — killed by signal {sig} ({_signal_name(sig)})", file=sys.stderr)
            else:
                print(f"\nExecution failed — exit code {exit_code}", file=sys.stderr)
            sys.exit(1)
        elapsed = time_mod.perf_counter() - start
        print(f"\nreal {elapsed:.3f}")
    else:
        try:
            subprocess.check_call(["time", "-p", str(exe_path)])
        except subprocess.CalledProcessError as e:
            exit_code = e.returncode
            if exit_code < 0:
                sig = -exit_code
                print(f"Killed by signal {sig} ({_signal_name(sig)})", file=sys.stderr)
            else:
                print(f"Execution failed — exit code {exit_code}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
