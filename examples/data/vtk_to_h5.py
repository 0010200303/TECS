#!/usr/bin/env python3
"""Convert VTK legacy files to HDF5 format."""
# AI generated!!!

import sys
import argparse
import h5py
import numpy as np


# VTK data type names -> numpy dtypes
_VTK_DTYPE = {
    "bit": np.bool_,
    "unsigned_char": np.uint8,
    "char": np.int8,
    "unsigned_short": np.uint16,
    "short": np.int16,
    "unsigned_int": np.uint32,
    "int": np.int32,
    "unsigned_long": np.uint64,
    "long": np.int64,
    "float": np.float32,
    "double": np.float64,
}


def parse_vtk_legacy(filepath):
    """Parse a legacy VTK file and return the data as a dict of numpy arrays."""
    data = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    lines = [l.rstrip() for l in lines]

    # Header: version, description, format
    idx = 0
    while idx < len(lines) and not lines[idx]:
        idx += 1
    data["version"] = lines[idx].strip()
    idx += 1

    while idx < len(lines) and not lines[idx]:
        idx += 1
    data["header"] = lines[idx].strip()
    idx += 1

    while idx < len(lines) and not lines[idx]:
        idx += 1
    fmt = lines[idx].strip().lower()
    data["format"] = fmt
    idx += 1

    # DATASET line
    while idx < len(lines) and (not lines[idx] or lines[idx].startswith("#")):
        idx += 1
    data["dataset_type"] = lines[idx].strip()
    idx += 1

    # Parse sections
    while idx < len(lines):
        line = lines[idx]
        idx += 1

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        keyword = parts[0].upper()

        if keyword == "POINTS":
            npoints = int(parts[1])
            dtype = _VTK_DTYPE.get(parts[2].lower(), np.float64)
            point_data = []
            while len(point_data) < npoints * 3:
                block = lines[idx].strip()
                idx += 1
                if not block or block.startswith("#"):
                    continue
                point_data.extend(float(v) for v in block.split())
            data[parts[0]] = np.array(point_data[:npoints * 3], dtype=dtype).reshape(-1, 3)

        elif keyword in ("VERTICES", "LINES", "POLYGONS", "TRIANGLE_STRIPS"):
            n_cells = int(parts[1])
            n_values = int(parts[2])
            cell_data = []
            while len(cell_data) < n_values:
                block = lines[idx].strip()
                idx += 1
                if not block or block.startswith("#"):
                    continue
                cell_data.extend(int(v) for v in block.split())
            key = parts[0]
            data[key] = np.array(cell_data[:n_values], dtype=np.int64)
            data[f"{key}_count"] = n_cells

        elif keyword == "CELL_TYPES":
            n_types = int(parts[1])
            type_data = []
            while len(type_data) < n_types:
                block = lines[idx].strip()
                idx += 1
                if not block or block.startswith("#"):
                    continue
                type_data.extend(int(v) for v in block.split())
            data[parts[0]] = np.array(type_data[:n_types], dtype=np.int32)

        elif keyword == "POINT_DATA":
            data["point_data_count"] = int(parts[1])

        elif keyword == "CELL_DATA":
            data["cell_data_count"] = int(parts[1])

        elif keyword == "SCALARS":
            scalars_name = parts[1]
            vtk_type = parts[2].lower() if len(parts) > 2 else "float"
            dtype = _VTK_DTYPE.get(vtk_type, np.float64)
            n_expected = data.get("point_data_count", data.get("cell_data_count", 0))
            scalars_vals = []
            while len(scalars_vals) < n_expected:
                block = lines[idx].strip()
                idx += 1
                if not block or block.startswith("#"):
                    continue
                if block.upper().startswith("LOOKUP_TABLE"):
                    continue
                if np.issubdtype(dtype, np.integer):
                    scalars_vals.extend(int(v) for v in block.split())
                else:
                    scalars_vals.extend(float(v) for v in block.split())
            data[scalars_name] = np.array(scalars_vals[:n_expected], dtype=dtype)

        elif keyword in ("VECTORS", "NORMALS"):
            vec_name = parts[1]
            vtk_type = parts[2].lower() if len(parts) > 2 else "float"
            dtype = _VTK_DTYPE.get(vtk_type, np.float64)
            n_expected = data.get("point_data_count", data.get("cell_data_count", 0))
            vec_vals = []
            while len(vec_vals) < n_expected * 3:
                block = lines[idx].strip()
                idx += 1
                if not block or block.startswith("#"):
                    continue
                if np.issubdtype(dtype, np.integer):
                    vec_vals.extend(int(v) for v in block.split())
                else:
                    vec_vals.extend(float(v) for v in block.split())
            data[vec_name] = np.array(vec_vals[:n_expected * 3], dtype=dtype).reshape(-1, 3)

        elif keyword == "TENSORS":
            tensor_name = parts[1]
            vtk_type = parts[2].lower() if len(parts) > 2 else "float"
            dtype = _VTK_DTYPE.get(vtk_type, np.float64)
            n_expected = data.get("point_data_count", data.get("cell_data_count", 0))
            tensor_vals = []
            while len(tensor_vals) < n_expected * 9:
                block = lines[idx].strip()
                idx += 1
                if not block or block.startswith("#"):
                    continue
                if np.issubdtype(dtype, np.integer):
                    tensor_vals.extend(int(v) for v in block.split())
                else:
                    tensor_vals.extend(float(v) for v in block.split())
            data[tensor_name] = np.array(tensor_vals[:n_expected * 9], dtype=dtype).reshape(-1, 3, 3)

        elif keyword == "FIELD":
            field_name = parts[1]
            num_arrays = int(parts[2])
            for _ in range(num_arrays):
                while idx < len(lines):
                    hdr = lines[idx].strip()
                    idx += 1
                    if not hdr or hdr.startswith("#"):
                        continue
                    break
                hdr_parts = hdr.split()
                arr_name = hdr_parts[0]
                arr_num_comp = int(hdr_parts[1])
                arr_num_tup = int(hdr_parts[2])
                arr_dtype = _VTK_DTYPE.get(arr_type.lower(), np.float64)
                arr_vals = []
                while len(arr_vals) < arr_num_tup * arr_num_comp:
                    block = lines[idx].strip()
                    idx += 1
                    if not block or block.startswith("#"):
                        continue
                    if np.issubdtype(arr_dtype, np.integer):
                        arr_vals.extend(int(v) for v in block.split())
                    else:
                        arr_vals.extend(float(v) for v in block.split())
                data[arr_name] = np.array(
                    arr_vals[:arr_num_tup * arr_num_comp],
                    dtype=arr_dtype
                ).reshape(-1, arr_num_comp)

    return data


def get_available_datasets(data):
    """Return list of all available dataset names from parsed VTK data."""
    skip_keys = {"version", "header", "format", "dataset_type",
                 "point_data_count", "cell_data_count"}
    return sorted(k for k in data if k not in skip_keys)


def filter_datasets(data, include, exclude):
    """Filter parsed VTK data to only include requested datasets.

    Parameters
    ----------
    data : dict
        Parsed VTK data from parse_vtk_legacy.
    include : set or None
        Dataset names to include. None means all.
    exclude : set or None
        Dataset names to exclude. None means none.

    Returns
    -------
    dict
        Filtered copy of data.
    """
    filtered = {}
    for key, val in data.items():
        if key in ("version", "header", "format", "dataset_type",
                   "point_data_count", "cell_data_count"):
            filtered[key] = val
            continue
        if include is not None and key not in include:
            continue
        if exclude is not None and key in exclude:
            continue
        filtered[key] = val
    return filtered


def vector3_to_polarity(vec3):
    """Convert 3D Cartesian vectors to 2D polarity (theta, phi).

    Based on tecs::Polarity_ constructor from Vector3:
      theta = acos(clamp(z / length, -1, 1))
      phi   = atan2(y, x)

    Parameters
    ----------
    vec3 : ndarray of shape (N, 3)
        3D vectors.

    Returns
    -------
    ndarray of shape (N, 2)
        Polarity values: [theta, phi] per row.
    """
    x, y, z = vec3[..., 0], vec3[..., 1], vec3[..., 2]
    length = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero: zero-length vectors get theta=0, phi=0
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.where(length > 0, np.arccos(np.clip(z / length, -1.0, 1.0)), 0.0)
        phi = np.where(length > 0, np.arctan2(y, x), 0.0)
    return np.column_stack((theta, phi))


def convert_vtk_to_h5(vtk_path, h5_path, include=None, exclude=None, convert_yalla_polarity=None):
    """Convert a VTK legacy file to HDF5.

    Parameters
    ----------
    vtk_path : str
        Path to input VTK file.
    h5_path : str
        Path to output HDF5 file.
    include : set or None
        Set of dataset names to include. None = all.
    exclude : set or None
        Set of dataset names to exclude. None = none.
    """
    data = parse_vtk_legacy(vtk_path)
    available = get_available_datasets(data)

    if include is not None:
        missing = include - set(available)
        if missing:
            print(f"Warning: requested datasets not found: {sorted(missing)}")

    data = filter_datasets(data, include, exclude)

    # Convert selected dataset from 3D vectors to 2D polarity
    if convert_yalla_polarity is not None:
        src_name = convert_yalla_polarity
        if src_name in data and isinstance(data[src_name], np.ndarray):
            src = data[src_name]
            if src.ndim == 2 and src.shape[1] == 3:
                pol_name = src_name
                data[pol_name] = vector3_to_polarity(src)
                print(f"  Converted '{src_name}' from 3D vectors to 2D polarity (theta, phi)")
            else:
                print(f"  Warning: '{src_name}' has shape {src.shape}, expected (N, 3). Skipping conversion.")
        else:
            print(f"  Warning: dataset '{src_name}' not found. Skipping polarity conversion.")

    selected = get_available_datasets(data)

    with h5py.File(h5_path, "w") as f:
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                f.create_dataset(key, data=val)

    print(f"Converted {vtk_path} -> {h5_path}")
    pts = data.get("POINTS")
    if pts is None:
        pts = data.get("points")
    print(f"  Points: {pts.shape if pts is not None else (0,)}")
    print(f"  Datasets ({len(selected)}): {selected}")


def list_datasets(vtk_path):
    """Parse VTK and print all available dataset names."""
    data = parse_vtk_legacy(vtk_path)
    available = get_available_datasets(data)
    print(f"Available datasets in {vtk_path}:")
    for name in available:
        val = data[name]
        if isinstance(val, np.ndarray):
            print(f"  {name}: {val.shape}  {val.dtype}")
        else:
            print(f"  {name}: {val}")
    pts = data.get("POINTS")
    if pts is None:
        pts = data.get("points")
    print(f"\nPoints: {pts.shape if pts is not None else (0,)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert VTK legacy files to HDF5 format."
    )
    parser.add_argument("input", help="Path to input .vtk file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Path to output .h5 file (default: <input>.h5)")
    parser.add_argument("--include", nargs="+", default=None,
                        help="Only include these datasets (e.g. points scalars_pressure)")
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="Exclude these datasets (e.g. polygons cell_types)")
    parser.add_argument("--list", action="store_true",
                        help="Print available datasets and exit")
    parser.add_argument("--convert-yalla-polarity", metavar="DATASET", default=None,
                        help="Convert a 3D vector dataset to 2D polarity (theta, phi) in-place")
    args = parser.parse_args()

    vtk_path = args.input
    h5_path = args.output or vtk_path.rsplit(".", 1)[0] + ".h5"

    if args.list:
        list_datasets(vtk_path)
        sys.exit(0)

    include = set(args.include) if args.include else None
    exclude = set(args.exclude) if args.exclude else None

    convert_vtk_to_h5(vtk_path, h5_path, include=include, exclude=exclude,
                      convert_yalla_polarity=args.convert_yalla_polarity)
