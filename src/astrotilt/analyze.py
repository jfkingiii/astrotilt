#!/usr/bin/env python3
"""
analyze_tilt.py — FITS Tilt Analysis via Star Eccentricity

Measures star eccentricity across a 3×3 grid in each FITS image.
Tilt causes stars near edges/corners to appear elongated (higher eccentricity).
"""

import argparse
import glob
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import mad_std
from tqdm import tqdm


def union_find_labels(rows, cols, shape):
    """Connected-component labeling using union-find (8-connectivity)."""
    n = len(rows)
    if n == 0:
        return np.array([], dtype=np.int32)

    # Map pixel coords to index
    coord_to_idx = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        coord_to_idx[(r, c)] = i

    parent = np.arange(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # 8-connectivity neighbors
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in coord_to_idx:
                    union(i, coord_to_idx[nb])

    # Relabel contiguously
    labels = np.array([find(i) for i in range(n)], dtype=np.int32)
    unique_roots, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int32)


def eccentricity_from_moments(rows, cols, vals):
    """Intensity-weighted 2nd moment eccentricity. Returns float in [0, 1]."""
    vals = vals.astype(np.float64)
    total = vals.sum()
    if total <= 0:
        return np.nan

    cy = np.dot(rows.astype(np.float64), vals) / total
    cx = np.dot(cols.astype(np.float64), vals) / total

    dr = rows.astype(np.float64) - cy
    dc = cols.astype(np.float64) - cx

    Ixx = np.dot(dc * dc, vals) / total
    Iyy = np.dot(dr * dr, vals) / total
    Ixy = np.dot(dr * dc, vals) / total

    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    disc = max(0.0, (trace / 2) ** 2 - det)

    lam_max = trace / 2 + sqrt(disc)
    lam_min = trace / 2 - sqrt(disc)

    if lam_max <= 0:
        return np.nan

    ratio = max(0.0, min(1.0, lam_min / lam_max))
    return sqrt(1.0 - ratio)


def analyze_cell(cell, nsigma=5.0, min_pixels=5, max_pixels=1000):
    """
    Detect stars in a cell image and return (n_stars, median_eccentricity).
    Returns (0, np.nan) if no valid stars found.
    """
    sky_median = np.median(cell)
    sky_std = mad_std(cell)

    if sky_std <= 0:
        return 0, np.nan

    above = cell > sky_median + nsigma * sky_std
    rows, cols = np.where(above)

    if len(rows) == 0:
        return 0, np.nan

    vals = cell[rows, cols]
    labels = union_find_labels(rows, cols, cell.shape)

    eccentricities = []
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        mask = labels == lbl
        size = mask.sum()
        if size < min_pixels or size > max_pixels:
            continue
        r = rows[mask]
        c = cols[mask]
        v = vals[mask]
        ecc = eccentricity_from_moments(r, c, v)
        if not np.isnan(ecc):
            eccentricities.append(ecc)

    if not eccentricities:
        return 0, np.nan

    return len(eccentricities), float(np.median(eccentricities))


def analyze_file(fits_path, nsigma=5.0, min_pixels=5, max_pixels=1000, verbose=False):
    """
    Analyze a single FITS file. Returns list of dicts (one per grid cell).
    """
    GRID = 3
    filename = os.path.basename(fits_path)

    if verbose:
        print(f"  Processing {filename} ...", file=sys.stderr, end="", flush=True)

    with fits.open(fits_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    H, W = data.shape
    cell_h = H // GRID
    cell_w = W // GRID

    rows_out = []
    for cell_row in range(GRID):
        for cell_col in range(GRID):
            r0 = cell_row * cell_h
            r1 = H if cell_row == GRID - 1 else (cell_row + 1) * cell_h
            c0 = cell_col * cell_w
            c1 = W if cell_col == GRID - 1 else (cell_col + 1) * cell_w

            cell = data[r0:r1, c0:c1]
            cell_y_center = (r0 + r1) // 2
            cell_x_center = (c0 + c1) // 2

            n_stars, median_ecc = analyze_cell(
                cell, nsigma=nsigma, min_pixels=min_pixels, max_pixels=max_pixels
            )

            rows_out.append({
                "filename": filename,
                "cell_row": cell_row,
                "cell_col": cell_col,
                "cell_x_center": cell_x_center,
                "cell_y_center": cell_y_center,
                "n_stars": n_stars,
                "median_eccentricity": median_ecc,
            })

    if verbose:
        total_stars = sum(r["n_stars"] for r in rows_out)
        print(f" {total_stars} stars found", file=sys.stderr)

    return rows_out


def collect_fits_files(path_arg):
    """Expand directory or glob pattern to list of FITS file paths."""
    if os.path.isdir(path_arg):
        files = sorted(glob.glob(os.path.join(path_arg, "*.fits")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_arg, "*.fit")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_arg, "*.FITS")))
    else:
        files = sorted(glob.glob(path_arg))
    return files


def print_summary_grids(df, verbose=False):
    """Print median_eccentricity as 3×3 grids to stderr. Mean and std only with verbose."""
    grp = df.groupby(["cell_row", "cell_col"])["median_eccentricity"]
    stats = {"Median across subs": grp.median()}
    if verbose:
        stats["Mean across subs"] = grp.mean()
        stats["Standard deviation across subs"] = grp.std()

    col_w = 8  # width per cell value
    header = "  ".join(f"col {c}".center(col_w) for c in range(3))
    separator = "  ".join("-" * col_w for _ in range(3))

    for title, series in stats.items():
        print(f"\n{title}", file=sys.stderr)
        print(f"         {header}", file=sys.stderr)
        print(f"         {separator}", file=sys.stderr)
        for r in range(3):
            vals = "  ".join(
                f"{series.get((r, c), float('nan')):.4f}".center(col_w)
                for c in range(3)
            )
            print(f"  row {r}  {vals}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze telescope optical tilt via star eccentricity in a 3×3 grid."
    )
    parser.add_argument(
        "input",
        help="Directory containing *.fits files, or a glob pattern (e.g. 'data/*.fits')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        metavar="NSIGMA",
        help="Detection threshold in sigma above background (default: 5.0)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=5,
        metavar="N",
        help="Minimum pixels per star (default: 5)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum pixels per star (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Output CSV file (default: stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr",
    )
    args = parser.parse_args()

    files = collect_fits_files(args.input)
    if not files:
        print(f"Error: no FITS files found at '{args.input}'", file=sys.stderr)
        sys.exit(1)

    n = len(files)
    all_rows = []
    bar = tqdm(files, desc="Analyzing", unit="image", file=sys.stderr, disable=args.verbose)
    for fits_path in bar:
        if args.verbose:
            print(f"  Processing {os.path.basename(fits_path)} ...", file=sys.stderr, end="", flush=True)
        rows = analyze_file(
            fits_path,
            nsigma=args.threshold,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            verbose=False,
        )
        if args.verbose:
            total_stars = sum(r["n_stars"] for r in rows)
            print(f" {total_stars} stars found", file=sys.stderr)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "filename", "cell_row", "cell_col",
        "cell_x_center", "cell_y_center",
        "n_stars", "median_eccentricity",
    ])

    print(f"\nAnalyzed {n} file(s)", file=sys.stderr)

    print_summary_grids(df, verbose=args.verbose)

    if args.output:
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
