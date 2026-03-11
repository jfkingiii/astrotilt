"""
stars.py — Pure star detection math (no I/O).

Functions for connected-component labeling, eccentricity calculation,
and per-cell star analysis.
"""

from math import sqrt

import numpy as np
from astropy.stats import mad_std


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
