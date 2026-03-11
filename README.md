# astrotilt

Measure telescope optical tilt by analyzing star eccentricity across a 3×3 grid in FITS images. Stars near tilted edges appear elongated (higher eccentricity), revealing the direction and magnitude of tilt.

## Installation

```bash
pipx install git+https://github.com/USER/astrotilt
```

Or for development:

```bash
pip install -e .
```

## Usage

```bash
# Analyze all .fits files in a directory, print grid summary to stderr
astrotilt samples/

# Save per-cell results to CSV with verbose progress
astrotilt "data/*.fits" --output results.csv --verbose

# Adjust detection parameters
astrotilt img.fits --threshold 4.0 --min-pixels 3 --max-pixels 500
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold NSIGMA` | 5.0 | Detection threshold (sigma above background) |
| `--min-pixels N` | 5 | Minimum pixels per star blob |
| `--max-pixels N` | 1000 | Maximum pixels per star blob |
| `--output FILE` | — | Write CSV output to FILE |
| `--verbose` | off | Print per-file progress to stderr |

## Output

The tool always prints a 3×3 eccentricity grid summary to stderr (median, mean, std across all input frames):

```
Median across subs
          col 0     col 1     col 2
         --------  --------  --------
  row 0   0.3210    0.2845    0.3501
  row 1   0.2901    0.2634    0.2978
  row 2   0.3412    0.2790    0.3689
```

With `--output`, it also writes a CSV with one row per file per cell:

| Column | Description |
|--------|-------------|
| `filename` | FITS file basename |
| `cell_row` / `cell_col` | Grid position (0–2) |
| `cell_x_center` / `cell_y_center` | Pixel center of cell |
| `n_stars` | Stars detected in cell |
| `median_eccentricity` | Median eccentricity (0 = round, 1 = line) |

## Requirements

- Python 3.9+
- FITS images from a cooled astronomy camera (monochrome, single HDU)
- Enough stars per frame for reliable eccentricity statistics (typically 10+ per cell)
