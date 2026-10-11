import argparse
import json
import sys

from scidata import __version__
from scidata.io import DataError, load_columns
from scidata.stats import describe, linear_fit, moving_average, zscore_outliers


def build_parser():
    parser = argparse.ArgumentParser(
        prog="scidata",
        description="Process and visualize experimental time series stored in CSV files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("file", help="path to the CSV file")
    parser.add_argument("--x", required=True, help="name of the independent variable column")
    parser.add_argument("--y", required=True, help="name of the dependent variable column")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--window", type=int, default=5, help="moving average window size")
    parser.add_argument("--threshold", type=float, default=3.0, help="z score outlier threshold")
    parser.add_argument("--plot", help="save a PNG chart to this path")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def analyze(x, y, window, threshold):
    smoothed = moving_average(y, min(window, y.size))
    outliers = zscore_outliers(y - smoothed, threshold)
    fit = linear_fit(x, y)
    report = {
        "summary": describe(y).as_dict(),
        "linear_fit": fit.as_dict(),
        "outliers": {
            "count": int(outliers.sum()),
            "x": [float(v) for v in x[outliers]],
        },
    }
    return report, smoothed, fit, outliers


def format_text(report):
    s = report["summary"]
    f = report["linear_fit"]
    o = report["outliers"]
    lines = [
        f"samples:    {s['count']}",
        f"mean:       {s['mean']:.4f}",
        f"std:        {s['std']:.4f}",
        f"min/median/max: {s['min']:.4f} / {s['median']:.4f} / {s['max']:.4f}",
        f"linear fit: y = {f['slope']:.4f} * x + {f['intercept']:.4f} (R^2 = {f['r_squared']:.4f})",
        f"outliers:   {o['count']}",
    ]
    return "\n".join(lines)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        x, y = load_columns(args.file, args.x, args.y, args.delimiter)
        report, smoothed, fit, outliers = analyze(x, y, args.window, args.threshold)
    except (DataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.plot:
        from scidata.plot import plot_series

        plot_series(
            x,
            y,
            args.plot,
            smoothed=smoothed,
            fit=fit,
            outliers=outliers,
            title=f"{args.y} vs {args.x}",
            xlabel=args.x,
            ylabel=args.y,
        )
        report["plot"] = args.plot
    print(json.dumps(report, indent=2) if args.json else format_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
