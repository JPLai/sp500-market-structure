"""Command line interface: ``market-structure step1``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .clustering import LINKAGES
from .report import write_outputs
from .step1 import Step1Config, run_step1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-structure",
        description="S&P 500 market structure: correlation, distance and hierarchical clustering of stocks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    step1 = sub.add_parser(
        "step1",
        help="build the market map (downloads data on the first run, then reuses the saved copy)",
    )
    step1.add_argument("--start", default="", help="first date, YYYY-MM-DD (default: one year before --end)")
    step1.add_argument("--end", default="", help="end date, exclusive (default: today)")
    step1.add_argument("--data-dir", type=Path, default=Path("data"), help="where downloaded data is kept")
    step1.add_argument(
        "--output-dir", type=Path, default=Path("output/step1"), help="where results are written"
    )
    step1.add_argument("--linkage", choices=LINKAGES, help="force a linkage (default: best silhouette)")
    step1.add_argument("--k", type=int, help="force the number of clusters (default: best silhouette)")
    step1.add_argument("--k-min", type=int, default=5, help="smallest k considered when choosing (default 5)")
    step1.add_argument(
        "--k-max", type=int, default=30, help="largest k considered when choosing (default 30)"
    )
    step1.add_argument(
        "--bootstrap",
        type=int,
        default=100,
        help="resamples for the stability check, 0 to skip (default 100)",
    )
    step1.add_argument("--seed", type=int, default=0, help="random seed for the bootstrap (default 0)")
    step1.add_argument(
        "--top-pairs",
        type=int,
        default=20,
        help="pairs listed in the strongest/most negative tables",
    )
    step1.add_argument(
        "--refresh-data", action="store_true", help="download again instead of using saved data"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Step1Config(
        start=args.start,
        end=args.end,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        linkage=args.linkage,
        k=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        bootstrap=args.bootstrap,
        seed=args.seed,
        top_pairs=args.top_pairs,
        refresh_data=args.refresh_data,
    )
    try:
        result = run_step1(config)
        folder = write_outputs(result)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"{len(result.symbols)} stocks, {result.days} days "
        f"({result.first_return_date} to {result.last_return_date}); "
        f"{result.linkage} linkage, {result.k} clusters."
    )
    print(f"Results written to {folder} (start with step1_report.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
