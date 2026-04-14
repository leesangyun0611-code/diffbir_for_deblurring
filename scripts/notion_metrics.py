import argparse
import csv
from pathlib import Path

import pandas as pd


def read_average_metrics(metrics_csv: Path) -> tuple[float | None, float | None]:
    if not metrics_csv.exists():
        return None, None

    with metrics_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None, None

    average_rows = [row for row in rows if row.get("file_name") == "AVERAGE"]
    row = average_rows[-1] if average_rows else rows[-1]
    return float(row["psnr"]), float(row["ssim"])


def summarize_experiment(exp_dir: Path, iqa_csv_name: str, metrics_csv_name: str) -> dict:
    iqa_csv = exp_dir / iqa_csv_name
    metrics_csv = exp_dir / metrics_csv_name

    row = {
        "experiment": exp_dir.name,
        "images": "",
        "avg_maniqa": "",
        "avg_lpips": "",
        "avg_psnr": "",
        "avg_ssim": "",
    }

    if iqa_csv.exists():
        iqa_df = pd.read_csv(iqa_csv)
        row["images"] = len(iqa_df)
        if "maniqa" in iqa_df.columns:
            row["avg_maniqa"] = iqa_df["maniqa"].mean()
        if "lpips" in iqa_df.columns:
            row["avg_lpips"] = iqa_df["lpips"].mean()

    avg_psnr, avg_ssim = read_average_metrics(metrics_csv)
    if avg_psnr is not None:
        row["avg_psnr"] = avg_psnr
    if avg_ssim is not None:
        row["avg_ssim"] = avg_ssim

    return row


def format_value(value, digits: int) -> str:
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def display_name(experiment_name: str) -> str:
    if experiment_name.startswith("exp"):
        parts = experiment_name.split("_", 2)
        if len(parts) == 3 and parts[0][3:].isdigit() and parts[1].isdigit():
            return f"{parts[0][3:]}-{parts[1]}: {parts[2]}"
    return experiment_name


def print_tsv(rows: list[dict], digits: int, no_header: bool) -> None:
    columns = ["experiment", "avg_lpips", "avg_psnr", "avg_ssim"]

    if not no_header:
        print("\t".join(columns))

    for row in rows:
        values = [
            row["experiment"],
            format_value(row["avg_lpips"], digits),
            format_value(row["avg_psnr"], digits),
            format_value(row["avg_ssim"], digits),
        ]
        print("\t".join(values))


def print_blocks(rows: list[dict], digits: int) -> None:
    for idx, row in enumerate(rows):
        if idx > 0:
            print()
            print()

        print(display_name(row["experiment"]))
        print()
        print(f"Average LPIPS: {format_value(row['avg_lpips'], digits)}")
        print()
        print(f"Average PSNR: {format_value(row['avg_psnr'], digits)}")
        print(f"Average SSIM: {format_value(row['avg_ssim'], digits)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Notion-friendly TSV summaries for DiffBIR experiment metrics."
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment result directories, e.g. results/exp1_1_naive_denoise.",
    )
    parser.add_argument("--iqa_csv", default="iqa_results.csv")
    parser.add_argument("--metrics_csv", default="metrics_psnr_ssim.csv")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument(
        "--format",
        choices=["tsv", "blocks"],
        default="tsv",
        help="Output format. Use 'blocks' for the paragraph style shown in notes.",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="Do not print the TSV header row.",
    )
    args = parser.parse_args()

    rows = [
        summarize_experiment(Path(exp), args.iqa_csv, args.metrics_csv)
        for exp in args.experiments
    ]

    if args.format == "blocks":
        print_blocks(rows, args.digits)
    else:
        print_tsv(rows, args.digits, args.no_header)


if __name__ == "__main__":
    main()
