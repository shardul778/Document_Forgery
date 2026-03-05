import argparse
import pickle
from pathlib import Path

# Use a non-interactive backend so this works on servers/CLI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score"]


def load_metrics(path: Path) -> dict:
    with open(path, "rb") as f:
        m = pickle.load(f)
    if not isinstance(m, dict):
        raise TypeError(f"Metrics file {path} did not contain a dict")
    out = {}
    for k in METRIC_KEYS:
        v = m.get(k, None)
        out[k] = float(v) if v is not None else float("nan")
    return out


def print_table(title: str, svm: dict, rf: dict) -> None:
    print("\n" + title)
    print("-" * len(title))
    print("{:<10} | {:>10} | {:>10}".format("metric", "SVM", "RF"))
    print("-" * 36)
    for k in METRIC_KEYS:
        print("{:<10} | {:>10.4f} | {:>10.4f}".format(k, svm[k], rf[k]))


def plot_bar(title: str, svm: dict, rf: dict, out_path: Path) -> None:
    x = list(range(len(METRIC_KEYS)))
    svm_vals = [svm[k] for k in METRIC_KEYS]
    rf_vals = [rf[k] for k in METRIC_KEYS]

    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width / 2 for i in x], svm_vals, width=width, label="SVM")
    ax.bar([i + width / 2 for i in x], rf_vals, width=width, label="Random Forest")

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_KEYS)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for i, v in enumerate(svm_vals):
        ax.text(i - width / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(rf_vals):
        ax.text(i + width / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare SVM vs Random Forest metrics from saved *.pkl files and plot a bar chart."
    )
    ap.add_argument(
        "--variant",
        choices=["base", "fantasyid", "real_world"],
        default="fantasyid",
        help="Which saved model metrics to compare (default: fantasyid).",
    )
    ap.add_argument("--models-dir", default="models", help="Directory containing the *.pkl metric files.")
    ap.add_argument("--out", default="reports", help="Output directory for generated plots.")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out)

    if args.variant == "base":
        svm_path = models_dir / "svm_metrics.pkl"
        rf_path = models_dir / "rf_metrics.pkl"
        title = "SVM vs Random Forest (base)"
        out_file = out_dir / "compare_svm_rf_base.png"
    elif args.variant == "fantasyid":
        svm_path = models_dir / "fantasyid_svm_metrics.pkl"
        rf_path = models_dir / "fantasyid_rf_metrics.pkl"
        title = "SVM vs Random Forest (FantasyID)"
        out_file = out_dir / "compare_svm_rf_fantasyid.png"
    else:
        svm_path = models_dir / "real_world_svm_metrics.pkl"
        rf_path = models_dir / "real_world_rf_metrics.pkl"
        title = "SVM vs Random Forest (Real-World)"
        out_file = out_dir / "compare_svm_rf_real_world.png"

    if not svm_path.exists():
        raise FileNotFoundError(f"Missing: {svm_path}")
    if not rf_path.exists():
        raise FileNotFoundError(f"Missing: {rf_path}")

    svm = load_metrics(svm_path)
    rf = load_metrics(rf_path)

    print_table(title, svm, rf)
    plot_bar(title, svm, rf, out_file)
    print(f"\nSaved plot: {out_file.resolve()}")


if __name__ == "__main__":
    main()

