import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

import cblind.cblind as cb


def _solstice_palette(fractions):
    """Sample cblind's colorblind-safe solstice colormap at the given 0..1 fractions."""
    cmap = cb.cbmap("cb.solstice")
    return [cmap(f) for f in fractions]

marker_for_benchmark = {
    "NaiveFor":                             "o",
    "NaiveForDouble":                       "v",
    "NaiveForSymmetric":                    "<",
    "NaiveParallelReduce":                  "s",
    "NaiveParallelReduceDouble":            "P",
    "NaiveParallelReduceSymmetric":         "X",
    "NaiveSpread":                          "^",
    "BinnedGabrielReduceFor":               "D",
    "BinnedGabrielReduceForDouble":         "d",
    "BinnedGabrielReduceForSymmetric":      "p",
    "BinnedGabrielReduceParallel":          "h",
    "BinnedGabrielReduceParallelDouble":    "H",
    "BinnedGabrielReduceParallelSymmetric": "*",
}

# Colourblind-safe colours sampled from cblind's solstice colormap.
# Naive family = cool (blue) end, Binned family = warm (yellow/red) end, so the
# two families stay distinct without relying on red-vs-green.
_NAIVE_COLORS = _solstice_palette(np.linspace(0.0, 0.42, 7))
_BINNED_COLORS = _solstice_palette(np.linspace(0.55, 1.0, 6))

style_for_benchmark = {
    "NaiveFor":                             {"linestyle": "-",  "color": _NAIVE_COLORS[0], "alpha": 0.7, "linewidth": 1.5},
    "NaiveForDouble":                       {"linestyle": "--", "color": _NAIVE_COLORS[1], "alpha": 0.8, "linewidth": 2.0},
    "NaiveForSymmetric":                    {"linestyle": ":",  "color": _NAIVE_COLORS[2], "alpha": 0.9, "linewidth": 2.5},
    "NaiveParallelReduce":                  {"linestyle": "-",  "color": _NAIVE_COLORS[3], "alpha": 0.7, "linewidth": 1.5},
    "NaiveParallelReduceDouble":            {"linestyle": "--", "color": _NAIVE_COLORS[4], "alpha": 0.8, "linewidth": 2.0},
    "NaiveParallelReduceSymmetric":         {"linestyle": ":",  "color": _NAIVE_COLORS[5], "alpha": 0.9, "linewidth": 2.5},
    "NaiveSpread":                          {"linestyle": "-",  "color": _NAIVE_COLORS[6], "alpha": 0.7, "linewidth": 1.5},
    "BinnedGabrielReduceFor":               {"linestyle": "-",  "color": _BINNED_COLORS[0], "alpha": 0.7, "linewidth": 1.5},
    "BinnedGabrielReduceForDouble":         {"linestyle": "--", "color": _BINNED_COLORS[1], "alpha": 0.8, "linewidth": 2.0},
    "BinnedGabrielReduceForSymmetric":      {"linestyle": ":",  "color": _BINNED_COLORS[2], "alpha": 0.9, "linewidth": 2.5},
    "BinnedGabrielReduceParallel":          {"linestyle": "-",  "color": _BINNED_COLORS[3], "alpha": 0.7, "linewidth": 1.5},
    "BinnedGabrielReduceParallelDouble":    {"linestyle": "--", "color": _BINNED_COLORS[4], "alpha": 0.8, "linewidth": 2.0},
    "BinnedGabrielReduceParallelSymmetric": {"linestyle": ":",  "color": _BINNED_COLORS[5], "alpha": 0.9, "linewidth": 2.5},
}

def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def fmt_typst_number(value: float, precision: int = 5) -> str:
    """Format a number for Typst. Uses a fixed number of decimals.

    A trivially small value is clamped to 0 to keep the table clean.
    """
    if abs(value) < 10 ** (-precision):
        return "0"
    return f"{value:.{precision}f}"


def render_typst_table(avg_by_benchmark, benchmark_order, precision: int = 5) -> str:
    """Return a complete, standalone Typst #table[...] code block.

    Columns are the benchmarks, rows are the agent counts, and each cell
    contains the time_per_step_ms (ms) averaged over all machines.
    """
    # Restrict to the benchmarks explicitly requested for this table, in the
    # provided order. (avg_by_benchmark may contain other families.)
    data = {b: avg_by_benchmark[b] for b in benchmark_order if b in avg_by_benchmark}
    available = list(data.keys())

    # Deterministic row order: all agent counts that the selected benchmarks report.
    all_agents = sorted({a for pairs in data.values() for a, _ in pairs})

    value_by = {}
    for benchmark, pairs in data.items():
        for agents, val in pairs:
            value_by[(benchmark, agents)] = val

    lines = []
    lines.append("#table(")
    lines.append("  columns: (auto, " + ", ".join("auto" for _ in available) + "),")
    lines.append("  align: (left, " + ", ".join("right" for _ in available) + "),")
    lines.append("  stroke: 0.5pt + gray,")
    lines.append("  inset: 6pt,")
    lines.append("  table.header(")
    lines.append("    [*Agents*], " + ", ".join(f"[*{b}*]" for b in available) + ",")
    lines.append("  ),")
    for agents in all_agents:
        cells = []
        for benchmark in available:
            val = value_by.get((benchmark, agents))
            text = fmt_typst_number(val, precision) if val is not None else "--"
            cells.append(f"[{text}]")
        lines.append(f"  [*{agents}*], " + ", ".join(cells) + ",")
        lines.append("  table.hline(),")
    lines.append(")")
    return "\n".join(lines) + "\n"


def write_typst_tables(avg_by_benchmark, out_dir: Path):
    """Write copy-pasteable Typst tables, split into a Naive and a Binned family."""
    naive_order = [
        "NaiveFor", "NaiveForDouble", "NaiveForSymmetric",
        "NaiveParallelReduce", "NaiveParallelReduceDouble", "NaiveParallelReduceSymmetric",
        "NaiveSpread",
    ]
    binned_order = [
        "BinnedGabrielReduceFor", "BinnedGabrielReduceForDouble", "BinnedGabrielReduceForSymmetric",
        "BinnedGabrielReduceParallel", "BinnedGabrielReduceParallelDouble", "BinnedGabrielReduceParallelSymmetric",
    ]
    for name, order in (("naive", naive_order), ("binned", binned_order)):
        typst = render_typst_table(avg_by_benchmark, order)
        out_path = out_dir / f"table_{name}.typ"
        out_path.write_text(typst)
        print(f"saved {out_path}")


def _geomean(pairs):
    """Geometric mean of the times in ``[(agents, time), ...]`` (scale-invariant
    overall performance score; lower = better)."""
    if not pairs:
        return float("nan")
    logsum = sum(__import__("math").log(t) for _, t in pairs)
    return __import__("math").exp(logsum / len(pairs))


def _best_and_symmetric(data, prefix):
    """Return (best, best_symmetric) for a family, based on geometric mean.

    ``best`` is the fastest benchmark of the family; ``best_symmetric`` is the
    symmetric counterpart of ``best`` when it exists, otherwise ``best`` itself.
    ``data`` is ``{benchmark: [(agents, time), ...]}``.
    """
    benches = [b for b in data if b.startswith(prefix)]
    if not benches:
        return None, None
    best = min(benches, key=lambda b: _geomean(data[b]))

    # the symmetric sibling shares everything but the trailing "Symmetric"
    if best.endswith("Symmetric"):
        sibling = best[: -len("Symmetric")]
    else:
        sibling = best + "Symmetric"
    if sibling in data:
        return best, sibling
    return best, best


def plot_benchmarks(ax, data, title):
    """Single plot with ALL benchmarks drawn faint, then the best-of-family and
    its symmetric counterpart highlighted for both families.
    `data` is {benchmark: [(agents, time), ...]}."""

    # every benchmark faintly, so the full picture stays present.
    # These background lines go into the SECOND legend (bottom right).
    faint_lines, faint_labels = [], []
    for bench, pairs in sorted(data.items()):
        pairs = sorted(pairs)
        sty = style_for_benchmark.get(bench, {})
        line, = ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                        marker=marker_for_benchmark.get(bench, "."), markersize=3,
                        linewidth=1.0, linestyle=sty.get("linestyle", "--"),
                        color=sty.get("color", None), alpha=0.45, zorder=2)
        faint_lines.append(line)
        faint_labels.append(bench)

    # family colours for the highlighted lines (colorblind-safe, solstice)
    # These emphasized lines go into the FIRST legend (top left).
    hi_colors = {"Naive": _NAIVE_COLORS[0], "Binned": _BINNED_COLORS[-1]}
    hi_lines, hi_labels = [], []
    for prefix in ("Naive", "Binned"):
        best, sym = _best_and_symmetric(data, prefix)
        if best is None:
            continue
        color = hi_colors.get(prefix, "gray")
        # best = solid + strong; symmetric sibling = dashed + slightly dimmer
        for bench, lw, ls, mks, al, lbl in (
            (best, 2.6, "-", 6, 0.95, f"{best} (best)"),
            (sym, 1.7, "--", 4, 0.6, f"{sym} (symmetric)"),
        ):
            if sym == best:
                break  # best is its own symmetric sibling -> draw only once
            if bench is None:
                continue
            pairs = sorted(data.get(bench, []))
            if not pairs:
                continue
            line, = ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                            marker=marker_for_benchmark.get(bench, "o"), markersize=mks,
                            linewidth=lw, linestyle=ls, color=color, alpha=al,
                            zorder=5)
            hi_lines.append(line)
            hi_labels.append(lbl)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(title + "\n"
                 + "best + its symmetric sibling per family emphasized\n"
                 + " (solid = best, dashed = symmetric)", fontsize=14)
    ax.set_xlabel("agents", fontsize=14)
    ax.set_ylabel("time_per_step_ms", fontsize=14)
    ax.grid(True, alpha=0.3)
    # Split into two legends so each stays compact and readable.
    # NOTE: each ax.legend() call replaces the previous legend, so the first is
    # re-added as a separate artist (ax.add_artist) to keep both visible.
    if hi_lines:
        ax.add_artist(ax.legend(handles=hi_lines, labels=hi_labels, loc="upper left", fontsize=10))
    if faint_lines:
        ax.legend(handles=faint_lines, labels=faint_labels, loc="lower right", fontsize=8)


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "results.csv"
    out_dir = base_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    # Increased tick label sizes (title/label sizes set in plot_benchmarks).
    plt.rcParams.update({
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    rows = load_rows(csv_path)

    cleaned = []
    for r in rows:
        if not r.get("machine"):
            continue
        agents_str = r.get("agents")
        if not agents_str:
            continue
        try:
            agents_int = int(agents_str)
            time_val = float(r.get("time_per_step_ms") or 0.0)
        except (ValueError, TypeError):
            continue
        r["_agents_int"] = agents_int
        r["_time"] = time_val
        cleaned.append(r)
    rows = cleaned

    # averages over all machines
    avg_acc = defaultdict(list)
    for r in rows:
        key = (r["benchmark"], r["_agents_int"])
        avg_acc[key].append(r["_time"])

    avg_by_benchmark = defaultdict(list)
    for (bench, agents), vals in avg_acc.items():
        avg_by_benchmark[bench].append((agents, mean(vals)))

    # Write a copy-pasteable Typst table of the averaged results.
    write_typst_tables(avg_by_benchmark, out_dir)

    # --- per-machine plots ---
    by_machine = defaultdict(list)
    for row in rows:
        if not row.get("machine"):
            continue
        by_machine[row["machine"]].append(row)

    # All benchmarks drawn faint, then best-of-family and its symmetric
    # counterpart highlighted, for every machine and for the average.
    for machine, machine_rows in by_machine.items():
        m_bm = defaultdict(list)
        for row in machine_rows:
            m_bm[row["benchmark"]].append((row["_agents_int"], row["_time"]))
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_benchmarks(ax, m_bm, machine)
        fig.tight_layout()
        out_path = out_dir / f"{machine}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"saved {out_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_benchmarks(ax, avg_by_benchmark, "Average across machines")
    fig.tight_layout()
    out_path = out_dir / "average.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()
