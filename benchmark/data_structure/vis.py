import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

import cblind.cblind as cb


def solstice_colors(n: int):
    """Return ``n`` colorblind-safe colours sampled from cblind's solstice cmap."""
    cmap = cb.cbmap("cb.solstice")
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]

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
    # Deterministic order: all agent counts that any benchmark reports.
    agents_per_bench = defaultdict(list)
    for benchmark, pairs in avg_by_benchmark.items():
        for agents, _ in pairs:
            agents_per_bench[benchmark].append(agents)
    all_agents = sorted({a for vals in agents_per_bench.values() for a in vals})

    value_by = {}
    for benchmark, pairs in avg_by_benchmark.items():
        for agents, val in pairs:
            value_by[(benchmark, agents)] = val

    # Select benchmarks present in the data, in a stable order.
    available = [b for b in benchmark_order if b in avg_by_benchmark]
    available += sorted(b for b in avg_by_benchmark if b not in benchmark_order)

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


def write_typst_table(avg_by_benchmark, out_path: Path):
    """Write a copy-pasteable Typst table next to the CSV."""
    benchmark_order = ["ViewOfVectors", "ViewOfArrays", "ViewOfArraysRaw", "ViewOfScalars"]
    typst = render_typst_table(avg_by_benchmark, benchmark_order)
    out_path.write_text(typst)
    print(f"saved {out_path}")

def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "results.csv"
    out_dir = base_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    # Increased font sizes for all plot text.
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
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
    write_typst_table(avg_by_benchmark, out_dir / "table.typ")

    by_machine = defaultdict(list)
    for row in rows:
        if not row.get("machine"):
            continue
        by_machine[row["machine"]].append(row)

    marker_by_benchmark = {
        "ViewOfVectors": "o",
        "ViewOfArrays": "s",
        "ViewOfArraysRaw": None,
        "ViewOfScalars": "^",
    }
    # Colourblind-safe colours sampled from cblind's solstice colormap.
    _solstice = solstice_colors(4)
    color_by_benchmark = {
        "ViewOfVectors": _solstice[0],
        "ViewOfArrays": _solstice[1],
        "ViewOfArraysRaw": _solstice[2],
        "ViewOfScalars": _solstice[3],
    }

    for machine, machine_rows in by_machine.items():
        by_benchmark = defaultdict(list)
        for row in machine_rows:
            by_benchmark[row["benchmark"]].append(row)

        plt.figure(figsize=(10, 6))
        for benchmark, bench_rows in sorted(by_benchmark.items()):
            bench_rows.sort(key=lambda r: r["_agents_int"])
            agents = [r["_agents_int"] for r in bench_rows]
            times = [r["_time"] for r in bench_rows]
            plt.plot(
                agents,
                times,
                marker=marker_by_benchmark.get(benchmark, "o"),
                linewidth=2,
                linestyle="--" if benchmark == "ViewOfArraysRaw" else "-",
                color=color_by_benchmark.get(benchmark, None),
                label=benchmark,
            )

        plt.title(machine)
        plt.xlabel("agents")
        plt.ylabel("time_per_step_ms")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"{machine}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"saved {out_path}")

    plt.figure(figsize=(10, 6))
    for benchmark, pairs in sorted(avg_by_benchmark.items()):
        pairs.sort(key=lambda p: p[0])
        agents = [p[0] for p in pairs]
        times = [p[1] for p in pairs]
        plt.plot(
            agents,
            times,
            marker=marker_by_benchmark.get(benchmark, "o"),
            linewidth=2.5,
            linestyle="--" if benchmark == "ViewOfArraysRaw" else "-",
            color=color_by_benchmark.get(benchmark, None),
            alpha=0.8,
            label=f"{benchmark} (avg)",
        )

    plt.title("Average across machines")
    plt.xlabel("agents")
    plt.ylabel("time_per_step_ms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "average.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()
