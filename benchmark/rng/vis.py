import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "results.csv"
    out_dir = base_dir / "plots"
    out_dir.mkdir(exist_ok=True)

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

    by_machine = defaultdict(list)
    for row in rows:
        if not row.get("machine"):
            continue
        by_machine[row["machine"]].append(row)

    marker_by_benchmark = {
        "Single_RNG": "o",
        "RNG_RNG": "s",
        "Fused": "*",
        "Fused_RNG": "^",
    }
    color_by_benchmark = {
        "Single_RNG": "red",
        "RNG_RNG": "blue",
        "Fused": "orange",
        "Fused_RNG": "green",
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
                linestyle="--",
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
            linestyle="--",
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
