#!/usr/bin/env python3
"""
Performance regression gate for Qwen3-0.6B Q4_0 x86 CI.

Parses nntr_causallm output for prefill TPS, generation TPS, and e2e wall time,
computes medians across iterations, and compares against a committed baseline
JSON. Exits 0 on pass or null baseline, 1 on regression.
"""

import argparse
import json
import re
import statistics
import sys

PREFILL_RE = re.compile(
    r"prefill:\s+(\d+)\s+tokens,\s+(\d+)\s+ms,\s+([\d.]+)\s+TPS"
)
GENERATION_RE = re.compile(
    r"generation:\s+(\d+)\s+tokens,\s+(\d+)\s+ms,\s+([\d.]+)\s+TPS"
)
E2E_RE = re.compile(r"\[e2e time\]:\s+(\d+)\s+ms")


def parse_log(path):
    """Parse run.log into per-iteration metric dicts."""
    iterations = []
    current = {}

    with open(path) as f:
        for line in f:
            m = PREFILL_RE.search(line)
            if m:
                current["prefill_tokens"] = int(m.group(1))
                current["prefill_ms"] = int(m.group(2))
                current["prefill_tps"] = float(m.group(3))

            m = GENERATION_RE.search(line)
            if m:
                current["gen_tokens"] = int(m.group(1))
                current["gen_ms"] = int(m.group(2))
                current["gen_tps"] = float(m.group(3))

            m = E2E_RE.search(line)
            if m:
                current["e2e_ms"] = int(m.group(1))
                iterations.append(current)
                current = {}

    if current and "e2e_ms" in current:
        iterations.append(current)

    return iterations


def median_or_none(values):
    if not values:
        return None
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to run.log")
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON")
    parser.add_argument("--summary-out", default="perf_summary.json",
                        help="Output summary JSON path")
    parser.add_argument("--require-baseline", action="store_true",
                        help="Fail if baseline metrics are null")
    args = parser.parse_args()

    iterations = parse_log(args.log)
    if not iterations:
        print("ERROR: No iterations parsed from log. Check run.log format.")
        sys.exit(1)

    prefill_vals = [it["prefill_tps"] for it in iterations if "prefill_tps" in it]
    gen_vals = [it["gen_tps"] for it in iterations if "gen_tps" in it]
    e2e_vals = [it["e2e_ms"] for it in iterations if "e2e_ms" in it]

    med_prefill = median_or_none(prefill_vals)
    med_gen = median_or_none(gen_vals)
    med_e2e = median_or_none(e2e_vals)

    print(f"Parsed {len(iterations)} iteration(s)")
    print(f"  Prefill TPS  : {prefill_vals} -> median {med_prefill}")
    print(f"  Generation TPS: {gen_vals} -> median {med_gen}")
    print(f"  E2E ms       : {e2e_vals} -> median {med_e2e}")
    print()

    with open(args.baseline) as f:
        baseline = json.load(f)

    metrics = baseline["metrics"]
    tolerance = baseline["tolerance"]

    summary = {
        "iterations": len(iterations),
        "observed": {
            "prefill_tps_median": med_prefill,
            "generation_tps_median": med_gen,
            "e2e_ms_median": med_e2e,
        },
        "baseline": metrics,
        "tolerance": tolerance,
        "result": "unknown",
        "details": [],
    }

    baseline_seeded = all(
        metrics.get(k) is not None
        for k in ("prefill_tps_min", "generation_tps_min", "e2e_ms_max")
    )

    if not baseline_seeded:
        if args.require_baseline:
            print("FAIL: --require-baseline set but baseline metrics are null")
            summary["result"] = "fail_no_baseline"
            with open(args.summary_out, "w") as f:
                json.dump(summary, f, indent=2)
            sys.exit(1)

        print("=" * 60)
        print("BASELINE NOT YET SEEDED — reporting observed values only.")
        print("=" * 60)
        print()
        print("To seed the baseline, update tools/ci/qwen3_0.6b_q40_baseline.json:")
        if med_prefill is not None:
            pct = tolerance.get("prefill_tps_pct", 20)
            suggested = round(med_prefill * (1 - pct / 100), 2)
            print(f'  "prefill_tps_min": {suggested}   (observed {med_prefill}, -{pct}%)')
        if med_gen is not None:
            pct = tolerance.get("generation_tps_pct", 20)
            suggested = round(med_gen * (1 - pct / 100), 2)
            print(f'  "generation_tps_min": {suggested}   (observed {med_gen}, -{pct}%)')
        if med_e2e is not None:
            pct = tolerance.get("e2e_ms_pct", 25)
            suggested = round(med_e2e * (1 + pct / 100))
            print(f'  "e2e_ms_max": {suggested}   (observed {med_e2e}, +{pct}%)')

        summary["result"] = "pass_no_baseline"
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
        sys.exit(0)

    failures = []

    prefill_pct = tolerance.get("prefill_tps_pct", 20)
    prefill_min = metrics["prefill_tps_min"]
    prefill_threshold = prefill_min * (1 - prefill_pct / 100)
    if med_prefill is not None and med_prefill < prefill_threshold:
        delta = (med_prefill - prefill_min) / prefill_min * 100
        failures.append(
            f"Prefill TPS regression: {med_prefill:.2f} < {prefill_threshold:.2f} "
            f"(baseline {prefill_min}, tolerance -{prefill_pct}%, delta {delta:+.1f}%)"
        )

    gen_pct = tolerance.get("generation_tps_pct", 20)
    gen_min = metrics["generation_tps_min"]
    gen_threshold = gen_min * (1 - gen_pct / 100)
    if med_gen is not None and med_gen < gen_threshold:
        delta = (med_gen - gen_min) / gen_min * 100
        failures.append(
            f"Generation TPS regression: {med_gen:.2f} < {gen_threshold:.2f} "
            f"(baseline {gen_min}, tolerance -{gen_pct}%, delta {delta:+.1f}%)"
        )

    e2e_pct = tolerance.get("e2e_ms_pct", 25)
    e2e_max = metrics["e2e_ms_max"]
    e2e_threshold = e2e_max * (1 + e2e_pct / 100)
    if med_e2e is not None and med_e2e > e2e_threshold:
        delta = (med_e2e - e2e_max) / e2e_max * 100
        failures.append(
            f"E2E time regression: {med_e2e} ms > {e2e_threshold:.0f} ms "
            f"(baseline {e2e_max}, tolerance +{e2e_pct}%, delta {delta:+.1f}%)"
        )

    print("=" * 60)
    print("Performance Gate Results")
    print("=" * 60)
    print(f"{'Metric':<22} {'Observed':>12} {'Baseline':>12} {'Threshold':>12} {'Status':>8}")
    print("-" * 70)

    def status_str(observed, threshold, higher_is_better):
        if observed is None:
            return "N/A"
        if higher_is_better:
            return "PASS" if observed >= threshold else "FAIL"
        return "PASS" if observed <= threshold else "FAIL"

    print(f"{'Prefill TPS':<22} {med_prefill or 0:>12.2f} {prefill_min:>12.2f} "
          f"{prefill_threshold:>12.2f} {status_str(med_prefill, prefill_threshold, True):>8}")
    print(f"{'Generation TPS':<22} {med_gen or 0:>12.2f} {gen_min:>12.2f} "
          f"{gen_threshold:>12.2f} {status_str(med_gen, gen_threshold, True):>8}")
    print(f"{'E2E time (ms)':<22} {med_e2e or 0:>12} {e2e_max:>12} "
          f"{e2e_threshold:>12.0f} {status_str(med_e2e, e2e_threshold, False):>8}")
    print()

    summary["details"] = failures
    if failures:
        summary["result"] = "fail"
        for f_msg in failures:
            print(f"REGRESSION: {f_msg}")
        print()
        print("To update the baseline after an intentional change, edit:")
        print("  tools/ci/qwen3_0.6b_q40_baseline.json")
    else:
        summary["result"] = "pass"
        print("All metrics within tolerance. PASSED.")

    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
