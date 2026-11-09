"""Judge the AMD PR test or nightly and write its report: the PR review guideline's
checks (rules, scope, accuracy, kernel microbenchmark, E2E performance, tests, doc).

    python scripts/ci/report.py --scope scope.json --results results/ --out report.md
    python scripts/ci/report.py --results results/ --model deepseek_v41   # one model
    python scripts/ci/report.py --nightly --results results/ --out report.md

Exits 1 when a check fails. A check that could not run here (no checkpoint on the
CI nodes, a gated dataset) is a warning: its numbers come from the author, as the
guideline asks until the CI covers them.

Accuracy compares the monokernels on and off, on the same build and server
recipe. GSM8K gates: an exact one-sided McNemar test on the questions the two runs
answer differently fails a significant loss with the monokernels on. The other
benchmarks only warn.
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    ALPHA,
    EVALS,
    GAIN_TYPES,
    MODELS,
    PERF_GAIN,
    PERF_LOSS,
    PR_EVALS,
)

PASS, WARN, FAIL = "pass", "warn", "fail"
ICON = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
MARKER = "<!-- sgl-amd-mono-ci -->"


def worst(*states):
    return FAIL if FAIL in states else WARN if WARN in states else PASS


def mcnemar(off, on):
    """(lost, gained, p): questions right off and wrong on, the reverse, and the
    exact one-sided p-value of losing at least ``lost`` of the ``lost + gained``
    discordant questions by chance."""
    common = off.keys() & on.keys()
    lost = sum(1 for q in common if round(off[q]) and not round(on[q]))
    gained = sum(1 for q in common if not round(off[q]) and round(on[q]))
    n = lost + gained
    p = sum(comb(n, k) for k in range(lost, n + 1)) / 2**n if n else 1.0
    return lost, gained, p


def scope_check(key, r):
    m = MODELS.get(key, {})
    if r.get("skipped", {}).get("serving"):
        return WARN, f"serving not run: {r['skipped']['serving']}"
    if r.get("perf", {}).get("on", {}).get("error"):
        return FAIL, f"monokernel server: {r['perf']['on']['error']}"
    routes = r.get("routing", {}).get("on", [])
    if not routes:
        return (
            FAIL,
            "the monokernel server logged no routing: every step ran SGLang's own path",
        )
    return PASS, f"TP{m.get('tp')} on {r.get('node', '?')}; " + "; ".join(
        f"`{x}`" for x in routes
    )


def accuracy_check(key, r):
    m = MODELS.get(key, {})
    names = m.get("evals", [])
    if r.get("mode", "pr") == "pr":
        names = [e for e in PR_EVALS if e in names]
    if not names:
        return WARN, "no accuracy benchmarks for this model in CI yet"
    if r.get("skipped", {}).get("serving"):
        return WARN, "not run (no server here): author-provided results needed"
    if r.get("tier") == "smoke":
        if not m.get("e2e_tests"):
            return (
                WARN,
                "smoke tier (shared code changed): no accuracy check for this model",
            )
        return (
            PASS,
            "smoke tier (shared code changed): the serving tests' GSM8K and KL bounds (Tests)",
        )
    acc = r.get("accuracy", {})
    states, parts = [], []
    for name in names:
        on = acc.get("on", {}).get(name, {})
        off = acc.get("off", {}).get(name, {})
        spec = EVALS[name]
        if "skipped" in on:
            states.append(WARN)
            parts.append(f"{name}: not run ({on['skipped']})")
            continue
        if "score" not in on:
            states.append(FAIL if spec.get("gate") else WARN)
            parts.append(f"{name}: monokernels on: {on.get('error', 'no result')}")
            continue
        if "score" not in off:
            states.append(WARN)
            why = off.get("error", off.get("skipped", "missing"))
            parts.append(f"{name}: on {on['score']:.4f} (no off score: {why})")
            continue
        head = f"{name}: off {off['score']:.4f} → on {on['score']:.4f}"
        if off.get("cached"):
            head += " (off cached)"
        if spec.get("gate"):
            lost, gained, p = mcnemar(
                off.get("per_question", {}), on.get("per_question", {})
            )
            bad = lost > gained and p < ALPHA
            states.append(FAIL if bad else PASS)
            parts.append(
                f"{head}; {lost} lost / {gained} gained, p={p:.3f}"
                + (f" < {ALPHA}: significant loss" if bad else "")
            )
        else:
            low = on["score"] < off["score"] - spec["warn"]
            states.append(WARN if low else PASS)
            parts.append(
                f"{head}"
                + (f" (more than {spec['warn']} lower; warning)" if low else "")
            )
    return worst(*states), " · ".join(parts)


def perf_check(key, r, needs_gain):
    """PR mode: base branch → PR, both monokernels on, with the gain/loss gate.
    Nightly: monokernels off → on, for the record."""
    m = MODELS.get(key, {})
    nightly = r.get("mode") == "nightly"
    if not m.get("concurrency"):
        return WARN, "no speed test for this model in CI yet", []
    if r.get("skipped", {}).get("serving"):
        return (
            WARN,
            "not run (no server here): author-provided InferenceX results needed",
            [],
        )
    perf = r.get("perf", {})
    base, new = perf.get("off" if nightly else "base", {}), perf.get("on", {})
    if "error" in new:
        return FAIL, f"monokernels on: {new['error']}", []
    rows, states, gate_gains = [], [], []
    for c in m["gate_concurrency" if r.get("tier") == "smoke" else "concurrency"]:
        p, b = new.get(str(c), {}), base.get(str(c), {})
        if not p.get("median_tpot_ms"):
            states.append(FAIL)
            rows.append(
                (c, b.get("median_tpot_ms"), None, None, p.get("error", "no result"))
            )
            continue
        if not b.get("median_tpot_ms"):
            states.append(WARN)
            rows.append((c, None, p["median_tpot_ms"], None, "no base result"))
            continue
        gain = b["median_tpot_ms"] / p["median_tpot_ms"] - 1
        if gain < -PERF_LOSS and (not nightly or c in m["gate_concurrency"]):
            states.append(WARN if nightly else FAIL)
        if c in m["gate_concurrency"]:
            gate_gains.append(gain)
        rows.append((c, b["median_tpot_ms"], p["median_tpot_ms"], gain, ""))
    best = max(gate_gains, default=None)
    if needs_gain and not nightly:
        if best is None:
            states.append(WARN)
        elif best < PERF_GAIN:
            states.append(FAIL)
    summary = ", ".join(
        f"conc {c}: {b:.2f} → {p:.2f} ms ({1 + g:.2f}×)"
        for c, b, p, g, _ in rows
        if g is not None
    )
    if nightly:
        rule = "monokernels off → on; a loss at a covered concurrency warns"
        sides = "off → on"
    else:
        rule = (
            f"needs ≥{PERF_GAIN:.0%} median-TPOT gain at conc {m['gate_concurrency']}"
            if needs_gain
            else f"no loss over {PERF_LOSS:.0%}"
        )
        sides = "base → PR"
    return (
        worst(PASS, *states),
        f"median TPOT, {sides}: {summary or 'n/a'} ({rule})",
        rows,
    )


def tests_check(key, r):
    m = MODELS.get(key, {})
    t = r.get("tests", {})
    expected = ["unit"] + (
        ["kernels"] if key == "common" or m.get("kernel_tests") else []
    )
    if m.get("e2e_tests") and not r.get("skipped", {}).get("e2e"):
        expected.append("e2e")
    states, parts = [], []
    for name in expected:
        c = t.get(name)
        if c is None:
            states.append(FAIL)
            parts.append(f"{name}: did not run")
            continue
        states.append(FAIL if c["failed"] else PASS)
        s = f"{name} {c['passed']} passed"
        if c["failed"]:
            s += f", **{c['failed']} failed**" + (
                " (timed out)" if c.get("timeout") else ""
            )
        if c["skipped"]:
            s += f", {c['skipped']} skipped"
        parts.append(s)
    if r.get("skipped", {}).get("e2e"):
        states.append(WARN)
        parts.append(f"e2e not run: {r['skipped']['e2e']}")
    return worst(*states), " · ".join(parts)


def micro_check(key, r):
    m = MODELS.get(key, {})
    if not m.get("microbench"):
        return WARN, "no kernel-only timing script for this model in CI yet"
    mb = r.get("microbench", {})
    pr = mb.get("pr", [])
    if not pr:
        return FAIL, "did not run"
    bad = [x["cmd"] for x in pr if x["rc"]]
    if bad:
        return FAIL, f"failed: `{bad[0]}`"
    if r.get("mode") == "nightly":
        sides = "main"
    else:
        sides = "base and PR" if mb.get("base") else "PR (no base package)"
    return PASS, f"{len(pr)} script(s) on {sides}; output below"


def model_section(key, r, needs_gain):
    checks = [("Scope", *scope_check(key, r))] if key != "common" else []
    if key != "common":
        checks += [
            ("Accuracy", *accuracy_check(key, r)),
            ("Kernel microbenchmark", *micro_check(key, r)),
        ]
        state, text, rows = perf_check(key, r, needs_gain)
        checks.append(("E2E performance", state, text))
    else:
        rows = []
    checks.append(("Tests", *tests_check(key, r)))
    if r.get("error"):
        checks.append(("Run", FAIL, f"`{r['error']}`"))
    lines = [
        f"### {r.get('name', key)} (`{key}`)",
        "",
        f"Node `{r.get('node', '?')}` · baseline: {r.get('baseline', 'n/a')} · SGLang `{r.get('sglang', '?')}`",
        "",
        "| | Check | Result |",
        "|---|---|---|",
    ]
    lines += [f"| {ICON[s]} | {n} | {t} |" for n, s, t in checks]
    if rows:
        left, right = ("off", "on") if r.get("mode") == "nightly" else ("base", "PR")
        lines += [
            "",
            "<details><summary>E2E performance (InferenceX fixed length: 8k/1k, range ratio 0.8)</summary>",
            "",
            f"| conc | {left} TPOT (ms) | {right} TPOT (ms) | speedup | |",
            "|---|---|---|---|---|",
        ]
        for c, b, p, g, note in rows:
            fmt = lambda v: f"{v:.2f}" if v is not None else "–"  # noqa: E731
            lines.append(
                f"| {c} | {fmt(b)} | {fmt(p)} | {f'{1 + g:.3f}×' if g is not None else '–'} | {note} |"
            )
        lines.append("\n</details>")
    for side in ("base", "pr"):
        for x in r.get("microbench", {}).get(side, []):
            tail = "\n".join(x["tail"].splitlines()[-25:])
            lines += [
                "",
                f"<details><summary>Kernel microbenchmark, {side}: <code>{x['cmd']}</code> (exit {x['rc']})</summary>",
                "",
                "```",
                tail,
                "```",
                "</details>",
            ]
    return worst(*(s for _, s, _ in checks)), "\n".join(lines)


def load(results: Path, key):
    p = results / key / "result.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"model": key, "error": "no result.json: the job did not finish"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--scope")
    ap.add_argument("--nightly", action="store_true", help="every model in --results")
    ap.add_argument("--model", help="judge one model only (its job's status)")
    ap.add_argument("--title-type", default="")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    results = Path(args.results)

    if args.model:
        state, text = model_section(
            args.model, load(results, args.model), args.title_type in GAIN_TYPES
        )
        print(text)
        sys.exit(1 if state == FAIL else 0)

    run = f" · [run]({args.run_url})" if args.run_url else ""
    if args.nightly:
        keys = sorted(p.name for p in results.iterdir() if p.is_dir())
        states, sections = [], []
        for key in keys:
            s, text = model_section(key, load(results, key), False)
            states.append(s)
            sections.append(text)
        overall = worst(*states)
        head = [
            MARKER,
            f"## AMD nightly (main): {ICON[overall]} {overall}",
            "",
            f"Models: {', '.join(f'`{k}`' for k in keys)} · monokernels on vs off{run}",
        ]
    else:
        scope = json.loads(Path(args.scope).read_text())
        needs_gain = scope["needs_gain"]
        rules = scope["rules"]
        states, sections = [], []
        for key in sorted(scope["models"]):
            s, text = model_section(key, load(results, key), needs_gain)
            states.append(s)
            why = scope["models"][key]
            sections.append(
                text.replace(
                    "\n\n", f"\n\nRuns because {why['why']} ({why['tier']} tier). ", 1
                )
            )
        rule_rows = [
            ("Rules: CODEOWNERS", *rules["codeowners"]),
            ("Doc", *rules["doc"]),
        ]
        states += [s for _, s, _ in rule_rows]
        overall = worst(*states)
        gain_note = (
            f"`[{scope['title_type']}]` PR: needs a ≥{PERF_GAIN:.0%} median-TPOT gain at a gate concurrency."
            if needs_gain
            else f"`[{scope['title_type'] or '?'}]` PR: no speed gain needed; no loss over {PERF_LOSS:.0%}."
        )
        head = [
            MARKER,
            f"## AMD PR test: {ICON[overall]} {overall}",
            "",
            f"Models: {', '.join(f'`{k}`' for k in sorted(scope['models']))} · {gain_note}{run}",
            "",
            "| | Check | Result |",
            "|---|---|---|",
        ]
        head += [f"| {ICON[s]} | {n} | {t} |" for n, s, t in rule_rows]
    report = "\n".join(head) + "\n\n" + "\n\n".join(sections) + "\n"
    if args.out:
        Path(args.out).write_text(report)
    print(report)
    sys.exit(1 if overall == FAIL else 0)


if __name__ == "__main__":
    main()
