"""One model's AMD PR test or nightly, inside the SGLang ROCm container on a whole
CI node.

    python3 scripts/ci/run_model.py --mode pr --model deepseek_v41 \\
        --pr /ci/pr --base /ci/base --out /ci/out --cache /cache

Each checkout is installed in turn over the image's sgl-amd-mono (editable), and
every server takes the PR's recipe (``test/e2e``), so the sides of a comparison
differ only in the package or in ``--enable-amd-monokernel``. Sides:

* ``base``: the base branch, monokernels on (PR mode): the timing scripts and the
  speed test;
* ``on``: the PR, monokernels on: the speed test and the accuracy benchmarks;
* ``off``: the PR, monokernels off (SGLang's own path): the accuracy benchmarks.
  It does not depend on the package, so a result for the same image and recipe
  is cached and reused (the nightly fills the cache).

PR mode: base, then the unit, kernel and timing tests, on and off (GSM8K only),
then the model's serving tests. A model new in the PR has no base package; its
off side runs the speed test as its base. The smoke tier (a model that runs only
because shared code changed) runs no accuracy benchmark, whose bound its serving
tests hold, and times only the gate concurrencies.

Nightly mode (``--pr`` and ``--base`` both main): the tests, then on and off with
the model's benchmarks but the weekly ones, and the speed test on both sides.
Weekly mode: the same with every benchmark.

Every step runs whatever an earlier one did; ``result.json`` (rewritten after each
step) and ``logs/`` hold the outcome. ``report.py`` judges it.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    COMMON_KERNEL_TESTS,
    EVALS,
    GPU,
    ISL,
    MODELS,
    OSL,
    PR_EVALS,
    PROMPTS_PER_SLOT,
    RANGE_RATIO,
)

PORT = 31000
LAUNCH_TIMEOUT = 3600
SGL_EVAL = "sgl-eval==0.1.3"
# Trial runs of the CI itself: fewer examples a benchmark, other concurrencies.
NUM_EXAMPLES = os.environ.get("MONO_CI_NUM_EXAMPLES")
CONCURRENCY = os.environ.get("MONO_CI_CONCURRENCY")


def per_question(run_dir):
    """{question id: fraction of its repeats answered right}, from sgl-eval's
    per-sample dumps (output-rs<repeat>.jsonl), for the paired GSM8K test."""
    right, seen = {}, {}
    for f in run_dir.glob("output-rs*.jsonl"):
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ok = row.get("symbolic_correct", row.get("is_correct"))
            if ok is None:
                continue
            q = str(row["id"])
            right[q] = right.get(q, 0) + bool(ok)
            seen[q] = seen.get(q, 0) + 1
    return {q: right[q] / seen[q] for q in seen}


class Run:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.key = args.model
        self.m = MODELS.get(args.model, {})
        self.result = {
            "model": args.model,
            "mode": args.mode,
            "tier": args.tier,
            "name": self.m.get("name", "common"),
            "image": os.environ.get("CI_IMAGE", ""),
            "node": os.uname().nodename,
            "pr_sha": os.environ.get("CI_PR_SHA", ""),
            "base_sha": os.environ.get("CI_BASE_SHA", ""),
            "skipped": {},
            "tests": {},
            "microbench": {},
            "perf": {},
            "accuracy": {},
            "routing": {},
        }
        for k, v in self.m.get("paths", {}).items():
            os.environ[k] = v

    # -- plumbing -------------------------------------------------------------

    def save(self):
        (self.out / "result.json").write_text(json.dumps(self.result, indent=2))

    def log(self, msg):
        print(f"[mono-ci {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def sh(self, cmd, name, cwd=None, env=None, timeout=None, echo=True):
        """Run ``cmd`` (a string, through bash) with its output in logs/<name>.log,
        and echoed when ``echo``. Returns the exit code; 124 on timeout."""
        path = self.logs / f"{name}.log"
        self.log(f"$ {cmd}  (cwd={cwd or os.getcwd()}, log={path.name})")
        argv = ["bash", "-c", cmd]
        if timeout:
            argv = ["timeout", "--kill-after=60", str(timeout), *argv]
        with open(path, "a") as f:
            p = subprocess.Popen(
                argv,
                cwd=cwd,
                env={**os.environ, **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            for line in p.stdout:
                f.write(line)
                if echo:
                    sys.stdout.write(line)
            rc = p.wait()
        try:  # what the step left running (torchrun ranks, servers)
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.log(f"exit {rc}{' (timed out)' if rc == 124 else ''}: {name}")
        return rc

    def install(self, src):
        """Make ``src`` the installed sgl-amd-mono, with its own kernel cache."""
        side = Path(src).name
        os.environ["SGLANG_AMD_MONO_CACHE_DIR"] = f"/tmp/sgl-amd-mono-cache-{side}"
        rc = self.sh(
            "pip uninstall -y -q sgl-amd-mono; "
            f"pip install -q --no-deps --no-build-isolation -e {shlex.quote(str(src))}",
            f"install-{side}",
        )
        if rc:
            raise RuntimeError(f"installing {src} failed")

    # -- tests ----------------------------------------------------------------

    def pytest(self, name, files, timeout):
        files = [f for f in files if (Path(self.args.pr) / f).exists()]
        if not files:
            return
        xml = self.logs / f"{name}.xml"
        rc = self.sh(
            f"python3 -m pytest -v -rA --junitxml={xml} " + " ".join(files),
            name,
            cwd=self.args.pr,
            timeout=timeout,
        )
        counts = {"passed": 0, "failed": 0, "skipped": 0, "rc": rc}
        if xml.exists():
            for case in ET.parse(xml).getroot().iter("testcase"):
                tags = {c.tag for c in case}
                if tags & {"failure", "error"}:
                    counts["failed"] += 1
                elif "skipped" in tags:
                    counts["skipped"] += 1
                else:
                    counts["passed"] += 1
        elif rc:
            counts["failed"] = 1  # collection or the run itself failed
        if rc == 124:
            counts["failed"] = max(counts["failed"], 1)
            counts["timeout"] = True
        self.result["tests"][name] = counts
        self.save()

    # -- servers --------------------------------------------------------------

    def recipe(self):
        """(model path, env, args) of the model's server, from the PR's tree."""
        m = self.m
        if "recipe" in m:
            mod, attr = m["recipe"]
            path = Path(self.args.pr) / "test" / "e2e" / f"{mod}.py"
            spec = importlib.util.spec_from_file_location(mod, path)
            r = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(r)
            return r.MODEL, dict(r.ENV), list(getattr(r, attr))
        return os.environ[m["model_env"]], dict(m["env"]), list(m["args"])

    def model_ready(self):
        """Why this model's server cannot run here, or None."""
        if self.m.get("blocked"):
            return self.m["blocked"]
        if GPU not in self.m.get("gpus", [GPU]):
            return f"the launches cover {', '.join(self.m['gpus'])} only; CI nodes are {GPU}"
        if not self.m.get("paths") or not ("recipe" in self.m or "args" in self.m):
            return "no checkpoint or serving recipe on the CI nodes"
        missing = [p for p in self.m["paths"].values() if not Path(p).exists()]
        if missing:
            return f"checkpoint not on the CI nodes: {', '.join(missing)}"
        return None

    def serve(self, side, mono=True):
        model, env, args = self.recipe()
        args = [*args, "--host", "127.0.0.1", "--port", str(PORT)]
        if mono:
            args.append("--enable-amd-monokernel")
        log = open(self.logs / f"server-{side}.log", "w")
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            *args,
        ]
        self.log(f"server {side}: {shlex.join(cmd)}  env={env}")
        p = subprocess.Popen(
            cmd,
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + LAUNCH_TIMEOUT
        while time.time() < deadline:
            if p.poll() is not None:
                self.log(f"server {side} exited {p.returncode} during startup")
                return None, model, args
            try:
                if requests.get(
                    f"http://127.0.0.1:{PORT}/health_generate", timeout=60
                ).ok:
                    self.log(f"server {side} ready")
                    return p, model, args
            except requests.RequestException:
                pass
            time.sleep(10)
        self.log(f"server {side} not ready after {LAUNCH_TIMEOUT}s")
        self.stop(p)
        return None, model, args

    def stop(self, p):
        if p is None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=120)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        subprocess.run("pkill -9 -f sglang.launch_server; sleep 15", shell=True)

    def routing(self, side):
        """The startup lines that say which steps the launches serve."""
        text = (self.logs / f"server-{side}.log").read_text(errors="replace")
        lines = sorted(set(re.findall(r"monokernel \S+ on \S+ runs [^\n]*", text)))
        self.result["routing"][side] = lines
        self.save()

    def perf(self, side, model):
        res = self.result["perf"].setdefault(side, {})
        smoke = self.args.tier == "smoke"
        levels = self.m["gate_concurrency" if smoke else "concurrency"]
        if CONCURRENCY:
            levels = [int(c) for c in CONCURRENCY.split(",")]
        for c in levels:
            out = self.logs / f"bench-{side}-c{c}.jsonl"
            out.unlink(missing_ok=True)
            n = max(c * PROMPTS_PER_SLOT, PROMPTS_PER_SLOT)
            rc = self.sh(
                f"python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 "
                f"--port {PORT} --model {shlex.quote(model)} --dataset-name random "
                f"--random-input-len {ISL} --random-output-len {OSL} "
                f"--random-range-ratio {RANGE_RATIO} --max-concurrency {c} "
                f"--num-prompts {n} --warmup-requests {c} --output-file {out}",
                f"bench-{side}-c{c}",
                timeout=3 * 3600,
            )
            rows = out.read_text().splitlines() if out.exists() else []
            if rc or not rows:
                res[str(c)] = {"error": f"exit {rc}"}
            else:
                d = json.loads(rows[-1])
                res[str(c)] = {
                    k: d.get(k)
                    for k in (
                        "median_tpot_ms",
                        "median_ttft_ms",
                        "median_itl_ms",
                        "output_throughput",
                        "completed",
                    )
                }
            self.save()

    def evals(self, side, threads, names):
        res = self.result["accuracy"].setdefault(side, {})
        for name in names:
            spec = EVALS[name]
            out = self.logs / f"eval-{side}-{name}"
            cmd = (
                f"sgl-eval run {name} --base-url http://127.0.0.1:{PORT}/v1 "
                f"--num-threads {threads} --out-dir {out} " + " ".join(spec["args"])
            )
            if NUM_EXAMPLES:
                cmd += f" --num-examples {NUM_EXAMPLES}"
            log_path = self.logs / f"eval-{side}-{name}.log"
            rc = self.sh(cmd, f"eval-{side}-{name}", timeout=8 * 3600, echo=False)
            log = log_path.read_text(errors="replace")
            if "gated dataset" in log:
                # a gated dataset the job cached with the CI's token: read the cache
                log_path.unlink()
                rc = self.sh(
                    cmd,
                    f"eval-{side}-{name}",
                    env={"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
                    timeout=8 * 3600,
                    echo=False,
                )
                log = log_path.read_text(errors="replace")
            metrics = sorted(out.glob("*/metrics.json"))
            if "gated dataset" in log or (rc and "offline mode" in log.lower()):
                res[name] = {
                    "skipped": "gated dataset: the CI needs an HF token for it"
                }
            elif rc or not metrics:
                res[name] = {"error": f"exit {rc}"}
            else:
                d = json.loads(metrics[-1].read_text())
                agg = d.get("aggregate", {})
                # sgl-eval's headline: pass@1 (the mean over repeats) with repeats
                score = agg.get("pass@1", agg.get("score"))
                if d.get("partial") or not d.get("total_completion_tokens"):
                    res[name] = {"error": "no completions (partial or empty run)"}
                else:
                    res[name] = {
                        "score": score,
                        "n_repeats": d.get("n_repeats"),
                        "num_examples": d.get("num_examples"),
                        "no_answer": agg.get("no_answer"),
                        "per_question": per_question(metrics[-1].parent),
                    }
            self.log(f"{side} {name}: {res[name]}")
            self.save()

    def microbench(self, side, src):
        for i, cmd in enumerate(self.m.get("microbench", [])):
            name = f"microbench-{side}-{i}"
            rc = self.sh(cmd, name, cwd=src, timeout=3600)
            tail = (self.logs / f"{name}.log").read_text(errors="replace")[-4000:]
            self.result["microbench"].setdefault(side, []).append(
                {"cmd": cmd, "rc": rc, "tail": tail}
            )
            self.save()

    # -- the run --------------------------------------------------------------

    def off_cache(self, name):
        """The off side's cached result of benchmark ``name``: monokernels off, it
        depends only on the image and the server recipe. None on a trial run."""
        if NUM_EXAMPLES:
            return None
        image = re.sub(r"[^A-Za-z0-9._-]", "_", self.result["image"] or "unknown")
        recipe = hashlib.sha256(json.dumps(self.recipe()).encode()).hexdigest()[:16]
        return (
            Path(self.args.cache) / "off" / self.key / image / f"{name}-{recipe}.json"
        )

    def side(self, side, mono, perf=True, evals=()):
        """One server: its routing lines, the speed test and the benchmarks."""
        # the off side's cached benchmarks are not run again
        cached = {}
        if side == "off":
            for name in evals:
                path = self.off_cache(name)
                if path and path.is_file():
                    cached[name] = {**json.loads(path.read_text()), "cached": True}
            self.result["accuracy"].setdefault("off", {}).update(cached)
            evals = [e for e in evals if e not in cached]
        if not (perf or evals):
            self.save()
            return
        p, model, args = self.serve(side, mono)
        if p is None:
            if perf:
                self.result["perf"][side] = {"error": "server did not start"}
            for name in evals:
                self.result["accuracy"].setdefault(side, {})[name] = {
                    "error": "server did not start"
                }
            self.save()
            return
        try:
            self.routing(side)
            if perf:
                self.perf(side, model)
            if evals:
                # With the monokernels on, no more requests in flight than the
                # launches cover: wider steps take SGLang's own path, and the score
                # would not be the kernels'. Off, as many as the recipe runs.
                threads = max(self.m["gate_concurrency"], default=1)
                if not mono and "--max-running-requests" in args:
                    threads = int(args[args.index("--max-running-requests") + 1])
                self.evals(side, threads, evals)
        finally:
            self.stop(p)
        if side == "off":
            for name in evals:
                got = self.result["accuracy"]["off"].get(name, {})
                path = self.off_cache(name)
                if path and "score" in got:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(got))

    def run(self):
        a = self.args
        # the checkouts belong to the job's user, not the container's root
        self.sh("git config --global --add safe.directory '*'", "setup")
        self.sh(f"pip install -q {SGL_EVAL}", "setup")
        self.result["sglang"] = subprocess.run(
            "git -C /sgl-workspace/sglang log -1 --format='%h %cs %s'",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if self.key == "common":
            self.install(a.pr)
            self.pytest("unit", ["test/unit"], 1800)
            self.pytest("kernels", COMMON_KERNEL_TESTS, 3600)
            return
        why = self.model_ready()
        if why:
            self.result["skipped"]["serving"] = why
            self.skip_e2e(why)
        if a.mode in ("nightly", "weekly"):
            self.install(a.pr)
            self.tests()
            if not why:
                evals = [
                    e
                    for e in self.m.get("evals", [])
                    if a.mode == "weekly" or not EVALS[e].get("weekly")
                ]
                self.side("on", True, evals=evals)
                self.side("off", False, evals=evals)
                self.e2e()
            return

        base_has_pkg = all(
            (Path(a.base) / "sgl_amd_mono" / p).is_dir() for p in self.m["packages"]
        )
        self.result["baseline"] = (
            "base branch, monokernels on"
            if base_has_pkg
            else "monokernels off (new model)"
        )
        evals = [e for e in PR_EVALS if e in self.m.get("evals", [])]
        if a.tier != "full":
            evals = []
        if base_has_pkg:
            self.install(a.base)
            self.microbench("base", a.base)
            if not why:
                self.side("base", True)
        self.install(a.pr)
        self.tests()
        if not why:
            self.side("on", True, evals=evals)
            # a new model's speed baseline is SGLang's own path
            self.side("off", False, perf=not base_has_pkg, evals=evals)
            if not base_has_pkg and "off" in self.result["perf"]:
                self.result["perf"]["base"] = self.result["perf"].pop("off")
            self.e2e()

    def tests(self):
        """The unit, kernel and timing tests of the installed PR."""
        self.pytest("unit", ["test/unit"], 1800)
        self.pytest("kernels", self.m.get("kernel_tests", []), 2 * 3600)
        self.microbench("pr", self.args.pr)

    def e2e(self):
        self.pytest("e2e", self.m.get("e2e_tests", []), 4 * 3600)

    def skip_e2e(self, why):
        if self.m.get("e2e_tests"):
            self.result["skipped"]["e2e"] = why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pr", required=True, help="the PR's checkout")
    ap.add_argument("--base", required=True, help="the base branch's checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cache", required=True, help="persistent: pip, datasets, baselines"
    )
    ap.add_argument("--tier", choices=("full", "smoke"), default="full")
    ap.add_argument("--mode", choices=("pr", "nightly", "weekly"), default="pr")
    args = ap.parse_args()
    if args.model != "common" and args.model not in MODELS:
        sys.exit(f"unknown model {args.model}")
    run = Run(args)
    try:
        run.run()
    except Exception as e:  # report what ran; the report marks the error
        run.result["error"] = f"{type(e).__name__}: {e}"
        run.log(run.result["error"])
    finally:
        run.result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run.save()


if __name__ == "__main__":
    main()
