# AMD PR test and nightly

Two workflows run the checks of the PR review guideline on two dedicated MI355X
nodes:

- `pr-test-amd.yml`, the **PR test**: a maintainer starts it. It gates the PR on
  its tests, GSM8K with the monokernels on and off, and speed against the base
  branch.
- `nightly-amd.yml`, the **nightly**: every day, on main and the newest SGLang
  ROCm image, every accuracy benchmark and the speed test with the monokernels on
  and off, for the models marked `nightly` in `models.py`. Its off results fill the
  cache the PR test reads.

## Running it

- Add the `run-ci` label to the PR. The run removes the label when it starts, so a
  later push needs a maintainer again.
- Or run the workflow (Actions, "PR Test (AMD)", "Run workflow") with the PR number.
  `models` runs the named models instead of the PR's scope (`common` for the unit
  and common kernel tests); `image` pins the SGLang image.

The run sets the `AMD PR test` status on the PR's head commit and keeps one PR
comment up to date with the report. Each model's logs (servers, benchmarks,
tests) are in the run's `mono-ci-<model>` artifact. The nightly writes its report
to the run's summary.

## What runs

`scope.py` maps a PR's changed files to models (`models.py`):

- a model whose own package (`sgl_amd_mono/<pkg>/`, or its tests) changed runs
  the **full** tier;
- a change to `sgl_amd_mono/common/`, `sgl_amd_mono/__init__.py` or
  `pyproject.toml` runs every other model's **smoke** tier, and `common`;
- a PR that changes no model runs `common`: the unit tests and the common kernel
  tests.

Every server is the model's recipe under `test/e2e`, from the PR. `run_model.py`
installs each checkout in turn over the image's sgl-amd-mono and serves it on
these sides:

- **base**: the base branch, monokernels on (PR test only);
- **on**: the PR (main in the nightly), monokernels on;
- **off**: the same build, monokernels off (SGLang's own path). It does not
  depend on the package, so its results are cached per image and recipe.

| Check | PR test, full tier | PR test, smoke tier | Nightly |
| --- | --- | --- | --- |
| Tests | unit, the model's kernel tests and serving tests (`test/e2e`) | same | same |
| Kernel microbenchmark | the model's timing scripts, base and PR | same | main |
| Accuracy | sgl-eval GSM8K (1319 questions), on and off | the serving tests' GSM8K and KL bounds | GSM8K, GPQA Diamond, AIME26 x4 (and MMMU-Pro for vision models), on and off |
| E2E performance | InferenceX fixed length at the model's concurrencies, base and on | the gate concurrencies only | off and on |
| Scope | the server's startup lines that say which steps the launches serve | same | same |

E2E performance is `sglang.benchmark.serving`: random 8k in / 1k out, range
ratio 0.8, 10 prompts a slot. With the monokernels on, the accuracy benchmarks
send no more requests at once than the launches cover (`gate_concurrency`): a
wider step takes SGLang's own path, and its score would not be the kernels'.

A model new in the PR has no base package: its speed is compared against the
off side.

## The gate

The PR test's status fails when:

- a test fails, or a serving test, benchmark or server does not finish;
- the monokernel server logs no routing (every step ran SGLang's own path);
- GSM8K loses significantly with the monokernels on: of the questions the on and
  off runs answer differently, the on run gets more wrong, and an exact one-sided
  McNemar test gives p < 0.05;
- median TPOT is more than 2% worse than the base branch at any concurrency;
- for a `[Perf]` or `[Feat]` PR, no gate concurrency gains 5% or more in median
  TPOT;
- the PR changes existing lines of `.github/CODEOWNERS` (a new model's added
  owner line is a warning for the reviewer).

The nightly fails the same way on tests and GSM8K. GPQA, AIME26 and MMMU-Pro have
too few questions to gate on: a score more than its `warn` margin below the off
run's is a warning, as is a slowdown with the monokernels on.

Warnings do not fail it: a model without a checkpoint on the CI nodes, a GPU the
nodes do not have, a model whose SGLang support has not merged (`blocked`), a
benchmark the CI cannot run yet, or a model package without a `README.md` (Doc).
Until the CI covers a check, the author runs it and gives the results in the PR,
and the reviewer verifies them. OCRBench and LiveCodeBench are not in the CI yet
(sgl-eval does not run them); SWE-bench Verified stays with the author.

## Infrastructure

- Nodes: two MI355X nodes of a Slurm cluster, kept for this CI.
- Runners: two, on the cluster's login node (`runner.sh`), label
  `sgl-amd-mono-mi355x`, each pinned to one node. A runner takes one job at a time,
  so the two nodes run two models at once and never two jobs on one node.
- `slurm_launch.sh` (on the runner) submits `slurm_job.sh` as a whole-node job,
  streams its log and collects its results; `slurm_job.sh` (on the node) fetches
  the PR and base commits onto node-local disk, starts the SGLang ROCm container
  and runs `run_model.py` in it. The CI scripts come from the base branch; only
  the package and its tests come from the PR.
- Image: the newest `rocm/sgl-dev:v*-rocm10-mi35x-YYYYMMDD`, unless the run pins one.
- Checkpoints: under `$CI_MODEL_ROOT` on shared storage (`paths` in `models.py`).
  Shared storage reads too slowly to load servers from, so a job first mirrors its
  model's checkpoints to the node's local disk (once; again when the source's
  config or index changes) and the container sees the mirror at the same paths.
  `CI_STAGE_ONLY=1` makes a job only fill the mirror.
- GPQA is a gated dataset: the job fills its cache with the token in
  `~/.sgl_amd_mono_ci/hf_token` on the runner account, before any PR code runs.

## Adding a model

Add an entry to `MODELS` in `models.py`: its packages, the server (a recipe module
under `test/e2e` and its argument list), the checkpoint paths on the CI nodes, its
kernel and serving tests, timing scripts, accuracy benchmarks and concurrencies
(`gate_concurrency`: the ones its launches cover).

## Trial runs

`MONO_CI_NUM_EXAMPLES` (examples a benchmark) and `MONO_CI_CONCURRENCY`
(comma-separated) shorten a run of the CI itself; such a run does not touch the
accuracy cache.
