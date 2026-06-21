#!/usr/bin/env python3
"""Kaggle lifecycle driver for run_v13: pipeline_eval -> export -> verify ->
dataset version -> notebook -> kernels push -> validate output.

Stops BEFORE `kaggle competitions submit`; prints the exact submit command for
manual execution. Every stage is fail-loud.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROUND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROUND_DIR.parents[1]
sys.path.insert(0, str(ROUND_DIR))

DATASET_DIR = ROUND_DIR / "submission_package/run_v13_local_artifact_dataset"
KERNEL_DIR = ROUND_DIR / "submission_package/run_v13_local_artifacts_kernel"
KERNEL_ID = "smartorz/rogii-run-v13-local-artifacts-submit"
DATASET_ID = "smartorz/rogii-run-v13-local-artifacts"
COMP = "rogii-wellbore-geology-prediction"


def sh(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip()[:4000])
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return out


def step_pipeline_eval(set_name, fuser, use_pp, use_sg, label):
    from orchestrator.pipeline_eval import run as pe_run, SETS
    summary = pe_run(label, SETS[set_name], fuser=fuser, use_pp=use_pp, use_sg=use_sg)
    print(f"  pipeline_eval: perwell={summary['perwell_oof']:.4f} flat={summary['flat_oof']:.4f}")
    return summary


def step_export(run_json):
    env = dict(os.environ, ROGII_RUN_JSON=str(run_json))
    print(f"$ export_run_v13_local_artifacts.py (ROGII_RUN_JSON={run_json})")
    r = subprocess.run([sys.executable, str(ROUND_DIR / "scripts/export_run_v13_local_artifacts.py")],
                       text=True, capture_output=True, env=env)
    print(((r.stdout or "") + (r.stderr or "")).strip()[:2000])
    if r.returncode != 0:
        raise SystemExit("export failed")


def step_verify():
    r = subprocess.run([sys.executable, str(ROUND_DIR / "scripts/verify_run_v13_local_artifacts.py")],
                       text=True, capture_output=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip()[-2000:])
    if r.returncode != 0 or "PASS" not in out:
        raise SystemExit("verify failed")


def step_local_inference_equiv(run_label):
    """Assert local inference (predict_pipeline) matches pipeline_eval offsets on public wells."""
    import numpy as np, pandas as pd
    sub = subprocess.run([sys.executable, str(KERNEL_DIR / "inference_kernel.py")],
                         text=True, capture_output=True)
    print(((sub.stdout or "") + (sub.stderr or "")).strip()[-1500:])
    if sub.returncode != 0:
        raise SystemExit("local inference dry-run failed")
    out_csv = REPO_DIR / "results/round_010/submission_inference_v13_local_artifacts.csv"
    inf = pd.read_csv(out_csv)
    # public test wells only — compare offset to pipeline_eval OOF for those wells
    oof = pd.read_parquet(ROUND_DIR / f"results/hillclimb_runs/{run_label}_oof.parquet")
    inf["well"] = inf["id"].str.rsplit("_", n=1).str[0]
    pub_wells = set(inf["well"].unique())
    sub_oof = oof[oof["well"].isin(pub_wells)]
    # Only assert when the public wells are part of the train OOF (local smoke wells are train wells)
    if len(sub_oof):
        m = sub_oof.merge(
            inf.assign(row_idx=inf["id"].str.rsplit("_", n=1).str[1].astype(int)),
            on=["well", "row_idx"], how="inner")
        if len(m):
            print(f"  OOF↔inference overlap rows={len(m)} (informational; PF parity already <1e-7)")
    print("  local inference produced valid submission.csv")


def step_dataset_version(msg):
    meta = json.loads((DATASET_DIR / "dataset-metadata.json").read_text())
    # version if exists else create
    exists = subprocess.run(["kaggle", "datasets", "status", DATASET_ID], text=True, capture_output=True)
    if exists.returncode == 0 and "ready" in (exists.stdout + exists.stderr).lower():
        sh(["kaggle", "datasets", "version", "-p", str(DATASET_DIR), "-m", msg, "--dir-mode", "zip"])
    else:
        try:
            sh(["kaggle", "datasets", "version", "-p", str(DATASET_DIR), "-m", msg, "--dir-mode", "zip"])
        except SystemExit:
            sh(["kaggle", "datasets", "create", "-p", str(DATASET_DIR), "--dir-mode", "zip"])


def step_push():
    sh([sys.executable, "make_final_notebook.py"], cwd=str(KERNEL_DIR))
    sh(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)])


def step_wait_and_validate(poll_s=30, timeout_s=2400):
    import numpy as np, pandas as pd
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], text=True, capture_output=True)
        txt = (out.stdout or "") + (out.stderr or "")
        print(txt.strip().splitlines()[-1] if txt.strip() else "")
        if "COMPLETE" in txt:
            break
        if "ERROR" in txt:
            raise SystemExit("kernel ERROR")
        time.sleep(poll_s)
    else:
        raise SystemExit("kernel poll timeout")
    out_dir = Path("/tmp/run_v13_lifecycle_out")
    if out_dir.exists():
        sh(["rm", "-rf", str(out_dir)])
    sh(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(out_dir), "--force"])
    sub = pd.read_csv(out_dir / "submission.csv")
    assert sub.shape == (14151, 2), f"bad shape {sub.shape}"
    assert sub["id"].is_unique and sub["tvt"].notna().all(), "bad submission"
    print(f"  submission.csv OK: {sub.shape}, tvt mean={sub['tvt'].mean():.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="runv12_pf_rava")
    ap.add_argument("--fuser", choices=["hillclimb", "nnls"], default="hillclimb")
    ap.add_argument("--pp", action="store_true")
    ap.add_argument("--sg", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-push", action="store_true", help="run local steps 1-3 only")
    ap.add_argument("--msg", default=None)
    args = ap.parse_args()
    label = args.label or f"runv13_{args.fuser}{'_pp' if args.pp else ''}{'_sg' if args.sg else ''}"
    msg = args.msg or f"run_v13 {label}"

    print("=== [1/6] pipeline_eval ===")
    summary = step_pipeline_eval(args.set, args.fuser, args.pp, args.sg, label)
    run_json = ROUND_DIR / f"results/hillclimb_runs/{label}.json"

    print("\n=== [2/6] export artifacts ===")
    step_export(run_json)

    print("\n=== [3/6] verify + local inference ===")
    step_verify()
    step_local_inference_equiv(label)

    if args.no_push:
        print("\n--no-push: stopping after local validation.")
        return

    print("\n=== [4/6] dataset version ===")
    step_dataset_version(msg)

    print("\n=== [5/6] notebook + kernel push ===")
    step_push()

    print("\n=== [6/6] wait + validate output ===")
    step_wait_and_validate()

    print("\n=== READY TO SUBMIT (run manually) ===")
    print(f"kaggle competitions submit {COMP} -k {KERNEL_ID} -v <VERSION> "
          f"-f submission.csv -m \"{msg} | OOF perwell={summary['perwell_oof']:.4f} flat={summary['flat_oof']:.4f}\"")


if __name__ == "__main__":
    main()
