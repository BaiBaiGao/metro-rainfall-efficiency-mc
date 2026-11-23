"""Checkpointed batch runner, identical to 投稿结果重算_两步选路_20261118\\checkpoint.py
except that the 17:20 office-hours pause is removed (cloud runners run to completion)."""
import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAUSE = ROOT / "暂停.request"


def stop_requested():
    return PAUSE.exists()


def checked_batches(parallel, tasks, directory, count, batch_size, sim_duration):
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        checkpoint = directory / f"runs_{start + 1:03d}_{end:03d}.pickle"
        if checkpoint.exists():
            with checkpoint.open("rb") as handle:
                results = pickle.load(handle)
        else:
            if stop_requested():
                print(f"安全暂停，下一批为 {directory.name} Run_{start + 1}–Run_{end}", flush=True)
                raise SystemExit(75)
            results = parallel(tasks(run) for run in range(start, end))
        if [result[0] for result in results] != list(range(start + 1, end + 1)):
            raise ValueError(f"Run identifiers do not match {checkpoint}")
        for result in results:
            if result[1] is None:
                raise RuntimeError(f"Run_{result[0]}: {result[2]}")
            if len(result[1]) != sim_duration:
                raise ValueError(f"Run_{result[0]} has an incomplete efficiency series")
        if not checkpoint.exists():
            pending = checkpoint.with_suffix(".pending")
            with pending.open("wb") as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            pending.replace(checkpoint)
        print(f"已校验落盘 {directory.name} Run_{start + 1}–Run_{end}", flush=True)
        yield from results
