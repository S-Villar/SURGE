# -*- coding: utf-8 -*-
"""Console/log helpers for live ROSE+SURGE demo progress."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from dataset_utils import workspace_dir


def default_log_path(example_name: str) -> Path:
    path = workspace_dir(example_name) / "execution.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reset_log_file(path: str | Path) -> Path:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    return log_path


@contextlib.contextmanager
def capture_output_to_log(path: str | Path) -> Iterator[None]:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            yield


class LiveProgress:
    """Small wrapper around tqdm so demos can update one clean console block."""

    def __init__(
        self,
        *,
        total: int,
        desc: str,
        enabled: bool = True,
        unit: str = "task",
    ) -> None:
        self.enabled = enabled
        self._bar = None
        if enabled:
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                leave=True,
                file=sys.__stderr__,
            )

    def update(self, advance: int = 1, **postfix: object) -> None:
        if not self._bar:
            return
        if postfix:
            self._bar.set_postfix(postfix, refresh=False)
        self._bar.update(advance)
        self._bar.refresh()

    def message(self, text: str) -> None:
        if self._bar:
            self._bar.write(text, file=sys.__stderr__)
        else:
            print(text, flush=True)

    def close(self) -> None:
        if self._bar:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "LiveProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
