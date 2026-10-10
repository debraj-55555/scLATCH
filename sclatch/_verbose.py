"""Output control for scLATCH.

Levels
------
0  silent   — nothing except errors
1  quiet    — milestones only (default): what was loaded, training progress bar,
              convergence, what was written
2  full     — every internal step (the historical behaviour; useful for debugging
              and for reproducing the logs referenced in the paper)

Use ``sclatch.set_verbosity(2)`` to restore the full logs.
"""
import os
import sys

_LEVEL = int(os.environ.get("SCLATCH_VERBOSITY", "1"))


def set_verbosity(level: int) -> None:
    """Set output level: 0 silent, 1 quiet (default), 2 full."""
    global _LEVEL
    _LEVEL = int(level)


def get_verbosity() -> int:
    return _LEVEL


def log(msg="", level=2, **kw):
    """Print ``msg`` only when the current verbosity is at least ``level``."""
    if _LEVEL >= level:
        print(msg, **kw)


def milestone(msg="", **kw):
    """A headline event the user should always see at the default level."""
    log(msg, level=1, **kw)


def quiet_external_warnings() -> None:
    """Silence noisy third-party warnings that say nothing about scLATCH.

    Covers the pynvml deprecation emitted by torch.cuda, anndata/scanpy
    FutureWarnings, and the h5py/numpy binary-compat notices.
    """
    import warnings
    warnings.filterwarnings("ignore", message=".*pynvml.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="anndata.*")
    warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")
    warnings.filterwarnings("ignore", message=".*NVML.*")
    warnings.filterwarnings("ignore", message=".*zero-centering a sparse array.*")
    warnings.filterwarnings("ignore", message=".*densifies it.*")
    warnings.filterwarnings("ignore", message=".*is_categorical_dtype.*")
    warnings.filterwarnings("ignore", message=".*observed=False.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def progress(iterable, total=None, desc="training", disable=None):
    """tqdm progress bar that respects the verbosity level.

    Falls back to the plain iterable if tqdm is unavailable.
    """
    if disable is None:
        disable = _LEVEL < 1
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    # When stdout is not a terminal (notebook via subprocess, log file, CI) the
    # carriage-return redraws become separate lines, so throttle hard: a handful of
    # updates instead of one per step. On a TTY keep it smooth.
    _tty = sys.stdout.isatty()
    return tqdm(iterable, total=total, desc=desc, disable=disable,
                leave=True, dynamic_ncols=_tty,
                ncols=None if _tty else 80,
                mininterval=0.1 if _tty else 20.0,
                miniters=0 if _tty else max(1, (total or 1) // 8),
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")
