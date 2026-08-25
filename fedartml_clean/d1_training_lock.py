from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "fedartml_clean" / ".d1_training.lock"
PID_FILE = LOCK_DIR / "pid"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire() -> None:
    if LOCK_DIR.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
        except Exception:
            raise RuntimeError(
                f"Invalid D1 training lock: {LOCK_DIR}"
            )

        if pid_alive(existing_pid):
            raise RuntimeError(
                f"REFUSE: D1 training already active as PID {existing_pid}"
            )

        shutil.rmtree(LOCK_DIR)

    LOCK_DIR.mkdir(exist_ok=False)
    PID_FILE.write_text(f"{os.getpid()}\n")


def release() -> None:
    if not LOCK_DIR.exists():
        return

    try:
        owner_pid = int(PID_FILE.read_text().strip())
    except Exception:
        return

    if owner_pid == os.getpid():
        shutil.rmtree(LOCK_DIR)
