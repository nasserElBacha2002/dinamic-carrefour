import subprocess
import threading
import queue
from typing import Dict, Optional, Iterator

class ProcessRunner:
    def __init__(self):
        self._procs: Dict[str, subprocess.Popen] = {}
        self._logs: Dict[str, "queue.Queue[str]"] = {}

    def start(self, run_id: str, cmd: list[str], cwd: Optional[str] = None, env: Optional[dict] = None):
        q: "queue.Queue[str]" = queue.Queue()
        self._logs[run_id] = q

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self._procs[run_id] = proc

        def _reader():
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line.rstrip("\n"))
            q.put("__PROCESS_DONE__")

        threading.Thread(target=_reader, daemon=True).start()

    def iter_logs(self, run_id: str) -> Iterator[str]:
        q = self._logs.get(run_id)
        if not q:
            return iter(())
        while True:
            line = q.get()
            yield line
            if line == "__PROCESS_DONE__":
                break

    def is_running(self, run_id: str) -> bool:
        proc = self._procs.get(run_id)
        return bool(proc and proc.poll() is None)

    def return_code(self, run_id: str) -> Optional[int]:
        proc = self._procs.get(run_id)
        return None if not proc else proc.poll()