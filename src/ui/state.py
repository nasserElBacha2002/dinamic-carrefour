from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional
import json
import sys

# Asegurar que la raíz del proyecto esté en el path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

STATE_FILE = project_root / "src" / "ui" / "runs.json"

@dataclass
class RunState:
    run_id: str
    video: str
    output_dir: Optional[str] = None
    status: str = "created"   # created|running|done|error
    reviewed: bool = False
    absorbed: bool = False

class StateStore:
    def __init__(self):
        self.runs: Dict[str, RunState] = {}
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for r in data:
                self.runs[r["run_id"]] = RunState(**r)

    def _save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps([asdict(r) for r in self.runs.values()], indent=2),
            encoding="utf-8"
        )

    def upsert(self, run: RunState):
        self.runs[run.run_id] = run
        self._save()

    def get(self, run_id: str) -> Optional[RunState]:
        return self.runs.get(run_id)