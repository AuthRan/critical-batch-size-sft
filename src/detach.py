"""Start a command that outlives this SSH session, with its output in two files; prints its process id.

  python src/detach.py runs/x.log runs/x.err -- python src/sweep.py configs/a.json --shard queue

Windows OpenSSH puts a session's processes in a job object and ends them all when the session closes, which most
likely ended the sweeps on 2026-10-20, 10-25 and 10-27. Start-Process cannot leave that job;
CREATE_BREAKAWAY_FROM_JOB can, and the job on the GPU machine allows it (tested 2026-10-27).
"""
from __future__ import annotations

import subprocess
import sys

# CREATE_BREAKAWAY_FROM_JOB | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP; a hidden console, not DETACHED_PROCESS,
# because powershell.exe exits at once without a console
FLAGS = 0x01000000 | 0x08000000 | 0x00000200


def main(argv: list[str]) -> int:
    out, err, sep, *cmd = argv
    if sep != "--" or not cmd:
        sys.exit("usage: detach.py OUT ERR -- COMMAND [ARG ...]")
    with open(out, "wb") as o, open(err, "wb") as e:
        if sys.platform == "win32":
            p = subprocess.Popen(cmd, stdout=o, stderr=e, stdin=subprocess.DEVNULL, creationflags=FLAGS)
        else:
            p = subprocess.Popen(cmd, stdout=o, stderr=e, stdin=subprocess.DEVNULL, start_new_session=True)
    print(p.pid)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
