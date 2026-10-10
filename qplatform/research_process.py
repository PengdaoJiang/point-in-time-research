"""Owned process containment for the research supervisor (Windows first)."""
from __future__ import annotations

import ctypes
import os
import signal
from ctypes import wintypes


class ProcessGroup:
    """Assign the blocked bootstrap before it is allowed to spawn science code."""

    def __init__(self, pid: int, max_memory_bytes: int) -> None:
        self.pid = pid
        self.handle = None
        if os.name != "nt":
            return
        size = ctypes.c_size_t

        class Basic(ctypes.Structure):
            _fields_ = [
                ("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_working_set", size),
                ("max_working_set", size), ("active_process_limit", wintypes.DWORD),
                ("affinity", size), ("priority", wintypes.DWORD),
                ("scheduling_class", wintypes.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Basic), ("io", IO), ("process_memory", size),
                ("job_memory", size), ("peak_process_memory", size), ("peak_job_memory", size),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel = kernel
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        info = Extended()
        # Kill-on-close covers grandchildren even if the controller crashes.
        info.basic.flags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        info.job_memory = max_memory_bytes
        process = None
        try:
            if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            process = kernel.OpenProcess(0x0001 | 0x0100, False, pid)
            if not process or not kernel.AssignProcessToJobObject(handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise
        finally:
            if process:
                kernel.CloseHandle(process)

    def close(self) -> None:
        if os.name == "nt":
            if self.handle is not None:
                self.kernel.CloseHandle(self.handle)
                self.handle = None
        else:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> int:
    import json
    import subprocess
    import sys
    import threading

    # No work or child process exists until the parent has assigned its job.
    message = sys.stdin.readline()
    if not message:
        return 125
    command = json.loads(message)
    if os.name != "nt":
        def parent_watch() -> None:
            sys.stdin.read()
            os.killpg(os.getpgrp(), signal.SIGKILL)
        threading.Thread(target=parent_watch, daemon=True).start()
    return subprocess.call(command, stdin=subprocess.DEVNULL, shell=False)


if __name__ == "__main__":
    raise SystemExit(main())
