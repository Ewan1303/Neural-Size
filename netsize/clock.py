"""A CPU-time clock.

Training time is measured as CPU time (time this process actually spent on a
core), not wall-clock time, so that somebody using the computer while the
experiments run does not distort the numbers.

Python's own time.process_time() only ticks every 15.6 ms on Windows, too coarse
for the smallest networks here. So on Windows the clock reads the processor's
cycle counter for this process (QueryProcessCycleTime) and converts cycles to
seconds. Everywhere else time.process_time() is already fine-grained and is used
directly.
"""
import os
import time

_ENV = "NEURAL_SIZE_CYCLES_PER_SECOND"


def _windows_cycle_clock():
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.QueryProcessCycleTime.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ulonglong)]
    k.QueryProcessCycleTime.restype = wintypes.BOOL
    handle = k.GetCurrentProcess()
    buf = ctypes.c_ulonglong()

    def cycles():
        if not k.QueryProcessCycleTime(handle, ctypes.byref(buf)):
            raise OSError("QueryProcessCycleTime failed")
        return buf.value

    rate = float(os.environ.get(_ENV, 0) or 0)
    if rate <= 0:
        # Cycles per second: spin for a moment and compare with the wall clock. If the
        # spin gets interrupted the ratio comes out low, so the highest of a few tries is kept.
        for _ in range(10):
            c0, t0 = cycles(), time.perf_counter()
            while time.perf_counter() - t0 < 0.03:
                pass
            c1, t1 = cycles(), time.perf_counter()
            rate = max(rate, (c1 - c0) / (t1 - t0))
        if rate <= 0:
            raise OSError("could not calibrate the cycle counter")
        os.environ[_ENV] = repr(rate)      # worker processes inherit the same calibration
    return lambda: cycles() / rate


def _make_clock():
    if os.name == "nt":
        try:
            return _windows_cycle_clock()
        except Exception:
            pass
    return time.process_time


cpu_seconds = _make_clock()
