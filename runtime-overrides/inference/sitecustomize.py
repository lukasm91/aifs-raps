"""Replace Anemoi Runner's call profiler with an elapsed timer."""

import cProfile
import os
import time


if os.environ.get("ANEMOI_DISABLE_RUNNER_CPROFILE") == "1":
    class _ElapsedProfile:
        """Preserve cProfile's report boundary without tracing Python calls."""

        def __init__(self, *args, **kwargs):
            self.stats = None
            self._started = None
            self._elapsed = 0.0

        def __enter__(self):
            self.enable()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.disable()
            return False

        def enable(self):
            if self._started is None:
                self._started = time.perf_counter()

        def disable(self):
            if self._started is not None:
                self._elapsed += time.perf_counter() - self._started
                self._started = None

        def create_stats(self):
            self.disable()
            # pstats rejects an empty mapping. A single pseudo-call retains its
            # familiar exact-seconds header while avoiding call instrumentation.
            self.stats = {
                ("~", 0, "forecast elapsed time (call profiling disabled)"):
                    (1, 1, self._elapsed, self._elapsed, {})
            }

    cProfile.Profile = _ElapsedProfile
