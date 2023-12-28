"""Animated spinner for long-running API calls.

Displays a Braille-pattern animation on stderr while basic_turn() runs.
Monkey-patches builtins.print so debug output from sub-modules prints cleanly
without garbling the spinner line.
"""

import sys
import threading
import itertools
import builtins

# Braille spinner frames — render well across macOS Terminal, iTerm2, VS Code, Windows Terminal
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_INTERVAL = 0.08  # seconds between frames


class Spinner:
    """Animated terminal spinner that cooperates with print() calls.

    Usage:
        with Spinner("🤔 Thinking..."):
            result = some_blocking_call()

    While active, any print() call from any module will:
    1. Clear the spinner line
    2. Print the output normally
    3. Let the spinner redraw on the next frame

    Skips animation entirely if stderr is not a terminal (e.g., piped/redirected).
    """

    def __init__(self, message: str = "🤔 Thinking..."):
        self._message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._original_print = builtins.print
        self._last_line_len = 0
        self._active = False

    def __enter__(self):
        self._stop_event.clear()

        # Skip spinner in non-interactive mode (piped/redirected stderr)
        if not sys.stderr.isatty():
            return self

        self._active = True
        builtins.print = self._patched_print
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if not self._active:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        # Clear the spinner line
        self._clear_line()

        # Restore original print
        builtins.print = self._original_print
        self._active = False

    def _spin(self):
        """Animation loop — runs on a daemon thread."""
        frames = itertools.cycle(SPINNER_FRAMES)
        while not self._stop_event.is_set():
            frame = next(frames)
            line = f"\r  {frame} {self._message}"
            with self._lock:
                sys.stderr.write(line)
                sys.stderr.flush()
                self._last_line_len = len(line)
            self._stop_event.wait(SPINNER_INTERVAL)

    def _clear_line(self):
        """Erase the spinner line from the terminal."""
        with self._lock:
            sys.stderr.write("\r" + " " * self._last_line_len + "\r")
            sys.stderr.flush()
            self._last_line_len = 0

    def _patched_print(self, *args, **kwargs):
        """Wrapper around print() that clears the spinner before writing output."""
        with self._lock:
            # Clear spinner line on stderr
            sys.stderr.write("\r" + " " * self._last_line_len + "\r")
            sys.stderr.flush()
            # Call the real print (writes to stdout)
            self._original_print(*args, **kwargs)
