"""Explicit character windows; the adapter supplies the actual processor budget."""

from bisect import bisect_right
from collections.abc import Callable

from .config import MAX_WINDOWS
from .errors import AdapterContractError, TraceLimitError
from .observations import Window


def plan_windows(
    text: str,
    spans: list[tuple[int, int]],
    fits: Callable[[str], bool],
    target: int = 192,
    overlap: int = 32,
) -> tuple[Window, ...]:
    if not text or text.isspace():
        return ()
    # Ends of real processor words include every inter-word gap in the next slice.
    ends = sorted({end for start, end in spans if 0 <= start < end <= len(text)} | {len(text)})
    windows = []
    start = 0
    while start < len(text):
        if len(windows) >= MAX_WINDOWS:
            raise TraceLimitError("model")
        pos = bisect_right(ends, start)
        options = ends[pos : pos + target]
        end = options[-1]
        if not fits(text[start:end]):
            # Prefer whole words, then bisect an oversized single word by code points.
            while len(options) > 1 and not fits(text[start : options[-1]]):
                options = options[: max(1, len(options) // 2)]
            end = options[-1]
            if not fits(text[start:end]):
                low, high = start, end
                while low + 1 < high:
                    mid = (low + high) // 2
                    if fits(text[start:mid]):
                        low = mid
                    else:
                        high = mid
                end = low
                if end == start:
                    raise AdapterContractError()
        if not fits(text[start:end]):
            raise AdapterContractError()
        windows.append(Window(f"w{len(windows) + 1}", start, end))
        if end == len(text):
            break
        # At most 32 word boundaries, at most half of the shrunken character window.
        covered = [point for point in ends if start < point < end]
        next_start = covered[max(0, len(covered) - overlap)] if covered and overlap else end
        start = max(start + 1, start + (end - start + 1) // 2, next_start)
    return tuple(windows)
