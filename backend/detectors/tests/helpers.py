"""Synthetic detector runs shared by the CPU tests."""

from backend.detectors.app.observations import ModelMetadata, ModelRun, Window


def model_run(profile, observations=(), windows=None, blank=False):
    if windows is None:
        windows = (Window("w1", 0, 100, len(observations)),)
    return ModelRun(
        tuple(observations),
        ModelMetadata(
            "hivetrace/gliner-guard-omni",
            "1" * 40,
            "cuda:0",
            "float32",
            "skipped_blank" if blank else "ok",
            profile.emit_threshold,
            profile.accept_threshold,
            profile.labels,
            tuple(windows),
        ),
    )


class EmptyModel:
    def __init__(self, profile):
        self.profile = profile
        self.calls = []

    def detect(self, text):
        self.calls.append(text)
        blank = not text or text.isspace()
        return model_run(
            self.profile, windows=() if blank else (Window("w1", 0, len(text)),), blank=blank
        )
