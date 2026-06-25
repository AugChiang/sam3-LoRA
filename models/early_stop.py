class EarlyStopper:
    """Track a validation metric and stop after repeated non-improvements."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max") -> None:
        if patience < 0:
            raise ValueError("early stopping patience must be non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("early stopping mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        """Update the stopper and return True when training should stop."""

        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta
