class IMetric(ABC):
    def __init__(self, compute_on_call: bool = True):
        self.compute_on_call = compute_on_call

    def reset(self) -> None:
        """Resets the metric to it's initial state."""
        pass

    def update(self, *args, **kwargs) -> Any:
        """Updates the metrics state using the passed data."""
        pass

    def update_key_value(self, *args, **kwargs) -> Dict:
        """
        Updates the metrics state using the passed data
        Returns results into key-value storage.
        """
        pass

    def compute(self) -> Any:
        """Computes the metric based on it's accumulated state."""
        pass

    def compute_key_value(self) -> Dict[str, float]:
        """
        Computes the metric based on it's accumulated state.
        Returns results into key-value storage.
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        value = self.update(*args, **kwargs)
        return self.compute() if self.compute_on_call else value