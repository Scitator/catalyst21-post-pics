class IEngine:

    def rank(self) -> int:
        """Process rank for distributed training."""
        pass

    def world_size(self) -> int:
        """Process world size for distributed training."""
        pass

    def sync_device(self, tensor_or_module: Any) -> Any:
        """Moves ``tensor_or_module`` to Engine's device."""
        pass

    def sync_tensor(self, tensor: Any, mode: str) -> Any:
        """Syncs ``tensor`` over ``world_size`` in distributed mode."""
        pass

    def init_components(self, ...):
        """Inits the runs components."""
        pass

    def deinit_components(self):
        """Deinits the runs components. Destroys process group in distributed run.
        """
        pass

    def zero_grad(self, loss, model, optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        pass

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        pass

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        pass

    def pack_checkpoint(self, model, criterion, optimizer, scheduler) -> Dict:
        """Packs ``model``, ... to torch-based checkpoint."""
        pass

    def unpack_checkpoint(self, checkpoint, model, criterion, optimizer, scheduler) -> None:
        """Unpacks checkpoint content to ``model``, ..."""
        pass

    def save_checkpoint(self, checkpoint: Dict, path: str) -> None:
        """Saves checkpoint to a file."""
        pass

    def load_checkpoint(self, path: str) -> Dict:
        """Loads checkpoint by path."""
        pass
