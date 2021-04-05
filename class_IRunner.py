class IRunner:
    # hardware accelerator setup
    def get_engine(self) -> IEngine: pass

    # expriment components setup: the data, the model, etc
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]": pass
    def get_model(self, stage: str) -> Model: pass
    def get_criterion(self, stage: str) -> Optional[Criterion]: pass
    def get_optimizer(self, stage: str, model: Model) -> Optional[Optimizer]: pass
    def get_scheduler(self, stage: str, optimizer: Optimizer) -> Optional[Scheduler]: pass

    # extra logic setup: metrics and deep learning tricks
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]": pass

    # monitoring systems setup
    def get_loggers(self) -> Dict[str, ILogger]: pass
    def log_metrics(self, *args, **kwargs) -> None: pass
    def log_image(self, *args, **kwargs) -> None: pass
    def log_hparams(self, *args, **kwargs) -> None: pass

    def _run_event(self, event: str) -> None:
        for callback in self.callbacks.values():
            getattr(callback, event)(self)

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        """Inner method to handle specified data batch."""
        pass

    def _run_batch(self) -> None:
        self._run_event("on_batch_start")
        self.handle_batch(batch=self.batch)
        self._run_event("on_batch_end")

    def _run_loader(self) -> None:
        self._run_event("on_loader_start")
        for self.loader_batch_step, self.batch in enumerate(self.loader):
            self._run_batch()
        self._run_event("on_loader_end")

    def _run_epoch(self) -> None:
        self._run_event("on_epoch_start")
        for self.loader_key, self.loader in self.loaders.items():
            self._run_loader()
        self._run_event("on_epoch_end")

    def _run_stage(self, rank: int = -1, world_size: int = 1) -> None:
        self._run_event("on_stage_start")
        while self.stage_epoch_step < self.stage_epoch_len:
            self._run_epoch()
        self._run_event("on_stage_end")

    def _run_experiment(self) -> None:
        self._run_event("on_experiment_start")
        for self.stage_key in self.stages:
            self._run_stage()
        self._run_event("on_experiment_end")

    def run(self) -> "IRunner":
        self._run_experiment()
        return self