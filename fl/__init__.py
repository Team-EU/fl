import torch
import inspect

__all__ = ['Module']


def count_params(model, mode='all'):
    if mode == 'all':
        return sum(p.numel() for p in model.parameters())
    elif mode == 'trainable':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == 'non-trainable':
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)
    else:
        raise ValueError(f"unknwon mode: {mode}")


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._round = 0

        # setup
        self._infer = None
        self._server_setup = None
        self._client_setup = None

        # main procedures
        self._training_step = None
        self._aggregation_step = None

        # hooks
        self._on_training_start = None
        self._on_aggregation_start = None
        self._on_training_end = None
        self._on_aggregation_end = None

    def forward(self, *x, **y):
        return self._infer(self, *x, **y)

    def init(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self']
        func(self)
        return func

    def infer(self, func):
        self._infer = func
        return func

    def server_setup(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self']
        self._server_setup = func
        return func

    def client_setup(self, func):
        argspec = inspect.getargspec(func)
        assert argspec.args == ['self']
        self._client_setup = func
        return func

    def training_step(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self', 'dataloader']
        self._training_step = func
        return func

    def aggregation_step(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self', 'results']
        self._aggregation_step = func
        return func

    def on_training_start(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self']
        self._on_training_start = func
        return func

    def on_training_end(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self']
        self._on_training_end = func
        return func

    def on_aggregation_start(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self']
        self._on_aggregation_start = func
        return func

    def on_aggregation_end(self, func):
        argspec = inspect.getfullargspec(func)
        assert argspec.args == ['self']
        self._on_aggregation_end = func
        return func

    def count_params(self):
        modes = ['all', 'trainable', 'non-trainable']
        return {x: count_params(self, mode=x) for x in modes}

    def count_children_params(self):
        return [(name, type(x).__name__, count_params(x)) for name, x in self.named_children()]
