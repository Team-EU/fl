import torch


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

    def client_setup(self):
        pass

    def training_step(self, train_loader):
        raise NotImplementedError()

    def on_training_end(self):
        pass

    def aggregation_step(self, results):
        raise NotImplementedError()

    def on_aggregation_end(self):
        pass

    def count_params(self):
        modes = ['all', 'trainable', 'non-trainable']
        return {x: count_params(self, mode=x) for x in modes}

    def count_children_params(self):
        return [(name, type(x).__name__, count_params(x)) for name, x in self.named_children()]
