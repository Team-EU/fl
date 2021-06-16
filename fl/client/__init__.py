import io
import time
import requests
import dill

__all__ = ['FLClient']


def send_result(url, round, result):
    stream = io.BytesIO()
    dill.dump(result, stream)
    stream.seek(0)
    r = requests.post(url, files={'result': stream}, data={'round': round})
    r.raise_for_status()


class FLClient:
    def __init__(self, url, verbose=True):
        self.url = url
        self.verbose = verbose
        self.fl_module = None

    @property
    def round(self):
        if self.fl_module is None:
            return -1
        return self.fl_module._round

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def pull(self, round=None):
        """ Pull the current model

        Args:
            - round (default: None)
                - if specified, pull the latest model after the round
                - if the current server round is less than this round, wait
        """
        if round is not None:
            while True:
                resp = requests.get(self.url + "/round")
                data = resp.json()
                if data['round'] >= round:
                    break
                time.sleep(5)

        resp = requests.get(self.url + "/model")
        self.fl_module = dill.load(io.BytesIO(resp.content))

        if self.fl_module._client_setup:
            self.fl_module._client_setup(self.fl_module)

    def run(self, train_loader, **kwargs):
        if self.fl_module._on_training_start:
            self.fl_module._on_training_start(self.fl_module, **kwargs)

        self.log(f"[round {self.round:03d}] running...", end='\t')
        result = self.fl_module._training_step(self.fl_module, train_loader, **kwargs)
        self.log("complete!")

        if self.fl_module._on_training_end:
            self.fl_module._on_training_end(self.fl_module, **kwargs)

        self.log(f"[round {self.round:03d}] sending result...", end="\t")
        send_result(self.url + '/upload', self.round, result)
        self.log("complete!")
