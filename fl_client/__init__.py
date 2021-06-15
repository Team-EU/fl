import io
import time
import requests
import torch
import dill

__all__ = ['FLClient']


def get_fl_module(url, round=0):
    url += f"/model?round={round}"
    response = requests.get(url)
    while response.status_code != 200:
        time.sleep(1)
        response = requests.get(url)
    stream = io.BytesIO(response.content)
    return dill.load(stream)


def send_result(url, result, round):
    stream = io.BytesIO()
    dill.dump(result, stream)
    stream.seek(0)
    r = requests.post(url, files={'result': stream}, data={'round': round})
    r.raise_for_status()


class FLClient():
    def __init__(self, url, round=0, verbose=True):
        self.url = url
        self.round = round
        self.verbose = verbose

        if self.verbose:
            print("get the initial fl_module...", end="\t")
        self.fl_module = get_fl_module(url, self.round)
        if self.verbose:
            print("complete!")

        if self.fl_module._client_setup:
            self.fl_module._client_setup(self.fl_module)

    def run(self, train_loader):
        if self.verbose:
            print("checking current round...", end="\t")
        self.fl_module = get_fl_module(self.url, self.round)
        if self.verbose:
            print("complete!")

        if self.verbose:
            print("running...")
        if self.fl_module._on_training_start:
            self.fl_module._on_training_start(self.fl_module)
        result = self.fl_module._training_step(self.fl_module, train_loader)
        if self.fl_module._on_training_end:
            self.fl_module._on_training_end(self.fl_module)

        if self.verbose:
            print("sending result...", end="\t")
        send_result(self.url + '/upload', result, self.round)
        if self.verbose:
            print("complete!")

        self.round += 1
