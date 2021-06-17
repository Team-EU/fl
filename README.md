# AISys FL project by Team EU

## Test ENV
* CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
* GPU: TitanXP * 4 / CUDA 11.2
* OS: Ubuntu 16.04
* Python: 3.8

## How to reporduce

### Installing FL package
* from base dir
* ```pip install -e .```

### Installing dependencies
* Necessary dependencies are in requirements.txt (```pip install -r requirements.txt```)
* Specified versions in requirements.txt are from actual test ENV (latest versions may work)

### IID Sync test 
#### Server side
* from test/cifar10 dir
* ```python test_server.py --host 0.0.0.0 --port 5000 --n_requests 2```

#### Client side (need as musch clients as n_requests in server, in this case 2)
* from test/cifar10 dir
* ```python test_client.py --host 0.0.0.0 --port 5000 --sync```

if specifying gpu device
* ```CUDA_VISIBLE_DEVICES=1 python test_client.py --host test.com --port 5000 --sync```

### IID Async test
#### Server side
* from test/cifar10 dir
* ```python test_server.py --host 0.0.0.0 --port 5000 --n_requests 2```

#### Client side (need multiple clients)
* from test/cifar10 dir
* ```python test_client.py --host 0.0.0.0 --port 5000```

### Non-IID Sync test
#### Server side
* from test/cifar10 dir
* ```python test_server.py --host 0.0.0.0 --port 5000 --n_requests 2```

#### Client side (need as musch clients as n_requests in server, in this case 2)
on one client (using dataset in label 01234)
* from test/cifar10 dir
* ```python test_client.py --host 0.0.0.0 --port 5000 --classes 0 1 2 3 4```

on other clients (using dataset in label 56789)
* from test/cifar10 dir
* ```python test_client.py --host 0.0.0.0 --port 5000 --classes 5 6 7 8 9```

### Running tensorboard
* from test/cifar10 dir
* ```tensorboard --logdir runs```
