# AISys FL project by Team EU

## How to reporduce

### Installing FL package
* from base dir
* pip install -e .

### Installing dependencies
* Necessary dependencies are in requirements.txt (pip install -r requirements.txt)
* Specified versions in requirements.txt are from actual test ENV

### IID Sync test 
#### Server side
* from test/cifar10 dir
* python test_server.py --host test.com --port 5000 --n_requests 2

#### Client side (need 2 clients)
* from test/cifar10 dir
* python test_client.py --host test.com --port 5000 --sync

### IID Async test
#### Server side
* from test/cifar10 dir
* python test_server.py --host test.com --port 5000 --n_requests 2

#### Client side (need multiple clients)
* from test/cifar10 dir
* python test_client.py --host test.com --port 5000

### Non-IID Sync test
#### Server side
* from test/cifar10 dir
* python test_server.py --host test.com --port 5000 --n_requests 2

#### Client side (need 2 clients)
* from test/cifar10 dir
* python test_client.py --host test.com --port 5000 --classes 0 1 2 3 4

on other clients
* from test/cifar10 dir
* python test_client.py --host test.com --port 5000 --classes 5 6 7 8 9
