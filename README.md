# AISys FL project by Team EU

## How to reporduce
### Server side
* from test/cifar10 dir
* python test_server.py --n_requests 2

### Client side
* from test/cifar10 dir
* python test_client.py --host test.com --port 5000 --classes 0 1 2 3 5
