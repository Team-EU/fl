# AISys FL project by Team EU

## How to reporduce
### IID Sync test 
#### Server side
Aggregate per 2 requests
* from test/cifar10 dir
* python test_server.py --n_requests 2

#### Client side
run two cilents!
* from test/cifar10 dir
* python test_client.py --host test.com --port 5000
