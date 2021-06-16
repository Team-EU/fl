import dill
import threading
from blinker import Signal
from flask import abort
from flask import request, send_file, jsonify, Response
from flask import current_app

sem_fl_module = threading.Semaphore()
sem_signal_send = threading.Semaphore()
aggregation_signal = Signal('aggregation')


def checkpoint_model(fl_module, path_format):
    filename = path_format.format(fl_module._round)
    with open(filename, 'wb') as file:
        dill.dump(fl_module, file)


@aggregation_signal.connect
def receive_aggregation_signal(sender):
    fl_module = sender['fl_module']
    results = sender['results']
    path_format = sender['path_format']

    sem_fl_module.acquire()
    if fl_module._on_aggregation_start:
        fl_module._on_aggregation_start(fl_module)
    fl_module._aggregation_step(fl_module, results)
    if fl_module._on_aggregation_end:
        fl_module._on_aggregation_end(fl_module)
    fl_module._round += 1
    checkpoint_model(fl_module, path_format)
    sem_fl_module.release()


def send_aggregation_signal(call_round=None):
    sem_signal_send.acquire()
    current_round = current_app.config['MODEL_OBJ']._round

    if (call_round is not None) and (call_round != current_round):
        return

    if current_app.config['RECEIVED_RESULTS']:
        aggregation_signal.send({
            'fl_module': current_app.config['MODEL_OBJ'],
            'results': current_app.config['RECEIVED_RESULTS'],
            'path_format': current_app.config['MODEL_PATH'],
        })
        current_app.config['RECEIVED_RESULTS'] = []
        send_timeout_aggregation_signal(current_round + 1)
    else:
        send_timeout_aggregation_signal(current_round)

    sem_signal_send.release()


def _appcontext_send_signal(call_round):
    thread = AppContextThread(target=send_aggregation_signal, args=[call_round])
    thread.daemon = True
    thread.start()


def send_timeout_aggregation_signal(call_round):
    timeout = current_app.config['ROUND_TIMEOUT']
    if timeout is not None:
        timer = threading.Timer(timeout, _appcontext_send_signal, args=[call_round])
        timer.daemon = True
        timer.start()


def add_model_communication(app):
    @app.route('/round')
    def get_round():
        """ Return the current round """
        current_round = current_app.config['MODEL_OBJ']._round
        return jsonify(round=current_round)

    @app.route('/model')
    def get_model():
        """ Return the current model """
        sem_fl_module.acquire()
        current_round = current_app.config['MODEL_OBJ']._round
        sem_fl_module.release()
        return send_file(current_app.config['MODEL_PATH'].format(current_round))

    @app.route('/upload', methods=['POST'])
    def upload():
        sem_fl_module.acquire()
        current_round = current_app.config['MODEL_OBJ']._round
        sem_fl_module.release()

        # if current_round != int(request.form['round']):
        #     abort(404)

        result = dill.load(request.files['result'].stream)

        current_app.config['RECEIVED_RESULTS'].append(result)

        if len(current_app.config['RECEIVED_RESULTS']) == current_app.config['NUM_REQUESTS']:
            send_aggregation_signal()

        return Response(status=200)

    return app
