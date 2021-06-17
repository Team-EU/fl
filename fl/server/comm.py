import os
import dill
import threading
from blinker import Signal
from flask import abort
from flask import request, send_file, jsonify, Response
from flask import current_app

sem_fl_module = threading.Semaphore()
aggregation_signal = Signal('aggregation')


def checkpoint_model(fl_module, ckpt_dir):
    path = os.path.join(ckpt_dir, f'{fl_module._round:05d}.ckpt')
    with open(path, 'wb') as file:
        dill.dump(fl_module, file)


@aggregation_signal.connect
def recv_aggregation_signal(sender, fl_module, results, ckpt_dir):
    with sem_fl_module:
        if fl_module._on_aggregation_start:
            fl_module._on_aggregation_start(fl_module)
        fl_module._aggregation_step(fl_module, results)
        if fl_module._on_aggregation_end:
            fl_module._on_aggregation_end(fl_module)
        fl_module._round += 1
        checkpoint_model(fl_module, ckpt_dir)


def add_model_communication(app, fl_module):
    context = {'results': []}
    sem_context = threading.Semaphore()

    def send_aggregation_signal(call_round=None):
        with sem_context:
            with sem_fl_module:
                current_round = fl_module._round

            if (call_round is not None) and (call_round != current_round):
                return

            if context['results']:
                aggregation_signal.send(
                    fl_module=fl_module,
                    results=context['results'],
                    ckpt_dir=app.config['CKPT_DIR'])
                context['results'] = []
                round_timeout_start(current_round + 1)
            else:
                round_timeout_start(current_round)

    def round_timeout_start(call_round=0):
        timeout = app.config['ROUND_TIMEOUT']
        if timeout is not None:
            timer = threading.Timer(timeout, send_aggregation_signal, args=[call_round])
            timer.daemon = True
            timer.start()

    @app.route('/round')
    def get_round():
        """ Return the current round """
        current_round = fl_module._round
        return jsonify(round=current_round)

    @app.route('/model')
    def get_model():
        """ Return the current model """
        with sem_fl_module:
            current_round = fl_module._round
        return send_file(os.path.join(current_app.config['CKPT_DIR'], f'{current_round:05d}.ckpt'))

    @app.route('/upload', methods=['POST'])
    def upload():
        if current_app.config['ROUND_STALENESS'] is not None:
            with sem_fl_module:
                current_round = fl_module._round
            staled_round = current_round - current_app.config['ROUND_STALENESS']
            if not (staled_round <= int(request.form['round']) <= current_round):
                abort(404)

        result = dill.load(request.files['result'].stream)

        with sem_context:
            context['results'].append(result)

        if len(context['results']) == current_app.config['NUM_REQUESTS']:
            send_aggregation_signal()

        return Response(status=200)

    round_timeout_start()

    return app
