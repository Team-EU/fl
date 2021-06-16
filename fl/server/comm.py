import dill
import threading
from flask import abort
from flask import request, send_file, jsonify, Response
from flask import current_app

sem = threading.Semaphore()


def checkpoint_model(fl_module, path_format):
    filename = path_format.format(fl_module._round)
    with open(filename, 'wb') as file:
        dill.dump(fl_module, file)


def run_aggregation_step(fl_module, results, path_format):
    sem.acquire()
    if fl_module._on_aggregation_start:
        fl_module._on_aggregation_start(fl_module)
    fl_module._aggregation_step(fl_module, results)
    if fl_module._on_aggregation_end:
        fl_module._on_aggregation_end(fl_module)
    fl_module._round += 1
    checkpoint_model(fl_module, path_format)
    sem.release()


def add_model_communication(app):
    @app.route('/round')
    def get_round():
        """ Return the current round """
        sem.acquire()
        current_round = current_app.config['MODEL_OBJ']._round
        sem.release()
        return jsonify(round=current_round)

    @app.route('/model')
    def get_model():
        """ Return the current model """
        sem.acquire()
        current_round = current_app.config['MODEL_OBJ']._round
        sem.release()
        return send_file(current_app.config['MODEL_PATH'].format(current_round))

    @app.route('/upload', methods=['POST'])
    def upload():
        sem.acquire()
        current_round = current_app.config['MODEL_OBJ']._round
        sem.release()

        # if current_round != int(request.form['round']):
        #     abort(404)

        result = dill.load(request.files['result'].stream)

        current_app.config['RECEIVED_RESULTS'].append(result)

        if len(current_app.config['RECEIVED_RESULTS']) == current_app.config['NUM_REQUESTS']:
            thread = threading.Thread(
                target=run_aggregation_step,
                kwargs={
                    'fl_module': current_app.config['MODEL_OBJ'],
                    'results': current_app.config['RECEIVED_RESULTS'],
                    'path_format': current_app.config['MODEL_PATH'],
                })
            thread.daemon = True
            thread.start()

            current_app.config['RECEIVED_RESULTS'] = []

        return Response(status=200)

    return app
