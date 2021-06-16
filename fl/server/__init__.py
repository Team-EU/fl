import os
from flask import Flask
from flask import current_app
from .comm import add_model_communication, checkpoint_model, send_timeout_aggregation_signal
from .viewer import add_model_viewer


__all__ = ['create_app']


def create_app(fl_module,
               n_requests,
               round_timeout=None,
               instance_path=None,
               ):

    # create and configure the app
    app = Flask(__name__, instance_path=instance_path)

    app.config.from_mapping(
        SECRET_KEY=os.urandom(24),
        MODEL_OBJ=fl_module,
        MODEL_PATH=os.path.join(app.instance_path, 'model-{:05d}.pth'),
        NUM_REQUESTS=n_requests,
        ROUND_TIMEOUT=round_timeout,
        RECEIVED_RESULTS=[],
    )

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
        print("create the instance:", app.instance_path)
    except OSError:
        print("use the existing instance:", app.instance_path)

    app = add_model_viewer(app)

    app = add_model_communication(app)

    with app.app_context():
        if fl_module._server_setup:
            fl_module._server_setup(fl_module, current_app)
        checkpoint_model(fl_module, current_app.config['MODEL_PATH'])
        send_timeout_aggregation_signal(0)
    return app
