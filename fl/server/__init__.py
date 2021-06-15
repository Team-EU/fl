import os
from flask import Flask
from flask import current_app
from .comm import add_model_communication, checkpoint_model
from .viewer import add_model_viewer


__all__ = ['create_app']


def create_app(fl_module,
               n_requests,
               n_requests_per_client=None,
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
        NUM_REQUESTS_PER_CLIENT=n_requests_per_client,
        ROUND_TIMEOUT=round_timeout,
        RECEIVED_RESULTS=[],
    )

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
        print("create the instance:", app.instance_path)
    except OSError:
        print("use the existing instance:", app.instance_path)

    with app.app_context():
        checkpoint_model(fl_module, current_app.config['MODEL_PATH'])

    app = add_model_viewer(app)

    app = add_model_communication(app)

    if fl_module._server_setup:
        fl_module._server_setup(fl_module)

    return app