import os
import time
from flask import Flask
from flask import current_app
from .comm import add_model_communication, checkpoint_model
from .viewer import add_model_viewer


__all__ = ['create_app']


def create_app(fl_module,
               n_requests,
               round_timeout=None,
               round_staleness=0,
               instance_path=None,
               ):
    now = time.strftime("%m%d%H%M", time.localtime())

    # create and configure the app
    app = Flask(__name__, instance_path=instance_path)

    app.config.from_mapping(
        SECRET_KEY=os.urandom(24),
        CKPT_DIR=os.path.join(app.instance_path, now),
        NUM_REQUESTS=n_requests,
        ROUND_TIMEOUT=round_timeout,
        ROUND_STALENESS=round_staleness,
    )

    try:
        os.makedirs(app.config['CKPT_DIR'])
        print("create the instance:", app.instance_path)
    except OSError:
        print("use the existing instance:", app.instance_path)

    checkpoint_model(fl_module, app.config['CKPT_DIR'])

    if fl_module._server_setup:
        with app.app_context():
            fl_module._server_setup(fl_module, current_app)

    app = add_model_viewer(app)

    app = add_model_communication(app, fl_module)

    return app
