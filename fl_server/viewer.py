from flask import current_app
from flask import render_template

server_config_list = [
    'NUM_REQUESTS',
    'NUM_REQUESTS_PER_CLIENT',
    'ROUND_TIMEOUT',
]


def add_model_viewer(app):
    @app.route('/')
    def hello():
        fl_module = current_app.config['MODEL_OBJ']
        model_configs = {
            'params': fl_module.count_params(),
            'children': fl_module.count_children_params(),
        }
        server_configs = {x: current_app.config[x] for x in server_config_list}
        server_configs['MODEL_ROUND'] = fl_module._round
        return render_template('index.html', server_configs=server_configs, model_configs=model_configs)

    return app