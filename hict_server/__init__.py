from genericpath import exists
import multiprocessing
import multiprocessing.managers
from pathlib import Path
from typing import NamedTuple, Optional, Tuple
from flask import Flask
from flask_cors import CORS
import json
from hict_server.api_controller.dev_demo_server import API_Server
from flask_classful import FlaskView


class HiCT_Server_Options(NamedTuple):
    verbose: Optional[bool] = None
    debug: Optional[bool] = None
    log_level: Optional[str] = None
    api_server_options: Optional[API_Server.API_Server_Options] = None


def create_hict_server_instance(
    hict_server_options: Optional[HiCT_Server_Options] = None
) -> Tuple[Flask, multiprocessing.managers.SyncManager]:
    app = Flask(__name__)

    # api_server_options = hict_server_options.api_server_options

    # if api_server_options is not None:
    #     mp_manager = api_server_options.mp_manager

    # if mp_manager is None:
    #     app.logger.info(
    #         "No synchronization manager was passed so creating new")
    #     mp_manager = multiprocessing.Manager()

    # assert (mp_manager is not None), "Manager must be initialized at this step"

    # if api_server_options is None:
    #     if exists("hict_server.config.json"):
    #         app.logger.info("Loading configuration from JSON")
    #         app.config.from_file("hict_server.config.json", load=json.load)
    #     else:
    #         app.logger.warning(
    #             "Can't find JSON configuration file, falling back to environment-based configuration")
    #         app.config.from_prefixed_env(prefix="HICT_SERVER")
    #     api_server_options = API_Server.API_Server_Options(
    #         data_path=Path(app.config["API_SERVER"]["data_path"]),
    #         mp_manager=mp_manager
    #     )

    # assert (api_server_options is not None), "Options must be initialized at this step"

    # API_Server.register(
    #     app,
    #     route_prefix="/api",
    #     init_argument=api_server_options
    # )

    CORS(app)

    # log_level_str: str
    # if hict_server_options.verbose:
    #     log_level_str = 'DEBUG'
    # elif hict_server_options.log_level is not None:
    #     log_level_str = hict_server_options.log_level
    # else:
    #     log_level_str = 'INFO'

    # app.logger.setLevel(log_level_str)
    # app.logger.info(
    #     f"Using '{api_server_options.data_path}' as data directory"
    # )
    
    class QuotesView(FlaskView):
        def __init__(self, my_init_argument):
            self._my_init_argument = my_init_argument

        def index(self):
            return self._my_init_argument

    QuotesView.register(app, "Fistro diodenarl de abajorl")

    return app, None #mp_manager
