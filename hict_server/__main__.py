import argparse
import multiprocessing
import os
from pathlib import Path
from hict_server.api_controller.dev_demo_server import API_Server
from hict_server import HiCT_Server_Options, create_hict_server_instance


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run development version of HiCT tile server.",
        epilog="Visit https://github.com/ctlab/HiCT for more info."
    )

    def dir_checker(arg_path: str) -> bool:
        if os.path.isdir(arg_path):
            return arg_path
        else:
            raise ValueError(
                f'Path {arg_path} does not point to any directory')

    parser.add_argument('--data-path', default='./data', type=dir_checker)
    parser.add_argument(
        '--log-level', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], default='INFO', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    arguments: argparse.Namespace = parser.parse_args()
    data_path = Path(os.path.abspath(arguments.data_path))
    log_level_str: str
    if arguments.verbose:
        log_level_str = 'DEBUG'
    else:
        log_level_str = arguments.log_level

    mp_manager = multiprocessing.Manager()

    app, mp_manager = create_hict_server_instance(
        HiCT_Server_Options(
            verbose=arguments.verbose,
            log_level=log_level_str,
            api_server_options=API_Server.API_Server_Options(
                data_path=data_path,
                mp_manager=mp_manager
            )
        )
    )
    app.run(debug=bool(arguments.debug))


if __name__ == '__main__':
    main()
