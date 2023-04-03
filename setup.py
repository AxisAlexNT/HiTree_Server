from setuptools import setup

setup(
    name='hict_server',
    version='0.1.1rc1',
    packages=['hict_server', 'hict_server.api_controller',
              'hict_server.api_controller.dto'],
    url='https://genome.ifmo.ru',
    license='MIT',
    author='Alexander Serdiukov, Anton Zamyatin and CT Lab ITMO University team',
    author_email='',
    description='Development version of API and tiling server for HiCT interactive Hi-C scaffolding tool.',
    install_requires=[
        'hict>=0.1.1rc1,<1.0',
        'matplotlib>=3.5.2',
        'flask>=2.1.3',
        'pathos>=0.2.9',
        'Pillow>=9.2.0',
        'Werkzeug>=2.1.2',
        'flask-cors>=3.0.10',
        'argparse>=1.4',
        'setuptools>=63.2.0',
        'wheel>=0.37.1',
        'flask_classful>=0.14.2'
    ],
    entry_points={
        'console_scripts': ['hict_server=hict_server:main'],
    }
)
