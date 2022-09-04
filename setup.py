from setuptools import setup

setup(
    name='hict_server',
    version='0.1.1rc1',
    packages=['hict_server', 'hict_server.api_controller'],
    url='https://genome.ifmo.ru',
    license='',
    author='Alexander Serdiukov, Anton Zamyatin and CT Lab ITMO University team',
    author_email='',
    description='Development version of API and tiling server for HiCT interactive Hi-C scaffolding tool.',
    install_requires=[
        'hict~=0.1.1rc1',
        'matplotlib==3.5.2',
        'flask~=2.1.3',
        'scikit-image~=0.19.3',
        'pathos~=0.2.9',
        'Pillow==9.2.0',
        'Werkzeug~=2.1.2',
        'flask-cors~=3.0.10',
        'argparse~=1.4.0',      
        'setuptools~=63.2.0',
        'wheel~=0.37.1',  
    ],
    entry_points = {
          'console_scripts': ['dev_demo_server=hict_server.api_controller.dev_demo_server:main'],
      }
)
