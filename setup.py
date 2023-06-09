from typing import List
from setuptools import find_packages, setup


requirements: List[str] = []
with open("requirements.txt", mode="rt", encoding="utf-8") as f:
    requirements = f.readlines()

setup(
    name='hict_server',
    version='0.1.3rc1',
    packages=list(set(['hict_server', 'hict_server.api_controller',
              'hict_server.api_controller.dto']).union(find_packages())),
    url='https://genome.ifmo.ru',
    license='MIT',
    author='Alexander Serdiukov, Anton Zamyatin and CT Lab ITMO University team',
    author_email='',
    description='Development version of API and tiling server for HiCT interactive Hi-C scaffolding tool.',
    install_requires=list(set([
        'hict>=0.1.3rc1,<1.0',
        'hict_utils>=0.1.3rc1,<1.0',
    ]).union(requirements)),
    entry_points={
        'console_scripts': ['hict_server=hict_server.api_controller.dev_demo_server:main'],
    }
)
