import pathlib

from setuptools import setup, find_packages

_THIS_DIR = pathlib.Path(__file__).parent


def _get_requirements():
    with (_THIS_DIR / 'requirements.txt').open() as fp:
        return fp.read()


setup(
    name='dialog_model',
    version='0.0.1',
    install_requires=_get_requirements(),
    package_dir={'dialog_model': 'dialog_model'},
    packages=find_packages(exclude=['tests', 'tests.*'])
)
