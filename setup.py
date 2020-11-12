import atexit
import subprocess
import sys

from setuptools import setup, find_packages

from setuptools.command.develop import develop
from setuptools.command.install import install

import logging.config
from logging_conf import LOGGING_CONF

log = logging.getLogger()
logging.config.dictConfig(LOGGING_CONF)

log.info("Started!")


def _post_install():
    log.info("POST INSTALL")
    # download_stanford_models()
    download_stanza_models()


def _pre_install():
    # need to install torch from here
    log.info("PRE INSTALL")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==1.4.0+cpu",
            "torchvision==0.5.0+cpu",
            "-f",
            "https://download.pytorch.org/whl/torch_stable.html",
        ]
    )


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        atexit.register(_post_install)

    def run(self):
        # _pre_install()
        install.run(self)
        _post_install()
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION


def download_stanza_models():
    import stanza

    for lang in ["it"]:
        try:
            stanza.Pipeline(lang)
        except:
            stanza.download(lang)


setup(
    name="evalita-ghigliottinai",
    version="1.0",
    url="",
    license="",
    author="nazareno.defrancesco",
    author_email="nazareno.defrancesco@celi.it",
    description="",
    # cmdclass={"develop": PostDevelopCommand, "install": PostInstallCommand,},
    python_requires=">=3.6",
    install_requires=[
        "numpy==1.18.1",
        "pandas==1.0.3",
        "deap==1.3.1",
        "gensim==3.7.3",
        # "spacy-stanza==0.2.1",
        # "stanza==1.0.1",
        "matplotlib==3.2.2",
        "scikit-learn==0.23.1",
        "nltk==3.5",
        "numba==0.50.1",
        "stop_words",
        "py7zr",
        "Flask==1.1.2",
        "requests==2.24.0"
    ],
)

