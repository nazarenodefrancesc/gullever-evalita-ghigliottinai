"""
Logger config file
"""

LOGGING_CONF = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s.%(funcName)s() - line: %(lineno)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        # "info_file_handler": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "level": "INFO",
        #     "formatter": "simple",
        #     "filename": "info.log",
        #     "maxBytes": 10485760,
        #     "backupCount": 20,
        #     "encoding": "utf8",
        # },
        # "error_file_handler": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "level": "ERROR",
        #     "formatter": "simple",
        #     "filename": "errors.log",
        #     "maxBytes": 10485760,
        #     "backupCount": 20,
        #     "encoding": "utf8",
        # },
    },
    "loggers": {
        "my_module": {"level": "ERROR", "handlers": ["console"], "propagate": False}
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],  # , "info_file_handler", "error_file_handler"],
    },
}
