import sys

import logbook

from sagol import config


def setup_log_handlers():
    logbook.StreamHandler(sys.stdout, level=config.LOGGING_LEVEL, bubble=True).push_application()
