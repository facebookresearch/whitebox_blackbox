import logging
import time
import datetime
from date import GMT1

class LogFormatter():
    def __init__(self):
        self.start_time = time.time()
        self.gmt = GMT1()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            datetime.datetime.now(tz=self.gmt).strftime('%x %X'),
            datetime.timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath="", vb=2):
    """
    Create a logger.
    """
    # create log formatter
    log_formatter = LogFormatter()

    if filepath != "":
        # create file handler and set level to debug
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    log_level = logging.DEBUG if vb == 2 else logging.INFO if vb == 1 else logging.WARNING
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath != "":
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger
