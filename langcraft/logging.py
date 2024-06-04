import os
import logging
import datetime


"""
This module contains the logging configuration for the langcraft module.
"""

##############################################
def _create_logger():
    """
    Create and configure a logger for the langcraft module.

    Returns:
        logging.Logger: The configured logger object.
    """
    log_directory = os.getenv(
        "LANGCRAFT_LOG_DIRECTORY",
        os.path.join(os.getcwd(), "logs"),
    )

    os.makedirs(log_directory, exist_ok=True)

    llm_logger = logging.getLogger("langcraft")
    llm_logger.setLevel(logging.INFO)
    llm_formatter = logging.Formatter(
        "============================================================\n"
        "%(asctime)s %(name)s %(levelname)s:\n%(message)s\n"
        "============================================================\n",
        "%Y-%m-%d %H:%M:%S",
    )
    llm_file_handler = logging.FileHandler(
        f"{log_directory}/langcraft_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
    )
    llm_file_handler.setFormatter(llm_formatter)
    llm_logger.addHandler(llm_file_handler)
    llm_logger.propagate = False


##############################################
_create_logger()
