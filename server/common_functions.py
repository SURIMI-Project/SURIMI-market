import logging


def log_and_abort(context, status_code, message):
    """
    Logs an error message and aborts the gRPC call with the given status code.
    """
    logger = logging.getLogger("common_functions")
    logger.error(message)
    context.abort(status_code, message)
