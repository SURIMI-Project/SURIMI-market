from logger import get_logger

logger = get_logger("common_functions")


def log_and_abort(context, status_code, message):
    """
    Logs an error message and aborts the gRPC call with the given status code.
    """
    logger.error(message)
    context.abort(status_code, message)
