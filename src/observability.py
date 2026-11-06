import logging
import os


DEBUG_MODE = (
    os.getenv(
        "FINSIGHT_DEBUG",
        "false",
    ).lower()
    == "true"
)


logger = logging.getLogger(
    "finsight"
)


if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "[FinSight] %(message)s"
    )

    handler.setFormatter(
        formatter
    )

    logger.addHandler(
        handler
    )


logger.setLevel(
    logging.DEBUG
    if DEBUG_MODE
    else logging.WARNING
)


def trace(
    message: str,
    **data,
) -> None:
    """
    Emits agent execution information only when
    FINSIGHT_DEBUG=true.
    """

    if not DEBUG_MODE:
        return

    if data:
        logger.debug(
            "%s | %s",
            message,
            data,
        )
    else:
        logger.debug(
            message
        )