class DCSAMPError(Exception):
    """Base-class for all DCS-AMP related exceptions."""


class InitError(DCSAMPError):
    """Raised when some parameters validation failed."""