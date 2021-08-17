
class SVGDNaNError(Exception):
    # raised when a SVGD particles contain NaNs and SVGD exits
    pass


class SVGDNaNErrorLatent(Exception):
    # raised when a SVGD latent representation contain NaNs and SVGD exits
    pass


class SVGDNaNErrorTheta(Exception):
    # raised when a SVGD theta contain NaNs and SVGD exits
    pass


class InvalidCPDAGError(Exception):
    # raised when a "CPDAG" returned by a learning alg does not admit a random extension
    pass


class ContinualInvalidCPDAGError(Exception):
    # raised when a "CPDAG" returned by a learning alg does not admit a random extension
    # even after repeating several times with hard confidence thresholds
    pass


class BaselineTimeoutError(Exception):
    # raised when a baseline method times out
    pass




