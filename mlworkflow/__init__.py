import pkgutil
__path__ = pkgutil.extend_path(__path__, f"{__name__}.v1")
from .v1 import *
