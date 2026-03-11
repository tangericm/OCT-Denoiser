# Import model files so they register themselves.
from . import resunet_pseudo3d  # noqa: F401
from . import resunet_pseudo3d_multilevel  # noqa: F401
from . import resunet_multilevel_1d  # noqa: F401
from . import spectrum_unet_1d   # noqa: F401
from . import spectrum_resunet_1d  # noqa: F401
from . import physics_oct_net    # noqa: F401
from .registry import create_model, list_models  # noqa: F401
