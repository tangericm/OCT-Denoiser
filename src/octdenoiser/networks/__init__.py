# Import model modules so they self-register via @register_model.
from . import (
    aniso_resunet,  # noqa: F401
    deform_fusion,  # noqa: F401
    dncnn,  # noqa: F401
    ffc_resunet,  # noqa: F401
    nafnet,  # noqa: F401
    restormer,  # noqa: F401
    resunet_pseudo3d,  # noqa: F401
    resunet_pseudo3d_multilevel,  # noqa: F401
    unet2d,  # noqa: F401
)
from .registry import create_model, list_models  # noqa: F401
