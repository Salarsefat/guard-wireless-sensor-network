"""Protocol registry."""
from .backpressure import BCP, CBPR
from .cetx import ACSRO, CETX, FSFG, RLCC
from .ganaco import GANACO
from .twinloop import TwinLoop
from .guard import GUARD
from .v2 import V2

REGISTRY = {
    "guard": GUARD,
    "twinloop": TwinLoop,
    "cetx": CETX,
    "bcp": BCP,
    "cbpr": CBPR,
    "ganaco": GANACO,
    "acsro": ACSRO,
    "fsfg": FSFG,
    "rlcc": RLCC,
    "v2": V2,
}

LABELS = {
    "guard": "GUARD (proposed)",
    "twinloop": "TwinLoop v1",
    "cetx": "C-ETX",
    "bcp": "BCP",
    "cbpr": "CBPR-like",
    "ganaco": "GAN-ACO",
    "acsro": "ACSRO-like",
    "fsfg": "FSFG-like",
    "rlcc": "RL-CC",
    "v2": "v2 (dev)",
}


def make_protocol(method: str, params: dict | None = None):
    params = dict(params or {})
    cls = REGISTRY[method]
    if method == "rlcc":
        q = params.pop("qtable", None)
        return cls(qtable=q, **params)
    return cls(**params)
