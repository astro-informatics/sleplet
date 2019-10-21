from .earth import Earth
from .wmap import WMAP


def maps():
    return {"earth": Earth, "wmap": WMAP}
