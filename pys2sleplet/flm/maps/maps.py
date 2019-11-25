from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.flm.maps.wmap import WMAP


def maps():
    return {"earth": Earth, "wmap": WMAP}
