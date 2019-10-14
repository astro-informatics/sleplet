@staticmethod
def calc_samples(L: int) -> int:
    """
    calculate appropriate sample number for given L
    chosen such that have a two samples less than 0.1deg
    """
    if L == 1:
        samples = 1801
    elif L < 4:
        samples = 901
    elif L < 8:
        samples = 451
    elif L < 16:
        samples = 226
    elif L < 32:
        samples = 113
    elif L < 64:
        samples = 57
    elif L < 128:
        samples = 29
    elif L < 256:
        samples = 15
    elif L < 512:
        samples = 8
    elif L < 1024:
        samples = 4
    elif L < 2048:
        samples = 2
    else:
        samples = 1
    return samples
