def calc_integration_resolution(L: int) -> int:
    """
    calculate appropriate sample number for given L
    chosen such that have a two samples less than 0.1deg
    """
    sample_dict = {
        1: 1801,
        2: 901,
        3: 451,
        4: 226,
        5: 113,
        6: 57,
        7: 29,
        8: 15,
        9: 8,
        10: 4,
        11: 2,
    }

    for log_bandlimit, samples in sample_dict.items():
        if L < 2 ** log_bandlimit:
            return samples

    # above L = 2048 just use 1 sample
    return 1
