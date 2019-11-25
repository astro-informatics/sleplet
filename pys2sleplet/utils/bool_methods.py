def is_polar_cap(phi_min: int, phi_max: int, theta_min: int, theta_max: int) -> bool:
    switch = False
    if phi_min == 0 and phi_max == 360 and theta_min == 0 and theta_max != 180:
        switch = True
    return switch
