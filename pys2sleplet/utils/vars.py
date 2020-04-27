from dataclasses import Field, field

PHI_MIN_DEFAULT: int = 0
PHI_MAX_DEFAULT: int = 360
THETA_MIN_DEFAULT: int = 0
THETA_MAX_DEFAULT: int = 180
DC_VAR_NOT_INIT: Field = field(init=False, repr=False)
