import pydantic


class Validation:
    arbitrary_types_allowed = True
    smart_union = True
    validate_all = True
    validate_assignment = True


validation = pydantic.ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=True,
    validate_default=True,
)
