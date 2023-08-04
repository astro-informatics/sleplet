from pydantic import ConfigDict

validation = ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=True,
    validate_default=True,
)
