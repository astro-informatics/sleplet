from dataclasses import dataclass


@dataclass
class Config:
    L: int = 128
    sampling: str = 'MW'
    ncpu: int = 1
    auto_open: bool = True
    save_fig: bool = False
