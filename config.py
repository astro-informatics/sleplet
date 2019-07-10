from dataclasses import dataclass


@dataclass
class Config:
    L: int = 16
    ncpu: int = 4
    auto_open: bool = True
    save_fig: bool = False
