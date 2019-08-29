from dataclasses import dataclass


@dataclass
class Config:
    L: int = 16
    auto_open: bool = True
    save_fig: bool = False
