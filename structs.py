from __future__ import annotations
from msgspec import Struct
import msgspec

Location = tuple[float, float, float]


class Message(Struct, array_like=True):
    stage: int
    cpu: Fighter
    opp: Fighter


class Fighter(Struct, array_like=True):
    location: Location
    damage: float
    is_shield: bool
    shield_strength: float
    attack: Attack
    grounded_ke: Location
    situation: str
    status: str


class Attack(Struct, array_like=True):
    is_attack: bool
    is_landed: bool
    is_grab: bool
    power: float
    knockback_growth: int
    fixed_knockback: int
    bonus_knockback: int
    bb1: Location
    bb2: Location


def into_dict(struct: Struct):
    d = msgspec.structs.asdict(struct)
    for k, v in d.items():
        if isinstance(v, Struct):
            d[k] = into_dict(v)
    return d
