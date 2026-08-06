from __future__ import annotations

from enum import StrEnum
from typing import Self

import msgspec
from msgspec import Struct

Location = tuple[float, float, float]
Primitive = float | Location | str | bool | int
StructDiff = tuple[*tuple[str, ...], tuple[Primitive, Primitive]]


class Situation(StrEnum):
    Nothing = "None"
    Air = "Air"
    Odd = "Odd"
    Term = "Term"
    Cliff = "Cliff"
    Ground = "Ground"
    Ladder = "Ladder"
    Outfield = "Outfield"
    Restraint = "Restraint"
    Water = "Water"


class Status(StrEnum):
    Unmapped = "Unmapped"
    Ice = "Ice"
    Run = "Run"
    Bury = "Bury"
    Fall = "Fall"
    Dash = "Dash"
    Jump = "Jump"
    Win = "Win"
    Lose = "Lose"
    Walk = "Walk"
    Grab = "Grab"
    Sleep = "Sleep"
    CliffWait = "CliffWait"
    CliffCatch = "CliffCatch"
    CliffClimb = "CliffClimb"
    CliffJump1 = "CliffJump1"
    CliffJump2 = "CliffJump2"
    CliffJump3 = "CliffJump3"
    CliffAttack = "CliffAttack"
    CliffEscape = "CliffEscape"
    EscapeAir = "EscapeAir"
    Escape = "Escape"
    EscapeB = "EscapeB"
    EscapeF = "EscapeF"
    Dead = "Dead"


class Message(Struct, array_like=True):
    stage: int
    cpu: Fighter
    opp: Fighter

    @classmethod
    def default(cls) -> Self:
        return cls(stage=0, cpu=Fighter.default(), opp=Fighter.default())


class Fighter(Struct, array_like=True):
    location: Location
    damage: float
    is_shield: bool
    attack: Attack
    grounded_ke: Location
    situation: str
    status: str

    @classmethod
    def default(cls) -> Self:
        return cls(
            location=(0, 0, 0),
            damage=0.0,
            is_shield=False,
            attack=Attack.default(),
            grounded_ke=(0, 0, 0),
            situation="None",
            status="Unmapped",
        )


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

    @classmethod
    def default(cls) -> Self:
        return cls(
            is_attack=False,
            is_landed=False,
            is_grab=False,
            power=0.0,
            knockback_growth=0,
            fixed_knockback=0,
            bonus_knockback=0,
            bb1=(0, 0, 0),
            bb2=(0, 0, 0),
        )


def into_dict(struct: Struct, int_enums: bool  = False):
    d = msgspec.structs.asdict(struct)
    for k, v in d.items():
        if isinstance(v, Struct):
            d[k] = into_dict(v, int_enums) 
        if int_enums:
            if k == "situation":
                d[k] = list(s.value for s in Situation).index(v)
            elif k == "status":
                d[k] = list(s.value for s in Status).index(v)
    return d
