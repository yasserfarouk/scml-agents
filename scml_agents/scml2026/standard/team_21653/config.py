\
\
\
\
\
   

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar


@dataclass(frozen=True)
class HorizonAwareAgentConfig:
    experiment_summary: ClassVar[str] = (
        "HorizonAwareAgentの大幅な機能改修中。各戦略機能の有効/無効を切り替えながら、個別の有効性と相互作用を判定する"
    )

                                                                                  
    searched_param_fields: ClassVar[tuple[str, ...]] = ('adjust_last_level_proposal_price', 'allowable_excess_sell_margin', 'first_level_future_offer_divisor', 'first_level_staggered_future_sell', 'lastlevel_inventory_buffer_days', 'middle_level_negotiation_mode', 'middlelevel_inventory_buffer_days', 'productivity', 'ptoday', 'sell_partner_filter_enabled', 'threshold', 'use_bankruptcy_forecast')
    param_search_queue: ClassVar[tuple[str, ...]] = ('first_level_sell_mode', 'last_level_negotiation_mode')
    param_search_plan: ClassVar[tuple[tuple[str, ...], ...]] = ()

    threshold: float | None = field(
        default=None,
        metadata={
            "candidates": (None, 0.75, 1.0, 1.25),
            "note": "",
        },
    )
    ptoday: tuple[float, float] = field(
        default=(0.5, 0.5),
        metadata={
            "candidates": (
                (0.5, 0.5),
                (0.6, 0.6),
                (0.7, 0.7),
                (0.8, 0.8),
                (1.0, 0.5),
                (0.9, 0.5),
                (0.8, 0.5),
            ),
            "note": "",
        },
    )
    productivity: float = field(
        default=0.7,
        metadata={
            "candidates": (0.60, 0.70, 0.80),
            "note": "",
        },
    )
    lastlevel_inventory_buffer_days: int = field(
        default=1,
        metadata={
            "candidates": (0, 1, 2, 3, 4),
            "note": "",
        },
    )
    last_level_negotiation_mode: int = field(
        default=3,
        metadata={
            "candidates": (0, 3, 4, 5),
            "note": "",
        },
    )
    adjust_last_level_proposal_price: bool = field(
        default=False,
        metadata={
            "candidates": (False, True),
            "note": "",
        },
    )
    middle_level_negotiation_mode: int = field(
        default=9,
        metadata={
            "candidates": (0, 6, 7, 8, 9),
            "note": "",
        },
    )
    middlelevel_inventory_buffer_days: int = field(
        default=0,
        metadata={
            "candidates": (0, 1, 2, 3, 4),
            "note": "",
        },
    )
    first_level_sell_mode: int = field(
        default=3,
        metadata={
            "candidates": (2, 3),
            "note": "",
        },
    )
    first_level_future_offer_divisor: int = field(
        default=1,
        metadata={
            "candidates": (1, 2, 3, 4),
            "note": "",
        },
    )
    first_level_staggered_future_sell: int = field(
        default=0,
        metadata={
            "candidates": (0, 1, 2),
            "note": "",
        },
    )
    sell_partner_filter_enabled: bool = field(
        default=True,
        metadata={
            "candidates": (False, True),
            "note": "",
        },
    )
    use_bankruptcy_forecast: bool = field(
        default=True,
        metadata={
            "candidates": (False, True),
            "note": "",
        },
    )
    allowable_excess_sell_margin: int = field(
        default=1,
        metadata={
            "candidates": (10, 0, 1, 2, 3),
            "note": "",
        },
    )

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "HorizonAwareAgentConfig":
        return cls(**{key: values[key] for key in cls.field_names() if key in values})

    @classmethod
    def default_values(cls) -> dict[str, Any]:
        return {f.name: f.default for f in fields(cls)}

    @classmethod
    def candidate_values(cls) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = {}
        for f in fields(cls):
            candidates = f.metadata.get("candidates", ())
            if candidates:
                result[f.name] = list(candidates)
        return result

    @classmethod
    def field_notes(cls) -> dict[str, str]:
        result: dict[str, str] = {}
        for f in fields(cls):
            note = f.metadata.get("note")
            if note:
                result[f.name] = str(note)
        return result

    @classmethod
    def searched_fields(cls) -> set[str]:
        return set(cls.searched_param_fields)


                                                               
PenguinAgentConfig = HorizonAwareAgentConfig
