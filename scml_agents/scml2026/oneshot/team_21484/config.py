from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar


class _LatticeAgentConfigMeta(type):
    def __getattr__(cls, name: str) -> Any:
        if name == "param_search_plan":
            first_overordering_queue = (
                "first_overordering_gap_scale",
                "first_overordering_gap_positive_only",
            )
            first_overordering_grid_fields = (
                "first_overordering_gap_scale",
                "first_overordering_gap_positive_only",
            )
            counter_distribution_queue = (
                "use_incoming_quantity_counter_distribution",
                "counter_distribution_always_min_one",
            )
            counter_distribution_grid_fields = (
                "use_incoming_quantity_counter_distribution",
                "counter_distribution_always_min_one",
            )
            counter_trinary_queue = (
                "use_counter_trinary_dp_decision",
                "use_counter_trinary_same_sign_smoothing",
                "deactivate_acceptanve_gate",
            )
            counter_trinary_grid_fields = (
                "use_counter_trinary_dp_decision",
                "use_counter_trinary_same_sign_smoothing",
                "deactivate_acceptanve_gate",
            )
            counter_trinary_quantity_queue = (
                "counter_trinary_total_quantity_mode",
                "counter_trinary_dp_lookahead_rounds",
                "counter_acceptance_prior_weight",
                "counter_acceptance_rate_warmup_steps",
            )
            counter_trinary_prior_grid_fields = (
                "counter_acceptance_prior_weight",
                "counter_acceptance_rate_warmup_steps",
            )
            first_trinary_grid_fields = (
                "first_overshoot_lockin_penalty",
                "first_counter_continuation_value",
                "first_proposal_warmup_steps",
            )
            counter_price_learning_queue = (
                "counter_price_warmup_steps",
                "counter_price_min_sample_per_side",
            )
            counter_price_learning_grid_fields = (
                "counter_price_warmup_steps",
                "counter_price_min_sample_per_side",
            )
            searched_fields = set(cls.searched_param_fields)
            if (
                cls.param_search_queue == first_overordering_queue
                and not searched_fields.intersection(first_overordering_grid_fields)
            ):
                return (first_overordering_grid_fields,)
            if (
                cls.param_search_queue == counter_distribution_queue
                and not searched_fields.intersection(counter_distribution_grid_fields)
            ):
                return (counter_distribution_grid_fields,)
            if (
                cls.param_search_queue[: len(counter_trinary_queue)]
                == counter_trinary_queue
                and not searched_fields.intersection(counter_trinary_grid_fields)
            ):
                return (counter_trinary_grid_fields,)
            if (
                cls.param_search_queue == counter_trinary_quantity_queue
                and set(counter_trinary_quantity_queue[:2]).issubset(searched_fields)
                and not searched_fields.intersection(
                    counter_trinary_prior_grid_fields
                )
            ):
                return (counter_trinary_prior_grid_fields,)
            if (
                cls.param_search_queue == counter_trinary_prior_grid_fields
                and not searched_fields.intersection(
                    counter_trinary_prior_grid_fields
                )
            ):
                return (counter_trinary_prior_grid_fields,)
            if (
                cls.param_search_queue == first_trinary_grid_fields
                and not searched_fields.intersection(first_trinary_grid_fields)
            ):
                return (first_trinary_grid_fields,)
            if (
                cls.param_search_queue == counter_price_learning_queue
                and not searched_fields.intersection(
                    counter_price_learning_grid_fields
                )
            ):
                return (counter_price_learning_grid_fields,)
        raise AttributeError(name)


@dataclass(frozen=True)
class LatticeAgentConfig(metaclass=_LatticeAgentConfigMeta):
    experiment_summary: ClassVar[str] = ""
    counter_trinary_neutral_lambda_prior: ClassVar[dict[float, float]] = {
        0.0: 0.35,
        0.2: 0.40,
        0.4: 0.20,
        0.6: 0.05,
    }
    counter_trinary_neutral_lambda_prior_by_level_side_delta: ClassVar[
        dict[str, dict[str, dict[str, dict[float, float]]]]
    ] = {
        "L0": {
            "sell": {
                "__overall__": {0.0: 0.682, 0.2: 0.007, 0.4: 0.005, 0.6: 0.306},
                "large_decrease": {0.0: 0.906, 0.2: 0.028, 0.4: 0.009, 0.6: 0.057},
                "large_increase": {0.0: 1.000, 0.2: 0.000, 0.4: 0.000, 0.6: 0.000},
                "small_decrease": {0.0: 0.636, 0.2: 0.003, 0.4: 0.004, 0.6: 0.357},
                "small_increase": {0.0: 0.800, 0.2: 0.000, 0.4: 0.000, 0.6: 0.200},
            },
        },
        "L1": {
            "buy": {
                "__overall__": {0.0: 0.693, 0.2: 0.057, 0.4: 0.054, 0.6: 0.196},
                "large_decrease": {0.0: 0.394, 0.2: 0.206, 0.4: 0.112, 0.6: 0.287},
                "large_increase": {0.0: 0.714, 0.2: 0.190, 0.4: 0.048, 0.6: 0.048},
                "medium_increase": {0.0: 0.873, 0.2: 0.000, 0.4: 0.032, 0.6: 0.095},
                "small_decrease": {0.0: 0.728, 0.2: 0.031, 0.4: 0.051, 0.6: 0.189},
                "small_increase": {0.0: 0.834, 0.2: 0.000, 0.4: 0.000, 0.6: 0.166},
            },
        },
    }
    first_trinary_response_prior: ClassVar[dict[str, float]] = {
        "accept": 1.0 / 3.0,
        "counter": 1.0 / 3.0,
        "reject": 1.0 / 3.0,
    }
    first_trinary_response_prior_weight: ClassVar[float] = 10.0
    counter_acceptance_rate_prior_by_level_delta: ClassVar[
        dict[str, dict[int, float]]
    ] = {
        "L0": {
            -9: 0.986,
            -8: 0.993,
            -7: 0.993,
            -6: 0.986,
            -5: 0.973,
            -4: 0.905,
            -3: 0.839,
            -2: 0.647,
            -1: 0.594,
            0: 0.525,
            1: 0.496,
            2: 0.454,
            3: 0.340,
            4: 0.333,
            5: 0.162,
            6: 0.034,
            7: 0.075,
            8: 0.000,
            9: 0.000,
        },
        "L1": {
            -9: 0.909,
            -8: 0.964,
            -7: 0.991,
            -6: 0.990,
            -5: 0.992,
            -4: 0.984,
            -3: 0.968,
            -2: 0.885,
            -1: 0.743,
            0: 0.750,
            1: 0.617,
            2: 0.626,
            3: 0.582,
            4: 0.545,
            5: 0.529,
            6: 0.500,
            7: 0.338,
            8: 0.321,
            9: 0.218,
        },
    }
    counter_trinary_rate_prior_by_level_delta: ClassVar[
        dict[str, dict[int, dict[str, float]]]
    ] = {
        "L0": {
            -9: {"accept": 0.419, "neutral": 0.578, "reject": 0.003},
            -8: {"accept": 0.466, "neutral": 0.530, "reject": 0.004},
            -7: {"accept": 0.468, "neutral": 0.527, "reject": 0.005},
            -6: {"accept": 0.462, "neutral": 0.531, "reject": 0.007},
            -5: {"accept": 0.434, "neutral": 0.554, "reject": 0.012},
            -4: {"accept": 0.312, "neutral": 0.645, "reject": 0.044},
            -3: {"accept": 0.249, "neutral": 0.676, "reject": 0.075},
            -2: {"accept": 0.160, "neutral": 0.745, "reject": 0.095},
            -1: {"accept": 0.483, "neutral": 0.303, "reject": 0.213},
            0: {"accept": 0.513, "neutral": 0.077, "reject": 0.410},
            1: {"accept": 0.508, "neutral": 0.049, "reject": 0.443},
            2: {"accept": 0.425, "neutral": 0.098, "reject": 0.478},
            3: {"accept": 0.210, "neutral": 0.361, "reject": 0.429},
            4: {"accept": 0.170, "neutral": 0.472, "reject": 0.358},
            5: {"accept": 0.032, "neutral": 0.621, "reject": 0.347},
            6: {"accept": 0.029, "neutral": 0.771, "reject": 0.200},
            7: {"accept": 0.015, "neutral": 0.691, "reject": 0.294},
            9: {"accept": 0.000, "neutral": 0.722, "reject": 0.278},
        },
        "L1": {
            -9: {"accept": 0.158, "neutral": 0.815, "reject": 0.027},
            -8: {"accept": 0.391, "neutral": 0.596, "reject": 0.012},
            -7: {"accept": 0.472, "neutral": 0.522, "reject": 0.006},
            -6: {"accept": 0.477, "neutral": 0.517, "reject": 0.006},
            -5: {"accept": 0.500, "neutral": 0.493, "reject": 0.007},
            -4: {"accept": 0.495, "neutral": 0.490, "reject": 0.015},
            -3: {"accept": 0.488, "neutral": 0.485, "reject": 0.027},
            -2: {"accept": 0.444, "neutral": 0.495, "reject": 0.062},
            -1: {"accept": 0.504, "neutral": 0.357, "reject": 0.140},
            0: {"accept": 0.381, "neutral": 0.214, "reject": 0.405},
            1: {"accept": 0.401, "neutral": 0.373, "reject": 0.227},
            2: {"accept": 0.453, "neutral": 0.341, "reject": 0.205},
            3: {"accept": 0.381, "neutral": 0.398, "reject": 0.222},
            4: {"accept": 0.369, "neutral": 0.398, "reject": 0.234},
            5: {"accept": 0.398, "neutral": 0.335, "reject": 0.267},
            6: {"accept": 0.361, "neutral": 0.345, "reject": 0.294},
            7: {"accept": 0.239, "neutral": 0.407, "reject": 0.354},
            8: {"accept": 0.188, "neutral": 0.394, "reject": 0.418},
            9: {"accept": 0.147, "neutral": 0.449, "reject": 0.404},
        },
    }
    counter_trinary_rate_prior_by_level_attempt_delta: ClassVar[
        dict[str, dict[str, dict[int, dict[str, float]]]]
    ] = {
        "L0": {
            "1": {
                -9: {"accept": 0.236, "neutral": 0.723, "reject": 0.041},
                -8: {"accept": 0.394, "neutral": 0.590, "reject": 0.017},
                -7: {"accept": 0.401, "neutral": 0.582, "reject": 0.017},
                -6: {"accept": 0.386, "neutral": 0.572, "reject": 0.042},
                -5: {"accept": 0.397, "neutral": 0.537, "reject": 0.066},
                -4: {"accept": 0.369, "neutral": 0.524, "reject": 0.107},
                -3: {"accept": 0.340, "neutral": 0.471, "reject": 0.189},
                -2: {"accept": 0.350, "neutral": 0.292, "reject": 0.358},
                -1: {"accept": 0.439, "neutral": 0.141, "reject": 0.420},
                0: {"accept": 0.347, "neutral": 0.137, "reject": 0.516},
                1: {"accept": 0.378, "neutral": 0.057, "reject": 0.565},
                2: {"accept": 0.265, "neutral": 0.073, "reject": 0.662},
                3: {"accept": 0.166, "neutral": 0.127, "reject": 0.707},
                4: {"accept": 0.075, "neutral": 0.159, "reject": 0.766},
                5: {"accept": 0.049, "neutral": 0.169, "reject": 0.783},
                6: {"accept": 0.025, "neutral": 0.171, "reject": 0.804},
                7: {"accept": 0.009, "neutral": 0.164, "reject": 0.827},
                8: {"accept": 0.006, "neutral": 0.270, "reject": 0.724},
                9: {"accept": 0.000, "neutral": 0.240, "reject": 0.760},
            },
            "2": {
                -9: {"accept": 0.148, "neutral": 0.852, "reject": 0.000},
                -8: {"accept": 0.307, "neutral": 0.691, "reject": 0.002},
                -7: {"accept": 0.286, "neutral": 0.712, "reject": 0.002},
                -6: {"accept": 0.280, "neutral": 0.716, "reject": 0.004},
                -5: {"accept": 0.275, "neutral": 0.721, "reject": 0.005},
                -4: {"accept": 0.261, "neutral": 0.726, "reject": 0.013},
                -3: {"accept": 0.251, "neutral": 0.726, "reject": 0.023},
                -2: {"accept": 0.188, "neutral": 0.775, "reject": 0.037},
                -1: {"accept": 0.263, "neutral": 0.699, "reject": 0.038},
                0: {"accept": 0.667, "neutral": 0.111, "reject": 0.222},
                1: {"accept": 0.062, "neutral": 0.798, "reject": 0.140},
                2: {"accept": 0.095, "neutral": 0.701, "reject": 0.204},
                3: {"accept": 0.100, "neutral": 0.630, "reject": 0.270},
                4: {"accept": 0.063, "neutral": 0.627, "reject": 0.309},
                5: {"accept": 0.026, "neutral": 0.622, "reject": 0.353},
                6: {"accept": 0.016, "neutral": 0.648, "reject": 0.335},
                7: {"accept": 0.010, "neutral": 0.656, "reject": 0.334},
                8: {"accept": 0.004, "neutral": 0.591, "reject": 0.405},
                9: {"accept": 0.003, "neutral": 0.590, "reject": 0.407},
            },
            "3": {
                -9: {"accept": 0.062, "neutral": 0.938, "reject": 0.000},
                -8: {"accept": 0.200, "neutral": 0.800, "reject": 0.000},
                -7: {"accept": 0.217, "neutral": 0.783, "reject": 0.000},
                -6: {"accept": 0.194, "neutral": 0.805, "reject": 0.001},
                -5: {"accept": 0.176, "neutral": 0.822, "reject": 0.002},
                -4: {"accept": 0.162, "neutral": 0.834, "reject": 0.004},
                -3: {"accept": 0.151, "neutral": 0.843, "reject": 0.005},
                -2: {"accept": 0.082, "neutral": 0.900, "reject": 0.018},
                -1: {"accept": 0.455, "neutral": 0.523, "reject": 0.022},
                0: {"accept": 0.000, "neutral": 1.000, "reject": 0.000},
                1: {"accept": 0.010, "neutral": 0.937, "reject": 0.054},
                2: {"accept": 0.024, "neutral": 0.908, "reject": 0.067},
                3: {"accept": 0.013, "neutral": 0.881, "reject": 0.106},
                4: {"accept": 0.008, "neutral": 0.835, "reject": 0.157},
                5: {"accept": 0.006, "neutral": 0.826, "reject": 0.168},
                6: {"accept": 0.007, "neutral": 0.821, "reject": 0.173},
                7: {"accept": 0.003, "neutral": 0.822, "reject": 0.175},
                8: {"accept": 0.000, "neutral": 0.762, "reject": 0.238},
                9: {"accept": 0.000, "neutral": 0.783, "reject": 0.217},
            },
            "4+": {
                -9: {"accept": 0.208, "neutral": 0.792, "reject": 0.000},
                -8: {"accept": 0.225, "neutral": 0.775, "reject": 0.000},
                -7: {"accept": 0.213, "neutral": 0.787, "reject": 0.000},
                -6: {"accept": 0.211, "neutral": 0.789, "reject": 0.001},
                -5: {"accept": 0.208, "neutral": 0.792, "reject": 0.001},
                -4: {"accept": 0.213, "neutral": 0.786, "reject": 0.001},
                -3: {"accept": 0.208, "neutral": 0.791, "reject": 0.002},
                -2: {"accept": 0.085, "neutral": 0.858, "reject": 0.057},
                -1: {"accept": 0.252, "neutral": 0.729, "reject": 0.018},
                0: {"accept": 1.000, "neutral": 0.000, "reject": 0.000},
                1: {"accept": 0.033, "neutral": 0.902, "reject": 0.066},
                2: {"accept": 0.022, "neutral": 0.880, "reject": 0.097},
                3: {"accept": 0.010, "neutral": 0.846, "reject": 0.144},
                4: {"accept": 0.012, "neutral": 0.813, "reject": 0.174},
                5: {"accept": 0.015, "neutral": 0.806, "reject": 0.179},
                6: {"accept": 0.012, "neutral": 0.806, "reject": 0.181},
                7: {"accept": 0.013, "neutral": 0.798, "reject": 0.189},
                8: {"accept": 0.010, "neutral": 0.797, "reject": 0.194},
                9: {"accept": 0.011, "neutral": 0.813, "reject": 0.175},
            },
        },
        "L1": {
            "1": {
                -9: {"accept": 0.397, "neutral": 0.246, "reject": 0.358},
                -8: {"accept": 0.461, "neutral": 0.443, "reject": 0.096},
                -7: {"accept": 0.492, "neutral": 0.484, "reject": 0.024},
                -6: {"accept": 0.496, "neutral": 0.476, "reject": 0.028},
                -5: {"accept": 0.495, "neutral": 0.477, "reject": 0.028},
                -4: {"accept": 0.488, "neutral": 0.469, "reject": 0.044},
                -3: {"accept": 0.475, "neutral": 0.453, "reject": 0.072},
                -2: {"accept": 0.465, "neutral": 0.357, "reject": 0.178},
                -1: {"accept": 0.564, "neutral": 0.123, "reject": 0.313},
                0: {"accept": 0.559, "neutral": 0.088, "reject": 0.352},
                1: {"accept": 0.417, "neutral": 0.268, "reject": 0.314},
                2: {"accept": 0.559, "neutral": 0.078, "reject": 0.363},
                3: {"accept": 0.534, "neutral": 0.057, "reject": 0.409},
                4: {"accept": 0.482, "neutral": 0.079, "reject": 0.439},
                5: {"accept": 0.403, "neutral": 0.124, "reject": 0.473},
                6: {"accept": 0.366, "neutral": 0.154, "reject": 0.479},
                7: {"accept": 0.309, "neutral": 0.194, "reject": 0.497},
                8: {"accept": 0.281, "neutral": 0.223, "reject": 0.496},
                9: {"accept": 0.185, "neutral": 0.301, "reject": 0.514},
            },
            "2": {
                -9: {"accept": 0.163, "neutral": 0.837, "reject": 0.000},
                -8: {"accept": 0.473, "neutral": 0.527, "reject": 0.000},
                -7: {"accept": 0.492, "neutral": 0.508, "reject": 0.000},
                -6: {"accept": 0.496, "neutral": 0.504, "reject": 0.000},
                -5: {"accept": 0.486, "neutral": 0.514, "reject": 0.000},
                -4: {"accept": 0.480, "neutral": 0.520, "reject": 0.000},
                -3: {"accept": 0.420, "neutral": 0.580, "reject": 0.000},
                -2: {"accept": 0.333, "neutral": 0.666, "reject": 0.002},
                -1: {"accept": 0.253, "neutral": 0.734, "reject": 0.013},
                0: {"accept": 0.374, "neutral": 0.443, "reject": 0.183},
                1: {"accept": 0.108, "neutral": 0.814, "reject": 0.079},
                2: {"accept": 0.211, "neutral": 0.754, "reject": 0.035},
                3: {"accept": 0.117, "neutral": 0.844, "reject": 0.038},
                4: {"accept": 0.084, "neutral": 0.874, "reject": 0.042},
                5: {"accept": 0.035, "neutral": 0.929, "reject": 0.036},
                6: {"accept": 0.015, "neutral": 0.942, "reject": 0.043},
                7: {"accept": 0.007, "neutral": 0.938, "reject": 0.055},
                8: {"accept": 0.001, "neutral": 0.960, "reject": 0.039},
                9: {"accept": 0.000, "neutral": 0.972, "reject": 0.028},
            },
            "3": {
                -9: {"accept": 0.097, "neutral": 0.903, "reject": 0.000},
                -8: {"accept": 0.444, "neutral": 0.556, "reject": 0.000},
                -7: {"accept": 0.458, "neutral": 0.542, "reject": 0.000},
                -6: {"accept": 0.472, "neutral": 0.528, "reject": 0.000},
                -5: {"accept": 0.458, "neutral": 0.542, "reject": 0.000},
                -4: {"accept": 0.421, "neutral": 0.579, "reject": 0.000},
                -3: {"accept": 0.339, "neutral": 0.660, "reject": 0.001},
                -2: {"accept": 0.234, "neutral": 0.766, "reject": 0.001},
                -1: {"accept": 0.183, "neutral": 0.811, "reject": 0.006},
                0: {"accept": 0.174, "neutral": 0.783, "reject": 0.043},
                1: {"accept": 0.047, "neutral": 0.896, "reject": 0.057},
                2: {"accept": 0.106, "neutral": 0.880, "reject": 0.013},
                3: {"accept": 0.037, "neutral": 0.947, "reject": 0.017},
                4: {"accept": 0.020, "neutral": 0.964, "reject": 0.016},
                5: {"accept": 0.011, "neutral": 0.975, "reject": 0.014},
                6: {"accept": 0.004, "neutral": 0.982, "reject": 0.014},
                7: {"accept": 0.003, "neutral": 0.984, "reject": 0.013},
                8: {"accept": 0.000, "neutral": 0.982, "reject": 0.018},
                9: {"accept": 0.000, "neutral": 0.993, "reject": 0.007},
            },
            "4+": {
                -9: {"accept": 0.284, "neutral": 0.716, "reject": 0.000},
                -8: {"accept": 0.446, "neutral": 0.554, "reject": 0.000},
                -7: {"accept": 0.433, "neutral": 0.567, "reject": 0.000},
                -6: {"accept": 0.460, "neutral": 0.540, "reject": 0.000},
                -5: {"accept": 0.414, "neutral": 0.586, "reject": 0.000},
                -4: {"accept": 0.382, "neutral": 0.618, "reject": 0.000},
                -3: {"accept": 0.341, "neutral": 0.659, "reject": 0.000},
                -2: {"accept": 0.121, "neutral": 0.837, "reject": 0.042},
                -1: {"accept": 0.182, "neutral": 0.804, "reject": 0.014},
                0: {"accept": 0.000, "neutral": 0.875, "reject": 0.125},
                1: {"accept": 0.049, "neutral": 0.936, "reject": 0.015},
                2: {"accept": 0.069, "neutral": 0.867, "reject": 0.064},
                3: {"accept": 0.046, "neutral": 0.857, "reject": 0.098},
                4: {"accept": 0.033, "neutral": 0.828, "reject": 0.138},
                5: {"accept": 0.025, "neutral": 0.811, "reject": 0.164},
                6: {"accept": 0.020, "neutral": 0.804, "reject": 0.176},
                7: {"accept": 0.014, "neutral": 0.804, "reject": 0.182},
                8: {"accept": 0.007, "neutral": 0.804, "reject": 0.189},
                9: {"accept": 0.000, "neutral": 0.807, "reject": 0.193},
            },
        },
    }
    layout_parameter_fields: ClassVar[dict[tuple[int, int], dict[str, str]]] = {
        (4, 4): {
            "first_overordering_scale": "first_overordering_scale_l0_4_l1_4",
            "undermismatch_sell": "undermismatch_sell_l0_4_l1_4",
            "overmismatch_sell": "overmismatch_sell_l0_4_l1_4",
        },
        (4, 5): {
            "first_overordering_scale": "first_overordering_scale_l0_4_l1_5",
            "undermismatch_sell": "undermismatch_sell_l0_4_l1_5",
            "overmismatch_sell": "overmismatch_sell_l0_4_l1_5",
        },
        (4, 6): {
            "first_overordering_scale": "first_overordering_scale_l0_4_l1_6",
            "undermismatch_sell": "undermismatch_sell_l0_4_l1_6",
            "overmismatch_sell": "overmismatch_sell_l0_4_l1_6",
        },
        (4, 7): {
            "first_overordering_scale": "first_overordering_scale_l0_4_l1_7",
            "undermismatch_sell": "undermismatch_sell_l0_4_l1_7",
            "overmismatch_sell": "overmismatch_sell_l0_4_l1_7",
        },
        (5, 4): {
            "first_overordering_scale": "first_overordering_scale_l0_5_l1_4",
            "undermismatch_sell": "undermismatch_sell_l0_5_l1_4",
            "overmismatch_sell": "overmismatch_sell_l0_5_l1_4",
        },
        (5, 5): {
            "first_overordering_scale": "first_overordering_scale_l0_5_l1_5",
            "undermismatch_sell": "undermismatch_sell_l0_5_l1_5",
            "overmismatch_sell": "overmismatch_sell_l0_5_l1_5",
        },
        (5, 6): {
            "first_overordering_scale": "first_overordering_scale_l0_5_l1_6",
            "undermismatch_sell": "undermismatch_sell_l0_5_l1_6",
            "overmismatch_sell": "overmismatch_sell_l0_5_l1_6",
        },
        (5, 7): {
            "first_overordering_scale": "first_overordering_scale_l0_5_l1_7",
            "undermismatch_sell": "undermismatch_sell_l0_5_l1_7",
            "overmismatch_sell": "overmismatch_sell_l0_5_l1_7",
        },
        (6, 4): {
            "first_overordering_scale": "first_overordering_scale_l0_6_l1_4",
            "undermismatch_sell": "undermismatch_sell_l0_6_l1_4",
            "overmismatch_sell": "overmismatch_sell_l0_6_l1_4",
        },
        (6, 5): {
            "first_overordering_scale": "first_overordering_scale_l0_6_l1_5",
            "undermismatch_sell": "undermismatch_sell_l0_6_l1_5",
            "overmismatch_sell": "overmismatch_sell_l0_6_l1_5",
        },
        (6, 6): {
            "first_overordering_scale": "first_overordering_scale_l0_6_l1_6",
            "undermismatch_sell": "undermismatch_sell_l0_6_l1_6",
            "overmismatch_sell": "overmismatch_sell_l0_6_l1_6",
        },
        (6, 7): {
            "first_overordering_scale": "first_overordering_scale_l0_6_l1_7",
            "undermismatch_sell": "undermismatch_sell_l0_6_l1_7",
            "overmismatch_sell": "overmismatch_sell_l0_6_l1_7",
        },
        (7, 4): {
            "first_overordering_scale": "first_overordering_scale_l0_7_l1_4",
            "undermismatch_sell": "undermismatch_sell_l0_7_l1_4",
            "overmismatch_sell": "overmismatch_sell_l0_7_l1_4",
        },
        (7, 5): {
            "first_overordering_scale": "first_overordering_scale_l0_7_l1_5",
            "undermismatch_sell": "undermismatch_sell_l0_7_l1_5",
            "overmismatch_sell": "overmismatch_sell_l0_7_l1_5",
        },
        (7, 6): {
            "first_overordering_scale": "first_overordering_scale_l0_7_l1_6",
            "undermismatch_sell": "undermismatch_sell_l0_7_l1_6",
            "overmismatch_sell": "overmismatch_sell_l0_7_l1_6",
        },
        (7, 7): {
            "first_overordering_scale": "first_overordering_scale_l0_7_l1_7",
        },
    }

                                       
                                        
                                      
    searched_param_fields: ClassVar[tuple[str, ...]] = ('concentration_top_k', 'counter_accept_matching_offer_tolerance', 'counter_acceptance_prior_weight', 'counter_acceptance_rate_warmup_steps', 'counter_distribution_always_min_one', 'counter_offer_price_mode', 'counter_overordering_exp', 'counter_overordering_gap_scale', 'counter_overordering_scale', 'counter_price_accept_rate_margin', 'counter_price_min_sample_per_side', 'counter_price_warmup_steps', 'counter_trinary_dp_lookahead_rounds', 'counter_trinary_dp_margin', 'counter_trinary_total_quantity_mode', 'deactivate_acceptanve_gate', 'equal', 'first_counter_continuation_value', 'first_distribution_gap_pressure_mode', 'first_distribution_mode', 'first_overordering_gap_positive_only', 'first_overordering_gap_scale', 'first_overordering_scale', 'first_overordering_scale_l0_4_l1_4', 'first_overordering_scale_l0_4_l1_5', 'first_overordering_scale_l0_4_l1_6', 'first_overordering_scale_l0_4_l1_7', 'first_overordering_scale_l0_5_l1_4', 'first_overordering_scale_l0_5_l1_5', 'first_overordering_scale_l0_5_l1_6', 'first_overordering_scale_l0_5_l1_7', 'first_overordering_scale_l0_6_l1_4', 'first_overordering_scale_l0_6_l1_5', 'first_overordering_scale_l0_6_l1_6', 'first_overordering_scale_l0_6_l1_7', 'first_overordering_scale_l0_7_l1_4', 'first_overordering_scale_l0_7_l1_5', 'first_overordering_scale_l0_7_l1_6', 'first_overordering_use_sd_ratio', 'first_overshoot_lockin_penalty', 'first_price_accept_rate_margin', 'first_proposal_warmup_steps', 'future_partner_prior_weight', 'large_offer_acceptance_margin', 'late_phase_fraction', 'late_phase_step_cap', 'mismatch_exp', 'no_change_distribute', 'overmismatch_buy', 'overmismatch_buy_linear', 'overmismatch_sell', 'overmismatch_sell_l0_4_l1_4', 'overmismatch_sell_l0_4_l1_5', 'overmismatch_sell_l0_4_l1_6', 'overmismatch_sell_l0_4_l1_7', 'overmismatch_sell_l0_5_l1_4', 'overmismatch_sell_l0_5_l1_5', 'overmismatch_sell_l0_5_l1_6', 'overmismatch_sell_l0_5_l1_7', 'overmismatch_sell_l0_6_l1_4', 'overmismatch_sell_l0_6_l1_5', 'overmismatch_sell_l0_6_l1_6', 'overmismatch_sell_l0_6_l1_7', 'overmismatch_sell_l0_7_l1_4', 'overmismatch_sell_l0_7_l1_5', 'overmismatch_sell_l0_7_l1_6', 'overmismatch_sell_linear', 'overordering_max_buying', 'undermismatch_buy', 'undermismatch_sell', 'undermismatch_sell_l0_4_l1_4', 'undermismatch_sell_l0_4_l1_5', 'undermismatch_sell_l0_4_l1_6', 'undermismatch_sell_l0_4_l1_7', 'undermismatch_sell_l0_5_l1_4', 'undermismatch_sell_l0_5_l1_5', 'undermismatch_sell_l0_5_l1_6', 'undermismatch_sell_l0_5_l1_7', 'undermismatch_sell_l0_6_l1_4', 'undermismatch_sell_l0_6_l1_5', 'undermismatch_sell_l0_6_l1_6', 'undermismatch_sell_l0_6_l1_7', 'undermismatch_sell_l0_7_l1_4', 'undermismatch_sell_l0_7_l1_5', 'undermismatch_sell_l0_7_l1_6', 'use_cash_tiebreak', 'use_counter_trinary_dp_before_threshold', 'use_counter_trinary_dp_decision', 'use_counter_trinary_neutral_quantity_shrink', 'use_counter_trinary_same_sign_smoothing', 'use_future_partner_prior', 'use_gap_price', 'use_incoming_quantity_counter_distribution', 'use_large_shortage_offer_acceptance', 'use_utility_acceptance_choice', 'utility_fallback_relative_time')
    param_search_queue: ClassVar[tuple[str, ...]] = ()
    world_search_plan: ClassVar[tuple[dict[str, Any], ...]] = (
        {
            "layout": (7, 6),
            "summary": "L0側が最大混雑でLatticeの相対優位がほぼ消える代表",
            "param_search_plan": (
                ("first_overordering_scale_l0_7_l1_6",),
                (
                    "undermismatch_sell_l0_7_l1_6",
                    "overmismatch_sell_l0_7_l1_6",
                ),
            ),
        },
        {
            "layout": (4, 7),
            "summary": "L1側が最大混雑でLatticeが強い代表",
            "param_search_plan": (
                ("first_overordering_scale_l0_4_l1_7",),
                (
                    "undermismatch_sell_l0_4_l1_7",
                    "overmismatch_sell_l0_4_l1_7",
                ),
            ),
        },
        {
            "layout": (7, 4),
            "summary": "L0最大かつL1最小で全体スコアが低めの端点",
            "param_search_plan": (
                ("first_overordering_scale_l0_7_l1_4",),
                (
                    "undermismatch_sell_l0_7_l1_4",
                    "overmismatch_sell_l0_7_l1_4",
                ),
            ),
        },
        {
            "layout": (4, 4),
            "summary": "低混雑側の端点",
            "param_search_plan": (
                ("first_overordering_scale_l0_4_l1_4",),
                (
                    "undermismatch_sell_l0_4_l1_4",
                    "overmismatch_sell_l0_4_l1_4",
                ),
            ),
        },
        {
            "layout": (6, 7),
            "summary": "L1側が多い高混雑寄りの代表",
            "param_search_plan": (
                ("first_overordering_scale_l0_6_l1_7",),
                (
                    "undermismatch_sell_l0_6_l1_7",
                    "overmismatch_sell_l0_6_l1_7",
                ),
            ),
        },
        {
            "layout": (4, 6),
            "summary": "L0最小かつL1がやや多い配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_4_l1_6",),
                (
                    "undermismatch_sell_l0_4_l1_6",
                    "overmismatch_sell_l0_4_l1_6",
                ),
            ),
        },
        {
            "layout": (6, 4),
            "summary": "L0がやや多くL1最小の配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_6_l1_4",),
                (
                    "undermismatch_sell_l0_6_l1_4",
                    "overmismatch_sell_l0_6_l1_4",
                ),
            ),
        },
        {
            "layout": (5, 5),
            "summary": "中間的な均衡配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_5_l1_5",),
                (
                    "undermismatch_sell_l0_5_l1_5",
                    "overmismatch_sell_l0_5_l1_5",
                ),
            ),
        },
        {
            "layout": (7, 5),
            "summary": "L0最大かつL1中程度の配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_7_l1_5",),
                (
                    "undermismatch_sell_l0_7_l1_5",
                    "overmismatch_sell_l0_7_l1_5",
                ),
            ),
        },
        {
            "layout": (5, 7),
            "summary": "L1最大かつL0中程度の配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_5_l1_7",),
                (
                    "undermismatch_sell_l0_5_l1_7",
                    "overmismatch_sell_l0_5_l1_7",
                ),
            ),
        },
        {
            "layout": (5, 4),
            "summary": "L0中程度かつL1最小の配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_5_l1_4",),
                (
                    "undermismatch_sell_l0_5_l1_4",
                    "overmismatch_sell_l0_5_l1_4",
                ),
            ),
        },
        {
            "layout": (4, 5),
            "summary": "L0最小かつL1中程度の配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_4_l1_5",),
                (
                    "undermismatch_sell_l0_4_l1_5",
                    "overmismatch_sell_l0_4_l1_5",
                ),
            ),
        },
        {
            "layout": (6, 6),
            "summary": "高めの均衡配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_6_l1_6",),
                (
                    "undermismatch_sell_l0_6_l1_6",
                    "overmismatch_sell_l0_6_l1_6",
                ),
            ),
        },
        {
            "layout": (5, 6),
            "summary": "L1がやや多い中間配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_5_l1_6",),
                (
                    "undermismatch_sell_l0_5_l1_6",
                    "overmismatch_sell_l0_5_l1_6",
                ),
            ),
        },
        {
            "layout": (6, 5),
            "summary": "L0がやや多い中間配置",
            "param_search_plan": (
                ("first_overordering_scale_l0_6_l1_5",),
                (
                    "undermismatch_sell_l0_6_l1_5",
                    "overmismatch_sell_l0_6_l1_5",
                ),
            ),
        },
    )

                                              
                                                                          
    equal: bool = field(
        default=False,
        metadata={"candidates": (False,)},
    )
    first_overordering_scale: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_4_l1_4: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_4_l1_5: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_4_l1_6: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_4_l1_7: float = field(
        default=0.4,
        metadata={
            "candidates": (0.4,),
            "note": "",
        },
    )
    first_overordering_scale_l0_5_l1_4: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_5_l1_5: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_5_l1_6: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_5_l1_7: float = field(
        default=0.3,
        metadata={
            "candidates": (0.3,),
            "note": "",
        },
    )
    first_overordering_scale_l0_6_l1_4: float = field(
        default=0.2,
        metadata={
            "candidates": (0.2,),
            "note": "",
        },
    )
    first_overordering_scale_l0_6_l1_5: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_6_l1_6: float = field(
        default=0.3,
        metadata={
            "candidates": (0.3,),
            "note": "",
        },
    )
    first_overordering_scale_l0_6_l1_7: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_7_l1_4: float = field(
        default=0.4,
        metadata={
            "candidates": (0.4,),
            "note": "",
        },
    )
    first_overordering_scale_l0_7_l1_5: float = field(
        default=0.25,
        metadata={
            "candidates": (0.25,),
            "note": "",
        },
    )
    first_overordering_scale_l0_7_l1_6: float = field(
        default=0.3,
        metadata={
            "candidates": (0.3,),
            "note": "",
        },
    )
    first_overordering_scale_l0_7_l1_7: float = field(
        default=0.25,
        metadata={
            "candidates": (),
            "note": "",
        },
    )
    first_overordering_use_sd_ratio: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    first_overordering_gap_scale: float = field(
        default=0.0,
        metadata={
            "candidates": (0.02, 0.04, 0.06),
            "note": "",
        },
    )
    first_overordering_gap_positive_only: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    first_distribution_mode: int = field(
        default=0,
        metadata={
            "candidates": (0,),
            "note": "",
        },
    )
    first_distribution_gap_pressure_mode: int = field(
        default=1,
        metadata={
            "candidates": (1,),
            "note": "",
        },
    )
    first_overshoot_lockin_penalty: float = field(
        default=2.0,
        metadata={
            "candidates": (2.0,),
            "note": "",
        },
    )
    first_counter_continuation_value: float = field(
        default=0.5,
        metadata={
            "candidates": (0.5,),
            "note": "",
        },
    )
    first_proposal_warmup_steps: int = field(
        default=25,
        metadata={
            "candidates": (25,),
            "note": "",
        },
    )
    first_trinary_outcome_eval_mode: int = field(
        default=2,
        metadata={
            "candidates": (2,),
            "note": "",
        },
    )
    first_trinary_allocation_candidate_mode: int = field(
        default=1,
        metadata={
            "candidates": (1,),
            "note": "",
        },
    )
    overordering_max_buying: float = field(
        default=0.15,
        metadata={
            "candidates": (0.15,),
            "note": "",
        },
    )
    overordering_min: float = field(
        default=0.0,
        metadata={"candidates": ()},                               
    )
    counter_overordering_scale: float = field(
        default=0.1,
        metadata={
            "candidates": (0.1,),
            "note": "",
        },
    )
    counter_overordering_exp: float = field(
        default=0.4,
        metadata={
            "candidates": (0.4,),
            "note": "",
        },
    )
    counter_overordering_gap_scale: float = field(
        default=0.0,
        metadata={
            "candidates": (0.02, 0.04, 0.06),
            "note": "",
        },
    )
    use_incoming_quantity_counter_distribution: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    counter_distribution_always_min_one: bool = field(
        default=False,
        metadata={
            "candidates": (False,),
            "note": "",
        },
    )
    use_large_shortage_offer_acceptance: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    large_offer_acceptance_margin: float = field(
        default=0,
        metadata={
            "candidates": (0,),
            "note": "",
        },
    )
                                                                  
    mismatch_exp: float = field(
        default=2.0,
        metadata={"candidates": (2.0,)},
    )
    undermismatch_sell: float = field(
        default=-1,
        metadata={
            "candidates": (-1,),
            "note": "",
        },
    )
    overmismatch_sell: float = field(
        default=-1,
        metadata={
            "candidates": (-1,),
            "note": "",
        },
    )
    undermismatch_sell_l0_4_l1_4: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_4_l1_4: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_4_l1_5: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_4_l1_5: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_4_l1_6: float = field(
        default=0.23,
        metadata={
            "candidates": (0.23,),
            "note": "",
        },
    )
    overmismatch_sell_l0_4_l1_6: float = field(
        default=0.1,
        metadata={
            "candidates": (0.1,),
            "note": "",
        },
    )
    undermismatch_sell_l0_4_l1_7: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_4_l1_7: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_5_l1_4: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_5_l1_4: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_5_l1_5: float = field(
        default=0.33,
        metadata={
            "candidates": (0.33,),
            "note": "",
        },
    )
    overmismatch_sell_l0_5_l1_5: float = field(
        default=0.3,
        metadata={
            "candidates": (0.3,),
            "note": "",
        },
    )
    undermismatch_sell_l0_5_l1_6: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_5_l1_6: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_5_l1_7: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_5_l1_7: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_6_l1_4: float = field(
        default=0.23,
        metadata={
            "candidates": (0.23,),
            "note": "",
        },
    )
    overmismatch_sell_l0_6_l1_4: float = field(
        default=0.3,
        metadata={
            "candidates": (0.3,),
            "note": "",
        },
    )
    undermismatch_sell_l0_6_l1_5: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_6_l1_5: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_6_l1_6: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_6_l1_6: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_6_l1_7: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_6_l1_7: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_sell_l0_7_l1_4: float = field(
        default=0.28,
        metadata={
            "candidates": (0.28,),
            "note": "",
        },
    )
    overmismatch_sell_l0_7_l1_4: float = field(
        default=0.1,
        metadata={
            "candidates": (0.1,),
            "note": "",
        },
    )
    undermismatch_sell_l0_7_l1_5: float = field(
        default=0.23,
        metadata={
            "candidates": (0.23,),
            "note": "",
        },
    )
    overmismatch_sell_l0_7_l1_5: float = field(
        default=0.2,
        metadata={
            "candidates": (0.2,),
            "note": "",
        },
    )
    undermismatch_sell_l0_7_l1_6: float = field(
        default=-1,
        metadata={
            "candidates": (0.23, 0.28, 0.33),
            "note": "",
        },
    )
    overmismatch_sell_l0_7_l1_6: float = field(
        default=-1,
        metadata={
            "candidates": (0.1, 0.2, 0.3),
            "note": "",
        },
    )
    undermismatch_buy: float = field(
        default=-1,
        metadata={
            "candidates": (-1,),
            "note": "",
        },
    )
    overmismatch_buy: float = field(
        default=0.1,
        metadata={
            "candidates": (0.1,),
            "note": "",
        },
    )
    overmismatch_sell_linear: float = field(
        default=0.15,
        metadata={
            "candidates": (0.15,),
            "note": "",
        },
    )
    overmismatch_buy_linear: float = field(
        default=-1,
        metadata={
            "candidates": (-1,),
            "note": "",
        },
    )
    late_phase_fraction: float = field(
        default=0.5,
        metadata={"candidates": (0.5,)},
    )
    late_phase_step_cap: int = field(
        default=60,
        metadata={"candidates": (60,)},
    )
    concentration_top_k: int = field(
        default=1,
        metadata={
            "candidates": (1,),
            "note": "",
        },
    )
    no_change_distribute: bool = field(
        default=False,
        metadata={
            "candidates": (False,),
            "note": "",
        },
    )
    use_cash_tiebreak: bool = field(
        default=False,
        metadata={
            "candidates": (False,),
            "note": "",
        },
    )
    use_gap_price: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    counter_offer_price_mode: int = field(
        default=0,
        metadata={
            "candidates": (0,),
            "note": "",
        },
    )
    counter_price_warmup_steps: int = field(
        default=50,
        metadata={
            "candidates": (50,),
            "note": "",
        },
    )
    counter_price_min_sample_per_side: int = field(
        default=3,
        metadata={
            "candidates": (3,),
            "note": "",
        },
    )
    counter_price_accept_rate_margin: float = field(
        default=0.15,
        metadata={
            "candidates": (0.15,),
            "note": "",
        },
    )
    first_price_warmup_steps: int = field(
        default=25,
        metadata={
            "candidates": (25,),
            "note": "",
        },
    )
    first_price_min_sample_per_side: int = field(
        default=5,
        metadata={
            "candidates": (5,),
            "note": "",
        },
    )
    first_price_accept_rate_margin: float = field(
        default=-1.0,
        metadata={
            "candidates": (-1.0,),
            "note": "",
        },
    )
    gap_d_scaler: float = field(
        default=-1,
        metadata={
            "candidates": (0.25, 0.5, 1.0),
            "note": "",
        },
    )
    utility_fallback_relative_time: float = field(
        default=0.42,
        metadata={
            "candidates": (0.42,),
            "note": "",
        },
    )
    deactivate_acceptanve_gate: int = field(
        default=1,
        metadata={
            "candidates": (1,),
            "note": "",
        },
    )
    use_utility_acceptance_choice: bool = field(
        default=False,
        metadata={
            "candidates": (False,),
            "note": "",
        },
    )
    use_future_partner_prior: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    future_partner_prior_weight: float = field(
        default=1.0,
        metadata={
            "candidates": (1.0,),
            "note": "",
        },
    )
    counter_accept_matching_offer_tolerance: int = field(
        default=0,
        metadata={
            "candidates": (0,),
            "note": "",
        },
    )
    counter_acceptance_prior_weight: float = field(
        default=30.0,
        metadata={
            "candidates": (30.0,),
            "note": "",
        },
    )
    counter_acceptance_rate_warmup_steps: int = field(
        default=50,
        metadata={
            "candidates": (50,),
            "note": "",
        },
    )
    use_counter_trinary_dp_decision: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    use_counter_trinary_dp_before_threshold: bool = field(
        default=False,
        metadata={
            "candidates": (False,),
            "note": "",
        },
    )
    use_counter_trinary_same_sign_smoothing: bool = field(
        default=True,
        metadata={
            "candidates": (True,),
            "note": "",
        },
    )
    use_counter_trinary_neutral_quantity_shrink: bool = field(
        default=False,
        metadata={
            "candidates": (False,),
            "note": "",
        },
    )
    counter_trinary_approx_candidate_mode: int = field(
        default=1,
        metadata={
            "candidates": (1,),
            "note": "",
        },
    )
    counter_trinary_dp_margin: float = field(
        default=0.0,
        metadata={
            "candidates": (0.0,),
            "note": "",
        },
    )
    counter_trinary_dp_lookahead_rounds: int = field(
        default=3,
        metadata={
            "candidates": (3,),
            "note": "",
        },
    )
    counter_trinary_total_quantity_mode: int = field(
        default=0,
        metadata={
            "candidates": (0,),
            "note": "",
        },
    )
    counter_trinary_extended_total_quantity_multiplier: float = field(
        default=2.0,
        metadata={
            "candidates": (2.0,),
            "note": "",
        },
    )
    round_offer_update_rate: float = field(
        default=0.3,
        metadata={"candidates": ()},
    )

    def to_kwargs(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "LatticeAgentConfig":
        return cls(**{key: values[key] for key in cls.field_names() if key in values})

    @classmethod
    def default_values(cls) -> dict[str, Any]:
                                                                           
        return {f.name: f.default for f in fields(cls)}

    @classmethod
    def candidate_values(cls) -> dict[str, list[Any]]:
\
\
\
\
           
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
