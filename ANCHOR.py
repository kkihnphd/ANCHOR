# -*- coding: utf-8 -*-
"""
ANCHOR Fitting Assistant

A Tkinter desktop app for fitting time-resolved kinetic traces with:
    - standard kinetic models
    - direct zero-intermediate kinetic models (E + S -> P)
    - reversible Michaelis-Menten turnover models (E + S <-> ES -> P)
    - slow-apo kinetic models
    - optional linear-only diagnostic models
    - optional no-binding models
    - individual best-fit plots
    - species/apparent-species plots for every trace
    - all-runs and per-concentration combined plots
    - CSV summaries of model selection, best-fit parameters, direct-model velocities, reversible Michaelis-Menten turnover fits, and Michaelis-Menten post-fit curves

Run from a terminal with:
    python ANCHOR.py

Required packages:
    numpy pandas matplotlib scipy
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend because this app only saves figures to disk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk


# User-facing unit choices. ANCHOR converts all time values to minutes and all
# concentrations to micromolar/millimolar internally so the kinetic equations
# and historical output columns remain backward compatible.
TIME_UNIT_TO_MIN = {
    "seconds": 1.0 / 60.0,
    "minutes": 1.0,
}

CONCENTRATION_UNIT_TO_UM = {
    "nM": 1.0e-3,
    "µM": 1.0,
    "uM": 1.0,  # accepted for backward-compatible JSON/config files
    "mM": 1.0e3,
    "M": 1.0e6,
}

GUI_CONCENTRATION_UNITS = ("nM", "µM", "mM", "M")
GUI_TIME_UNITS = ("seconds", "minutes")


##############################################################################
# CONFIGURATION OBJECTS
##############################################################################


@dataclass
class ConcentrationGroup:
    concentration_uM: float = 50.0
    n_samples: int = 9


@dataclass
class FittingConfig:
    # Data/output settings
    input_csv: str = ""
    output_parent: str = str(Path.home())
    output_folder_name: str = "ANCHOR_Fitting_Output"
    concentration_groups: List[ConcentrationGroup] = field(
        default_factory=lambda: [ConcentrationGroup(50.0, 9), ConcentrationGroup(200.0, 9)]
    )
    E0_uM: float = 5.0

    # User-facing input units. Values are converted to canonical internal units:
    # time -> minutes; concentrations -> µM (and mM where required by ODEs).
    input_time_unit: str = "minutes"
    enzyme_concentration_unit: str = "µM"
    substrate_concentration_unit: str = "µM"

    # High-level analysis mode. The GUI uses this to keep routine workflows
    # simple while preserving full manual control in custom mode.
    # Options: general_auto, turnover_auto, direct_only, reversible_mm_only, custom
    analysis_mode: str = "general_auto"
    auto_detect_turnover_mechanism: bool = False

    # Model-selection settings
    max_intermediates: int = 5
    include_zero_intermediate_kinetic_model: bool = True
    fit_zero_intermediate_only: bool = False
    loosen_zero_intermediate_bounds: bool = True
    include_reversible_mm_model: bool = False
    fit_reversible_mm_only: bool = False
    loosen_reversible_mm_bounds: bool = True
    use_standard_fit_mode: bool = True
    use_slow_apo_fit_mode: bool = True
    include_linear_only_model: bool = True
    include_no_binding_model: bool = True
    n_starts_screen: int = 12
    n_starts_refine: int = 25
    selection_stride: int = 3
    random_seed: int = 42

    # Data preprocessing / fitting window
    fit_start_time: float = 0.01
    early_max_window_min: float = 0.50

    # Smoothed-target settings
    use_smoothed_target: bool = True
    raw_weight: float = 1.0
    smooth_weight: float = 1.5
    smooth_window_fraction: float = 0.10
    smooth_polyorder: int = 3

    # Early-time weighting
    early_weight_strength: float = 3.0
    early_weight_tau: float = 0.25

    # Baseline / linear drift settings
    fit_baseline_offset: bool = True
    fit_baseline_slope: bool = True
    baseline_offset_min: float = -1.00
    baseline_offset_max: float = 1.00
    baseline_slope_min: float = -3.00
    baseline_slope_max: float = 3.00
    initialize_baseline_slope_from_data: bool = True
    linear_intercept_min: float = 0.0
    linear_intercept_max: float = 1.5
    linear_slope_min: float = -3.00
    linear_slope_max: float = 3.00

    # No-binding model settings
    # The no-binding model is a constant signal model. It is preferred when
    # more complex models only explain small changes that are not meaningful
    # relative to the trace noise.
    no_binding_intercept_min: float = 0.0
    no_binding_intercept_max: float = 1.5
    no_binding_bic_preference_tolerance: float = 6.0
    no_binding_noise_multiplier: float = 2.0
    no_binding_min_signal_change: float = 0.05
    no_binding_prefer_if_change_within_noise: bool = True

    # Hard shape constraints
    integrate_from_true_time_zero: bool = True
    fix_free_enzyme_response: bool = True
    free_enzyme_response: float = 1.0
    first_intermediate_must_not_exceed_E: bool = True
    first_intermediate_upper: float = 1.0
    prevent_initial_peak: bool = True
    auto_disable_initial_peak_penalty_for_increasing_traces: bool = True
    initial_peak_window_min: float = 0.12
    initial_peak_tolerance: float = 0.02
    initial_peak_penalty_weight: float = 100.0
    start_value_penalty_weight: float = 4.0

    # Parameter bounds and starting values
    standard_k1_min: float = 1e-8
    standard_k1_max: float = 1e6
    standard_k1_initial_guess: float = 10.0
    slow_apo_k1_min: float = 1e-8
    slow_apo_k1_max: float = 1e-1
    slow_apo_k1_initial_guess: float = 1e-3
    k_other_min: float = 1e-8
    k_other_max: float = 1e4
    k_other_initial_guess: float = 2.0
    response_min: float = 0.0
    response_max: float = 3.0

    # Reversible Michaelis-Menten turnover model bounds
    # Mechanism: E + S <-> ES -> P. Internal units use minutes and millimolar.
    mm_kon_min: float = 1e-8
    mm_kon_max: float = 1e6
    mm_kon_initial_guess: float = 10.0
    mm_koff_min: float = 1e-8
    mm_koff_max: float = 1e4
    mm_koff_initial_guess: float = 1.0
    mm_kcat_min: float = 1e-8
    mm_kcat_max: float = 1e4
    mm_kcat_initial_guess: float = 1.0

    # Michaelis-Menten post-fit analysis
    # When enabled, ANCHOR collects velocities from selected E + S -> P
    # direct 0-intermediate fits across substrate/reactant concentrations and
    # fits v = Vmax[S]/(Km + [S]). kcat is calculated as Vmax/E0 when the
    # raw trace units are concentration units matching E0_uM.
    fit_michaelis_menten: bool = True
    mm_use_fitted_direct_velocity: bool = True
    mm_fit_group_means: bool = True
    mm_min_points: int = 3

    # Output cleanup
    organize_outputs: bool = True

    @property
    def output_dir(self) -> str:
        return str(Path(self.output_parent).expanduser() / self.output_folder_name)

    @property
    def E0_mM(self) -> float:
        return self.E0_uM / 1000.0

    @staticmethod
    def normalize_concentration_unit(unit: str) -> str:
        unit = str(unit or "µM").strip()
        if unit == "uM":
            return "µM"
        return unit

    @staticmethod
    def concentration_to_uM(value: float, unit: str) -> float:
        unit_key = str(unit or "µM").strip()
        if unit_key not in CONCENTRATION_UNIT_TO_UM:
            raise ValueError(f"Unsupported concentration unit: {unit}")
        return float(value) * CONCENTRATION_UNIT_TO_UM[unit_key]

    @staticmethod
    def concentration_from_uM(value_uM: float, unit: str) -> float:
        unit_key = str(unit or "µM").strip()
        if unit_key not in CONCENTRATION_UNIT_TO_UM:
            raise ValueError(f"Unsupported concentration unit: {unit}")
        return float(value_uM) / CONCENTRATION_UNIT_TO_UM[unit_key]

    @staticmethod
    def time_to_minutes(values: np.ndarray, unit: str) -> np.ndarray:
        unit_key = str(unit or "minutes").strip().lower()
        if unit_key not in TIME_UNIT_TO_MIN:
            raise ValueError(f"Unsupported time unit: {unit}")
        return np.asarray(values, dtype=float) * TIME_UNIT_TO_MIN[unit_key]

    @property
    def D_values_uM(self) -> List[float]:
        values: List[float] = []
        for group in self.concentration_groups:
            values.extend([float(group.concentration_uM)] * int(group.n_samples))
        return values

    @property
    def kinetic_fit_modes(self) -> List[str]:
        modes: List[str] = []
        if self.use_standard_fit_mode:
            modes.append("standard")
        if self.use_slow_apo_fit_mode:
            modes.append("slow_apo")
        return modes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FittingConfig":
        clean_data = dict(data)
        groups_raw = clean_data.get("concentration_groups", [])
        groups: List[ConcentrationGroup] = []
        for item in groups_raw:
            if isinstance(item, ConcentrationGroup):
                groups.append(item)
            elif isinstance(item, dict):
                groups.append(ConcentrationGroup(**item))
        clean_data["concentration_groups"] = groups or cls().concentration_groups

        # Older setup files did not store unit preferences. They therefore
        # retain the historical defaults of minutes and micromolar.
        clean_data["input_time_unit"] = str(clean_data.get("input_time_unit", "minutes")).lower()
        clean_data["enzyme_concentration_unit"] = cls.normalize_concentration_unit(
            clean_data.get("enzyme_concentration_unit", "µM")
        )
        clean_data["substrate_concentration_unit"] = cls.normalize_concentration_unit(
            clean_data.get("substrate_concentration_unit", "µM")
        )

        valid_fields = set(cls.__dataclass_fields__.keys())
        clean_data = {key: value for key, value in clean_data.items() if key in valid_fields}
        return cls(**clean_data)


DEFAULT_CONFIG = FittingConfig()


##############################################################################
# FITTING ENGINE
##############################################################################


class CancelledError(RuntimeError):
    pass


class StoppedFlowFitter:
    def __init__(
        self,
        config: FittingConfig,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.cfg = config
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event
        self.rng = np.random.default_rng(self.cfg.random_seed)

    def log(self, message: str) -> None:
        if self.log_callback is not None:
            self.log_callback(message)
        else:
            print(message)

    def check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError("Fitting was cancelled by the user.")

    def apply_model_scope_overrides(self) -> None:
        """Apply high-level model-scope presets before validation/fitting.

        ANCHOR exposes a small set of user-facing analysis modes so routine
        users do not have to reason through every low-level model checkbox.
        Custom mode leaves the detailed switches untouched. The turnover-auto
        mode evaluates both direct product formation and reversible
        Michaelis-Menten turnover candidates without requiring the user to
        pre-specify which mechanism applies.
        """
        mode = str(getattr(self.cfg, "analysis_mode", "general_auto") or "general_auto")

        if mode == "direct_only":
            self.cfg.fit_zero_intermediate_only = True
            self.cfg.fit_reversible_mm_only = False
            self.cfg.auto_detect_turnover_mechanism = False
        elif mode == "reversible_mm_only":
            self.cfg.fit_reversible_mm_only = True
            self.cfg.fit_zero_intermediate_only = False
            self.cfg.auto_detect_turnover_mechanism = False
        elif mode == "turnover_auto":
            self.cfg.auto_detect_turnover_mechanism = True
            self.cfg.fit_zero_intermediate_only = False
            self.cfg.fit_reversible_mm_only = False
        elif mode == "general_auto":
            # General model-selection mode: use the standard ANCHOR candidate
            # library, but do not automatically include the reversible-MM
            # enzyme-turnover candidate unless the user enables it in custom mode.
            self.cfg.auto_detect_turnover_mechanism = False
            self.cfg.fit_zero_intermediate_only = False
            self.cfg.fit_reversible_mm_only = False

        if self.cfg.fit_reversible_mm_only:
            # Candidate-library override: only the reversible enzyme-turnover
            # candidate is evaluated. This is appropriate for known enzyme-
            # substrate progress curves where the goal is to fit kon, koff,
            # kcat, Km, and Vmax-related quantities directly from the trace.
            self.cfg.include_reversible_mm_model = True
            self.cfg.include_zero_intermediate_kinetic_model = False
            self.cfg.max_intermediates = 0
            self.cfg.use_standard_fit_mode = False
            self.cfg.use_slow_apo_fit_mode = False
            self.cfg.include_linear_only_model = False
            self.cfg.include_no_binding_model = False
            self.cfg.prevent_initial_peak = False

            if self.cfg.loosen_reversible_mm_bounds:
                self._apply_turnover_friendly_bounds()
                self.cfg.mm_kon_min = min(self.cfg.mm_kon_min, 1e-12)
                self.cfg.mm_kon_max = max(self.cfg.mm_kon_max, 1e6)
                self.cfg.mm_koff_max = max(self.cfg.mm_koff_max, 1e5)
                self.cfg.mm_kcat_max = max(self.cfg.mm_kcat_max, 1e5)
            return

        if self.cfg.fit_zero_intermediate_only:
            # Candidate-library override: only one kinetic candidate is evaluated.
            self.cfg.include_zero_intermediate_kinetic_model = True
            self.cfg.include_reversible_mm_model = False
            self.cfg.max_intermediates = 0
            self.cfg.use_standard_fit_mode = True
            self.cfg.use_slow_apo_fit_mode = False
            self.cfg.include_linear_only_model = False
            self.cfg.include_no_binding_model = False
            self.cfg.prevent_initial_peak = False

            if self.cfg.loosen_zero_intermediate_bounds:
                self._apply_turnover_friendly_bounds()
                self.cfg.standard_k1_min = min(self.cfg.standard_k1_min, 1e-12)
                self.cfg.standard_k1_max = max(self.cfg.standard_k1_max, 1e6)
                self.cfg.standard_k1_initial_guess = min(
                    max(self.cfg.standard_k1_initial_guess, self.cfg.standard_k1_min),
                    self.cfg.standard_k1_max,
                )
            return

        if self.cfg.auto_detect_turnover_mechanism:
            # Automatic enzyme-turnover discovery mode. ANCHOR compares the
            # direct product-formation candidate (E + S -> P) against the
            # reversible enzyme-substrate candidate (E + S <-> ES -> P)
            # while retaining no-binding and linear-only controls.
            self.cfg.include_zero_intermediate_kinetic_model = True
            self.cfg.include_reversible_mm_model = True
            self.cfg.max_intermediates = 0
            self.cfg.use_standard_fit_mode = True
            self.cfg.use_slow_apo_fit_mode = False
            self.cfg.include_linear_only_model = True
            self.cfg.include_no_binding_model = True
            self.cfg.prevent_initial_peak = False
            self.cfg.mm_use_fitted_direct_velocity = True
            self.cfg.mm_fit_group_means = True
            self.cfg.loosen_zero_intermediate_bounds = True
            self.cfg.loosen_reversible_mm_bounds = True
            self._apply_turnover_friendly_bounds()
            self.cfg.standard_k1_min = min(self.cfg.standard_k1_min, 1e-12)
            self.cfg.standard_k1_max = max(self.cfg.standard_k1_max, 1e6)
            self.cfg.mm_kon_min = min(self.cfg.mm_kon_min, 1e-12)
            self.cfg.mm_kon_max = max(self.cfg.mm_kon_max, 1e6)
            self.cfg.mm_koff_max = max(self.cfg.mm_koff_max, 1e5)
            self.cfg.mm_kcat_max = max(self.cfg.mm_kcat_max, 1e5)

    def _apply_turnover_friendly_bounds(self) -> None:
        """Broaden bounds for product-formation and initial-velocity-like traces."""
        self.cfg.fit_baseline_offset = True
        self.cfg.fit_baseline_slope = True
        self.cfg.baseline_offset_min = min(self.cfg.baseline_offset_min, -10.0)
        self.cfg.baseline_offset_max = max(self.cfg.baseline_offset_max, 10.0)
        self.cfg.baseline_slope_min = min(self.cfg.baseline_slope_min, -20.0)
        self.cfg.baseline_slope_max = max(self.cfg.baseline_slope_max, 20.0)
        self.cfg.linear_intercept_min = min(self.cfg.linear_intercept_min, -10.0)
        self.cfg.linear_intercept_max = max(self.cfg.linear_intercept_max, 10.0)
        self.cfg.linear_slope_min = min(self.cfg.linear_slope_min, -20.0)
        self.cfg.linear_slope_max = max(self.cfg.linear_slope_max, 20.0)
        self.cfg.no_binding_intercept_min = min(self.cfg.no_binding_intercept_min, -10.0)
        self.cfg.no_binding_intercept_max = max(self.cfg.no_binding_intercept_max, 10.0)
        self.cfg.response_max = max(self.cfg.response_max, 25.0)

    def validate_config(self) -> None:
        self.apply_model_scope_overrides()
        if not self.cfg.input_csv:
            raise ValueError("Please select an input CSV file.")
        if not os.path.isfile(self.cfg.input_csv):
            raise ValueError(f"Input CSV file does not exist:\n{self.cfg.input_csv}")
        if not self.cfg.output_parent:
            raise ValueError("Please select an output parent folder.")
        if not os.path.isdir(str(Path(self.cfg.output_parent).expanduser())):
            raise ValueError(f"Output parent folder does not exist:\n{self.cfg.output_parent}")
        if not self.cfg.output_folder_name.strip():
            raise ValueError("Please enter an output folder name.")
        if any(char in self.cfg.output_folder_name for char in '<>:"|?*') or '/' in self.cfg.output_folder_name or '\\' in self.cfg.output_folder_name:
            raise ValueError("Output folder name must be a single folder name and cannot contain path separators or invalid filename characters.")
        if self.cfg.E0_uM <= 0:
            raise ValueError("E0 must be greater than zero.")

        time_unit = str(getattr(self.cfg, "input_time_unit", "minutes") or "minutes").lower()
        if time_unit not in TIME_UNIT_TO_MIN:
            raise ValueError("Input time unit must be 'seconds' or 'minutes'.")
        self.cfg.input_time_unit = time_unit

        self.cfg.enzyme_concentration_unit = FittingConfig.normalize_concentration_unit(
            getattr(self.cfg, "enzyme_concentration_unit", "µM")
        )
        self.cfg.substrate_concentration_unit = FittingConfig.normalize_concentration_unit(
            getattr(self.cfg, "substrate_concentration_unit", "µM")
        )
        if self.cfg.enzyme_concentration_unit not in GUI_CONCENTRATION_UNITS:
            raise ValueError("Enzyme concentration unit must be nM, µM, mM, or M.")
        if self.cfg.substrate_concentration_unit not in GUI_CONCENTRATION_UNITS:
            raise ValueError("Substrate/reactant concentration unit must be nM, µM, mM, or M.")

        valid_analysis_modes = {"general_auto", "turnover_auto", "direct_only", "reversible_mm_only", "custom"}
        if str(getattr(self.cfg, "analysis_mode", "general_auto")) not in valid_analysis_modes:
            raise ValueError(f"Unknown analysis mode: {self.cfg.analysis_mode}")
        if not self.cfg.concentration_groups:
            raise ValueError("Please define at least one concentration.")
        for idx, group in enumerate(self.cfg.concentration_groups, start=1):
            if group.concentration_uM <= 0:
                raise ValueError(f"Concentration {idx} must be greater than zero.")
            if int(group.n_samples) <= 0:
                raise ValueError(f"Sample count for concentration {idx} must be greater than zero.")
        has_kinetic_candidates = bool(self.cfg.kinetic_fit_modes) and (
            self.cfg.max_intermediates >= 1 or self.cfg.include_zero_intermediate_kinetic_model
        )
        if (
            not has_kinetic_candidates
            and not self.cfg.include_reversible_mm_model
            and not self.cfg.include_linear_only_model
            and not self.cfg.include_no_binding_model
        ):
            raise ValueError("Enable at least one fitting model: direct 0-intermediate, reversible MM, standard, slow-apo, linear-only, or no-binding.")
        if self.cfg.selection_stride < 1:
            raise ValueError("Selection stride must be at least 1.")
        if self.cfg.max_intermediates < 0:
            raise ValueError("Max intermediates must be at least 0.")
        if self.cfg.n_starts_screen < 1 or self.cfg.n_starts_refine < 1:
            raise ValueError("The number of starts must be at least 1.")

    def get_k1_settings(self, fit_mode: str) -> Tuple[Tuple[float, float], float]:
        if fit_mode == "standard":
            return (
                (self.cfg.standard_k1_min, self.cfg.standard_k1_max),
                self.cfg.standard_k1_initial_guess,
            )
        if fit_mode == "slow_apo":
            return (
                (self.cfg.slow_apo_k1_min, self.cfg.slow_apo_k1_max),
                self.cfg.slow_apo_k1_initial_guess,
            )
        raise ValueError(f"Unknown kinetic fit mode: {fit_mode}")

    @staticmethod
    def concentration_label(D_value: float) -> str:
        D_float = float(D_value)
        if D_float.is_integer():
            D_text = str(int(D_float))
        else:
            D_text = str(D_float).replace(".", "p")
        return f"{D_text}uM"

    def kinetics_chain(self, t: float, y: np.ndarray, rates: np.ndarray, D_mM: float) -> np.ndarray:
        dydt = np.zeros_like(y)

        flux0 = rates[0] * y[0] * D_mM
        dydt[0] -= flux0
        dydt[1] += flux0

        for i in range(1, len(y) - 1):
            flux = rates[i] * y[i]
            dydt[i] -= flux
            dydt[i + 1] += flux

        return dydt

    def simulate_model(
        self,
        time_fit: np.ndarray,
        rates: np.ndarray,
        fitted_responses: np.ndarray,
        baseline_offset: float,
        baseline_slope: float,
        D_uM: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        D_mM = D_uM / 1000.0
        n_species = len(rates) + 1

        y0 = np.zeros(n_species)
        y0[0] = self.cfg.E0_mM

        if self.cfg.integrate_from_true_time_zero:
            t_span = [0.0, float(time_fit[-1])]
            t_eval = time_fit
            t_baseline = time_fit - time_fit[0]
        else:
            t_eval = time_fit - time_fit[0]
            t_span = [0.0, float(t_eval[-1])]
            t_baseline = t_eval

        sol = solve_ivp(
            self.kinetics_chain,
            t_span,
            y0,
            t_eval=t_eval,
            args=(rates, D_mM),
            method="LSODA",
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        species = sol.y
        species_frac = species / self.cfg.E0_mM

        if self.cfg.fix_free_enzyme_response:
            responses = np.concatenate([[self.cfg.free_enzyme_response], fitted_responses])
        else:
            responses = fitted_responses

        signal = np.sum(responses[:, None] * species_frac, axis=0)

        if self.cfg.fit_baseline_offset:
            signal += baseline_offset

        if self.cfg.fit_baseline_slope:
            signal += baseline_slope * t_baseline

        return signal, species

    @staticmethod
    def simulate_linear_only(time_fit: np.ndarray, intercept: float, slope: float) -> np.ndarray:
        t_local = time_fit - time_fit[0]
        return intercept + slope * t_local

    @staticmethod
    def simulate_no_binding(time_fit: np.ndarray, intercept: float) -> np.ndarray:
        return np.full_like(time_fit, float(intercept), dtype=float)

    def reversible_mm_ode(self, t: float, y: np.ndarray, kon: float, koff: float, kcat: float) -> np.ndarray:
        """ODEs for E + S <-> ES -> P.

        Species order is [E, S, ES, P], all in mM. Rate units are:
        kon: min^-1 mM^-1; koff: min^-1; kcat: min^-1.
        """
        E, S, ES, P = y
        association = kon * E * S
        dissociation = koff * ES
        turnover = kcat * ES
        return np.array([
            -association + dissociation + turnover,
            -association + dissociation,
            association - dissociation - turnover,
            turnover,
        ], dtype=float)

    def simulate_reversible_mm_model(
        self,
        time_fit: np.ndarray,
        params: np.ndarray,
        D_uM: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Simulate E + S <-> ES -> P and convert it to observed signal.

        The observed signal uses state fractions for E and ES and a
        substrate-normalized product fraction for P. This keeps product
        formation usable when P can exceed E0 during catalytic turnover.
        """
        kon, koff, kcat, fitted_responses, b0, b1 = self.unpack_reversible_mm_params(params)
        E0_mM = float(self.cfg.E0_mM)
        D0_mM = float(D_uM) / 1000.0

        if E0_mM <= 0 or D0_mM <= 0:
            raise ValueError("E0 and D0 must be positive for reversible Michaelis-Menten simulation.")

        y0 = np.array([E0_mM, D0_mM, 0.0, 0.0], dtype=float)

        if self.cfg.integrate_from_true_time_zero:
            t_span = [0.0, float(time_fit[-1])]
            t_eval = time_fit
            t_baseline = time_fit - time_fit[0]
        else:
            t_eval = time_fit - time_fit[0]
            t_span = [0.0, float(t_eval[-1])]
            t_baseline = t_eval

        sol = solve_ivp(
            self.reversible_mm_ode,
            t_span,
            y0,
            t_eval=t_eval,
            args=(kon, koff, kcat),
            method="LSODA",
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        species = sol.y
        E = species[0]
        ED = species[2]
        P = species[3]

        if self.cfg.fix_free_enzyme_response:
            response_E = float(self.cfg.free_enzyme_response)
            response_ED = float(fitted_responses[0]) if len(fitted_responses) >= 1 else 0.0
            response_P = float(fitted_responses[1]) if len(fitted_responses) >= 2 else 0.0
        else:
            response_E = float(fitted_responses[0]) if len(fitted_responses) >= 1 else 0.0
            response_ED = float(fitted_responses[1]) if len(fitted_responses) >= 2 else 0.0
            response_P = float(fitted_responses[2]) if len(fitted_responses) >= 3 else 0.0

        signal = (
            response_E * (E / E0_mM)
            + response_ED * (ED / E0_mM)
            + response_P * (P / D0_mM)
        )

        if self.cfg.fit_baseline_offset:
            signal += b0
        if self.cfg.fit_baseline_slope:
            signal += b1 * t_baseline

        Km_mM = (koff + kcat) / kon if kon > 0 else np.nan
        metrics = {
            "kon": float(kon),
            "koff": float(koff),
            "kcat": float(kcat),
            "Km_mM": float(Km_mM),
            "Km_uM": float(Km_mM * 1000.0) if np.isfinite(Km_mM) else np.nan,
            "Vmax_uM_per_min": float(kcat * self.cfg.E0_uM),
            "Vmax_uM_per_s": float(kcat * self.cfg.E0_uM / 60.0),
            "kcat_over_Km_mM_inv_min_inv": float(kcat / Km_mM) if np.isfinite(Km_mM) and Km_mM > 0 else np.nan,
            "kcat_over_Km_uM_inv_s_inv": float((kcat / 60.0) / (Km_mM * 1000.0)) if np.isfinite(Km_mM) and Km_mM > 0 else np.nan,
            "response_E": response_E,
            "response_ED": response_ED,
            "response_P": response_P,
            "baseline_offset": float(b0),
            "baseline_slope": float(b1),
        }
        return signal, species, metrics

    def unpack_reversible_mm_params(
        self,
        params: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray, float, float]:
        idx = 0
        kon = float(params[idx]); idx += 1
        koff = float(params[idx]); idx += 1
        kcat = float(params[idx]); idx += 1

        if self.cfg.fix_free_enzyme_response:
            n_response_params = 2  # ES and P
        else:
            n_response_params = 3  # E, ES, and P

        fitted_responses = np.asarray(params[idx:idx + n_response_params], dtype=float)
        idx += n_response_params

        if self.cfg.fit_baseline_offset:
            baseline_offset = float(params[idx])
            idx += 1
        else:
            baseline_offset = 0.0

        if self.cfg.fit_baseline_slope:
            baseline_slope = float(params[idx])
        else:
            baseline_slope = 0.0

        return kon, koff, kcat, fitted_responses, baseline_offset, baseline_slope

    def unpack_params(
        self,
        params: np.ndarray,
        n_intermediates: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        n_species = n_intermediates + 2
        n_rates = n_species - 1
        idx = 0

        rates = params[idx : idx + n_rates]
        idx += n_rates

        if self.cfg.fix_free_enzyme_response:
            n_response_params = n_species - 1
        else:
            n_response_params = n_species

        fitted_responses = params[idx : idx + n_response_params]
        idx += n_response_params

        if self.cfg.fit_baseline_offset:
            baseline_offset = float(params[idx])
            idx += 1
        else:
            baseline_offset = 0.0

        if self.cfg.fit_baseline_slope:
            baseline_slope = float(params[idx])
        else:
            baseline_slope = 0.0

        return rates, fitted_responses, baseline_offset, baseline_slope

    @staticmethod
    def unpack_linear_params(params: np.ndarray) -> Tuple[float, float]:
        return float(params[0]), float(params[1])

    @staticmethod
    def unpack_no_binding_params(params: np.ndarray) -> float:
        return float(params[0])

    def normalization_scale(self, y: np.ndarray, time: np.ndarray) -> float:
        """Return the raw-value scale used to normalize one trace.

        The scale is the maximum raw response observed between fit_start_time
        and early_max_window_min. Retaining this value lets ANCHOR convert
        normalized fitted slopes back into the original CSV response units.
        For SPpNA substrate-turnover datasets whose trace columns are product
        concentration in micromolar, this back-conversion yields micromolar per
        minute and micromolar per second values comparable to spreadsheet
        initial-velocity calculations.
        """
        mask = (time >= self.cfg.fit_start_time) & (time <= self.cfg.early_max_window_min)

        if not np.any(mask):
            raise ValueError("No points found in early normalization window.")

        scale = float(np.nanmax(y[mask]))

        if scale == 0 or np.isnan(scale):
            raise ValueError("Invalid early maximum; cannot normalize.")

        return scale

    def normalize_trace(self, y: np.ndarray, time: np.ndarray) -> np.ndarray:
        scale = self.normalization_scale(y, time)
        return y / scale

    @staticmethod
    def least_squares_slope(time_values: np.ndarray, y_values: np.ndarray) -> float:
        """Return the ordinary least-squares slope of y versus time.

        Time is centered at the first fitted point so the returned slope has
        units of y-units per minute. CSV time values may be supplied in seconds
        or minutes; ANCHOR converts them to minutes before this routine is called.
        """
        t = np.asarray(time_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]

        if t.size < 2:
            return np.nan

        t_local = t - t[0]
        denom = float(np.sum((t_local - np.mean(t_local)) ** 2))
        if denom <= 0:
            return np.nan

        return float(np.polyfit(t_local, y, 1)[0])

    def direct_zero_intermediate_velocity_metrics(
        self,
        time_fit: np.ndarray,
        sim_norm: np.ndarray,
        raw_trace_fit: np.ndarray,
        norm_trace_fit: np.ndarray,
        normalization_factor: float,
        rates: np.ndarray,
        fitted_responses: np.ndarray,
        baseline_slope: float,
        D_uM: float,
    ) -> Dict[str, Any]:
        """Calculate spreadsheet-comparable velocities for E + S -> P fits.

        The direct 0-intermediate model reports k1 and k_obs as rate constants
        for a normalized response model. For substrate-turnover data, users
        often need a value comparable to a spreadsheet initial velocity, i.e.,
        the slope of product concentration versus time. This routine reports
        two back-calculated slopes from the fitted direct-model curve:

        1. an instantaneous initial slope from the analytical derivative of
           the fitted E + S -> P observation model; and
        2. an ordinary least-squares slope of the fitted curve over the fitted
           time window, which is the closest direct comparison to a spreadsheet
           linear initial-velocity calculation over the same points.

        The normalized slopes are multiplied by the normalization factor to
        convert back into the original CSV response units. If the CSV trace
        columns are in micromolar product, the converted values are micromolar
        per minute and micromolar per second.
        """
        D_mM = float(D_uM) / 1000.0
        k1 = float(rates[0]) if len(rates) else np.nan
        k_obs = k1 * D_mM if np.isfinite(k1) else np.nan

        if self.cfg.fix_free_enzyme_response:
            response_E = float(self.cfg.free_enzyme_response)
            response_P = float(fitted_responses[0]) if len(fitted_responses) >= 1 else np.nan
        else:
            response_E = float(fitted_responses[0]) if len(fitted_responses) >= 1 else np.nan
            response_P = float(fitted_responses[1]) if len(fitted_responses) >= 2 else np.nan

        if self.cfg.integrate_from_true_time_zero:
            t_for_derivative = float(time_fit[0])
        else:
            t_for_derivative = 0.0

        if np.isfinite(k_obs) and np.isfinite(response_E) and np.isfinite(response_P):
            # For E + S -> P:
            #   E/E0 = exp(-k_obs*t)
            #   P/E0 = 1 - exp(-k_obs*t)
            #   y = R_E E/E0 + R_P P/E0 + b0 + b1*(t - t0)
            # so:
            #   dy/dt = (R_P - R_E)*k_obs*exp(-k_obs*t) + b1
            direct_initial_slope_norm = (
                (response_P - response_E)
                * k_obs
                * np.exp(-k_obs * t_for_derivative)
                + float(baseline_slope)
            )
        else:
            direct_initial_slope_norm = np.nan

        fit_linear_slope_norm = self.least_squares_slope(time_fit, sim_norm)
        raw_data_slope_raw = self.least_squares_slope(time_fit, raw_trace_fit)
        raw_data_slope_norm = self.least_squares_slope(time_fit, norm_trace_fit)

        def to_raw_per_min(norm_slope: float) -> float:
            if not np.isfinite(norm_slope) or not np.isfinite(normalization_factor):
                return np.nan
            return float(norm_slope * normalization_factor)

        direct_initial_velocity_raw_per_min = to_raw_per_min(direct_initial_slope_norm)
        direct_fit_velocity_raw_per_min = to_raw_per_min(fit_linear_slope_norm)

        return {
            "direct_model": "E + S -> P",
            "S_mM": D_mM,
            "k_obs_direct_min_inv": k_obs,
            "normalization_factor_raw_units": normalization_factor,
            "direct_initial_slope_norm_per_min": direct_initial_slope_norm,
            "direct_initial_velocity_raw_units_per_min": direct_initial_velocity_raw_per_min,
            "direct_initial_velocity_raw_units_per_s": direct_initial_velocity_raw_per_min / 60.0 if np.isfinite(direct_initial_velocity_raw_per_min) else np.nan,
            "direct_fit_linear_slope_norm_per_min": fit_linear_slope_norm,
            "direct_fit_linear_velocity_raw_units_per_min": direct_fit_velocity_raw_per_min,
            "direct_fit_linear_velocity_raw_units_per_s": direct_fit_velocity_raw_per_min / 60.0 if np.isfinite(direct_fit_velocity_raw_per_min) else np.nan,
            "raw_data_linear_slope_norm_per_min": raw_data_slope_norm,
            "raw_data_linear_slope_raw_units_per_min": raw_data_slope_raw,
            "raw_data_linear_slope_raw_units_per_s": raw_data_slope_raw / 60.0 if np.isfinite(raw_data_slope_raw) else np.nan,
            "velocity_units_note": "raw_units are the units of the input CSV trace columns; if those columns are uM product, velocities are uM/min and uM/s",
            "velocity_backcalculation_note": "direct_fit_linear_velocity is the fitted-curve OLS slope over the fitted window and is the preferred comparison to spreadsheet Vi; direct_initial_velocity is the analytical derivative at the first fitted time point",
        }

    def write_turnover_model_detection_summary(self, best_df: pd.DataFrame) -> None:
        """Write a compact summary of selected turnover-related model classes."""
        if not (
            getattr(self.cfg, "auto_detect_turnover_mechanism", False)
            or getattr(self.cfg, "include_reversible_mm_model", False)
            or getattr(self.cfg, "fit_zero_intermediate_only", False)
            or getattr(self.cfg, "fit_reversible_mm_only", False)
        ):
            return

        rows: List[Dict[str, Any]] = []
        for _, row in best_df.iterrows():
            model_type = str(row.get("model_type", ""))
            n_int = pd.to_numeric(row.get("best_n_intermediates", np.nan), errors="coerce")
            if model_type == "reversible_mm":
                selected_class = "reversible_MM_E_plus_S_reversible_ES_to_P"
            elif model_type == "kinetic" and np.isfinite(n_int) and int(n_int) == 0:
                selected_class = "direct_0_intermediate_E_plus_S_to_P"
            elif model_type == "kinetic":
                selected_class = f"sequential_kinetic_{int(n_int) if np.isfinite(n_int) else 'unknown'}_intermediates"
            else:
                selected_class = model_type
            rows.append({
                "Run": row.get("Run", np.nan),
                "S_uM": row.get("D_uM", np.nan),
                "selected_class": selected_class,
                "RSS": row.get("RSS", np.nan),
                "AIC": row.get("AIC", np.nan),
                "BIC": row.get("BIC", np.nan),
            })

        detail = pd.DataFrame(rows)
        detail.to_csv(os.path.join(self.cfg.output_dir, "turnover_model_detection_details.csv"), index=False)
        if not detail.empty:
            summary = (
                detail.groupby("selected_class", as_index=False)
                .agg(n_traces=("Run", "size"), mean_BIC=("BIC", "mean"), median_BIC=("BIC", "median"))
                .sort_values("n_traces", ascending=False)
            )
            summary.to_csv(os.path.join(self.cfg.output_dir, "turnover_model_detection_summary.csv"), index=False)
            self.log("Saved: turnover_model_detection_details.csv")
            self.log("Saved: turnover_model_detection_summary.csv")

    @staticmethod
    def michaelis_menten_model(S_uM: np.ndarray, Vmax: float, Km_uM: float) -> np.ndarray:
        """Michaelis-Menten equation using substrate concentration in micromolar."""
        S = np.asarray(S_uM, dtype=float)
        return (Vmax * S) / (Km_uM + S)

    def fit_michaelis_menten_outputs(self, best_df: pd.DataFrame) -> None:
        """Fit Michaelis-Menten parameters from selected turnover velocities.

        The post-fit analysis collects velocity estimates from the turnover
        mechanism selected for each trace. Direct 0-intermediate runs contribute
        direct_fit_linear_velocity_raw_units_per_s, whereas reversible-MM runs
        contribute mm_fit_linear_velocity_raw_units_per_s. A Michaelis-Menten
        plot is written even when the nonlinear fit is skipped, so users can
        see whether the issue was insufficient concentrations, absent turnover
        velocities, or optimizer failure.
        """
        if not self.cfg.fit_michaelis_menten:
            self.log("Michaelis-Menten post-fit analysis disabled by user.")
            return

        points_path = os.path.join(self.cfg.output_dir, "michaelis_menten_velocity_points.csv")
        summary_path = os.path.join(self.cfg.output_dir, "michaelis_menten_fit_summary.csv")
        grouped_path = os.path.join(self.cfg.output_dir, "michaelis_menten_grouped_velocity_summary.csv")
        plot_path = os.path.join(self.cfg.output_dir, "michaelis_menten_fit.png")

        def save_mm_plot(
            mm_points: Optional[pd.DataFrame] = None,
            grouped: Optional[pd.DataFrame] = None,
            Vmax: Optional[float] = None,
            Km_uM: Optional[float] = None,
            note: str = "",
        ) -> None:
            """Write a clean Michaelis-Menten plot or diagnostic placeholder."""
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            has_points = mm_points is not None and not mm_points.empty

            if has_points:
                legend_added = False

                if "selected_turnover_model" in mm_points.columns:
                    model_groups = list(mm_points.groupby("selected_turnover_model"))
                    if len(model_groups) == 1:
                        _, group_df = model_groups[0]
                        ax.scatter(
                            group_df["D_uM"],
                            group_df["velocity_raw_units_per_s"],
                            s=28,
                            alpha=0.35,
                            label="Individual trace velocities",
                        )
                        legend_added = True
                    else:
                        for model_label, group_df in model_groups:
                            if group_df.empty:
                                continue
                            label = (
                                "Direct E + S -> P"
                                if "direct" in str(model_label).lower()
                                else "Reversible E + S <-> ES -> P"
                            )
                            ax.scatter(
                                group_df["D_uM"],
                                group_df["velocity_raw_units_per_s"],
                                s=28,
                                alpha=0.35,
                                label=label,
                            )
                            legend_added = True
                else:
                    ax.scatter(
                        mm_points["D_uM"],
                        mm_points["velocity_raw_units_per_s"],
                        s=28,
                        alpha=0.35,
                        label="Individual trace velocities",
                    )
                    legend_added = True

                if grouped is not None and not grouped.empty:
                    yerr = grouped.get("velocity_sd_raw_units_per_s", pd.Series(dtype=float)).to_numpy(dtype=float)
                    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
                    ax.errorbar(
                        grouped["D_uM"],
                        grouped["velocity_mean_raw_units_per_s"],
                        yerr=yerr,
                        fmt="o",
                        markersize=8,
                        capsize=3,
                        elinewidth=1.2,
                        label="Concentration means",
                    )
                    legend_added = True

                if Vmax is not None and Km_uM is not None and np.isfinite(Vmax) and np.isfinite(Km_uM) and Km_uM > 0:
                    try:
                        x_max = max(float(np.nanmax(mm_points["D_uM"])), 1.0)
                    except Exception:
                        x_max = 1.0
                    x_curve = np.linspace(0, x_max * 1.05, 300)
                    y_curve = self.michaelis_menten_model(x_curve, float(Vmax), float(Km_uM))
                    ax.plot(x_curve, y_curve, "-", linewidth=2.8, label="Michaelis-Menten fit")
                    legend_added = True
                elif note:
                    ax.text(
                        0.03,
                        0.97,
                        note,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
                    )

                ax.set_xlabel("Substrate/reactant concentration (µM)")
                ax.set_ylabel("Velocity (raw units/s)")
                ax.set_title("Michaelis-Menten fit", pad=10)
                ax.grid(alpha=0.2, linewidth=0.6)
                if legend_added:
                    handles, labels = ax.get_legend_handles_labels()
                    seen = set()
                    filtered = []
                    for h, l in zip(handles, labels):
                        if l and l not in seen:
                            seen.add(l)
                            filtered.append((h, l))
                    if filtered:
                        ax.legend([h for h, _ in filtered], [l for _, l in filtered], frameon=True, loc="best")
                fig.tight_layout()
            else:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.58,
                    "Michaelis-Menten plot not fit",
                    ha="center",
                    va="center",
                    fontsize=16,
                    weight="bold",
                )
                ax.text(
                    0.5,
                    0.44,
                    note or "No selected turnover velocity points were available.",
                    ha="center",
                    va="center",
                    fontsize=11,
                    wrap=True,
                )
                fig.tight_layout()

            fig.savefig(plot_path, dpi=300)
            plt.close(fig)

        required_base = {"Run", "D_uM", "model_type"}
        missing_base = sorted(required_base - set(best_df.columns))
        if missing_base:
            note = "Required columns were missing: " + ", ".join(missing_base)
            pd.DataFrame([{"status": "skipped_missing_required_columns", "note": note}]).to_csv(summary_path, index=False)
            save_mm_plot(note=note)
            self.log("Michaelis-Menten fit skipped: " + note)
            self.log("Saved: michaelis_menten_fit.png (diagnostic placeholder)")
            return

        point_frames: List[pd.DataFrame] = []

        direct_velocity_col = (
            "direct_fit_linear_velocity_raw_units_per_s"
            if self.cfg.mm_use_fitted_direct_velocity
            else "raw_data_linear_slope_raw_units_per_s"
        )
        if direct_velocity_col in best_df.columns and "best_n_intermediates" in best_df.columns:
            direct_mask = (
                (best_df["model_type"].astype(str) == "kinetic")
                & (pd.to_numeric(best_df["best_n_intermediates"], errors="coerce") == 0)
            )
            direct_points = best_df.loc[direct_mask, ["Run", "D_uM", direct_velocity_col]].copy()
            if not direct_points.empty:
                direct_points = direct_points.rename(columns={direct_velocity_col: "velocity_raw_units_per_s"})
                direct_points["velocity_source_column"] = direct_velocity_col
                direct_points["selected_turnover_model"] = "direct_0_intermediate_E_plus_S_to_P"
                point_frames.append(direct_points)

        reversible_velocity_col = "mm_fit_linear_velocity_raw_units_per_s"
        if reversible_velocity_col in best_df.columns:
            reversible_mask = best_df["model_type"].astype(str) == "reversible_mm"
            reversible_points = best_df.loc[reversible_mask, ["Run", "D_uM", reversible_velocity_col]].copy()
            if not reversible_points.empty:
                reversible_points = reversible_points.rename(columns={reversible_velocity_col: "velocity_raw_units_per_s"})
                reversible_points["velocity_source_column"] = reversible_velocity_col
                reversible_points["selected_turnover_model"] = "reversible_MM_E_plus_S_reversible_ES_to_P"
                point_frames.append(reversible_points)

        if not point_frames:
            note = "No selected direct or reversible turnover velocities were available for Michaelis-Menten fitting."
            pd.DataFrame([{"status": "skipped_no_turnover_velocity_columns", "note": note}]).to_csv(summary_path, index=False)
            save_mm_plot(note=note)
            self.log("Michaelis-Menten fit skipped: no selected turnover velocities were available.")
            self.log("Saved: michaelis_menten_fit.png (diagnostic placeholder)")
            return

        mm_points = pd.concat(point_frames, ignore_index=True, sort=False)
        mm_points["D_uM"] = pd.to_numeric(mm_points["D_uM"], errors="coerce")
        mm_points["D_mM"] = mm_points["D_uM"] / 1000.0
        mm_points["velocity_raw_units_per_s"] = pd.to_numeric(mm_points["velocity_raw_units_per_s"], errors="coerce")
        mm_points = mm_points[np.isfinite(mm_points["D_uM"]) & np.isfinite(mm_points["velocity_raw_units_per_s"])].copy()
        mm_points = mm_points[(mm_points["D_uM"] > 0) & (mm_points["velocity_raw_units_per_s"] > 0)]
        mm_points_output = mm_points.rename(columns={"D_uM": "S_uM", "D_mM": "S_mM"})
        mm_points_output.to_csv(points_path, index=False)

        if mm_points.empty:
            note = "No positive finite selected turnover velocities were available for Michaelis-Menten fitting."
            pd.DataFrame([{"status": "skipped_no_positive_turnover_velocities", "note": note}]).to_csv(summary_path, index=False)
            save_mm_plot(note=note)
            self.log("Michaelis-Menten fit skipped: no positive finite selected turnover velocities were available.")
            self.log("Saved: michaelis_menten_fit.png (diagnostic placeholder)")
            return

        grouped = (
            mm_points.groupby("D_uM", as_index=False)
            .agg(
                D_mM=("D_mM", "first"),
                n_traces=("velocity_raw_units_per_s", "size"),
                velocity_mean_raw_units_per_s=("velocity_raw_units_per_s", "mean"),
                velocity_sd_raw_units_per_s=("velocity_raw_units_per_s", "std"),
                selected_models=("selected_turnover_model", lambda x: ";".join(sorted(set(map(str, x))))),
            )
            .sort_values("D_uM")
        )
        grouped["velocity_sem_raw_units_per_s"] = grouped["velocity_sd_raw_units_per_s"] / np.sqrt(grouped["n_traces"])
        grouped_output = grouped.rename(columns={"D_uM": "S_uM", "D_mM": "S_mM"})
        grouped_output.to_csv(grouped_path, index=False)

        if self.cfg.mm_fit_group_means:
            fit_x = grouped["D_uM"].to_numpy(dtype=float)
            fit_y = grouped["velocity_mean_raw_units_per_s"].to_numpy(dtype=float)
            fit_basis = "group_mean_selected_turnover_velocities"
        else:
            fit_x = mm_points["D_uM"].to_numpy(dtype=float)
            fit_y = mm_points["velocity_raw_units_per_s"].to_numpy(dtype=float)
            fit_basis = "individual_selected_turnover_velocities"

        valid = np.isfinite(fit_x) & np.isfinite(fit_y) & (fit_x > 0) & (fit_y > 0)
        fit_x = fit_x[valid]
        fit_y = fit_y[valid]

        if fit_x.size < int(self.cfg.mm_min_points) or np.unique(fit_x).size < 2:
            note = "Insufficient positive velocities at multiple concentrations for Michaelis-Menten fitting."
            pd.DataFrame([
                {
                    "status": "skipped_insufficient_points",
                    "fit_basis": fit_basis,
                    "n_fit_points": int(fit_x.size),
                    "n_unique_concentrations": int(np.unique(fit_x).size),
                    "minimum_points_required": int(self.cfg.mm_min_points),
                    "note": note,
                }
            ]).to_csv(summary_path, index=False)
            save_mm_plot(mm_points=mm_points, grouped=grouped, note=note)
            self.log("Michaelis-Menten fit skipped: insufficient positive selected turnover velocity points.")
            self.log("Saved: michaelis_menten_fit.png (diagnostic velocity plot)")
            return

        vmax0 = float(np.nanmax(fit_y) * 1.2)
        if not np.isfinite(vmax0) or vmax0 <= 0:
            vmax0 = 1.0
        km0 = float(np.nanmedian(fit_x))
        if not np.isfinite(km0) or km0 <= 0:
            km0 = 1.0

        start_guesses = [
            np.array([vmax0, km0], dtype=float),
            np.array([max(float(np.nanmax(fit_y)), 1e-12), max(float(np.nanmin(fit_x)), 1e-12)], dtype=float),
            np.array([max(float(np.nanmax(fit_y)) * 2.0, 1e-12), max(float(np.nanmedian(fit_x)), 1e-12)], dtype=float),
            np.array([max(float(np.nanmax(fit_y)) * 1.5, 1e-12), max(float(np.nanmax(fit_x)), 1e-12)], dtype=float),
        ]

        def residual_mm(params: np.ndarray) -> np.ndarray:
            Vmax, Km_uM = params
            return self.michaelis_menten_model(fit_x, Vmax, Km_uM) - fit_y

        best_result = None
        for guess in start_guesses:
            try:
                result = least_squares(
                    residual_mm,
                    guess,
                    bounds=([0.0, 1e-12], [np.inf, np.inf]),
                    loss="soft_l1",
                    max_nfev=10000,
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                )
            except Exception:
                continue
            if best_result is None or float(np.sum(residual_mm(result.x) ** 2)) < float(np.sum(residual_mm(best_result.x) ** 2)):
                best_result = result

        if best_result is None:
            note = "All Michaelis-Menten optimizer starts failed."
            pd.DataFrame([{"status": "failed_optimizer_error", "fit_basis": fit_basis, "note": note}]).to_csv(summary_path, index=False)
            save_mm_plot(mm_points=mm_points, grouped=grouped, note=note)
            self.log("Michaelis-Menten fit failed: optimizer error.")
            self.log("Saved: michaelis_menten_fit.png (diagnostic velocity plot)")
            return

        Vmax, Km_uM = [float(x) for x in best_result.x]
        pred_y = self.michaelis_menten_model(fit_x, Vmax, Km_uM)
        residuals = fit_y - pred_y
        rss = float(np.sum(residuals ** 2))
        tss = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
        r_squared = float(1.0 - rss / tss) if tss > 0 else np.nan
        n_fit = int(fit_y.size)
        p_fit = 2
        rss_safe = max(rss, 1e-300)
        aic = float(n_fit * np.log(rss_safe / n_fit) + 2 * p_fit)
        bic = float(n_fit * np.log(rss_safe / n_fit) + p_fit * np.log(n_fit))

        Vmax_se = np.nan
        Km_se = np.nan
        if n_fit > p_fit:
            try:
                jac = np.asarray(best_result.jac, dtype=float)
                cov = np.linalg.pinv(jac.T @ jac) * (rss / (n_fit - p_fit))
                se = np.sqrt(np.diag(cov))
                Vmax_se = float(se[0])
                Km_se = float(se[1])
            except Exception:
                pass

        kcat_per_s = Vmax / float(self.cfg.E0_uM) if self.cfg.E0_uM > 0 else np.nan
        kcat_se_per_s = Vmax_se / float(self.cfg.E0_uM) if np.isfinite(Vmax_se) and self.cfg.E0_uM > 0 else np.nan
        source_counts = mm_points["selected_turnover_model"].value_counts().to_dict()

        summary = {
            "status": "fit_successful",
            "velocity_source": "selected_turnover_models",
            "selected_model_counts": json.dumps(source_counts),
            "fit_basis": fit_basis,
            "n_fit_points": n_fit,
            "n_unique_concentrations": int(np.unique(fit_x).size),
            "E0_uM": float(self.cfg.E0_uM),
            "E0_input_value": self.cfg.concentration_from_uM(
                self.cfg.E0_uM, self.cfg.enzyme_concentration_unit
            ),
            "E0_input_unit": self.cfg.enzyme_concentration_unit,
            "substrate_input_unit": self.cfg.substrate_concentration_unit,
            "input_time_unit": self.cfg.input_time_unit,
            "internal_time_unit": "minutes",
            "Vmax_raw_units_per_s": Vmax,
            "Vmax_se_raw_units_per_s": Vmax_se,
            "Vmax_raw_units_per_min": Vmax * 60.0,
            "Vmax_se_raw_units_per_min": Vmax_se * 60.0 if np.isfinite(Vmax_se) else np.nan,
            "Km_uM": Km_uM,
            "Km_se_uM": Km_se,
            "Km_mM": Km_uM / 1000.0,
            "Km_se_mM": Km_se / 1000.0 if np.isfinite(Km_se) else np.nan,
            "kcat_app_s_inv": kcat_per_s,
            "kcat_app_se_s_inv": kcat_se_per_s,
            "kcat_app_min_inv": kcat_per_s * 60.0 if np.isfinite(kcat_per_s) else np.nan,
            "kcat_over_Km_app_uM_inv_s_inv": kcat_per_s / Km_uM if Km_uM > 0 and np.isfinite(kcat_per_s) else np.nan,
            "kcat_over_Km_app_mM_inv_s_inv": kcat_per_s / (Km_uM / 1000.0) if Km_uM > 0 and np.isfinite(kcat_per_s) else np.nan,
            "RSS": rss,
            "AIC": aic,
            "BIC": bic,
            "R_squared": r_squared,
            "equation": "v = Vmax*[S]/(Km + [S])",
            "concentration_units": "uM for Km; mM also reported",
            "velocity_units_note": "Vmax is in raw input trace units per second; if raw traces are uM product, Vmax is uM/s.",
            "kcat_units_note": "kcat_app_s_inv = Vmax_raw_units_per_s / E0_uM; this is s^-1 when raw trace units are uM product.",
        }
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        save_mm_plot(mm_points=mm_points, grouped=grouped, Vmax=Vmax, Km_uM=Km_uM)

        self.log("Saved: michaelis_menten_velocity_points.csv")
        self.log("Saved: michaelis_menten_grouped_velocity_summary.csv")
        self.log("Saved: michaelis_menten_fit_summary.csv")
        self.log("Saved: michaelis_menten_fit.png")
        self.log(f"Michaelis-Menten fit: Vmax={Vmax:.6g} raw units/s, Km={Km_uM:.6g} µM, kcat_app={kcat_per_s:.6g} s^-1")

    def write_key_results_summary(self, best_df: pd.DataFrame) -> None:
        """Write a compact, human-readable selected-model summary."""
        def selected_model_label(row: pd.Series) -> str:
            model_type = str(row.get("model_type", ""))
            n_int = pd.to_numeric(row.get("best_n_intermediates", np.nan), errors="coerce")
            if model_type == "reversible_mm":
                return "Reversible MM: E + S <-> ES -> P"
            if model_type == "kinetic" and np.isfinite(n_int) and int(n_int) == 0:
                return "Direct turnover: E + S -> P"
            if model_type == "kinetic" and np.isfinite(n_int):
                return f"Sequential kinetic: {int(n_int)} intermediate(s)"
            if model_type == "linear_only":
                return "Linear-only"
            if model_type == "no_binding":
                return "No-binding"
            return model_type or "unknown"

        def get_first(row: pd.Series, names: List[str]) -> Any:
            for name in names:
                if name in row.index:
                    val = row.get(name, np.nan)
                    try:
                        if pd.notna(val):
                            return val
                    except Exception:
                        if val is not None:
                            return val
            return np.nan

        rows: List[Dict[str, Any]] = []
        for _, row in best_df.iterrows():
            selected_model = selected_model_label(row)
            rows.append(
                {
                    "Run": row.get("Run", np.nan),
                    "S_uM": row.get("D_uM", np.nan),
                    "selected_model": selected_model,
                    "model_type": row.get("model_type", ""),
                    "fit_mode": row.get("fit_mode", ""),
                    "best_n_intermediates": row.get("best_n_intermediates", np.nan),
                    "RSS": row.get("RSS", np.nan),
                    "AIC": row.get("AIC", np.nan),
                    "BIC": row.get("BIC", np.nan),
                    "selected_velocity_raw_units_per_s": get_first(
                        row,
                        [
                            "direct_fit_linear_velocity_raw_units_per_s",
                            "mm_fit_linear_velocity_raw_units_per_s",
                            "raw_data_linear_slope_raw_units_per_s",
                        ],
                    ),
                    "k1_min_inv_mM_inv": row.get("k1", np.nan),
                    "k_obs_direct_min_inv": row.get("k_obs_direct_min_inv", np.nan),
                    "kon_min_inv_mM_inv": row.get("kon", np.nan),
                    "koff_min_inv": row.get("koff", np.nan),
                    "kcat_min_inv": row.get("kcat", np.nan),
                    "Km_uM": row.get("Km_uM", np.nan),
                    "kcat_app_s_inv": row.get("kcat_app_s_inv", np.nan),
                    "linear_slope": row.get("linear_slope", np.nan),
                    "no_binding_intercept": row.get("no_binding_intercept", np.nan),
                    "baseline_offset": row.get("baseline_offset", np.nan),
                    "baseline_slope": row.get("baseline_slope", np.nan),
                }
            )

        key_df = pd.DataFrame(rows)
        key_df.to_csv(os.path.join(self.cfg.output_dir, "ANCHOR_key_results.csv"), index=False)

        if not key_df.empty:
            counts = (
                key_df.groupby("selected_model", as_index=False)
                .agg(n_traces=("Run", "size"))
                .sort_values("n_traces", ascending=False)
            )
            counts.to_csv(os.path.join(self.cfg.output_dir, "ANCHOR_model_counts.csv"), index=False)
        self.log("Saved: ANCHOR_key_results.csv")
        self.log("Saved: ANCHOR_model_counts.csv")

    def write_output_guide(self) -> None:
        """Write a short guide to the cleaned output folder."""
        guide = f"""ANCHOR output guide
===================

Start here
----------
1. ANCHOR_key_results.csv
   Compact selected-model summary. Use this first for interpretation.

2. best_fit_summary.csv
   Complete selected-model parameters for every trace.

3. michaelis_menten_fit.png
   Michaelis-Menten plot from selected turnover velocities. If the nonlinear
   Michaelis-Menten fit could not be performed, this file is still written as
   a diagnostic plot or placeholder explaining why.

4. michaelis_menten_fit_summary.csv
   Vmax, Km, apparent kcat, kcat/Km, RSS, AIC, BIC, and R-squared when enough
   turnover velocity points are available.

Folders when output organization is enabled
-------------------------------------------
figures/overlays
   all_runs_best_fits*.png concentration-level and all-run overlays.

figures/per_trace_fits
   best_fit_run_*.png individual trace overlays.

figures/species
   species_run_*.png species or apparent-species plots.

figures/michaelis_menten
   Copy of michaelis_menten_fit.png.

tables/detailed
   Candidate-level model-selection tables, velocity-point tables, grouped
   velocity tables, and turnover detection summaries.

Configuration
-------------
run_configuration.json records all GUI and advanced settings used for this run.

Analysis mode used: {getattr(self.cfg, 'analysis_mode', 'unknown')}
"""
        with open(os.path.join(self.cfg.output_dir, "ANCHOR_output_guide.txt"), "w", encoding="utf-8") as handle:
            handle.write(guide)
        self.log("Saved: ANCHOR_output_guide.txt")

    def write_summary_workbook(self) -> None:
        """Write a consolidated Excel workbook when openpyxl is available."""
        output_path = os.path.join(self.cfg.output_dir, "ANCHOR_summary.xlsx")
        sheet_files = [
            ("Key Results", "ANCHOR_key_results.csv"),
            ("Model Counts", "ANCHOR_model_counts.csv"),
            ("Best Fit Full", "best_fit_summary.csv"),
            ("MM Fit", "michaelis_menten_fit_summary.csv"),
            ("MM Velocity Points", "michaelis_menten_velocity_points.csv"),
            ("MM Group Means", "michaelis_menten_grouped_velocity_summary.csv"),
            ("Turnover Summary", "turnover_model_detection_summary.csv"),
            ("Model Selection", "model_selection_summary.csv"),
        ]
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                pd.DataFrame(
                    [
                        {"Item": "Analysis mode", "Value": getattr(self.cfg, "analysis_mode", "unknown")},
                        {"Item": "Input time unit", "Value": self.cfg.input_time_unit},
                        {"Item": "Internal time unit", "Value": "minutes"},
                        {"Item": "E0_uM", "Value": self.cfg.E0_uM},
                        {"Item": "E0 input unit", "Value": self.cfg.enzyme_concentration_unit},
                        {"Item": "Substrate/reactant input unit", "Value": self.cfg.substrate_concentration_unit},
                        {"Item": "Notes", "Value": "Use Key Results first. Detailed candidate fits are in Model Selection."},
                    ]
                ).to_excel(writer, sheet_name="README", index=False)
                for sheet_name, filename in sheet_files:
                    path = os.path.join(self.cfg.output_dir, filename)
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            self.log("Saved: ANCHOR_summary.xlsx")
        except Exception as exc:
            self.log(f"Summary workbook skipped: {exc}")

    def organize_output_files(self) -> None:
        """Move verbose output files into predictable subfolders."""
        outdir = Path(self.cfg.output_dir)
        if not outdir.exists():
            return

        folders = {
            "trace_fits": outdir / "figures" / "per_trace_fits",
            "species": outdir / "figures" / "species",
            "overlays": outdir / "figures" / "overlays",
            "mm_figures": outdir / "figures" / "michaelis_menten",
            "detailed_tables": outdir / "tables" / "detailed",
        }
        for folder in folders.values():
            folder.mkdir(parents=True, exist_ok=True)

        def move_matching(pattern: str, destination: Path, keep_top_level: bool = False) -> None:
            for path in list(outdir.glob(pattern)):
                if not path.is_file():
                    continue
                target = destination / path.name
                if keep_top_level:
                    shutil.copy2(path, target)
                else:
                    if target.exists():
                        target.unlink()
                    shutil.move(str(path), str(target))

        move_matching("best_fit_run_*.png", folders["trace_fits"])
        move_matching("species_run_*.png", folders["species"])
        move_matching("all_runs_best_fits*.png", folders["overlays"])
        move_matching("michaelis_menten_fit.png", folders["mm_figures"], keep_top_level=True)

        for filename in [
            "model_selection_summary.csv",
            "michaelis_menten_velocity_points.csv",
            "michaelis_menten_grouped_velocity_summary.csv",
            "turnover_model_detection_details.csv",
            "turnover_model_detection_summary.csv",
        ]:
            path = outdir / filename
            if path.exists():
                target = folders["detailed_tables"] / filename
                if target.exists():
                    target.unlink()
                shutil.move(str(path), str(target))

        self.log("Organized outputs into figures/ and tables/detailed/ folders.")

    def smooth_trace(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        window = max(11, int(self.cfg.smooth_window_fraction * n))

        if window % 2 == 0:
            window += 1

        if window >= n:
            window = n - 1 if (n - 1) % 2 == 1 else n - 2

        if window < 5:
            return y.copy()

        polyorder = min(self.cfg.smooth_polyorder, window - 2)
        return savgol_filter(y, window_length=window, polyorder=polyorder)

    @staticmethod
    def estimate_trace_slope(time_fit: np.ndarray, y_fit: np.ndarray) -> float:
        duration = time_fit[-1] - time_fit[0]
        if duration <= 0:
            return 0.0
        return float((y_fit[-1] - y_fit[0]) / duration)

    def make_initial_guess(
        self,
        n_intermediates: int,
        y_fit: np.ndarray,
        time_fit: np.ndarray,
        fit_mode: str,
    ) -> np.ndarray:
        n_species = n_intermediates + 2
        n_rates = n_species - 1
        _, k1_guess = self.get_k1_settings(fit_mode)

        rate_guesses = [k1_guess] + [self.cfg.k_other_initial_guess] * (n_rates - 1)

        response_template = [0.33, 0.80, 0.60, 0.60, 0.60, 0.60]

        if self.cfg.fix_free_enzyme_response:
            n_response_params = n_species - 1
            response_guesses = [
                response_template[min(i, len(response_template) - 1)]
                for i in range(n_response_params)
            ]
        else:
            response_template_with_E = [
                max(min(float(y_fit[0]), 1.2), 0.3),
                0.33,
                0.80,
                0.60,
                0.60,
                0.60,
                0.60,
            ]
            response_guesses = [
                response_template_with_E[min(i, len(response_template_with_E) - 1)]
                for i in range(n_species)
            ]

        p0: List[float] = rate_guesses + response_guesses

        if self.cfg.fit_baseline_offset:
            p0.append(0.0)

        if self.cfg.fit_baseline_slope:
            if self.cfg.initialize_baseline_slope_from_data:
                slope0 = self.estimate_trace_slope(time_fit, y_fit)
                slope0 = np.clip(
                    slope0,
                    self.cfg.baseline_slope_min + 1e-12,
                    self.cfg.baseline_slope_max - 1e-12,
                )
                p0.append(float(slope0))
            else:
                p0.append(0.0)

        return np.array(p0, dtype=float)

    def make_zero_intermediate_start_vectors(
        self,
        time_fit: np.ndarray,
        y_fit: np.ndarray,
        D_uM: float,
        fit_mode: str,
    ) -> List[np.ndarray]:
        """Create data-driven starts for the direct E + S -> P model.

        The 0-intermediate model is often observed as an approximately linear
        rising or falling trace over the sampled time window. These starts seed
        combinations with broader baseline offset/slope and product response so
        the optimizer does not need to discover a near-linear direct conversion
        from the stopped-flow-oriented default guesses alone.
        """
        D_mM = max(float(D_uM) / 1000.0, 1e-12)
        k1_bounds, k1_default = self.get_k1_settings(fit_mode)
        lb, ub = self.make_bounds(0, fit_mode)

        t_local = np.asarray(time_fit, dtype=float) - float(time_fit[0])
        duration = max(float(t_local[-1]), 1e-12)
        y_fit = np.asarray(y_fit, dtype=float)
        y0 = float(y_fit[0])
        y_end = float(y_fit[-1])
        observed_slope = self.estimate_trace_slope(time_fit, y_fit)

        b0_candidates = [0.0]
        if self.cfg.fit_baseline_offset:
            if self.cfg.fix_free_enzyme_response:
                b0_candidates.append(y0 - float(self.cfg.free_enzyme_response))
            else:
                b0_candidates.append(0.0)
            b0_candidates.append(float(np.nanmedian(y_fit)) - 0.5 * (y0 + y_end))

        b1_candidates = [0.0]
        if self.cfg.fit_baseline_slope:
            b1_candidates.extend([observed_slope, 0.5 * observed_slope])

        # Trial apparent observed rates, in min^-1. Including slow values helps
        # the direct model approximate an initially linear trace; faster values
        # are useful for clearly curved direct conversions.
        kobs_trials = [
            max(1.0 / duration, 1e-6),
            max(0.2 / duration, 1e-6),
            max(0.05 / duration, 1e-6),
            abs(float(k1_default)) * D_mM,
        ]

        starts: List[np.ndarray] = []
        seen: set = set()

        for b0 in b0_candidates:
            if self.cfg.fit_baseline_offset:
                b0_use = float(np.clip(b0, self.cfg.baseline_offset_min + 1e-12, self.cfg.baseline_offset_max - 1e-12))
            else:
                b0_use = 0.0

            for b1 in b1_candidates:
                if self.cfg.fit_baseline_slope:
                    b1_use = float(np.clip(b1, self.cfg.baseline_slope_min + 1e-12, self.cfg.baseline_slope_max - 1e-12))
                else:
                    b1_use = 0.0

                terminal_response = y_end - b0_use - b1_use * duration

                if self.cfg.fix_free_enzyme_response:
                    free_response = float(self.cfg.free_enzyme_response)
                    # For increasing traces, seed a product response above the
                    # free-enzyme response. For decreasing traces, seed below it.
                    if observed_slope > 0:
                        terminal_response = max(terminal_response, free_response + 0.1)
                    elif observed_slope < 0:
                        terminal_response = min(terminal_response, free_response - 0.1)
                else:
                    free_response = y0 - b0_use

                rp_use = float(np.clip(terminal_response, self.cfg.response_min + 1e-12, self.cfg.response_max - 1e-12))

                kinetic_slope = observed_slope - b1_use
                response_gap = rp_use - free_response
                if abs(response_gap) > 1e-9 and abs(kinetic_slope) > 1e-12:
                    kobs_from_slope = abs(kinetic_slope / response_gap)
                    kobs_trials_local = [kobs_from_slope, 0.3 * kobs_from_slope, 3.0 * kobs_from_slope] + kobs_trials
                else:
                    kobs_trials_local = kobs_trials

                for kobs in kobs_trials_local:
                    k1_use = float(np.clip(kobs / D_mM, k1_bounds[0] + 1e-12, k1_bounds[1] - 1e-12))
                    values: List[float] = [k1_use]

                    if self.cfg.fix_free_enzyme_response:
                        values.append(rp_use)
                    else:
                        re_use = float(np.clip(free_response, self.cfg.response_min + 1e-12, self.cfg.response_max - 1e-12))
                        values.extend([re_use, rp_use])

                    if self.cfg.fit_baseline_offset:
                        values.append(b0_use)
                    if self.cfg.fit_baseline_slope:
                        values.append(b1_use)

                    arr = np.clip(np.array(values, dtype=float), lb + 1e-12, ub - 1e-12)
                    key = tuple(np.round(arr, decimals=10))
                    if key not in seen:
                        seen.add(key)
                        starts.append(arr)

        return starts

    def make_reversible_mm_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = [self.cfg.mm_kon_min, self.cfg.mm_koff_min, self.cfg.mm_kcat_min]
        ub = [self.cfg.mm_kon_max, self.cfg.mm_koff_max, self.cfg.mm_kcat_max]

        n_response_params = 2 if self.cfg.fix_free_enzyme_response else 3
        lb.extend([self.cfg.response_min] * n_response_params)
        ub.extend([self.cfg.response_max] * n_response_params)

        if self.cfg.fit_baseline_offset:
            lb.append(self.cfg.baseline_offset_min)
            ub.append(self.cfg.baseline_offset_max)

        if self.cfg.fit_baseline_slope:
            lb.append(self.cfg.baseline_slope_min)
            ub.append(self.cfg.baseline_slope_max)

        return np.array(lb, dtype=float), np.array(ub, dtype=float)

    def make_reversible_mm_initial_guess(
        self,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        D_uM: float,
    ) -> np.ndarray:
        D_mM = max(float(D_uM) / 1000.0, 1e-12)
        observed_slope = self.estimate_trace_slope(time_fit, y_raw_fit)
        y0 = float(y_raw_fit[0]) if len(y_raw_fit) else 0.0
        y_end = float(y_raw_fit[-1]) if len(y_raw_fit) else y0

        kon0 = float(self.cfg.mm_kon_initial_guess)
        koff0 = float(self.cfg.mm_koff_initial_guess)
        kcat0 = float(self.cfg.mm_kcat_initial_guess)

        responses: List[float] = []
        if not self.cfg.fix_free_enzyme_response:
            responses.append(float(np.clip(y0, self.cfg.response_min, self.cfg.response_max)))
        # ES response initialized between starting and ending response.
        responses.append(float(np.clip(0.5 * (y0 + y_end), self.cfg.response_min, self.cfg.response_max)))
        # P response initialized from the observed direction and amplitude.
        p_resp = y_end if observed_slope >= 0 else y0
        responses.append(float(np.clip(abs(p_resp), self.cfg.response_min, self.cfg.response_max)))

        values: List[float] = [kon0, koff0, kcat0] + responses

        if self.cfg.fit_baseline_offset:
            values.append(float(np.clip(y0 - self.cfg.free_enzyme_response, self.cfg.baseline_offset_min, self.cfg.baseline_offset_max)))
        if self.cfg.fit_baseline_slope:
            values.append(float(np.clip(observed_slope, self.cfg.baseline_slope_min, self.cfg.baseline_slope_max)))

        lb, ub = self.make_reversible_mm_bounds()
        return np.clip(np.array(values, dtype=float), lb + 1e-12, ub - 1e-12)

    def make_reversible_mm_start_vectors(
        self,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        D_uM: float,
    ) -> List[np.ndarray]:
        lb, ub = self.make_reversible_mm_bounds()
        base = self.make_reversible_mm_initial_guess(time_fit, y_raw_fit, D_uM)
        D_mM = max(float(D_uM) / 1000.0, 1e-12)
        starts: List[np.ndarray] = [base]
        seen = {tuple(np.round(base, 10))}

        observed_slope = abs(self.estimate_trace_slope(time_fit, y_raw_fit))
        duration = max(float(time_fit[-1] - time_fit[0]), 1e-6)
        # Try apparent first-order time scales around the observed window.
        kcat_trials = [self.cfg.mm_kcat_initial_guess, 0.1, 1.0, 10.0, 100.0, 1.0 / duration, 5.0 / duration]
        Km_trials = [D_mM / 10.0, D_mM, 10.0 * D_mM, 0.01, 0.1, 1.0, 10.0]
        koff_fracs = [0.0, 0.1, 1.0, 10.0]

        for kcat in kcat_trials:
            kcat = float(np.clip(abs(kcat), self.cfg.mm_kcat_min + 1e-12, self.cfg.mm_kcat_max - 1e-12))
            for Km in Km_trials:
                Km = max(float(Km), 1e-12)
                for frac in koff_fracs:
                    koff = float(np.clip(frac * kcat, self.cfg.mm_koff_min + 1e-12, self.cfg.mm_koff_max - 1e-12))
                    kon = float(np.clip((koff + kcat) / Km, self.cfg.mm_kon_min + 1e-12, self.cfg.mm_kon_max - 1e-12))
                    trial = base.copy()
                    trial[0:3] = [kon, koff, kcat]
                    # Give some starts a small baseline slope and others the observed one.
                    if self.cfg.fit_baseline_slope:
                        trial[-1] = float(np.clip(observed_slope if len(starts) % 2 == 0 else 0.0, self.cfg.baseline_slope_min + 1e-12, self.cfg.baseline_slope_max - 1e-12))
                    trial = np.clip(trial, lb + 1e-12, ub - 1e-12)
                    key = tuple(np.round(trial, 10))
                    if key not in seen:
                        seen.add(key)
                        starts.append(trial)

        return starts

    def reversible_mm_residuals(
        self,
        params: np.ndarray,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
        D_uM: float,
    ) -> np.ndarray:
        try:
            sim, _, _ = self.simulate_reversible_mm_model(time_fit, params, D_uM)
        except Exception:
            return np.full(len(y_raw_fit) * (2 if self.cfg.use_smoothed_target else 1) + 1, 1e6, dtype=float)

        t_local = time_fit - time_fit[0]
        early_weights = 1.0 + self.cfg.early_weight_strength * np.exp(-t_local / self.cfg.early_weight_tau)

        raw_resid = self.cfg.raw_weight * early_weights * (sim - y_raw_fit)
        residual_list = [raw_resid]

        if self.cfg.use_smoothed_target:
            smooth_resid = self.cfg.smooth_weight * early_weights * (sim - y_smooth_fit)
            residual_list.append(smooth_resid)

        start_resid = self.cfg.start_value_penalty_weight * (sim[0] - y_smooth_fit[0])
        residual_list.append(np.array([start_resid]))

        # Reversible MM product-formation traces often increase; apply the
        # early-peak penalty only when explicitly enabled and the trend is not
        # increasing.
        apply_peak_penalty = bool(self.cfg.prevent_initial_peak)
        if (
            apply_peak_penalty
            and self.cfg.auto_disable_initial_peak_penalty_for_increasing_traces
            and self.estimate_trace_slope(time_fit, y_smooth_fit) > 0
        ):
            apply_peak_penalty = False

        if apply_peak_penalty:
            early_mask = t_local <= self.cfg.initial_peak_window_min
            if np.any(early_mask):
                overshoot = np.maximum(sim[early_mask] - sim[0] - self.cfg.initial_peak_tolerance, 0.0)
                residual_list.append(self.cfg.initial_peak_penalty_weight * overshoot)

        return np.concatenate(residual_list)

    def fit_reversible_mm_model(
        self,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
        D_uM: float,
        n_starts: int,
    ) -> Dict[str, Any]:
        lb, ub = self.make_reversible_mm_bounds()
        explicit_starts = self.make_reversible_mm_start_vectors(time_fit, y_raw_fit, D_uM)
        if not explicit_starts:
            explicit_starts = [self.make_reversible_mm_initial_guess(time_fit, y_raw_fit, D_uM)]

        best: Optional[Dict[str, Any]] = None
        for start_idx in range(n_starts):
            self.check_cancelled()
            if start_idx < len(explicit_starts):
                p0 = explicit_starts[start_idx].copy()
            else:
                base = explicit_starts[start_idx % len(explicit_starts)]
                jitter = np.exp(self.rng.normal(0, 1.0, size=len(base)))
                p0 = base * jitter
            p0 = np.clip(p0, lb + 1e-12, ub - 1e-12)

            result = least_squares(
                self.reversible_mm_residuals,
                p0,
                bounds=(lb, ub),
                args=(time_fit, y_raw_fit, y_smooth_fit, D_uM),
                loss="soft_l1",
                max_nfev=5000,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )

            sim, species, metrics = self.simulate_reversible_mm_model(time_fit, result.x, D_uM)
            rss_raw = float(np.sum((sim - y_raw_fit) ** 2))

            if best is None or rss_raw < best["rss"]:
                best = {
                    "params": result.x,
                    "rss": rss_raw,
                    "success": result.success,
                    "model_type": "reversible_mm",
                    "fit_mode": "reversible_mm",
                    "metrics": metrics,
                }

        if best is None:
            raise RuntimeError("No reversible Michaelis-Menten fit was completed.")

        n = len(y_raw_fit)
        k = len(best["params"])
        rss_safe = max(float(best["rss"]), 1e-12)
        best["aic"] = float(n * np.log(rss_safe / n) + 2 * k)
        best["bic"] = float(n * np.log(rss_safe / n) + k * np.log(n))
        best["n_params"] = int(k)
        return best

    def make_bounds(self, n_intermediates: int, fit_mode: str) -> Tuple[np.ndarray, np.ndarray]:
        n_species = n_intermediates + 2
        n_rates = n_species - 1
        k1_bounds, _ = self.get_k1_settings(fit_mode)

        lb_rates = [k1_bounds[0]] + [self.cfg.k_other_min] * (n_rates - 1)
        ub_rates = [k1_bounds[1]] + [self.cfg.k_other_max] * (n_rates - 1)

        if self.cfg.fix_free_enzyme_response:
            n_response_params = n_species - 1
            lb_responses = [self.cfg.response_min] * n_response_params
            ub_responses = [self.cfg.response_max] * n_response_params

            # For n_intermediates == 0, the only fitted non-E response is the
            # terminal/product state P, not an intermediate. Do not cap it using
            # the first-intermediate upper bound.
            if (
                self.cfg.first_intermediate_must_not_exceed_E
                and n_intermediates >= 1
                and n_response_params >= 1
            ):
                ub_responses[0] = self.cfg.first_intermediate_upper
        else:
            lb_responses = [self.cfg.response_min] * n_species
            ub_responses = [self.cfg.response_max] * n_species

        lb = lb_rates + lb_responses
        ub = ub_rates + ub_responses

        if self.cfg.fit_baseline_offset:
            lb.append(self.cfg.baseline_offset_min)
            ub.append(self.cfg.baseline_offset_max)

        if self.cfg.fit_baseline_slope:
            lb.append(self.cfg.baseline_slope_min)
            ub.append(self.cfg.baseline_slope_max)

        return np.array(lb, dtype=float), np.array(ub, dtype=float)

    def residuals(
        self,
        params: np.ndarray,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
        D_uM: float,
        n_intermediates: int,
    ) -> np.ndarray:
        rates, responses, b0, b1 = self.unpack_params(params, n_intermediates)
        sim, _ = self.simulate_model(time_fit, rates, responses, b0, b1, D_uM)

        t_local = time_fit - time_fit[0]
        early_weights = 1.0 + self.cfg.early_weight_strength * np.exp(-t_local / self.cfg.early_weight_tau)

        raw_resid = self.cfg.raw_weight * early_weights * (sim - y_raw_fit)
        residual_list = [raw_resid]

        if self.cfg.use_smoothed_target:
            smooth_resid = self.cfg.smooth_weight * early_weights * (sim - y_smooth_fit)
            residual_list.append(smooth_resid)

        start_resid = self.cfg.start_value_penalty_weight * (sim[0] - y_smooth_fit[0])
        residual_list.append(np.array([start_resid]))

        apply_peak_penalty = bool(self.cfg.prevent_initial_peak)
        if (
            apply_peak_penalty
            and self.cfg.auto_disable_initial_peak_penalty_for_increasing_traces
            and self.estimate_trace_slope(time_fit, y_smooth_fit) > 0
        ):
            apply_peak_penalty = False

        if apply_peak_penalty:
            early_mask = t_local <= self.cfg.initial_peak_window_min
            if np.any(early_mask):
                overshoot = np.maximum(
                    sim[early_mask] - sim[0] - self.cfg.initial_peak_tolerance,
                    0.0,
                )
                peak_penalty = self.cfg.initial_peak_penalty_weight * overshoot
                residual_list.append(peak_penalty)

        return np.concatenate(residual_list)

    def _shared_nonkinetic_residuals(
        self,
        sim: np.ndarray,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
    ) -> np.ndarray:
        t_local = time_fit - time_fit[0]
        early_weights = 1.0 + self.cfg.early_weight_strength * np.exp(-t_local / self.cfg.early_weight_tau)

        raw_resid = self.cfg.raw_weight * early_weights * (sim - y_raw_fit)
        residual_list = [raw_resid]

        if self.cfg.use_smoothed_target:
            smooth_resid = self.cfg.smooth_weight * early_weights * (sim - y_smooth_fit)
            residual_list.append(smooth_resid)

        start_resid = self.cfg.start_value_penalty_weight * (sim[0] - y_smooth_fit[0])
        residual_list.append(np.array([start_resid]))

        return np.concatenate(residual_list)

    def linear_residuals(
        self,
        params: np.ndarray,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
    ) -> np.ndarray:
        intercept, slope = self.unpack_linear_params(params)
        sim = self.simulate_linear_only(time_fit, intercept, slope)
        return self._shared_nonkinetic_residuals(sim, time_fit, y_raw_fit, y_smooth_fit)

    def no_binding_residuals(
        self,
        params: np.ndarray,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
    ) -> np.ndarray:
        intercept = self.unpack_no_binding_params(params)
        sim = self.simulate_no_binding(time_fit, intercept)
        return self._shared_nonkinetic_residuals(sim, time_fit, y_raw_fit, y_smooth_fit)

    def fit_model(
        self,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
        D_uM: float,
        n_intermediates: int,
        n_starts: int,
        fit_mode: str,
    ) -> Dict[str, Any]:
        p0_base = self.make_initial_guess(n_intermediates, y_raw_fit, time_fit, fit_mode)
        lb, ub = self.make_bounds(n_intermediates, fit_mode)

        explicit_starts: List[np.ndarray] = [np.clip(p0_base.copy(), lb + 1e-12, ub - 1e-12)]
        if n_intermediates == 0:
            explicit_starts.extend(
                self.make_zero_intermediate_start_vectors(time_fit, y_raw_fit, D_uM, fit_mode)
            )

        best: Optional[Dict[str, Any]] = None

        for start_idx in range(n_starts):
            self.check_cancelled()

            if start_idx < len(explicit_starts):
                p0 = explicit_starts[start_idx].copy()
            else:
                base = explicit_starts[start_idx % len(explicit_starts)]
                jitter = np.exp(self.rng.normal(0, 1.0, size=len(base)))
                p0 = base * jitter

            p0 = np.clip(p0, lb + 1e-12, ub - 1e-12)

            result = least_squares(
                self.residuals,
                p0,
                bounds=(lb, ub),
                args=(time_fit, y_raw_fit, y_smooth_fit, D_uM, n_intermediates),
                loss="soft_l1",
                max_nfev=4000,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )

            rates, responses, b0, b1 = self.unpack_params(result.x, n_intermediates)
            sim, _ = self.simulate_model(time_fit, rates, responses, b0, b1, D_uM)
            rss_raw = float(np.sum((sim - y_raw_fit) ** 2))

            if best is None or rss_raw < best["rss"]:
                best = {
                    "params": result.x,
                    "rss": rss_raw,
                    "success": result.success,
                    "fit_mode": fit_mode,
                    "model_type": "kinetic",
                }

        if best is None:
            raise RuntimeError("No kinetic fit was completed.")

        n = len(y_raw_fit)
        k = len(best["params"])
        rss_safe = max(float(best["rss"]), 1e-12)
        best["aic"] = float(n * np.log(rss_safe / n) + 2 * k)
        best["bic"] = float(n * np.log(rss_safe / n) + k * np.log(n))
        best["n_params"] = int(k)
        return best

    def fit_linear_only(
        self,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
    ) -> Dict[str, Any]:
        self.check_cancelled()

        slope0 = self.estimate_trace_slope(time_fit, y_smooth_fit)
        slope0 = np.clip(
            slope0,
            self.cfg.linear_slope_min + 1e-12,
            self.cfg.linear_slope_max - 1e-12,
        )

        intercept0 = np.clip(
            y_smooth_fit[0],
            self.cfg.linear_intercept_min + 1e-12,
            self.cfg.linear_intercept_max - 1e-12,
        )

        p0 = np.array([intercept0, slope0], dtype=float)
        lb = np.array([self.cfg.linear_intercept_min, self.cfg.linear_slope_min], dtype=float)
        ub = np.array([self.cfg.linear_intercept_max, self.cfg.linear_slope_max], dtype=float)

        result = least_squares(
            self.linear_residuals,
            p0,
            bounds=(lb, ub),
            args=(time_fit, y_raw_fit, y_smooth_fit),
            loss="soft_l1",
            max_nfev=4000,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )

        intercept, slope = self.unpack_linear_params(result.x)
        sim = self.simulate_linear_only(time_fit, intercept, slope)
        rss_raw = float(np.sum((sim - y_raw_fit) ** 2))

        n = len(y_raw_fit)
        k = len(result.x)
        rss_safe = max(rss_raw, 1e-12)

        return {
            "params": result.x,
            "rss": rss_raw,
            "success": result.success,
            "model_type": "linear_only",
            "fit_mode": "linear_only",
            "aic": float(n * np.log(rss_safe / n) + 2 * k),
            "bic": float(n * np.log(rss_safe / n) + k * np.log(n)),
            "n_params": int(k),
        }

    def fit_no_binding(
        self,
        time_fit: np.ndarray,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
    ) -> Dict[str, Any]:
        self.check_cancelled()

        intercept0 = np.clip(
            float(np.nanmedian(y_smooth_fit)),
            self.cfg.no_binding_intercept_min + 1e-12,
            self.cfg.no_binding_intercept_max - 1e-12,
        )

        p0 = np.array([intercept0], dtype=float)
        lb = np.array([self.cfg.no_binding_intercept_min], dtype=float)
        ub = np.array([self.cfg.no_binding_intercept_max], dtype=float)

        result = least_squares(
            self.no_binding_residuals,
            p0,
            bounds=(lb, ub),
            args=(time_fit, y_raw_fit, y_smooth_fit),
            loss="soft_l1",
            max_nfev=4000,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )

        intercept = self.unpack_no_binding_params(result.x)
        sim = self.simulate_no_binding(time_fit, intercept)
        rss_raw = float(np.sum((sim - y_raw_fit) ** 2))

        n = len(y_raw_fit)
        k = len(result.x)
        rss_safe = max(rss_raw, 1e-12)

        return {
            "params": result.x,
            "rss": rss_raw,
            "success": result.success,
            "model_type": "no_binding",
            "fit_mode": "no_binding",
            "aic": float(n * np.log(rss_safe / n) + 2 * k),
            "bic": float(n * np.log(rss_safe / n) + k * np.log(n)),
            "n_params": int(k),
        }

    @staticmethod
    def robust_noise_sigma(y_raw_fit: np.ndarray, y_smooth_fit: np.ndarray) -> float:
        residual = np.asarray(y_raw_fit, dtype=float) - np.asarray(y_smooth_fit, dtype=float)
        residual = residual[np.isfinite(residual)]

        if residual.size < 3:
            return float(np.nanstd(y_raw_fit))

        median = float(np.nanmedian(residual))
        mad = float(np.nanmedian(np.abs(residual - median)))
        sigma = 1.4826 * mad

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(residual))

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(y_raw_fit))

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.0

        return sigma

    def choose_best_record(
        self,
        run_results: List[Dict[str, Any]],
        time_fit: np.ndarray,
        D_uM: float,
        y_raw_fit: np.ndarray,
        y_smooth_fit: np.ndarray,
    ) -> Dict[str, Any]:
        if not run_results:
            raise RuntimeError("No candidate models were fit.")

        best_by_bic = min(run_results, key=lambda x: x["fit"]["bic"])
        no_binding_records = [record for record in run_results if record["model_type"] == "no_binding"]

        if not no_binding_records:
            best_by_bic["selection_reason"] = "lowest_bic_no_no_binding_candidate"
            return best_by_bic

        no_binding_record = no_binding_records[0]
        no_binding_bic = float(no_binding_record["fit"]["bic"])
        best_bic = float(best_by_bic["fit"]["bic"])

        best_sim = self.get_fit_signal(time_fit, D_uM, best_by_bic)
        fitted_signal_change = float(np.nanmax(best_sim) - np.nanmin(best_sim))

        noise_sigma = self.robust_noise_sigma(y_raw_fit, y_smooth_fit)
        signal_change_threshold = max(
            float(self.cfg.no_binding_min_signal_change),
            float(self.cfg.no_binding_noise_multiplier) * noise_sigma,
        )

        # Rule 1: if the constant/no-binding model is statistically close to the
        # best BIC model AND the best nonconstant model changes only within the
        # estimated noise, prefer the simpler no-binding interpretation. This
        # avoids false no-binding calls for product-formation traces that have
        # meaningful positive slopes but similar BIC values.
        if (
            no_binding_bic <= best_bic + float(self.cfg.no_binding_bic_preference_tolerance)
            and best_by_bic["model_type"] != "no_binding"
            and fitted_signal_change < signal_change_threshold
        ):
            no_binding_record["selection_reason"] = "no_binding_within_bic_tolerance_and_change_within_noise"
            no_binding_record["no_binding_noise_sigma"] = noise_sigma
            no_binding_record["no_binding_signal_change_threshold"] = signal_change_threshold
            no_binding_record["best_nonconstant_signal_change"] = fitted_signal_change
            return no_binding_record

        # Rule 2: if the best nonconstant model only moves within the noise,
        # treat the trace as no binding rather than assigning a misleading
        # linear slope or kinetic transition.
        if (
            self.cfg.no_binding_prefer_if_change_within_noise
            and best_by_bic["model_type"] != "no_binding"
            and fitted_signal_change < signal_change_threshold
        ):
            no_binding_record["selection_reason"] = "best_model_change_within_noise"
            no_binding_record["no_binding_noise_sigma"] = noise_sigma
            no_binding_record["no_binding_signal_change_threshold"] = signal_change_threshold
            no_binding_record["best_nonconstant_signal_change"] = fitted_signal_change
            return no_binding_record

        best_by_bic["selection_reason"] = "lowest_bic_and_meaningful_signal_change"
        best_by_bic["no_binding_noise_sigma"] = noise_sigma
        best_by_bic["no_binding_signal_change_threshold"] = signal_change_threshold
        best_by_bic["best_nonconstant_signal_change"] = fitted_signal_change
        return best_by_bic

    def get_fit_signal(self, time_fit: np.ndarray, D_uM: float, best_record: Dict[str, Any]) -> np.ndarray:
        fit = best_record["fit"]

        if best_record["model_type"] == "no_binding":
            intercept = self.unpack_no_binding_params(fit["params"])
            return self.simulate_no_binding(time_fit, intercept)

        if best_record["model_type"] == "linear_only":
            intercept, slope = self.unpack_linear_params(fit["params"])
            return self.simulate_linear_only(time_fit, intercept, slope)

        if best_record["model_type"] == "reversible_mm":
            sim, _, _ = self.simulate_reversible_mm_model(time_fit, fit["params"], D_uM)
            return sim

        rates, responses, b0, b1 = self.unpack_params(
            fit["params"],
            best_record["n_intermediates"],
        )
        sim, _ = self.simulate_model(time_fit, rates, responses, b0, b1, D_uM)
        return sim

    def save_combined_plot(
        self,
        selected_indices: List[int],
        filename: str,
        title: str,
        best_fits: List[Dict[str, Any]],
        data: np.ndarray,
        time: np.ndarray,
        time_fit: np.ndarray,
        D_values_uM: List[float],
    ) -> None:
        if len(selected_indices) == 0:
            return

        plt.figure(figsize=(11, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

        for color_idx, run_idx in enumerate(selected_indices):
            best_record = best_fits[run_idx]
            y_norm = self.normalize_trace(data[:, run_idx], time)
            D_uM = D_values_uM[run_idx]
            sim = self.get_fit_signal(time_fit, D_uM, best_record)

            if best_record["model_type"] == "no_binding":
                label = f"Run {run_idx + 1}, no binding"
            elif best_record["model_type"] == "linear_only":
                label = f"Run {run_idx + 1}, linear"
            elif best_record["model_type"] == "reversible_mm":
                label = f"Run {run_idx + 1}, E + S <-> ES -> P"
            else:
                label = (
                    f"Run {run_idx + 1}, "
                    f"{'E + S -> P' if int(best_record['n_intermediates']) == 0 else best_record['fit_mode']}, "
                    f"{best_record['n_intermediates']} int."
                )

            plt.plot(time, y_norm, "o", markersize=2, alpha=0.30, color=colors[color_idx])
            plt.plot(time_fit, sim, "-", linewidth=2.5, color=colors[color_idx], label=label)

        plt.axvline(self.cfg.fit_start_time, linestyle=":", linewidth=1.5)
        plt.xlabel("Time (min)")
        plt.ylabel("Relative Intensity")
        plt.title(title, pad=10)
        plt.grid(alpha=0.2, linewidth=0.6)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, filename), dpi=300)
        plt.close()

    def run(self) -> str:
        self.validate_config()
        self.check_cancelled()

        np.random.seed(self.cfg.random_seed)
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        with open(os.path.join(self.cfg.output_dir, "run_configuration.json"), "w", encoding="utf-8") as handle:
            json.dump(self.cfg.to_dict(), handle, indent=2)

        self.log(f"Reading CSV: {self.cfg.input_csv}")
        df = pd.read_csv(self.cfg.input_csv)

        try:
            time_raw = pd.to_numeric(df.iloc[:, 0], errors="raise").to_numpy(dtype=float)
        except Exception as exc:
            raise ValueError("The first CSV column must contain numeric time values.") from exc
        data_raw = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        sort_idx = np.argsort(time_raw)
        time_input = time_raw[sort_idx]
        time = self.cfg.time_to_minutes(time_input, self.cfg.input_time_unit)
        data = data_raw[sort_idx, :]
        time = time - time[0]

        D_values_uM = self.cfg.D_values_uM

        if data.shape[1] != len(D_values_uM):
            raise ValueError(
                f"CSV has {data.shape[1]} trace columns, but the concentration setup expects "
                f"{len(D_values_uM)} traces. Update the concentration/sample counts or check the CSV."
            )

        fit_mask = time >= self.cfg.fit_start_time
        time_fit = time[fit_mask]

        if len(time_fit) < 5:
            raise ValueError("Too few points are available after fit_start_time. Lower fit_start_time.")

        total_runs = data.shape[1]
        self.log(f"Output folder: {self.cfg.output_dir}")
        self.log(
            f"Input time unit: {self.cfg.input_time_unit}; "
            "time is converted internally to minutes for fitting and rate constants."
        )
        self.log(
            f"Concentration input units: E0 in {self.cfg.enzyme_concentration_unit}; "
            f"substrate/reactant in {self.cfg.substrate_concentration_unit}. "
            "Concentrations are converted internally to µM/mM."
        )
        self.log(f"Detected {total_runs} traces across {len(self.cfg.concentration_groups)} concentration(s).")
        if self.cfg.fit_zero_intermediate_only:
            self.log("Direct-only mode active: fitting only E + S -> P, with n_intermediates = 0.")
        if self.cfg.fit_reversible_mm_only:
            self.log("Reversible Michaelis-Menten-only mode active: fitting only E + S <-> ES -> P.")
        self.log("Starting model selection and fitting...")

        model_summary: List[Dict[str, Any]] = []
        best_fits: List[Dict[str, Any]] = []

        for run_idx in range(total_runs):
            self.check_cancelled()
            D_uM = D_values_uM[run_idx]
            D_display = self.cfg.concentration_from_uM(D_uM, self.cfg.substrate_concentration_unit)
            self.log(
                f"Fitting run {run_idx + 1}/{total_runs} at "
                f"{D_display:g} {self.cfg.substrate_concentration_unit} "
                f"({D_uM:g} µM internal)"
            )

            y_norm = self.normalize_trace(data[:, run_idx], time)
            y_raw_fit = y_norm[fit_mask]
            y_smooth_fit = self.smooth_trace(y_raw_fit)

            screen_idx = np.arange(0, len(time_fit), self.cfg.selection_stride)
            time_screen = time_fit[screen_idx]
            y_raw_screen = y_raw_fit[screen_idx]
            y_smooth_screen = y_smooth_fit[screen_idx]

            run_results: List[Dict[str, Any]] = []

            kinetic_start = 0 if self.cfg.include_zero_intermediate_kinetic_model else 1
            for fit_mode in self.cfg.kinetic_fit_modes:
                for n_intermediates in range(kinetic_start, self.cfg.max_intermediates + 1):
                    self.check_cancelled()
                    fit_screen = self.fit_model(
                        time_screen,
                        y_raw_screen,
                        y_smooth_screen,
                        D_uM,
                        n_intermediates,
                        self.cfg.n_starts_screen,
                        fit_mode,
                    )

                    record = {
                        "model_type": "kinetic",
                        "fit_mode": fit_mode,
                        "n_intermediates": n_intermediates,
                        "fit": fit_screen,
                    }
                    run_results.append(record)

                    model_summary.append(
                        {
                            "Run": run_idx + 1,
                            "D_uM": D_uM,
                            "model_type": "kinetic",
                            "fit_mode": fit_mode,
                            "n_intermediates": n_intermediates,
                            "RSS": fit_screen["rss"],
                            "AIC": fit_screen["aic"],
                            "BIC": fit_screen["bic"],
                            "n_params": fit_screen["n_params"],
                            "fit_stage": "screen",
                        }
                    )

            if self.cfg.include_reversible_mm_model:
                fit_screen_mm = self.fit_reversible_mm_model(
                    time_screen,
                    y_raw_screen,
                    y_smooth_screen,
                    D_uM,
                    self.cfg.n_starts_screen,
                )
                record = {
                    "model_type": "reversible_mm",
                    "fit_mode": "reversible_mm",
                    "n_intermediates": 1,
                    "fit": fit_screen_mm,
                }
                run_results.append(record)

                model_summary.append(
                    {
                        "Run": run_idx + 1,
                        "D_uM": D_uM,
                        "model_type": "reversible_mm",
                        "fit_mode": "reversible_mm",
                        "n_intermediates": 1,
                        "mechanism": "E + S <-> ES -> P",
                        "RSS": fit_screen_mm["rss"],
                        "AIC": fit_screen_mm["aic"],
                        "BIC": fit_screen_mm["bic"],
                        "n_params": fit_screen_mm["n_params"],
                        "fit_stage": "screen",
                    }
                )

            if self.cfg.include_no_binding_model:
                fit_screen_no_binding = self.fit_no_binding(time_screen, y_raw_screen, y_smooth_screen)
                record = {
                    "model_type": "no_binding",
                    "fit_mode": "no_binding",
                    "n_intermediates": 0,
                    "fit": fit_screen_no_binding,
                }
                run_results.append(record)

                model_summary.append(
                    {
                        "Run": run_idx + 1,
                        "D_uM": D_uM,
                        "model_type": "no_binding",
                        "fit_mode": "no_binding",
                        "n_intermediates": 0,
                        "RSS": fit_screen_no_binding["rss"],
                        "AIC": fit_screen_no_binding["aic"],
                        "BIC": fit_screen_no_binding["bic"],
                        "n_params": fit_screen_no_binding["n_params"],
                        "fit_stage": "screen",
                    }
                )

            if self.cfg.include_linear_only_model:
                fit_screen_linear = self.fit_linear_only(time_screen, y_raw_screen, y_smooth_screen)
                record = {
                    "model_type": "linear_only",
                    "fit_mode": "linear_only",
                    "n_intermediates": 0,
                    "fit": fit_screen_linear,
                }
                run_results.append(record)

                model_summary.append(
                    {
                        "Run": run_idx + 1,
                        "D_uM": D_uM,
                        "model_type": "linear_only",
                        "fit_mode": "linear_only",
                        "n_intermediates": 0,
                        "RSS": fit_screen_linear["rss"],
                        "AIC": fit_screen_linear["aic"],
                        "BIC": fit_screen_linear["bic"],
                        "n_params": fit_screen_linear["n_params"],
                        "fit_stage": "screen",
                    }
                )

            best_screen_record = self.choose_best_record(
                run_results,
                time_screen,
                D_uM,
                y_raw_screen,
                y_smooth_screen,
            )

            if best_screen_record["model_type"] == "no_binding":
                best_fit = self.fit_no_binding(time_fit, y_raw_fit, y_smooth_fit)
                best_record = {
                    "model_type": "no_binding",
                    "fit_mode": "no_binding",
                    "n_intermediates": 0,
                    "fit": best_fit,
                    "selection_reason": best_screen_record.get("selection_reason", ""),
                    "no_binding_noise_sigma": best_screen_record.get("no_binding_noise_sigma", np.nan),
                    "no_binding_signal_change_threshold": best_screen_record.get("no_binding_signal_change_threshold", np.nan),
                    "best_nonconstant_signal_change": best_screen_record.get("best_nonconstant_signal_change", np.nan),
                }
            elif best_screen_record["model_type"] == "linear_only":
                best_fit = self.fit_linear_only(time_fit, y_raw_fit, y_smooth_fit)
                best_record = {
                    "model_type": "linear_only",
                    "fit_mode": "linear_only",
                    "n_intermediates": 0,
                    "fit": best_fit,
                    "selection_reason": best_screen_record.get("selection_reason", ""),
                    "no_binding_noise_sigma": best_screen_record.get("no_binding_noise_sigma", np.nan),
                    "no_binding_signal_change_threshold": best_screen_record.get("no_binding_signal_change_threshold", np.nan),
                    "best_nonconstant_signal_change": best_screen_record.get("best_nonconstant_signal_change", np.nan),
                }
            elif best_screen_record["model_type"] == "reversible_mm":
                best_fit = self.fit_reversible_mm_model(
                    time_fit,
                    y_raw_fit,
                    y_smooth_fit,
                    D_uM,
                    self.cfg.n_starts_refine,
                )
                best_record = {
                    "model_type": "reversible_mm",
                    "fit_mode": "reversible_mm",
                    "n_intermediates": 1,
                    "fit": best_fit,
                    "selection_reason": best_screen_record.get("selection_reason", ""),
                    "no_binding_noise_sigma": best_screen_record.get("no_binding_noise_sigma", np.nan),
                    "no_binding_signal_change_threshold": best_screen_record.get("no_binding_signal_change_threshold", np.nan),
                    "best_nonconstant_signal_change": best_screen_record.get("best_nonconstant_signal_change", np.nan),
                }
            else:
                best_fit = self.fit_model(
                    time_fit,
                    y_raw_fit,
                    y_smooth_fit,
                    D_uM,
                    best_screen_record["n_intermediates"],
                    self.cfg.n_starts_refine,
                    best_screen_record["fit_mode"],
                )
                best_record = {
                    "model_type": "kinetic",
                    "fit_mode": best_screen_record["fit_mode"],
                    "n_intermediates": best_screen_record["n_intermediates"],
                    "fit": best_fit,
                    "selection_reason": best_screen_record.get("selection_reason", ""),
                    "no_binding_noise_sigma": best_screen_record.get("no_binding_noise_sigma", np.nan),
                    "no_binding_signal_change_threshold": best_screen_record.get("no_binding_signal_change_threshold", np.nan),
                    "best_nonconstant_signal_change": best_screen_record.get("best_nonconstant_signal_change", np.nan),
                }

            best_fits.append(best_record)

            model_summary.append(
                {
                    "Run": run_idx + 1,
                    "D_uM": D_uM,
                    "model_type": best_record["model_type"],
                    "fit_mode": best_record["fit_mode"],
                    "n_intermediates": best_record["n_intermediates"],
                    "RSS": best_fit["rss"],
                    "AIC": best_fit["aic"],
                    "BIC": best_fit["bic"],
                    "n_params": best_fit["n_params"],
                    "fit_stage": "refine",
                    "selection_reason": best_record.get("selection_reason", ""),
                    "no_binding_noise_sigma": best_record.get("no_binding_noise_sigma", np.nan),
                    "no_binding_signal_change_threshold": best_record.get("no_binding_signal_change_threshold", np.nan),
                    "best_nonconstant_signal_change": best_record.get("best_nonconstant_signal_change", np.nan),
                }
            )

            sim = self.get_fit_signal(time_fit, D_uM, best_record)

            if best_record["model_type"] == "no_binding":
                fit_label = "Best fit: no binding"
            elif best_record["model_type"] == "linear_only":
                fit_label = "Best fit: linear-only"
            elif best_record["model_type"] == "reversible_mm":
                fit_label = "Best fit: E + S <-> ES -> P"
            else:
                fit_label = (
                    f"Best fit: {best_record['fit_mode']}, "
                    f"{best_record['n_intermediates']} intermediates"
                )

            self.log(
                f"  Selected {best_record['model_type']} / {best_record['fit_mode']} "
                f"(n_intermediates={best_record['n_intermediates']})"
            )

            plt.figure()
            plt.plot(time, y_norm, "o", markersize=3, alpha=0.45, label="Experimental")
            plt.plot(time_fit, y_smooth_fit, "--", linewidth=2, label="Smoothed target")
            plt.plot(time_fit, sim, "-", linewidth=3, label=fit_label)
            plt.axvline(self.cfg.fit_start_time, linestyle=":", linewidth=1.5, label="Fit start")
            plt.xlabel("Time (min)")
            plt.ylabel("Relative Intensity")
            plt.title(f"Run {run_idx + 1}, S={D_uM:g} µM", pad=10)
            plt.legend(frameon=True)
            plt.grid(alpha=0.2, linewidth=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.output_dir, f"best_fit_run_{run_idx + 1}.png"), dpi=300)
            plt.close()

            # Species / apparent-species plot
            #
            # Kinetic fits have true integrated species from the ODE model.
            # Linear-only fits do not have mechanistic species, but users still
            # need a species_run_*.png for every trace. For linear-only fits we
            # therefore plot an apparent first-order species proxy using the same
            # k_obs value reported in best_fit_summary.csv:
            #
            #     k_obs = -linear_slope / linear_intercept
            #
            # For no-binding fits, the only species is constant free enzyme.
            plt.figure()

            if best_record["model_type"] == "kinetic":
                rates, responses, b0, b1 = self.unpack_params(best_fit["params"], best_record["n_intermediates"])
                _, species = self.simulate_model(time_fit, rates, responses, b0, b1, D_uM)

                labels = ["E"] + [f"I{i}" for i in range(1, best_record["n_intermediates"] + 1)] + ["P"]
                for i, label in enumerate(labels):
                    plt.plot(time_fit, species[i], label=label)

                species_title = f"Run {run_idx + 1} species profile"

            elif best_record["model_type"] == "reversible_mm":
                _, species, metrics_mm = self.simulate_reversible_mm_model(time_fit, best_fit["params"], D_uM)
                labels = ["E", "S", "ES", "P"]
                for i, label in enumerate(labels):
                    plt.plot(time_fit, species[i], label=label)
                species_title = f"Run {run_idx + 1} reversible MM species profile"

            elif best_record["model_type"] == "linear_only":
                intercept, slope = self.unpack_linear_params(best_fit["params"])
                t_local = time_fit - time_fit[0]

                if abs(intercept) > 1e-12:
                    k_obs_linear = -slope / intercept
                else:
                    k_obs_linear = -slope / 1e-12

                # Apparent species proxy.
                #
                # A decreasing linear signal gives positive k_obs and can be
                # visualized as apparent E loss and apparent product formation.
                # A positive slope gives negative k_obs. That is still reported
                # in the table, but it is not physically meaningful as loss of E,
                # so the proxy species plot keeps E constant and annotates the
                # plot as an increasing/negative-k_obs linear fit.
                if k_obs_linear > 0:
                    E_app = self.cfg.E0_mM * np.exp(-k_obs_linear * t_local)
                    P_app = self.cfg.E0_mM - E_app
                    plt.plot(time_fit, E_app, label="E (apparent)")
                    plt.plot(time_fit, P_app, label="Product (apparent)")
                    species_title = f"Run {run_idx + 1} linear-only proxy"
                else:
                    E_app = np.full_like(time_fit, self.cfg.E0_mM, dtype=float)
                    P_app = np.maximum(0.0, slope * t_local)
                    plt.plot(time_fit, E_app, label="E (apparent)")
                    plt.plot(time_fit, P_app, label="Product (apparent)")
                    species_title = f"Run {run_idx + 1} linear-only proxy"


            elif best_record["model_type"] == "no_binding":
                E_app = np.full_like(time_fit, self.cfg.E0_mM, dtype=float)
                plt.plot(time_fit, E_app, label="E")
                species_title = f"Run {run_idx + 1} no-binding profile"

            else:
                plt.close()
                raise ValueError(f"Unknown model_type: {best_record['model_type']}")

            plt.xlabel("Time (min)")
            plt.ylabel("Concentration (mM)")
            plt.title(species_title, pad=10)
            plt.legend(frameon=True)
            plt.grid(alpha=0.2, linewidth=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(self.cfg.output_dir, f"species_run_{run_idx + 1}.png"), dpi=300)
            plt.close()

            if self.progress_callback is not None:
                self.progress_callback(run_idx + 1, total_runs)

        model_summary_df = pd.DataFrame(model_summary)
        if not model_summary_df.empty and "D_uM" in model_summary_df.columns:
            model_summary_df["S_input_value"] = model_summary_df["D_uM"].apply(
                lambda value: self.cfg.concentration_from_uM(value, self.cfg.substrate_concentration_unit)
            )
            model_summary_df["S_input_unit"] = self.cfg.substrate_concentration_unit
            model_summary_df["input_time_unit"] = self.cfg.input_time_unit
            model_summary_df["internal_time_unit"] = "minutes"
        model_summary_output = model_summary_df.rename(columns={"D_uM": "S_uM"})
        model_summary_output.to_csv(
            os.path.join(self.cfg.output_dir, "model_selection_summary.csv"),
            index=False,
        )

        best_rows: List[Dict[str, Any]] = []

        for run_idx, best_record in enumerate(best_fits):
            best_fit = best_record["fit"]
            D_uM_current = float(D_values_uM[run_idx])
            row: Dict[str, Any] = {
                "Run": run_idx + 1,
                "D_uM": D_uM_current,
                "S_input_value": self.cfg.concentration_from_uM(
                    D_uM_current, self.cfg.substrate_concentration_unit
                ),
                "S_input_unit": self.cfg.substrate_concentration_unit,
                "E0_uM": float(self.cfg.E0_uM),
                "E0_input_value": self.cfg.concentration_from_uM(
                    self.cfg.E0_uM, self.cfg.enzyme_concentration_unit
                ),
                "E0_input_unit": self.cfg.enzyme_concentration_unit,
                "input_time_unit": self.cfg.input_time_unit,
                "internal_time_unit": "minutes",
                "model_type": best_record["model_type"],
                "fit_mode": best_record["fit_mode"],
                "best_n_intermediates": best_record["n_intermediates"],
                "RSS": best_fit["rss"],
                "AIC": best_fit["aic"],
                "BIC": best_fit["bic"],
            }

            if best_record["model_type"] == "no_binding":
                intercept = self.unpack_no_binding_params(best_fit["params"])
                row["no_binding_intercept"] = intercept
                row["selection_reason"] = best_record.get("selection_reason", "")
                row["no_binding_noise_sigma"] = best_record.get("no_binding_noise_sigma", np.nan)
                row["no_binding_signal_change_threshold"] = best_record.get("no_binding_signal_change_threshold", np.nan)
                row["best_nonconstant_signal_change"] = best_record.get("best_nonconstant_signal_change", np.nan)
                row["k_note"] = "No k reported because no binding was detected."
            elif best_record["model_type"] == "linear_only":
                intercept, slope = self.unpack_linear_params(best_fit["params"])
                D_mM = float(D_values_uM[run_idx]) / 1000.0

                # Convert the linear fit into an apparent kinetic value.
                #
                # Linear-only model:
                #     signal = intercept + slope * t
                #
                # Near t = 0, a first-order exponential decrease has:
                #     signal ~= intercept * (1 - k_obs * t)
                # so:
                #     slope ~= -k_obs * intercept
                #     k_obs = -slope / intercept
                #
                # To keep the output table consistent with kinetic fits, also
                # report a concentration-normalized k1:
                #     k1 = k_obs / S_mM
                if abs(intercept) > 1e-12:
                    k_obs_linear = -slope / intercept
                    k_calc_note = "calculated"
                else:
                    # This should be extremely rare because normalized stopped-flow
                    # traces should not start at zero. Use a tiny floor so the k
                    # columns are still populated rather than silently omitted.
                    intercept_floor = 1e-12
                    k_obs_linear = -slope / intercept_floor
                    k_calc_note = "intercept_floor_1e-12_used"

                if D_mM > 0:
                    k1_linear = k_obs_linear / D_mM
                else:
                    k1_linear = np.nan
                    k_calc_note = "invalid_D_mM"

                row["linear_intercept"] = intercept
                row["linear_slope"] = slope
                row["k_obs_linear_min_inv"] = k_obs_linear
                row["k1"] = k1_linear
                row["k_obs_units"] = "min^-1"
                row["k1_units"] = "min^-1 mM^-1"
                row["linear_k_calculation"] = "-linear_slope / linear_intercept; k1 = k_obs / S_mM"
                row["linear_k_note"] = k_calc_note
            elif best_record["model_type"] == "reversible_mm":
                sim_mm, species_mm, metrics_mm = self.simulate_reversible_mm_model(
                    time_fit,
                    best_fit["params"],
                    D_values_uM[run_idx],
                )
                row["mechanism"] = "E + S <-> ES -> P"
                row["kon"] = metrics_mm.get("kon", np.nan)
                row["kon_units"] = "min^-1 mM^-1"
                row["koff"] = metrics_mm.get("koff", np.nan)
                row["koff_units"] = "min^-1"
                row["kcat"] = metrics_mm.get("kcat", np.nan)
                row["kcat_units"] = "min^-1"
                row["Km_mM"] = metrics_mm.get("Km_mM", np.nan)
                row["Km_uM"] = metrics_mm.get("Km_uM", np.nan)
                row["Vmax_uM_per_min_from_kcat_E0"] = metrics_mm.get("Vmax_uM_per_min", np.nan)
                row["Vmax_uM_per_s_from_kcat_E0"] = metrics_mm.get("Vmax_uM_per_s", np.nan)
                row["kcat_over_Km_mM_inv_min_inv"] = metrics_mm.get("kcat_over_Km_mM_inv_min_inv", np.nan)
                row["kcat_over_Km_uM_inv_s_inv"] = metrics_mm.get("kcat_over_Km_uM_inv_s_inv", np.nan)
                response_s_value = self.cfg.free_enzyme_response if self.cfg.fix_free_enzyme_response else metrics_mm.get("response_E", np.nan)
                row["response_E_fixed"] = response_s_value
                row["response_ES"] = metrics_mm.get("response_ED", np.nan)
                row["response_P"] = metrics_mm.get("response_P", np.nan)
                row["baseline_offset"] = metrics_mm.get("baseline_offset", np.nan)
                row["baseline_slope"] = metrics_mm.get("baseline_slope", np.nan)

                raw_trace = np.asarray(data[:, run_idx], dtype=float)
                raw_trace_fit = raw_trace[fit_mask]
                norm_factor = self.normalization_scale(raw_trace, time)
                row["normalization_factor_raw_units"] = norm_factor
                row["mm_fit_linear_slope_norm_per_min"] = self.least_squares_slope(time_fit, sim_mm)
                row["mm_fit_linear_velocity_raw_units_per_min"] = row["mm_fit_linear_slope_norm_per_min"] * norm_factor
                row["mm_fit_linear_velocity_raw_units_per_s"] = row["mm_fit_linear_velocity_raw_units_per_min"] / 60.0
                row["raw_data_linear_slope_raw_units_per_s"] = self.least_squares_slope(time_fit, raw_trace_fit) / 60.0

            else:
                rates, responses, b0, b1 = self.unpack_params(best_fit["params"], best_record["n_intermediates"])
                row["baseline_offset"] = b0
                row["baseline_slope"] = b1

                for i, value in enumerate(rates, start=1):
                    row[f"k{i}"] = value

                if self.cfg.fix_free_enzyme_response:
                    row["response_E_fixed"] = self.cfg.free_enzyme_response
                    for i, value in enumerate(responses, start=1):
                        row[f"response_species_{i}"] = value

                    # Also write explicit state labels. For n_intermediates == 0,
                    # the only fitted non-E response is P, not I1. The legacy
                    # response_species_* columns are retained for backward
                    # compatibility with existing output parsers.
                    n_int = int(best_record["n_intermediates"])
                    if n_int == 0 and len(responses) >= 1:
                        row["response_P"] = responses[0]
                    else:
                        for j in range(n_int):
                            if j < len(responses):
                                row[f"response_I{j + 1}"] = responses[j]
                        if len(responses) > n_int:
                            row["response_P"] = responses[n_int]
                else:
                    for i, value in enumerate(responses):
                        row[f"response_species_{i}"] = value

                    n_int = int(best_record["n_intermediates"])
                    if len(responses) >= 1:
                        row["response_E"] = responses[0]
                    if n_int == 0 and len(responses) >= 2:
                        row["response_P"] = responses[1]
                    else:
                        for j in range(n_int):
                            idx = j + 1
                            if idx < len(responses):
                                row[f"response_I{j + 1}"] = responses[idx]
                        final_idx = n_int + 1
                        if final_idx < len(responses):
                            row["response_P"] = responses[final_idx]

                # For the direct zero-intermediate kinetic model, report
                # spreadsheet-comparable back-calculated velocities. These
                # values convert the normalized fitted response slope back into
                # the original CSV trace units using the same scale that was
                # used for normalization. For SPpNA data whose CSV columns are
                # micromolar product, the raw-unit velocity columns can be
                # compared directly with Excel initial velocities.
                n_int = int(best_record["n_intermediates"])
                if n_int == 0:
                    raw_trace = np.asarray(data[:, run_idx], dtype=float)
                    raw_trace_fit = raw_trace[fit_mask]
                    norm_factor = self.normalization_scale(raw_trace, time)
                    norm_trace_fit = raw_trace_fit / norm_factor
                    sim_norm, _ = self.simulate_model(
                        time_fit,
                        rates,
                        responses,
                        b0,
                        b1,
                        D_values_uM[run_idx],
                    )
                    row.update(
                        self.direct_zero_intermediate_velocity_metrics(
                            time_fit=time_fit,
                            sim_norm=sim_norm,
                            raw_trace_fit=raw_trace_fit,
                            norm_trace_fit=norm_trace_fit,
                            normalization_factor=norm_factor,
                            rates=rates,
                            fitted_responses=responses,
                            baseline_slope=b1,
                            D_uM=D_values_uM[run_idx],
                        )
                    )

            best_rows.append(row)

        best_df = pd.DataFrame(best_rows)
        best_df_output = best_df.rename(columns={"D_uM": "S_uM", "D_mM": "S_mM"})
        best_df_output.to_csv(
            os.path.join(self.cfg.output_dir, "best_fit_summary.csv"),
            index=False,
        )

        self.write_turnover_model_detection_summary(best_df)
        self.fit_michaelis_menten_outputs(best_df)

        all_indices = list(range(len(best_fits)))
        self.save_combined_plot(
            all_indices,
            "all_runs_best_fits.png",
            "All Runs: Direct/Sequential Kinetic, Linear, and No-Binding Model Selection",
            best_fits,
            data,
            time,
            time_fit,
            D_values_uM,
        )

        unique_concentrations = sorted(set(D_values_uM))
        for D_plot in unique_concentrations:
            selected_indices = [i for i, D_val in enumerate(D_values_uM) if D_val == D_plot]
            D_label = self.concentration_label(D_plot)
            self.save_combined_plot(
                selected_indices,
                f"all_runs_best_fits_{D_label}.png",
                f"All Runs at {D_plot:g} µM: Standard, Slow-Apo, Linear, and No-Binding Model Selection",
                best_fits,
                data,
                time,
                time_fit,
                D_values_uM,
            )

        self.write_key_results_summary(best_df)
        self.write_output_guide()
        self.write_summary_workbook()
        if getattr(self.cfg, "organize_outputs", True):
            self.organize_output_files()

        self.log("Done.")
        self.log("Start with: ANCHOR_output_guide.txt")
        self.log("Key tables: ANCHOR_key_results.csv, best_fit_summary.csv")
        self.log("Key plot: michaelis_menten_fit.png")
        if getattr(self.cfg, "organize_outputs", True):
            self.log("Detailed tables: tables/detailed/")
            self.log("Fit overlays: figures/per_trace_fits/, figures/overlays/, figures/species/")
        if self.cfg.fit_michaelis_menten:
            self.log("Michaelis-Menten post-fit analysis was attempted when direct or reversible turnover velocities were available.")

        self.log("")
        self.log("Interpretation tip:")
        self.log("  model_type == no_binding means no meaningful concentration-dependent change was detected.")
        self.log("  model_type == linear_only means that trace is best treated as an approximately linear change.")
        self.log("  fit_mode == slow_apo with very small k1 supports a long-lived apo/free-enzyme population.")
        self.log("  reversible_mm rows report kon, koff, kcat, Km, Vmax from kcat*E0, and response coefficients for E + S <-> ES -> P.")
        self.log("  michaelis_menten_fit_summary.csv and michaelis_menten_fit.png report apparent Vmax, Km, and kcat when enough turnover velocities are available.")

        return self.cfg.output_dir


##############################################################################
# TKINTER GUI HELPERS
##############################################################################


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget, *args: Any, **kwargs: Any) -> None:
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)

        self.content.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            pass


##############################################################################
# GUI APPLICATION
##############################################################################


class StoppedFlowApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ANCHOR Time-Resolved Kinetic Fitting")
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)

        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.last_output_dir: Optional[str] = None

        self.advanced_entries: Dict[str, tk.Variable] = {}
        self.advanced_widgets: List[tk.Widget] = []
        self.concentration_rows: List[Dict[str, Any]] = []

        self._setup_style()
        self._create_variables(DEFAULT_CONFIG)
        self._build_layout()
        self._refresh_concentration_rows()
        self._toggle_advanced_state()
        self._drain_log_queue()

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        default_font = ("Helvetica", 11)
        heading_font = ("Helvetica", 15, "bold")
        subheading_font = ("Helvetica", 12, "bold")

        style.configure("TFrame", padding=4)
        style.configure("TLabel", font=default_font)
        style.configure("TButton", font=default_font, padding=6)
        style.configure("TCheckbutton", font=default_font)
        style.configure("TEntry", font=default_font)
        style.configure("Header.TLabel", font=heading_font)
        style.configure("Subheader.TLabel", font=subheading_font)
        style.configure("Hint.TLabel", font=("Helvetica", 10), foreground="#555555")
        style.configure("Danger.TButton", foreground="#9b1c1c")
        style.configure("Run.TButton", font=("Helvetica", 12, "bold"), padding=8)

    def _create_variables(self, cfg: FittingConfig) -> None:
        self.input_csv_var = tk.StringVar(value=cfg.input_csv)
        self.output_parent_var = tk.StringVar(value=cfg.output_parent)
        self.output_folder_name_var = tk.StringVar(value=cfg.output_folder_name)
        self.input_time_unit_var = tk.StringVar(value=getattr(cfg, "input_time_unit", "minutes"))
        self.E0_unit_var = tk.StringVar(value=getattr(cfg, "enzyme_concentration_unit", "µM"))
        self.substrate_unit_var = tk.StringVar(value=getattr(cfg, "substrate_concentration_unit", "µM"))
        self._last_E0_unit = self.E0_unit_var.get()
        self._last_substrate_unit = self.substrate_unit_var.get()
        self.E0_uM_var = tk.StringVar(
            value=str(cfg.concentration_from_uM(cfg.E0_uM, self.E0_unit_var.get()))
        )
        self.num_concentrations_var = tk.IntVar(value=len(cfg.concentration_groups))
        self.use_default_params_var = tk.BooleanVar(value=True)
        self.analysis_mode_var = tk.StringVar(value=cfg.analysis_mode)
        self.fit_michaelis_menten_var = tk.BooleanVar(value=bool(cfg.fit_michaelis_menten))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready")
        self.expected_traces_var = tk.StringVar(value="Expected trace columns: 18")
        self.output_preview_var = tk.StringVar(value="")

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        title = ttk.Label(container, text="ANCHOR Fitting Assistant", style="Header.TLabel")
        title.pack(anchor="w")

        subtitle = ttk.Label(
            container,
            text=(
                "Choose your CSV, define time and concentration units, select an analysis mode, "
                "or unlock custom parameters for full control."
            ),
            style="Hint.TLabel",
        )
        subtitle.pack(anchor="w", pady=(0, 8))

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)

        self.data_tab = ttk.Frame(self.notebook)
        self.params_tab = ttk.Frame(self.notebook)
        self.run_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.data_tab, text="1. Data setup")
        self.notebook.add(self.params_tab, text="2. Fitting parameters")
        self.notebook.add(self.run_tab, text="3. Run")

        self._build_data_tab()
        self._build_params_tab()
        self._build_run_tab()

    def _build_data_tab(self) -> None:
        frame = ttk.Frame(self.data_tab)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Input and output", style="Subheader.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        self._add_file_row(
            frame,
            row=1,
            label="CSV data file",
            variable=self.input_csv_var,
            browse_command=self._browse_input_csv,
        )

        self._add_file_row(
            frame,
            row=2,
            label="Output parent folder",
            variable=self.output_parent_var,
            browse_command=self._browse_output_parent,
        )

        ttk.Label(frame, text="Output folder name").grid(row=3, column=0, sticky="w", pady=5)
        output_name_entry = ttk.Entry(frame, textvariable=self.output_folder_name_var)
        output_name_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Preview", command=self._update_output_preview).grid(row=3, column=2, sticky="ew", pady=5)

        self.output_preview_var.set(self._output_preview_text())
        ttk.Label(frame, textvariable=self.output_preview_var, style="Hint.TLabel").grid(
            row=4, column=1, columnspan=2, sticky="w", padx=5, pady=(0, 12)
        )

        ttk.Separator(frame).grid(row=5, column=0, columnspan=3, sticky="ew", pady=12)

        ttk.Label(frame, text="Experiment design", style="Subheader.TLabel").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        ttk.Label(frame, text="CSV time unit").grid(row=7, column=0, sticky="w", pady=5)
        time_combo = ttk.Combobox(
            frame,
            textvariable=self.input_time_unit_var,
            values=GUI_TIME_UNITS,
            state="readonly",
            width=12,
        )
        time_combo.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(
            frame,
            text="Input time is converted internally to minutes; fitted rate constants remain in min-based units.",
            style="Hint.TLabel",
            wraplength=650,
        ).grid(row=7, column=2, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Initial enzyme E0").grid(row=8, column=0, sticky="w", pady=5)
        e0_frame = ttk.Frame(frame)
        e0_frame.grid(row=8, column=1, sticky="w", padx=5, pady=5)
        ttk.Entry(e0_frame, textvariable=self.E0_uM_var, width=14).pack(side="left")
        e0_unit_combo = ttk.Combobox(
            e0_frame,
            textvariable=self.E0_unit_var,
            values=GUI_CONCENTRATION_UNITS,
            state="readonly",
            width=7,
        )
        e0_unit_combo.pack(side="left", padx=(6, 0))
        e0_unit_combo.bind("<<ComboboxSelected>>", self._on_e0_unit_changed)

        ttk.Label(frame, text="Substrate/reactant concentration unit").grid(row=9, column=0, sticky="w", pady=5)
        substrate_unit_combo = ttk.Combobox(
            frame,
            textvariable=self.substrate_unit_var,
            values=GUI_CONCENTRATION_UNITS,
            state="readonly",
            width=10,
        )
        substrate_unit_combo.grid(row=9, column=1, sticky="w", padx=5, pady=5)
        substrate_unit_combo.bind("<<ComboboxSelected>>", self._on_substrate_unit_changed)
        ttk.Label(
            frame,
            text="nM, µM, mM, and M are accepted; ANCHOR converts concentrations internally to µM/mM.",
            style="Hint.TLabel",
            wraplength=650,
        ).grid(row=9, column=2, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Number of concentrations").grid(row=10, column=0, sticky="w", pady=5)
        concentration_spin = ttk.Spinbox(
            frame,
            from_=1,
            to=20,
            textvariable=self.num_concentrations_var,
            width=8,
            command=self._refresh_concentration_rows,
        )
        concentration_spin.grid(row=10, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(frame, text="Apply count", command=self._refresh_concentration_rows).grid(
            row=10, column=2, sticky="ew", pady=5
        )

        self.concentration_assignment_hint_var = tk.StringVar()
        self._update_concentration_hint()
        ttk.Label(
            frame,
            textvariable=self.concentration_assignment_hint_var,
            style="Hint.TLabel",
            wraplength=850,
        ).grid(row=11, column=0, columnspan=3, sticky="w", pady=(4, 8))

        self.conc_container = ttk.Frame(frame)
        self.conc_container.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.conc_container.columnconfigure(1, weight=1)

        ttk.Label(frame, textvariable=self.expected_traces_var, style="Hint.TLabel").grid(
            row=13, column=0, columnspan=3, sticky="w", pady=(4, 12)
        )

        ttk.Button(frame, text="Next: fitting parameters", command=lambda: self.notebook.select(self.params_tab)).grid(
            row=14, column=2, sticky="e", pady=18
        )

        for var in (self.output_parent_var, self.output_folder_name_var):
            var.trace_add("write", lambda *_: self._update_output_preview())

    def _add_file_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_command: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(parent, text="Browse...", command=browse_command).grid(row=row, column=2, sticky="ew", pady=5)

    def _build_params_tab(self) -> None:
        outer = ttk.Frame(self.params_tab)
        outer.pack(fill="both", expand=True, padx=8, pady=8)
        # Use a three-row grid so the navigation footer is always reserved and
        # remains visible even when a model selection changes the height of the
        # explanatory text or when the advanced-parameter area is long.
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky="ew")

        ttk.Label(top, text="Fitting behavior", style="Subheader.TLabel").pack(anchor="w")
        ttk.Label(
            top,
            text=(
                "The defaults are tuned for standard/slow-apo/linear/no-binding model selection. "
                "Uncheck the box below to edit advanced settings."
            ),
            style="Hint.TLabel",
            wraplength=950,
        ).pack(anchor="w", pady=(0, 8))

        mode_frame = ttk.LabelFrame(top, text="Recommended analysis mode")
        mode_frame.pack(fill="x", pady=(4, 10))
        mode_options = [
            ("General automatic model selection", "general_auto"),
            ("Auto-detect enzyme turnover: E + S -> P vs E + S <-> ES -> P", "turnover_auto"),
            ("Known direct turnover only: E + S -> P", "direct_only"),
            ("Known reversible MM only: E + S <-> ES -> P", "reversible_mm_only"),
            ("Custom advanced settings", "custom"),
        ]
        for label_text, value in mode_options:
            ttk.Radiobutton(
                mode_frame,
                text=label_text,
                variable=self.analysis_mode_var,
                value=value,
                command=self._on_analysis_mode_changed,
            ).pack(anchor="w", padx=6, pady=2)
        self.analysis_mode_hint_var = tk.StringVar(value=self._analysis_mode_hint())
        ttk.Label(mode_frame, textvariable=self.analysis_mode_hint_var, style="Hint.TLabel", wraplength=950).pack(
            anchor="w", padx=6, pady=(4, 6)
        )

        output_options_frame = ttk.LabelFrame(top, text="Output options")
        output_options_frame.pack(fill="x", pady=(4, 10))
        ttk.Checkbutton(
            output_options_frame,
            text="Run Michaelis-Menten post-fit analysis and output MM plot when turnover velocities are available",
            variable=self.fit_michaelis_menten_var,
        ).pack(anchor="w", padx=6, pady=(4, 2))
        ttk.Label(
            output_options_frame,
            text=(
                "Turn this off when you only want trace-level fits and do not want "
                "michaelis_menten_*.csv or michaelis_menten_fit.png outputs."
            ),
            style="Hint.TLabel",
            wraplength=950,
        ).pack(anchor="w", padx=6, pady=(0, 6))

        defaults_row = ttk.Frame(top)
        defaults_row.pack(fill="x", pady=(4, 10))
        ttk.Checkbutton(
            defaults_row,
            text="Use validated default numeric settings",
            variable=self.use_default_params_var,
            command=self._toggle_advanced_state,
        ).pack(side="left")
        ttk.Button(defaults_row, text="Restore fitting defaults", command=self._restore_advanced_defaults).pack(
            side="left", padx=12
        )

        scroll = ScrollableFrame(outer)
        scroll.grid(row=1, column=0, sticky="nsew")
        self.advanced_frame = scroll.content
        self.advanced_frame.columnconfigure(1, weight=1)
        self.advanced_frame.columnconfigure(3, weight=1)

        self._build_advanced_fields(self.advanced_frame)

        # Keep the Step 2 navigation outside the scrollable region and pin it to
        # the bottom of the tab. This prevents the Next button from disappearing
        # beneath the advanced settings on smaller displays or after changing
        # analysis modes.
        bottom = ttk.Frame(outer)
        bottom.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(bottom, text="Back", command=lambda: self.notebook.select(self.data_tab)).pack(side="left")
        ttk.Button(bottom, text="Next: run", command=lambda: self.notebook.select(self.run_tab)).pack(side="right")

    def _build_run_tab(self) -> None:
        frame = ttk.Frame(self.run_tab)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text="Run fitting", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text=(
                "Save or load a reusable setup, then run. The interface stays responsive "
                "while the fitting runs in a worker thread."
            ),
            style="Hint.TLabel",
            wraplength=950,
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, sticky="ew", pady=(4, 8))

        self.run_button = ttk.Button(actions, text="Run fitting", style="Run.TButton", command=self._start_run)
        self.run_button.pack(side="left")

        self.stop_button = ttk.Button(actions, text="Stop after current fit", style="Danger.TButton", command=self._request_stop)
        self.stop_button.pack(side="left", padx=8)
        self.stop_button.configure(state="disabled")

        ttk.Button(actions, text="Save setup JSON", command=self._save_config_dialog).pack(side="left", padx=8)
        ttk.Button(actions, text="Load setup JSON", command=self._load_config_dialog).pack(side="left")

        self.open_output_button = ttk.Button(actions, text="Open output folder", command=self._open_output_folder)
        self.open_output_button.pack(side="right")
        self.open_output_button.configure(state="disabled")

        status_row = ttk.Frame(frame)
        status_row.grid(row=3, column=0, sticky="ew", pady=(4, 4))
        status_row.columnconfigure(1, weight=1)
        ttk.Label(status_row, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_row, textvariable=self.status_var, style="Hint.TLabel").grid(row=0, column=1, sticky="w", padx=5)

        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        log_frame = ttk.LabelFrame(frame, text="Run log")
        log_frame.grid(row=5, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=18, wrap="word", font=("Menlo", 11))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self._append_log("Ready. Select your CSV and output folder, then run fitting.")

    def _build_advanced_fields(self, parent: ttk.Frame) -> None:
        self.advanced_entries.clear()
        self.advanced_widgets.clear()

        current_row = 0

        def add_heading(text: str) -> None:
            nonlocal current_row
            label = ttk.Label(parent, text=text, style="Subheader.TLabel")
            label.grid(row=current_row, column=0, columnspan=4, sticky="w", pady=(14, 5))
            self.advanced_widgets.append(label)
            current_row += 1

        def add_bool(attr: str, label_text: str) -> None:
            nonlocal current_row
            var = tk.BooleanVar(value=getattr(DEFAULT_CONFIG, attr))
            widget = ttk.Checkbutton(parent, text=label_text, variable=var)
            widget.grid(row=current_row, column=0, columnspan=4, sticky="w", pady=3)
            self.advanced_entries[attr] = var
            self.advanced_widgets.append(widget)
            current_row += 1

        def add_numeric(attr: str, label_text: str, width: int = 15) -> None:
            nonlocal current_row
            col = 0 if current_row % 2 == 0 else 2
            # Keep each numeric setting on its own row on smaller displays for readability.
            col = 0
            label = ttk.Label(parent, text=label_text)
            label.grid(row=current_row, column=col, sticky="w", pady=3)
            var = tk.StringVar(value=str(getattr(DEFAULT_CONFIG, attr)))
            entry = ttk.Entry(parent, textvariable=var, width=width)
            entry.grid(row=current_row, column=col + 1, sticky="w", padx=8, pady=3)
            self.advanced_entries[attr] = var
            self.advanced_widgets.extend([label, entry])
            current_row += 1

        add_heading("Model selection")
        add_numeric("max_intermediates", "Maximum intermediates")
        add_bool("auto_detect_turnover_mechanism", "Auto-detect enzyme turnover mechanism")
        add_bool("include_zero_intermediate_kinetic_model", "Include 0-intermediate kinetic model (E + S -> P)")
        add_bool("fit_zero_intermediate_only", "Fit ONLY direct 0-intermediate model (E + S -> P)")
        add_bool("loosen_zero_intermediate_bounds", "Loosen direct-model bounds for linear-like traces")
        add_bool("include_reversible_mm_model", "Include reversible MM model (E + S <-> ES -> P)")
        add_bool("fit_reversible_mm_only", "Fit ONLY reversible MM model (E + S <-> ES -> P)")
        add_bool("loosen_reversible_mm_bounds", "Loosen reversible-MM bounds for product-formation traces")
        add_bool("use_standard_fit_mode", "Include standard kinetic mode")
        add_bool("use_slow_apo_fit_mode", "Include slow-apo kinetic mode")
        add_bool("include_linear_only_model", "Include linear-only diagnostic model")
        add_bool("include_no_binding_model", "Include no-binding model")
        add_numeric("n_starts_screen", "Screening random starts")
        add_numeric("n_starts_refine", "Refinement random starts")
        add_numeric("selection_stride", "Screening downsample stride")
        add_numeric("random_seed", "Random seed")

        add_heading("Michaelis-Menten post-fit analysis")
        add_bool("mm_use_fitted_direct_velocity", "Use fitted direct-model velocity when direct model is selected")
        add_bool("mm_fit_group_means", "Fit Michaelis-Menten to concentration means")
        add_numeric("mm_min_points", "Minimum velocity points for Michaelis-Menten")

        add_heading("Output cleanup")
        add_bool("organize_outputs", "Organize outputs into summary/table/figure folders")

        add_heading("Fit window and normalization")
        add_numeric("fit_start_time", "Fit start time (min)")
        add_numeric("early_max_window_min", "Early max normalization window end (min)")

        add_heading("Smoothing and residual weighting")
        add_bool("use_smoothed_target", "Fit raw and smoothed target")
        add_numeric("raw_weight", "Raw residual weight")
        add_numeric("smooth_weight", "Smoothed residual weight")
        add_numeric("smooth_window_fraction", "Smoothing window fraction")
        add_numeric("smooth_polyorder", "Smoothing polynomial order")
        add_numeric("early_weight_strength", "Early-time weight strength")
        add_numeric("early_weight_tau", "Early-time weight tau (min)")

        add_heading("Baseline and linear drift")
        add_bool("fit_baseline_offset", "Fit baseline offset")
        add_bool("fit_baseline_slope", "Fit baseline slope")
        add_numeric("baseline_offset_min", "Baseline offset min")
        add_numeric("baseline_offset_max", "Baseline offset max")
        add_numeric("baseline_slope_min", "Baseline slope min")
        add_numeric("baseline_slope_max", "Baseline slope max")
        add_bool("initialize_baseline_slope_from_data", "Initialize baseline slope from observed trend")
        add_numeric("linear_intercept_min", "Linear-only intercept min")
        add_numeric("linear_intercept_max", "Linear-only intercept max")
        add_numeric("linear_slope_min", "Linear-only slope min")
        add_numeric("linear_slope_max", "Linear-only slope max")
        add_numeric("no_binding_intercept_min", "No-binding intercept min")
        add_numeric("no_binding_intercept_max", "No-binding intercept max")
        add_numeric("no_binding_bic_preference_tolerance", "No-binding BIC preference tolerance")
        add_numeric("no_binding_noise_multiplier", "No-binding noise multiplier")
        add_numeric("no_binding_min_signal_change", "No-binding minimum signal change")
        add_bool("no_binding_prefer_if_change_within_noise", "Prefer no-binding when best-model change is within noise")

        add_heading("Shape constraints")
        add_bool("integrate_from_true_time_zero", "Integrate from true time zero")
        add_bool("fix_free_enzyme_response", "Fix free enzyme response")
        add_numeric("free_enzyme_response", "Free enzyme response")
        add_bool("first_intermediate_must_not_exceed_E", "Prevent first intermediate from exceeding E")
        add_numeric("first_intermediate_upper", "First intermediate upper response")
        add_bool("prevent_initial_peak", "Prevent artificial early peak")
        add_bool("auto_disable_initial_peak_penalty_for_increasing_traces", "Auto-disable early-peak penalty for increasing traces")
        add_numeric("initial_peak_window_min", "Initial peak window (min)")
        add_numeric("initial_peak_tolerance", "Initial peak tolerance")
        add_numeric("initial_peak_penalty_weight", "Initial peak penalty weight")
        add_numeric("start_value_penalty_weight", "Start-value penalty weight")

        add_heading("Rate and response bounds")
        add_numeric("standard_k1_min", "Standard k1 min")
        add_numeric("standard_k1_max", "Standard k1 max")
        add_numeric("standard_k1_initial_guess", "Standard k1 initial guess")
        add_numeric("slow_apo_k1_min", "Slow-apo k1 min")
        add_numeric("slow_apo_k1_max", "Slow-apo k1 max")
        add_numeric("slow_apo_k1_initial_guess", "Slow-apo k1 initial guess")
        add_numeric("k_other_min", "Other rate min")
        add_numeric("k_other_max", "Other rate max")
        add_numeric("k_other_initial_guess", "Other rate initial guess")
        add_numeric("response_min", "Species response min")
        add_numeric("response_max", "Species response max")
        add_numeric("mm_kon_min", "MM kon min")
        add_numeric("mm_kon_max", "MM kon max")
        add_numeric("mm_kon_initial_guess", "MM kon initial guess")
        add_numeric("mm_koff_min", "MM koff min")
        add_numeric("mm_koff_max", "MM koff max")
        add_numeric("mm_koff_initial_guess", "MM koff initial guess")
        add_numeric("mm_kcat_min", "MM kcat min")
        add_numeric("mm_kcat_max", "MM kcat max")
        add_numeric("mm_kcat_initial_guess", "MM kcat initial guess")

    def _browse_input_csv(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select time-resolved kinetics CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filename:
            self.input_csv_var.set(filename)

    def _browse_output_parent(self) -> None:
        folder = filedialog.askdirectory(title="Select output parent folder")
        if folder:
            self.output_parent_var.set(folder)
            self._update_output_preview()

    def _output_preview_text(self) -> str:
        output_parent = self.output_parent_var.get().strip() or str(Path.home())
        output_name = self.output_folder_name_var.get().strip() or "ANCHOR_Fitting_Output"
        return f"Full output path: {Path(output_parent).expanduser() / output_name}"

    def _update_output_preview(self) -> None:
        self.output_preview_var.set(self._output_preview_text())

    def _update_concentration_hint(self) -> None:
        if not hasattr(self, "concentration_assignment_hint_var"):
            return
        unit = self.substrate_unit_var.get() or "µM"
        self.concentration_assignment_hint_var.set(
            "Trace columns are assigned in order. Example: if the first concentration has 9 samples "
            f"and the second has 9 samples, columns 2-10 use the first {unit} value and columns "
            f"11-19 use the second {unit} value."
        )

    @staticmethod
    def _format_unit_value(value: float) -> str:
        if not np.isfinite(value):
            return str(value)
        return f"{value:.12g}"

    def _on_e0_unit_changed(self, _event: Optional[tk.Event] = None) -> None:
        new_unit = self.E0_unit_var.get() or "µM"
        old_unit = getattr(self, "_last_E0_unit", new_unit)
        if new_unit == old_unit:
            return
        try:
            value = float(self.E0_uM_var.get())
            value_uM = FittingConfig.concentration_to_uM(value, old_unit)
            converted = FittingConfig.concentration_from_uM(value_uM, new_unit)
            self.E0_uM_var.set(self._format_unit_value(converted))
        except (ValueError, TypeError):
            pass
        self._last_E0_unit = new_unit

    def _on_substrate_unit_changed(self, _event: Optional[tk.Event] = None) -> None:
        new_unit = self.substrate_unit_var.get() or "µM"
        old_unit = getattr(self, "_last_substrate_unit", new_unit)
        if new_unit != old_unit:
            for row_info in self.concentration_rows:
                try:
                    value = float(row_info["conc_var"].get())
                    value_uM = FittingConfig.concentration_to_uM(value, old_unit)
                    converted = FittingConfig.concentration_from_uM(value_uM, new_unit)
                    row_info["conc_var"].set(self._format_unit_value(converted))
                except (ValueError, TypeError):
                    pass
        self._last_substrate_unit = new_unit
        self._refresh_concentration_rows()
        self._update_concentration_hint()

    def _refresh_concentration_rows(self) -> None:
        try:
            n = int(self.num_concentrations_var.get())
        except (tk.TclError, ValueError):
            n = 1
        n = max(1, min(n, 20))
        self.num_concentrations_var.set(n)

        old_values: List[Tuple[str, str]] = []
        for row_info in self.concentration_rows:
            old_values.append((row_info["conc_var"].get(), row_info["sample_var"].get()))

        for child in self.conc_container.winfo_children():
            child.destroy()

        self.concentration_rows = []
        unit = self.substrate_unit_var.get() or "µM"

        ttk.Label(self.conc_container, text="#", style="Subheader.TLabel").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(self.conc_container, text="Concentration", style="Subheader.TLabel").grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Label(self.conc_container, text="Number of samples/traces", style="Subheader.TLabel").grid(
            row=0, column=2, sticky="w", padx=4
        )

        for idx in range(n):
            if idx < len(old_values):
                conc_value, sample_value = old_values[idx]
            elif idx < len(DEFAULT_CONFIG.concentration_groups):
                group = DEFAULT_CONFIG.concentration_groups[idx]
                display_value = FittingConfig.concentration_from_uM(group.concentration_uM, unit)
                conc_value, sample_value = self._format_unit_value(display_value), str(group.n_samples)
            else:
                default_uM = 50.0 * (idx + 1)
                display_value = FittingConfig.concentration_from_uM(default_uM, unit)
                conc_value, sample_value = self._format_unit_value(display_value), "1"

            conc_var = tk.StringVar(value=conc_value)
            sample_var = tk.StringVar(value=sample_value)
            conc_var.trace_add("write", lambda *_: self._update_expected_traces())
            sample_var.trace_add("write", lambda *_: self._update_expected_traces())

            ttk.Label(self.conc_container, text=str(idx + 1)).grid(row=idx + 1, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(self.conc_container, textvariable=conc_var, width=18).grid(
                row=idx + 1, column=1, sticky="w", padx=4, pady=3
            )
            ttk.Entry(self.conc_container, textvariable=sample_var, width=18).grid(
                row=idx + 1, column=2, sticky="w", padx=4, pady=3
            )

            self.concentration_rows.append({"conc_var": conc_var, "sample_var": sample_var})

        self._update_expected_traces()
        self._update_concentration_hint()

    def _update_expected_traces(self) -> None:
        total = 0
        valid = True
        for row_info in self.concentration_rows:
            try:
                total += int(float(row_info["sample_var"].get()))
            except ValueError:
                valid = False
        if valid:
            self.expected_traces_var.set(f"Expected trace columns in CSV after the time column: {total}")
        else:
            self.expected_traces_var.set("Expected trace columns: sample counts need numeric values")

    def _analysis_mode_hint(self) -> str:
        mode = self.analysis_mode_var.get()
        if mode == "turnover_auto":
            return (
                "Compares direct product formation and reversible Michaelis-Menten turnover automatically. "
                "Use this for enzyme turnover/progress-curve datasets across substrate concentrations."
            )
        if mode == "direct_only":
            return "Fits only E + S -> P and then fits velocities to Michaelis-Menten when multiple concentrations are present."
        if mode == "reversible_mm_only":
            return "Fits only E + S <-> ES -> P and reports kon, koff, kcat, Km, and MM velocity plots."
        if mode == "custom":
            return "Uses the detailed advanced switches exactly as configured below."
        return "General ANCHOR model selection for no-binding, linear, direct, and multistep time-resolved kinetic traces."

    def _on_analysis_mode_changed(self) -> None:
        if hasattr(self, "analysis_mode_hint_var"):
            self.analysis_mode_hint_var.set(self._analysis_mode_hint())
        if self.analysis_mode_var.get() == "custom":
            self.use_default_params_var.set(False)
        self._toggle_advanced_state()

    def _toggle_advanced_state(self) -> None:
        state = "disabled" if self.use_default_params_var.get() else "normal"
        for widget in self.advanced_widgets:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _restore_advanced_defaults(self) -> None:
        for attr, var in self.advanced_entries.items():
            default_value = getattr(DEFAULT_CONFIG, attr)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(default_value))
            else:
                var.set(str(default_value))
        self._toggle_advanced_state()

    def _parse_float(self, value: str, label: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric.") from exc

    def _parse_int(self, value: str, label: str) -> int:
        try:
            parsed = int(float(value))
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        return parsed

    def _build_config_from_ui(self) -> FittingConfig:
        cfg = FittingConfig()
        cfg.input_csv = self.input_csv_var.get().strip()
        cfg.output_parent = self.output_parent_var.get().strip()
        cfg.output_folder_name = self.output_folder_name_var.get().strip()
        cfg.input_time_unit = self.input_time_unit_var.get().strip().lower()
        cfg.enzyme_concentration_unit = FittingConfig.normalize_concentration_unit(self.E0_unit_var.get())
        cfg.substrate_concentration_unit = FittingConfig.normalize_concentration_unit(self.substrate_unit_var.get())
        E0_display = self._parse_float(self.E0_uM_var.get(), "Initial enzyme E0")
        cfg.E0_uM = FittingConfig.concentration_to_uM(E0_display, cfg.enzyme_concentration_unit)
        cfg.analysis_mode = self.analysis_mode_var.get()
        cfg.fit_michaelis_menten = bool(self.fit_michaelis_menten_var.get())

        groups: List[ConcentrationGroup] = []
        for idx, row_info in enumerate(self.concentration_rows, start=1):
            conc_display = self._parse_float(row_info["conc_var"].get(), f"Concentration {idx}")
            conc_uM = FittingConfig.concentration_to_uM(conc_display, cfg.substrate_concentration_unit)
            samples = self._parse_int(row_info["sample_var"].get(), f"Sample count {idx}")
            groups.append(ConcentrationGroup(conc_uM, samples))
        cfg.concentration_groups = groups

        if not self.use_default_params_var.get():
            int_fields = {
                "max_intermediates",
                "n_starts_screen",
                "n_starts_refine",
                "selection_stride",
                "random_seed",
                "smooth_polyorder",
                "mm_min_points",
            }

            for attr, var in self.advanced_entries.items():
                if isinstance(var, tk.BooleanVar):
                    setattr(cfg, attr, bool(var.get()))
                elif attr in int_fields:
                    setattr(cfg, attr, self._parse_int(var.get(), attr))
                else:
                    setattr(cfg, attr, self._parse_float(var.get(), attr))

        return cfg

    def _apply_config_to_ui(self, cfg: FittingConfig, use_default_params: bool = False) -> None:
        self.input_csv_var.set(cfg.input_csv)
        self.output_parent_var.set(cfg.output_parent)
        self.output_folder_name_var.set(cfg.output_folder_name)

        time_unit = str(getattr(cfg, "input_time_unit", "minutes") or "minutes").lower()
        e0_unit = FittingConfig.normalize_concentration_unit(
            getattr(cfg, "enzyme_concentration_unit", "µM")
        )
        substrate_unit = FittingConfig.normalize_concentration_unit(
            getattr(cfg, "substrate_concentration_unit", "µM")
        )
        if time_unit not in GUI_TIME_UNITS:
            time_unit = "minutes"
        if e0_unit not in GUI_CONCENTRATION_UNITS:
            e0_unit = "µM"
        if substrate_unit not in GUI_CONCENTRATION_UNITS:
            substrate_unit = "µM"

        self.input_time_unit_var.set(time_unit)
        self.E0_unit_var.set(e0_unit)
        self.substrate_unit_var.set(substrate_unit)
        self._last_E0_unit = e0_unit
        self._last_substrate_unit = substrate_unit
        self.E0_uM_var.set(
            self._format_unit_value(FittingConfig.concentration_from_uM(cfg.E0_uM, e0_unit))
        )

        self.analysis_mode_var.set(getattr(cfg, "analysis_mode", "general_auto"))
        if hasattr(self, "fit_michaelis_menten_var"):
            self.fit_michaelis_menten_var.set(bool(getattr(cfg, "fit_michaelis_menten", True)))
        if hasattr(self, "analysis_mode_hint_var"):
            self.analysis_mode_hint_var.set(self._analysis_mode_hint())

        self.num_concentrations_var.set(len(cfg.concentration_groups))

        for child in self.conc_container.winfo_children():
            child.destroy()
        self.concentration_rows = []

        ttk.Label(self.conc_container, text="#", style="Subheader.TLabel").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(self.conc_container, text="Concentration", style="Subheader.TLabel").grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Label(self.conc_container, text="Number of samples/traces", style="Subheader.TLabel").grid(
            row=0, column=2, sticky="w", padx=4
        )

        for idx, group in enumerate(cfg.concentration_groups):
            display_value = FittingConfig.concentration_from_uM(group.concentration_uM, substrate_unit)
            conc_var = tk.StringVar(value=self._format_unit_value(display_value))
            sample_var = tk.StringVar(value=str(group.n_samples))
            conc_var.trace_add("write", lambda *_: self._update_expected_traces())
            sample_var.trace_add("write", lambda *_: self._update_expected_traces())

            ttk.Label(self.conc_container, text=str(idx + 1)).grid(row=idx + 1, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(self.conc_container, textvariable=conc_var, width=18).grid(
                row=idx + 1, column=1, sticky="w", padx=4, pady=3
            )
            ttk.Entry(self.conc_container, textvariable=sample_var, width=18).grid(
                row=idx + 1, column=2, sticky="w", padx=4, pady=3
            )
            self.concentration_rows.append({"conc_var": conc_var, "sample_var": sample_var})

        for attr, var in self.advanced_entries.items():
            value = getattr(cfg, attr, getattr(DEFAULT_CONFIG, attr))
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value))

        self.use_default_params_var.set(bool(use_default_params))
        self._update_expected_traces()
        self._update_concentration_hint()
        self._update_output_preview()
        self._toggle_advanced_state()

    def _save_config_dialog(self) -> None:
        try:
            cfg = self._build_config_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid setup", str(exc))
            return

        data = cfg.to_dict()
        data["use_default_fitting_parameters"] = bool(self.use_default_params_var.get())

        filename = filedialog.asksaveasfilename(
            title="Save fitting setup",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        self._append_log(f"Saved setup JSON: {filename}")

    def _load_config_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            title="Load fitting setup",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        try:
            with open(filename, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            use_default_params = bool(data.pop("use_default_fitting_parameters", False))
            cfg = FittingConfig.from_dict(data)
        except Exception as exc:
            messagebox.showerror("Could not load setup", str(exc))
            return

        self._apply_config_to_ui(cfg, use_default_params=use_default_params)
        self._append_log(f"Loaded setup JSON: {filename}")

    def _start_run(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Already running", "A fitting job is already running.")
            return

        try:
            cfg = self._build_config_from_ui()
            StoppedFlowFitter(cfg).validate_config()
        except ValueError as exc:
            messagebox.showerror("Invalid setup", str(exc))
            return

        output_path = Path(cfg.output_dir)
        if output_path.exists():
            if output_path.is_dir():
                try:
                    existing_count = sum(1 for _ in output_path.iterdir())
                except OSError:
                    existing_count = None

                detail = (
                    f"The output folder already exists:\n\n{output_path}\n\n"
                    "Running ANCHOR with this folder may overwrite files from a previous analysis.\n"
                )
                if existing_count is not None:
                    detail += f"Existing items in folder: {existing_count}\n\n"
                else:
                    detail += "The existing folder could not be inspected.\n\n"
                detail += "Proceed with this output folder?"

                proceed = messagebox.askyesno(
                    "Output folder already exists",
                    detail,
                    icon="warning",
                    default="no",
                )
                if not proceed:
                    self.status_var.set("Ready")
                    self._append_log("Run cancelled before start: output folder already exists.")
                    return
            else:
                messagebox.showerror(
                    "Invalid output folder",
                    f"The selected output path already exists but is not a folder:\n\n{output_path}",
                )
                return

        self.cancel_event.clear()
        self.progress_var.set(0.0)
        self.progress_bar.configure(maximum=max(1, len(cfg.D_values_uM)))
        self.status_var.set("Running")
        self.last_output_dir = cfg.output_dir
        self.open_output_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.notebook.select(self.run_tab)
        self._append_log("")
        self._append_log("==============================")
        self._append_log("Starting a new fitting run")
        self._append_log("==============================")

        def log_callback(message: str) -> None:
            self.log_queue.put(("log", message))

        def progress_callback(done: int, total: int) -> None:
            self.log_queue.put(("progress", done, total))

        def worker() -> None:
            fitter = StoppedFlowFitter(
                cfg,
                log_callback=log_callback,
                progress_callback=progress_callback,
                cancel_event=self.cancel_event,
            )
            try:
                output_dir = fitter.run()
                self.log_queue.put(("done", output_dir))
            except CancelledError as exc:
                self.log_queue.put(("cancelled", str(exc)))
            except Exception:
                self.log_queue.put(("error", traceback.format_exc()))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _request_stop(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Stopping after current fit finishes...")
        self._append_log("Stop requested. The current optimizer call will finish, then the run will stop.")

    def _open_output_folder(self) -> None:
        folder = self.last_output_dir or self._build_config_from_ui().output_dir
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("Output folder", "The output folder does not exist yet.")
            return

        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", folder], check=False)
            elif os.name == "nt":
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", folder], check=False)
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc))

    def _append_log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                event_type = item[0]

                if event_type == "log":
                    self._append_log(item[1])
                elif event_type == "progress":
                    done, total = item[1], item[2]
                    self.progress_bar.configure(maximum=max(1, total))
                    self.progress_var.set(done)
                    self.status_var.set(f"Running: {done}/{total} traces complete")
                elif event_type == "done":
                    output_dir = item[1]
                    self.last_output_dir = output_dir
                    self.status_var.set("Complete")
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.open_output_button.configure(state="normal")
                    self._append_log(f"Output folder: {output_dir}")
                    messagebox.showinfo("Fitting complete", f"Fitting complete.\n\nOutput folder:\n{output_dir}")
                elif event_type == "cancelled":
                    self.status_var.set("Cancelled")
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.open_output_button.configure(state="normal")
                    self._append_log(item[1])
                    messagebox.showinfo("Fitting cancelled", item[1])
                elif event_type == "error":
                    self.status_var.set("Error")
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.open_output_button.configure(state="normal")
                    self._append_log("ERROR:")
                    self._append_log(item[1])
                    messagebox.showerror("Fitting error", item[1])
        except queue.Empty:
            pass

        self.root.after(150, self._drain_log_queue)


##############################################################################
# ENTRY POINT
##############################################################################


def main() -> None:
    root = tk.Tk()
    app = StoppedFlowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


