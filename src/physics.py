"""State-continuous PyBaMM SPM reference solver."""

from __future__ import annotations

import os
import time

os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")

import numpy as np
import pybamm


class Cell:
    """One built Chen2020 SPM with explicit current inputs."""

    def __init__(self, initial_soc: float, config: dict):
        self.config = config
        self.nodes = int(config["radial_nodes_per_particle"])
        self.parameters = pybamm.ParameterValues(config["parameter_set"])
        self.parameters.update(
            {
                "Current function [A]": "[input]",
                "Ambient temperature [K]": config["temperature_K"],
                "Initial temperature [K]": config["temperature_K"],
            }
        )
        self.solver = pybamm.IDAKLUSolver(
            rtol=config["rtol"],
            atol=config["atol"],
            options={"num_threads": 1},
        )
        axial = int(config["axial_nodes"])
        simulation = pybamm.Simulation(
            pybamm.lithium_ion.SPM({"thermal": "isothermal"}),
            parameter_values=self.parameters,
            solver=self.solver,
            var_pts={
                "r_n": self.nodes,
                "r_p": self.nodes,
                "x_n": axial,
                "x_s": axial,
                "x_p": axial,
            },
        )
        simulation.build(
            initial_soc=float(initial_soc),
            inputs={"Current function [A]": 0.0},
        )
        self.model = simulation.built_model
        soc_parameters = self.parameters.copy()
        soc_parameters.update({"Current function [A]": 0.0})
        self.stoich_limits = pybamm.lithium_ion.get_min_max_stoichiometries(soc_parameters)
        p = self.parameters
        self.capacity_Ah = float(
            96485.33212
            * p["Electrode width [m]"]
            * p["Electrode height [m]"]
            * p["Negative electrode thickness [m]"]
            * p["Negative electrode active material volume fraction"]
            * p["Maximum concentration in negative electrode [mol.m-3]"]
            * (self.stoich_limits[1] - self.stoich_limits[0])
            / 3600.0
        )

    def step(self, previous, current_A: float, duration_s: float, sample_interval_s=None):
        grid = None
        if sample_interval_s is not None:
            count = int(round(duration_s / sample_interval_s))
            grid = np.linspace(0.0, duration_s, count + 1)
        return self.solver.step(
            previous,
            self.model,
            float(duration_s),
            inputs={"Current function [A]": float(current_A)},
            save=False,
            t_interp=grid,
        )

    def run_history(self, segments):
        solution = None
        for current_A, duration_s in segments:
            if duration_s <= 0.0:
                continue
            solution = self.step(solution, current_A, duration_s)
            if solution.termination != "final time":
                raise RuntimeError(f"Preparation history terminated: {solution.termination}")
        return solution

    def concentration_state(self, solution) -> np.ndarray:
        state = []
        for electrode in ["negative", "positive"]:
            name = f"X-averaged {electrode} particle concentration [mol.m-3]"
            concentration = np.asarray(solution[name].entries)[:, -1].reshape(-1)
            maximum = self.parameters[f"Maximum concentration in {electrode} electrode [mol.m-3]"]
            state.append(concentration / float(maximum))
        result = np.stack(state)
        if result.shape != (2, self.nodes) or not np.isfinite(result).all():
            raise RuntimeError("Invalid concentration snapshot.")
        return result

    def soc_from_state(self, state: np.ndarray) -> float:
        edges = np.linspace(0.0, 1.0, self.nodes + 1)
        weights = np.diff(edges**3)
        negative_mean = float(state[0] @ weights)
        positive_mean = float(state[1] @ weights)
        x0, x100, y100, y0 = self.stoich_limits
        soc_negative = (negative_mean - x0) / (x100 - x0)
        soc_positive = (y0 - positive_mean) / (y0 - y100)
        if abs(soc_negative - soc_positive) > 2e-7:
            raise RuntimeError("Electrode SOC mismatch in a conserved SPM state.")
        return float(0.5 * (soc_negative + soc_positive))

    def _initial_event(self, previous, current_A: float):
        t = float(previous.t[-1])
        y = np.asarray(previous.y[:, -1]).reshape(-1, 1)
        inputs = {"Current function [A]": float(current_A)}
        for event in self.model.events:
            if event.event_type != pybamm.EventType.TERMINATION:
                continue
            value = np.asarray(event.expression.evaluate(t, y, inputs=inputs))
            if np.isfinite(value).all() and (value <= 0.0).any():
                return "initial: " + event.name
        return None

    def pulse_transition(self, previous, current_A: float):
        """Advance one fixed pulse without reinitializing the internal state."""
        duration = float(self.config["pulse_duration_s"])
        spacing = float(self.config["sample_interval_s"])
        initial_event = self._initial_event(previous, current_A)
        if initial_event is not None:
            return {
                "status": "physical_cutoff",
                "event": initial_event,
                "end_s": 0.0,
                "time_s": np.array([], dtype=float),
                "voltage_V": np.array([], dtype=float),
                "state1": None,
                "solution": None,
                "solve_s": 0.0,
            }
        start_time = float(previous.t[-1])
        started = time.perf_counter()
        try:
            solution = self.step(previous, current_A, duration, spacing)
        except pybamm.SolverError as error:
            return {
                "status": "solver_failure",
                "event": str(error),
                "end_s": None,
                "time_s": np.array([], dtype=float),
                "voltage_V": np.array([], dtype=float),
                "state1": None,
                "solution": None,
                "solve_s": time.perf_counter() - started,
            }
        solve_s = time.perf_counter() - started
        time_s = np.asarray(solution.t, dtype=float) - start_time
        time_s[0] = 0.0
        voltage = np.asarray(solution["Terminal voltage [V]"].entries, dtype=float).reshape(-1)
        complete = solution.termination == "final time" and np.isclose(time_s[-1], duration, atol=1e-7)
        if not complete and not solution.termination.startswith("event:"):
            raise RuntimeError(f"Unexpected pulse termination: {solution.termination}")
        if not np.isfinite(voltage).all() or np.any(np.diff(time_s) <= 0.0):
            raise RuntimeError("Invalid pulse reference values.")
        if complete:
            expected = int(round(duration / spacing)) + 1
            if len(time_s) != expected:
                raise RuntimeError(f"Complete pulse has {len(time_s)} samples; expected {expected}.")
        return {
            "status": "complete" if complete else "physical_cutoff",
            "event": solution.termination,
            "end_s": float(time_s[-1]),
            "time_s": time_s,
            "voltage_V": voltage,
            "state1": self.concentration_state(solution) if complete else None,
            "solution": solution,
            "solve_s": float(solve_s),
        }
