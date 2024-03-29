"""This module contains the Model class, which is the primary point of access to `cubewalkers`."""

from __future__ import annotations

import random

# for generating model names when the user doesn't want to specify them
import string
import warnings
from typing import TYPE_CHECKING, Iterable

import cupy as cp  # type: ignore

from cubewalkers import initial_conditions, parser, simulation

# for default update scheme
from cubewalkers.update_schemes import asynchronous, synchronous

if TYPE_CHECKING:
    from cubewalkers.custom_typing import MaskFunctionType
    from cubewalkers.experiment import Experiment


class Model:
    """
    Stores a Boolean network and experimental conditions, as well as generates
    and stores the results of simulations on that network.
    """

    _automatic_model_name_length = 16
    _automatic_model_names_taken: set[str] = set()

    def __init__(
        self,
        rules: str | None = None,
        lookup_tables: cp.typing.NDArray | None = None,
        node_regulators: Iterable[Iterable[int]] | None = None,
        lookup_table_varnames: Iterable[str] | None = None,
        initial_biases: str = "",
        model_name: str | None = None,
        experiment: Experiment | None = None,
        comment_char: str = "#",
        n_time_steps: int = 1,
        n_walkers: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        rules : str, optional
            Rules to input. If skip_clean is `True` (not default), then these
            are assumed to have been cleaned. If not provided, then
            `lookup_tables` and `node_regulators` must both be provided instead.
        lookup_tables : cp.NDArray, optional
            A merged lookup table that contains the output column of each rule's
            lookup table (padded by `False` values). If not provided, rules must
            be provided instead.
        node_regulators : Iterable[Iterable[int]]
            Iterable i should contain the indicies of the nodes that regulate
            node i, optionally padded by negative values. If not provided, rules
            must be provided instead.
        lookup_table_varnames: Iterable[str], optional
            Iterable of variable names to associate with lookup table entries.
            If `None` (default) dummy variable names of the form 'x0', 'x1',
            etc., will be constructed.
        initial_biases : str
            Each line should be of the form - NodeName,bias where NodeName is
            the name of the node, and bias is the probability that the node will
            be initialized to `1` (instead of 0). Nodes whose names are not listed
            are given a bias of 0.5 by default.
        model_name : str
            A name for the kernel.
        experiment : Experiment | None, optional
            An :class:`Experiment<cubewalkers.experiment.Experiment>` object
            specifying experimental conditions, by default None, in which case
            no experimental conditions are incorporated into the rules.
            Currently not implemented for lookup-table-based Boolean networks.
        comment_char : str, optional
            In rules, empty lines and lines beginning with this character are
            ignored, by default `'#'`.
        n_time_steps: int, optional
            Number of timesteps to simulate, by default `1`.
        n_walkers: int, optional
            Number of ensemble walkers to simulate, by default `1`.
        """
        self.name = ""
        if model_name is None:
            g = None
            while g is None or g in Model._automatic_model_names_taken:
                g = "".join(
                    random.choices(
                        string.ascii_letters, k=Model._automatic_model_name_length
                    )
                )
            self.name = g
            Model._automatic_model_names_taken.add(self.name)
        else:
            self.name = str(model_name)

        self.raw_rules = rules  # uncleaned, comments retained, etc.
        self.rules = rules
        self.node_regulators = node_regulators
        self.lookup_tables = lookup_tables
        if rules is not None:
            self.mode = "algebraic"
            if (
                lookup_tables is not None
                or node_regulators is not None
                or lookup_table_varnames is not None
            ):
                warnings.warn(
                    "Lookup table data and rule data were both provided. "
                    "Lookup table data will be ignored in favor of rules."
                )

            self.rules = parser.clean_rules(rules, comment_char=comment_char)
            self.kernel, self.varnames, self.code = parser.bnet2rawkernel(
                self.rules, self.name, experiment=experiment, skip_clean=True
            )
        else:
            self.mode = "tabular"
            if self.node_regulators is None or lookup_tables is None:
                raise TypeError(
                    "Must specify node_regulators and lookup_tables if rules are not provided."
                )
            self.kernel, self.code = parser.regulators2lutkernel(
                self.node_regulators, self.name
            )
            if lookup_table_varnames is None:
                self.varnames = [f"x{i}" for i in range(len(lookup_tables))]
            else:
                self.varnames = list(lookup_table_varnames)

        self.vardict = {k: i for i, k in enumerate(self.varnames)}

        self.n_time_steps = n_time_steps
        self.n_walkers = n_walkers
        self.n_variables = len(self.varnames)
        self.comment_char = comment_char
        self.initial_biases = initial_biases
        self.initialize_walkers()

    def initialize_walkers(self) -> None:
        """
        Generates initial conditions from internally stored data. See the
        `initial_conditions` module for details.
        """
        self.initial_states = initial_conditions.initial_walker_states(
            self.initial_biases,
            self.vardict,
            self.n_walkers,
            comment_char=self.comment_char,
        )

    def simulate_ensemble(
        self,
        T_window: int | None = None,
        averages_only: bool = False,
        maskfunction: MaskFunctionType = synchronous,
        threads_per_block: tuple[int, int] = (32, 32),
        set_update_prob: float = 0.5,
    ) -> None:
        """
        Simulates a random ensemble of walkers on the internally stored Boolean
        network. Results are stored in the trajectories attribute.

        Parameters
        ----------
        T_window : int, optional
            Number of time points to keep (from t=T-T_window+1 to t=T). If
            `None` (default), keep all time points.
        averages_only : bool, optional
            If True, stores only average node values at each timestep.
            Otherwise, stores node values for each walker. By default `False`.
        maskfunction : MaskFunctionType, optional
            Function that returns a mask for selecting which node values to
            update. By default, uses the synchronous update scheme. See
            :mod:`cubewalkers.update_schemes` for examples.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for details.
        """

        if self.n_walkers != self.initial_states.shape[1]:
            self.initialize_walkers()

        self.averages_only = averages_only

        self.trajectories = simulation.simulate_ensemble(
            self.kernel,
            self.n_variables,
            self.n_time_steps,
            self.n_walkers,
            T_window=T_window,
            lookup_tables=self.lookup_tables,
            initial_states=self.initial_states,
            averages_only=averages_only,
            maskfunction=maskfunction,
            threads_per_block=threads_per_block,
            set_update_prob=set_update_prob,
        )

    def trajectory_variance(
        self,
        initial_state: cp.NDarray[cp.bool_],
        n_time_steps: int | None = None,
        n_walkers: int | None = None,
        maskfunction: MaskFunctionType = asynchronous,
        threads_per_block: tuple[int, int] = (32, 32),
    ) -> cp.NDarray:
        """
        Returns the variance of trajectories that begin at the specified initial
        state. Note that the covariances are not, in general, zero.

        Parameters
        ----------
        initial_state : cp.NDArray[cp.bool]
            The initial state to use. Will cast to `cupy.bool` if other dtype is
            provided.
        n_time_steps : int | None, optional
            Number of timesteps to simulate. By default, use internally stored
            variable `n_time_steps`, which itself defaults to `1`.
        n_walkers : int | None, optional
            How many walkers to use to estimate the impact. By default, use
            internally stored variable `n_walkers`, which itself defaults to `1`.
        maskfunction : MaskFunctionType, optional
            Function that returns a mask for selecting which node values to
            update. By default, uses the synchronous update scheme. See
            :mod:`cubewalkers.update_schemes` for examples.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for details.

        Returns
        -------
        cp.NDArray
            Variance of trajectories
        """

        if n_time_steps is None:
            n_time_steps = self.n_time_steps
        if n_walkers is None:
            n_walkers = self.n_walkers

        walkers_initial_state: cp.NDArray = cp.array(  # type: ignore
            [cp.bool_(initial_state) for _ in range(n_walkers)]
        ).T

        avgs = simulation.simulate_ensemble(
            self.kernel,
            self.n_variables,
            n_time_steps,
            n_walkers,
            lookup_tables=self.lookup_tables,
            initial_states=walkers_initial_state,
            averages_only=True,
            maskfunction=maskfunction,
            threads_per_block=threads_per_block,
        )

        # avgs[i] is P(node_i=1), and
        # variance of Bernoulli distribution is p*(1-p),
        # so avgs * (1-avgs) is the variance of each node
        return avgs * (1 - avgs)

    def dynamical_impact(
        self,
        source_var: str | list[str],
        n_time_steps: int | None = None,
        n_walkers: int | None = None,
        maskfunction: MaskFunctionType = synchronous,
        threads_per_block: tuple[int, int] = (32, 32),
    ) -> cp.NDArray:
        """
        Computes the dynamical impact of the source node index on all others
        (including itself, from time=0 to time=T).

        Parameters
        ----------
        source_var : str | list[str]
            Name(s) of variable(s) to find dynamical impact of.
        n_time_steps : int | None, optional
            Number of timesteps to simulate. By default, use internally stored
            variable `n_time_steps`, which itself defaults to `1`.
        n_walkers : int | None, optional
            How many walkers to use to estimate the impact. By default, use
            internally stored variable `n_walkers`, which itself defaults to `1`.
        maskfunction : MaskFunctionType, optional
            Function that returns a mask for selecting which node values to
            update. By default, uses the synchronous update scheme. See
            :mod:`cubewalkers.update_schemes` for examples. For dynamical
            impact, if the maskfunction is state-dependent, then the unperturbed
            trajectory is used.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for details.

        Returns
        -------
        cp.NDArray
            `(n_time_steps+1)` x `n_variables` array of dynamical impacts of the
            source at each time. Refer to `vardict` member variable to see
            ordering of variables. Note that the initial time impact is always
            maximal for the source node and minimal for all others.
        """
        if n_time_steps is None:
            n_time_steps = self.n_time_steps
        if n_walkers is None:
            n_walkers = self.n_walkers

        if isinstance(source_var, str):
            source: int | list[int] = self.vardict[source_var]
        else:
            source = [self.vardict[sv] for sv in source_var]

        return simulation.dynamical_impact(
            self.kernel,
            source,
            self.n_variables,
            n_time_steps,
            n_walkers,
            lookup_tables=self.lookup_tables,
            maskfunction=maskfunction,
            threads_per_block=threads_per_block,
        )

    def source_quasicoherence(
        self,
        source_var: str | list[str],
        n_time_steps: int | None = None,
        n_walkers: int | None = None,
        T_sample: int = 1,
        fuzzy_coherence: bool = False,
        maskfunction: MaskFunctionType = synchronous,
        threads_per_block: tuple[int, int] = (32, 32),
    ) -> cp.NDArray:
        """
        Computes the quasicoherence in response to perturbation of source node
        index, averaging trajectories from t=T-T_sample+1 to T.

        Parameters
        ----------
        source_var : str | list[str]
            Name(s) of variable(s) to find coherence with respect to.
        n_time_steps : int | None, optional
            Number of timesteps to simulate. By default, use internally stored
            variable `n_time_steps`, which itself defaults to `1`.
        n_walkers : int | None, optional
            How many walkers to use to estimate the coherence. By default, use
            internally stored variable `n_walkers`, which itself defaults to `1`.
        T_sample : int, optional
            Number of time points to use for averaging (t=T-T_sample+1 to t=T),
            by default, `1`.
        fuzzy_coherence : bool, optional
            If `False` (default), trajectroies are marked as either in agreement
            (`1`) or not in agreement (`0`) depending on whether fixed nodes are in
            agreement. If True, the average absolute difference between state
            vectors is used instead.
        maskfunction : MaskFunctionType, optional
            Function that returns a mask for selecting which node values to
            update. By default, uses the synchronous update scheme. See
            :mod:`cubewalkers.update_schemes` for examples. For coherence, if
            the maskfunction is state-dependent, then the unperturbed trajectory
            is used.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for details.

        Returns
        -------
        cp.NDArray
            The estimated value of the quasicoherence response to the source
            node perturbation.
        """
        if n_time_steps is None:
            n_time_steps = self.n_time_steps
        if n_walkers is None:
            n_walkers = self.n_walkers

        if isinstance(source_var, str):
            source: int | list[int] = self.vardict[source_var]
        else:
            source = [self.vardict[sv] for sv in source_var]

        trajU, trajP, _ = simulation.simulate_perturbation(
            self.kernel,
            source,
            self.n_variables,
            n_time_steps,
            n_walkers,
            T_sample=T_sample,
            lookup_tables=self.lookup_tables,
            maskfunction=maskfunction,
            threads_per_block=threads_per_block,
        )

        return simulation.source_quasicoherence(
            trajU, trajP, T_sample=T_sample, fuzzy_coherence=fuzzy_coherence
        )

    def quasicoherence(
        self,
        n_time_steps: int | None = None,
        n_walkers: int | None = None,
        T_sample: int = 1,
        fuzzy_coherence: bool = False,
        maskfunction: MaskFunctionType = synchronous,
        threads_per_block: tuple[int, int] = (32, 32),
    ) -> cp.NDArray:
        """
        Computes the quasicoherence in response to perturbation of single nodes,
        averaging trajectories from t=T-T_sample+1 to T.

        Parameters
        ----------
        n_time_steps : int | None, optional
            Number of timesteps to simulate. By default, use internally stored
            variable `n_time_steps`, which itself defaults to `1`.
        n_walkers : int | None, optional
            How many walkers to use to estimate the coherence. By default, use
            internally stored variable `n_walkers`, which itself defaults to
            `1`.
        T_sample : int, optional
            Number of time points to use for averaging (t=T-T_sample+1 to t=T),
            by default, `1`.
        fuzzy_coherence : bool, optional
            If `False` (default), trajectories are marked as either in agreement
            (`1`) or not in agreement (`0`) depending on whether fixed nodes are
            in agreement. If True, the average absolute difference between state
            vectors is used instead.
        maskfunction : MaskFunctionType, optional
            Function that returns a mask for selecting which node values to
            update. By default, uses the synchronous update scheme. See
            :mod:`cupewalkers.update_schemes` for examples. For coherence, if
            the maskfunction is state-dependent, then the unperturbed trajectory
            is used.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for
            details.

        Returns
        -------
        cp.NDArray
            The estimated value of the quasicoherence response to single node
            perturbations.
        """
        if n_time_steps is None:
            n_time_steps = self.n_time_steps
        if n_walkers is None:
            n_walkers = self.n_walkers

        c = 0
        for source_ind in range(self.n_variables):
            trajU, trajP, _ = simulation.simulate_perturbation(
                self.kernel,
                source_ind,
                self.n_variables,
                n_time_steps,
                n_walkers,
                T_sample=T_sample,
                lookup_tables=self.lookup_tables,
                maskfunction=maskfunction,
                threads_per_block=threads_per_block,
            )

            c += simulation.source_quasicoherence(
                trajU, trajP, T_sample=T_sample, fuzzy_coherence=fuzzy_coherence
            )

        return c / self.n_variables

    def source_final_hamming_distance(
        self,
        source_var: str | list[str],
        n_time_steps: int | None = None,
        n_walkers: int | None = None,
        T_sample: int = 1,
        maskfunction: MaskFunctionType = synchronous,
        threads_per_block: tuple[int, int] = (32, 32),
    ) -> cp.NDArray:
        """
        Computes the final hamming distance in response to perturbation of
        source node index, averaging hamming distances from t=T-T_sample+1 to T.

        Parameters
        ----------
        source_var : str | list[str]
            Name(s) of variable(s) to find the Hamming distance with respect to.
        n_time_steps : int | None, optional
            Number of timesteps to simulate. By default, use internally stored
            variable `n_time_steps`, which itself defaults to `1`.
        n_walkers : int | None, optional
            How many walkers to use to estimate the final Hamming distance. By
            default, use internally stored variable `n_walkers`, which itself
            defaults to `1`.
        T_sample : int, optional
            Number of time points to use for averaging (t=T-T_sample+1 to t=T),
            by default, `1`.
        maskfunction : MaskFunctionType, optional
            Function that returns a mask for selecting which node values to
            update. By default, uses the synchronous update scheme. See
            :mod:`cubewalkers.update_schemes` for examples.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for
            details.

        Returns
        -------
        cp.NDArray
            The estimated value of the final Hamming distance response to the
            source node perturbation.
        """
        if n_time_steps is None:
            n_time_steps = self.n_time_steps
        if n_walkers is None:
            n_walkers = self.n_walkers

        if isinstance(source_var, str):
            source: int | list[int] = self.vardict[source_var]
        else:
            source = [self.vardict[sv] for sv in source_var]

        _, _, diff = simulation.simulate_perturbation(
            self.kernel,
            source,
            self.n_variables,
            n_time_steps,
            n_walkers,
            T_sample=T_sample,
            lookup_tables=self.lookup_tables,
            maskfunction=maskfunction,
            threads_per_block=threads_per_block,
        )

        return simulation.source_final_hamming_distance(diff, T_sample=T_sample)

    def final_hamming_distance(
        self,
        n_time_steps: int | None = None,
        n_walkers: int | None = None,
        T_sample: int = 1,
        maskfunction: MaskFunctionType = synchronous,
        threads_per_block: tuple[int, int] = (32, 32),
    ) -> cp.NDArray:
        """
        Computes the final Hamming distance in response to perturbation of
        single nodes, averaging Hamming distances from t=T-T_sample+1 to T.

        Parameters
        ----------
        n_time_steps : int | None, optional
            Number of timesteps to simulate. By default, use internally stored
            variable `n_time_steps`, which itself defaults to `1`.
        n_walkers : int | None, optional
            How many walkers to use to estimate the final Hamming distance. By
            default, use internally stored variable `n_walkers`, which itself
            defaults to `1`.
        T_sample : int, optional
            Number of time points to use for averaging (t=T-T_sample+1 to t=T),
            by default, `1`.
        maskfunction : MaskFunctionType, optional
            Function that returns a mask for selecting which node values to
            update. By default, uses the synchronous update scheme. See
            :mod:`cubewalkers.update_schemes` for examples.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for
            details.

        Returns
        -------
        cp.NDArray
            The estimated value of the final Hamming distance response to single
            node perturbations.
        """
        if n_time_steps is None:
            n_time_steps = self.n_time_steps
        if n_walkers is None:
            n_walkers = self.n_walkers

        c = 0
        for source_ind in range(self.n_variables):
            _, _, diff = simulation.simulate_perturbation(
                self.kernel,
                source_ind,
                self.n_variables,
                n_time_steps,
                n_walkers,
                T_sample=T_sample,
                lookup_tables=self.lookup_tables,
                maskfunction=maskfunction,
                threads_per_block=threads_per_block,
            )

            c += simulation.source_final_hamming_distance(diff, T_sample=T_sample)

        return c / self.n_variables

    def derrida_coefficient(
        self,
        n_walkers: int | None = None,
        threads_per_block: tuple[int, int] = (32, 32),
    ) -> float:
        """
        Estimates the (synchronous) Derrida coefficient.

        The Derrida coefficent is computed as the mean Hamming distance after
        one synchronous update between trajectories with initial Hamming
        distance of one. For analogs using other update schemes, use
        :meth:`dynamical_impact`.

        Parameters
        ----------
        n_walkers : int | None, optional
            How many walkers to use to estimate the Coefficient. By default, use
            internally stored variable `n_walkers`, which itself defaults to
            `1`.
        threads_per_block : tuple[int, int], optional
            How many threads should be in each block for each dimension of the N
            x W array, by default `(32, 32)`. See CUDA documentation for
            details.

        Returns
        -------
        float
            Estimate of the Derrida coefficient.
        """
        if n_walkers is None:
            n_walkers = self.n_walkers

        return simulation.derrida_coefficient(
            self.kernel,
            self.n_variables,
            n_walkers,
            lookup_tables=self.lookup_tables,
            threads_per_block=threads_per_block,
        )
