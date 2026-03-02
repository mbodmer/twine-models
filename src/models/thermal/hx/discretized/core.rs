//! Discretized counterflow and parallel-flow heat exchanger modeling.
//!
//! A discretized heat exchanger divides the flow into a linear series of
//! constant-property sub-exchangers so thermodynamic properties can vary
//! along a linear array of nodes, supporting real-fluid behavior.

// This module is internal infrastructure for Model adapters (issue #14).
// Dead code warnings are expected until adapters consume this API.
#![allow(dead_code)]

mod given_ua;
mod heat_transfer_rate;
mod input;
mod metrics;
mod results;
mod solve;
mod traits;

#[cfg(test)]
mod test_support;

pub use given_ua::{GivenUaConfig, GivenUaError, GivenUaResults};
pub use heat_transfer_rate::HeatTransferRate;
pub use input::{Given, Inlets, Known, MassFlows, PressureDrops};
pub use results::{MinDeltaT, Results};
pub use solve::SolveError;
pub(crate) use traits::DiscretizedHxThermoModel;

use std::marker::PhantomData;

use uom::si::f64::ThermalConductance;

use given_ua::given_ua;
use solve::solve;
use traits::DiscretizedArrangement;

/// Entry point for solving a discretized heat exchanger.
///
/// The arrangement and node count are fixed by generics.
/// This type also provides the natural home for higher-level solve helpers,
/// such as iterating on UA to match target outlet states.
///
/// # Minimum Node Count
///
/// The node count `N` must be at least 2 (inlet and outlet).
/// This constraint is enforced at compile time via const assertions.
///
/// ```compile_fail
/// # use twine_models::models::thermal::hx::discretized::core::DiscretizedHx;
/// # use twine_models::support::hx::arrangement::ParallelFlow;
/// // This will fail to compile: N must be >= 2
/// let _ = DiscretizedHx::<ParallelFlow, 1>::solve(todo!(), todo!(), todo!(), todo!());
/// ```
pub struct DiscretizedHx<Arrangement, const N: usize> {
    _arrangement: PhantomData<Arrangement>,
}

impl<Arrangement, const N: usize> DiscretizedHx<Arrangement, N> {
    /// Solves a discretized heat exchanger with a fixed arrangement and node count.
    ///
    /// # Errors
    ///
    /// Returns a [`SolveError`] on non-physical results or thermodynamic model failures.
    pub fn solve<TopFluid, BottomFluid>(
        known: &Known<TopFluid, BottomFluid>,
        given: Given,
        thermo_top: &impl DiscretizedHxThermoModel<TopFluid>,
        thermo_bottom: &impl DiscretizedHxThermoModel<BottomFluid>,
    ) -> Result<Results<TopFluid, BottomFluid, N>, SolveError>
    where
        Arrangement: DiscretizedArrangement + Default,
        TopFluid: Clone,
        BottomFluid: Clone,
    {
        solve::<Arrangement, _, _, N>(known, given, thermo_top, thermo_bottom)
    }

    /// Solves a discretized heat exchanger when both streams share the same thermo model.
    ///
    /// This is a convenience wrapper around [`DiscretizedHx::solve`].
    ///
    /// # Errors
    ///
    /// Returns a [`SolveError`] on non-physical results or thermodynamic model failures.
    pub fn solve_same<Fluid, Model>(
        known: &Known<Fluid, Fluid>,
        given: Given,
        thermo: &Model,
    ) -> Result<Results<Fluid, Fluid, N>, SolveError>
    where
        Arrangement: DiscretizedArrangement + Default,
        Fluid: Clone,
        Model: DiscretizedHxThermoModel<Fluid>,
    {
        solve::<Arrangement, _, _, N>(known, given, thermo, thermo)
    }

    /// Solves a discretized heat exchanger given a target conductance (UA).
    ///
    /// Accepts bare `ThermalConductance`. Non-positive values skip the solve and
    /// return zero-transfer results immediately.
    ///
    /// Iterates on the top outlet temperature to achieve the specified
    /// thermal conductance.
    ///
    /// # Errors
    ///
    /// Returns a [`GivenUaError`] on non-physical results, thermodynamic model failures,
    /// or if the solver fails to converge.
    pub fn given_ua<TopFluid, BottomFluid>(
        known: &Known<TopFluid, BottomFluid>,
        target_ua: ThermalConductance,
        config: GivenUaConfig,
        thermo_top: &impl DiscretizedHxThermoModel<TopFluid>,
        thermo_bottom: &impl DiscretizedHxThermoModel<BottomFluid>,
    ) -> Result<GivenUaResults<TopFluid, BottomFluid, N>, GivenUaError>
    where
        Arrangement: DiscretizedArrangement + Default,
        TopFluid: Clone,
        BottomFluid: Clone,
    {
        given_ua::<Arrangement, _, _, N>(known, target_ua, config, thermo_top, thermo_bottom)
    }

    /// Solves a discretized heat exchanger given a target UA when both streams share the same thermo model.
    ///
    /// This is a convenience wrapper around [`DiscretizedHx::given_ua`].
    ///
    /// # Errors
    ///
    /// Returns a [`GivenUaError`] on non-physical results, thermodynamic model failures,
    /// or if the solver fails to converge.
    pub fn given_ua_same<Fluid, Model>(
        known: &Known<Fluid, Fluid>,
        target_ua: ThermalConductance,
        config: GivenUaConfig,
        thermo: &Model,
    ) -> Result<GivenUaResults<Fluid, Fluid, N>, GivenUaError>
    where
        Arrangement: DiscretizedArrangement + Default,
        Fluid: Clone,
        Model: DiscretizedHxThermoModel<Fluid>,
    {
        given_ua::<Arrangement, _, _, N>(known, target_ua, config, thermo, thermo)
    }
}
