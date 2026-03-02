//! Iterative solver for target thermal conductance (UA).
//!
//! This module provides iterative solving to match a target UA by varying
//! the top stream outlet temperature until the achieved conductance converges
//! to the desired value.

mod config;
mod error;
mod problem;

pub use config::GivenUaConfig;
pub use error::GivenUaError;

use twine_solvers::equation::{EvalError, bisection};
use uom::{
    ConstZero,
    si::{
        f64::ThermalConductance, thermal_conductance::watt_per_kelvin,
        thermodynamic_temperature::kelvin,
    },
};

use super::{
    Given, HeatTransferRate, Known, Results, SolveError,
    traits::{DiscretizedArrangement, DiscretizedHxThermoModel},
};

use problem::{GivenUaModel, GivenUaProblem};

/// Results from a `given_ua` solve, including the node states and iteration count.
#[derive(Debug, Clone)]
pub struct GivenUaResults<TopFluid, BottomFluid, const N: usize> {
    /// Heat exchanger node states and performance metrics.
    pub results: Results<TopFluid, BottomFluid, N>,

    /// Number of bisection iterations performed.
    pub iterations: usize,
}

/// Solves a discretized heat exchanger given a target conductance (UA).
///
/// Accepts bare `ThermalConductance`. Non-positive UA means no heat transfer —
/// the solver is skipped and zero-transfer results are returned immediately.
///
/// Uses bisection to iteratively find the top stream outlet temperature that
/// achieves the specified thermal conductance.
///
/// # Errors
///
/// Returns [`GivenUaError`] on non-physical results, thermodynamic model failures,
/// or if the solver fails to converge.
pub(super) fn given_ua<Arrangement, TopFluid, BottomFluid, const N: usize>(
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
    const {
        assert!(
            N >= 2,
            "discretized heat exchanger requires at least 2 nodes (inlet and outlet)"
        );
    };

    if target_ua < ThermalConductance::ZERO {
        return Err(GivenUaError::NegativeUa(target_ua));
    }

    if target_ua == ThermalConductance::ZERO {
        let results = super::DiscretizedHx::<Arrangement, N>::solve(
            known,
            Given::HeatTransferRate(HeatTransferRate::None),
            thermo_top,
            thermo_bottom,
        )?;
        return Ok(GivenUaResults {
            results,
            iterations: 0,
        });
    }

    let model = GivenUaModel::<Arrangement, _, _, _, _, N>::new(known, thermo_top, thermo_bottom);

    let problem = GivenUaProblem::new(target_ua);

    let solution = bisection::solve(
        &model,
        &problem,
        [
            known.inlets.top.temperature.get::<kelvin>(),
            known.inlets.bottom.temperature.get::<kelvin>(),
        ],
        &config.bisection(),
        |event: &bisection::Event<'_, _, _>| {
            if let Err(EvalError::Model(SolveError::SecondLawViolation { .. })) = event.result() {
                return Some(bisection::Action::assume_positive());
            }
            None
        },
    )?;

    let iterations = solution.iters;

    if solution.status != bisection::Status::Converged {
        return Err(GivenUaError::MaxIters {
            residual: ThermalConductance::new::<watt_per_kelvin>(solution.residual),
            iters: iterations,
        });
    }

    Ok(GivenUaResults {
        results: solution.snapshot.output,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use uom::si::{
        f64::{MassRate, ThermodynamicTemperature},
        mass_rate::kilogram_per_second,
        thermal_conductance::kilowatt_per_kelvin,
        thermodynamic_temperature::kelvin,
    };

    use crate::models::thermal::hx::discretized::core::{
        DiscretizedHx, Given, HeatTransferRate, Inlets, Known, MassFlows, PressureDrops,
        test_support::{TestThermoModel, state},
    };
    use crate::support::hx::arrangement::CounterFlow;

    #[test]
    fn roundtrip() {
        let model = TestThermoModel::new();

        let known = Known {
            inlets: Inlets {
                top: state(400.0),
                bottom: state(300.0),
            },
            m_dot: MassFlows::new_unchecked(
                MassRate::new::<kilogram_per_second>(2.0),
                MassRate::new::<kilogram_per_second>(3.0),
            ),
            dp: PressureDrops::default(),
        };

        let target = DiscretizedHx::<CounterFlow, 5>::solve(
            &known,
            Given::TopOutletTemp(ThermodynamicTemperature::new::<kelvin>(360.0)),
            &model,
            &model,
        )
        .expect("baseline solve should succeed");

        let result = given_ua::<CounterFlow, _, _, 5>(
            &known,
            target.ua,
            GivenUaConfig::default(),
            &model,
            &model,
        )
        .expect("ua solve should succeed");

        assert_relative_eq!(
            result.results.top[4].temperature.get::<kelvin>(),
            target.top[4].temperature.get::<kelvin>(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn zero_returns_no_heat_transfer() {
        let model = TestThermoModel::new();

        let known = Known {
            inlets: Inlets {
                top: state(400.0),
                bottom: state(300.0),
            },
            m_dot: MassFlows::new_unchecked(
                MassRate::new::<kilogram_per_second>(2.0),
                MassRate::new::<kilogram_per_second>(3.0),
            ),
            dp: PressureDrops::default(),
        };

        let result = given_ua::<CounterFlow, _, _, 5>(
            &known,
            ThermalConductance::ZERO,
            GivenUaConfig::default(),
            &model,
            &model,
        )
        .expect("zero ua solve should succeed");

        // With zero UA, no heat transfer occurs
        assert_eq!(result.results.q_dot, HeatTransferRate::None);
        assert_eq!(result.results.ua, ThermalConductance::ZERO);
        assert_eq!(result.iterations, 0);

        // Outlet temperatures should match inlet temperatures
        assert_relative_eq!(result.results.top[4].temperature.get::<kelvin>(), 400.0);
        assert_relative_eq!(result.results.bottom[0].temperature.get::<kelvin>(), 300.0);
    }

    #[test]
    fn negative_ua_returns_error() {
        let model = TestThermoModel::new();

        let known = Known {
            inlets: Inlets {
                top: state(400.0),
                bottom: state(300.0),
            },
            m_dot: MassFlows::new_unchecked(
                MassRate::new::<kilogram_per_second>(2.0),
                MassRate::new::<kilogram_per_second>(3.0),
            ),
            dp: PressureDrops::default(),
        };

        let result = given_ua::<CounterFlow, _, _, 5>(
            &known,
            ThermalConductance::new::<watt_per_kelvin>(-1.0),
            GivenUaConfig::default(),
            &model,
            &model,
        );

        assert!(matches!(result, Err(GivenUaError::NegativeUa(_))));
    }

    #[test]
    fn handles_second_law_violations_during_iteration() {
        let model = TestThermoModel::new();

        // Unbalanced flow rates create challenging conditions for the solver.
        // The bottom stream has much lower flow, so it experiences larger temperature changes.
        // This imbalance causes many top outlet candidates to violate the second law.
        let known = Known {
            inlets: Inlets {
                top: state(400.0),
                bottom: state(300.0),
            },
            m_dot: MassFlows::new_unchecked(
                MassRate::new::<kilogram_per_second>(2.0),
                MassRate::new::<kilogram_per_second>(0.5),
            ),
            dp: PressureDrops::default(),
        };

        let result = given_ua::<CounterFlow, _, _, 5>(
            &known,
            ThermalConductance::new::<kilowatt_per_kelvin>(2.0),
            GivenUaConfig::default(),
            &model,
            &model,
        )
        .expect("solver should converge despite violations during iteration");

        assert_relative_eq!(
            result.results.ua.get::<kilowatt_per_kelvin>(),
            2.0,
            epsilon = 1e-12
        );
    }
}
