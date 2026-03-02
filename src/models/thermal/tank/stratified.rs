//! Stratified thermal energy storage tank model.
//!
//! A [`StratifiedTank`] discretizes a vertical tank into `N` fully mixed nodes
//! and computes per-node temperature derivatives from fluid flow, auxiliary heat,
//! and conduction.
//!
//! ## Quick start
//!
//! ```
//! use twine_models::models::thermal::tank::stratified::{
//!     AuxHeatFlow, Environment, Fluid, Geometry, Insulation, Location,
//!     PortFlow, PortLocation, StratifiedTank, StratifiedTankInput,
//! };
//! use uom::si::{
//!     f64::{Length, MassDensity, SpecificHeatCapacity, ThermalConductivity,
//!           ThermodynamicTemperature, VolumeRate},
//!     length::meter,
//!     mass_density::kilogram_per_cubic_meter,
//!     specific_heat_capacity::kilojoule_per_kilogram_kelvin,
//!     thermal_conductivity::watt_per_meter_kelvin,
//!     thermodynamic_temperature::degree_celsius,
//!     volume_rate::liter_per_minute,
//! };
//!
//! let fluid = Fluid {
//!     density: MassDensity::new::<kilogram_per_cubic_meter>(1000.0),
//!     specific_heat: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.186),
//!     thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(0.6),
//! };
//!
//! let geometry = Geometry::VerticalCylinder {
//!     diameter: Length::new::<meter>(0.5),
//!     height: Length::new::<meter>(1.8),
//! };
//!
//! let port_locations = [PortLocation {
//!     inlet: Location::tank_bottom(),
//!     outlet: Location::tank_top(),
//! }];
//!
//! let tank = StratifiedTank::new::<5>(
//!     fluid, geometry, Insulation::Adiabatic, [], port_locations,
//! ).expect("valid configuration");
//!
//! let t_init = ThermodynamicTemperature::new::<degree_celsius>(20.0);
//! let ambient = Environment { bottom: t_init, side: t_init, top: t_init };
//!
//! let input = StratifiedTankInput {
//!     temperatures: [t_init; 5],
//!     port_flows: [PortFlow::new(VolumeRate::new::<liter_per_minute>(10.0), t_init).unwrap()],
//!     aux_heat_flows: [],
//!     environment: ambient,
//! };
//!
//! let output = tank.evaluate(&input);
//! assert_eq!(output.temperatures.len(), 5);
//! ```
//!
//! ## Full simulation example
//!
//! The `stratified_tank` example demonstrates a complete transient simulation
//! with thermostat control, a daily draw schedule, and interactive plotting:
//!
//! ```sh
//! cargo run --example stratified_tank --release
//! ```

mod core;

use std::convert::Infallible;

use twine_core::{DerivativeOf, Model, OdeProblem, StepIntegrable};
use uom::si::{
    f64::{ThermodynamicTemperature, Time},
    temperature_interval::kelvin as delta_kelvin,
    thermodynamic_temperature::kelvin,
};

pub use core::{
    AuxHeatFlow, Environment, Fluid, Geometry, Insulation, Location, PortFlow, PortLocation,
    StratifiedTank, StratifiedTankError, StratifiedTankInput, StratifiedTankOutput,
    TemperatureRate, ValidatedPower,
};

impl<const N: usize, const P: usize, const Q: usize> Model for StratifiedTank<N, P, Q> {
    type Input = StratifiedTankInput<N, P, Q>;
    type Output = StratifiedTankOutput<N>;
    type Error = Infallible;

    fn call(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        Ok(self.evaluate(input))
    }
}

/// Temperatures of all `N` nodes, used as the ODE state for time integration.
#[derive(Debug, Clone, Copy)]
pub struct TankState<const N: usize> {
    /// Node temperatures from bottom (index 0) to top (index N-1).
    pub temperatures: [ThermodynamicTemperature; N],
}

/// Time derivatives of all `N` node temperatures (K/s).
#[derive(Debug, Clone, Copy)]
pub struct TankDerivative<const N: usize> {
    /// Per-node temperature rates from bottom to top.
    pub rates: [TemperatureRate; N],
}

impl<const N: usize> StepIntegrable<Time> for TankState<N> {
    type Derivative = TankDerivative<N>;

    fn step(&self, derivative: TankDerivative<N>, delta: Time) -> Self {
        TankState {
            temperatures: std::array::from_fn(|i| {
                // TemperatureRate * Time = TemperatureInterval (in kelvin).
                let delta_t = (derivative.rates[i] * delta).get::<delta_kelvin>();
                let t_k = self.temperatures[i].get::<kelvin>();
                ThermodynamicTemperature::new::<kelvin>(t_k + delta_t)
            }),
        }
    }
}

/// Adapts a [`StratifiedTank`] for use with `twine_solvers::transient::euler::solve`.
///
/// The state is the vector of node temperatures. On each step, the solver
/// extracts temperatures from the input, steps them forward using the
/// derivatives from the model output, and rebuilds the input for the next
/// evaluation.
///
/// # Example
///
/// ```
/// use twine_models::models::thermal::tank::stratified::{
///     AuxHeatFlow, Environment, Fluid, Geometry, Insulation, Location,
///     PortFlow, PortLocation, StratifiedTank, StratifiedTankInput,
///     TankOdeProblem,
/// };
/// use twine_solvers::transient::euler;
/// use uom::si::{
///     f64::{Length, MassDensity, SpecificHeatCapacity, ThermalConductivity,
///           ThermodynamicTemperature, Time, VolumeRate},
///     length::meter,
///     mass_density::kilogram_per_cubic_meter,
///     specific_heat_capacity::kilojoule_per_kilogram_kelvin,
///     thermal_conductivity::watt_per_meter_kelvin,
///     thermodynamic_temperature::degree_celsius,
///     time::second,
///     volume_rate::liter_per_minute,
/// };
///
/// let fluid = Fluid {
///     density: MassDensity::new::<kilogram_per_cubic_meter>(1000.0),
///     specific_heat: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.186),
///     thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(0.6),
/// };
///
/// let tank = StratifiedTank::new::<3>(
///     fluid,
///     Geometry::VerticalCylinder {
///         diameter: Length::new::<meter>(0.5),
///         height: Length::new::<meter>(1.5),
///     },
///     Insulation::Adiabatic,
///     [],
///     [PortLocation {
///         inlet: Location::tank_bottom(),
///         outlet: Location::tank_top(),
///     }],
/// ).unwrap();
///
/// let t_init = ThermodynamicTemperature::new::<degree_celsius>(60.0);
/// let ambient = Environment { bottom: t_init, side: t_init, top: t_init };
/// let initial = StratifiedTankInput {
///     temperatures: [t_init; 3],
///     port_flows: [PortFlow::new(VolumeRate::new::<liter_per_minute>(5.0), t_init).unwrap()],
///     aux_heat_flows: [],
///     environment: ambient,
/// };
///
/// let dt = Time::new::<second>(60.0);
/// let solution = euler::solve_unobserved(
///     &tank,
///     &TankOdeProblem::<3, 1, 0>,
///     initial,
///     dt,
///     10,
/// ).unwrap();
/// assert_eq!(solution.steps, 10);
/// ```
pub struct TankOdeProblem<const N: usize, const P: usize, const Q: usize>;

impl<const N: usize, const P: usize, const Q: usize> OdeProblem for TankOdeProblem<N, P, Q> {
    type Input = StratifiedTankInput<N, P, Q>;
    type Output = StratifiedTankOutput<N>;
    type Delta = Time;
    type State = TankState<N>;
    type Error = Infallible;

    fn state(&self, input: &Self::Input) -> Result<TankState<N>, Infallible> {
        Ok(TankState {
            temperatures: input.temperatures,
        })
    }

    fn derivative(
        &self,
        _input: &Self::Input,
        output: &Self::Output,
    ) -> Result<DerivativeOf<TankState<N>, Time>, Infallible> {
        Ok(TankDerivative {
            rates: output.derivatives,
        })
    }

    fn build_input(
        &self,
        base: &Self::Input,
        state: &Self::State,
        _delta: &Time,
    ) -> Result<Self::Input, Infallible> {
        Ok(StratifiedTankInput {
            temperatures: state.temperatures,
            port_flows: base.port_flows,
            aux_heat_flows: base.aux_heat_flows,
            environment: base.environment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use twine_core::Model;
    use twine_solvers::transient::euler;
    use uom::si::{
        f64::{Length, MassDensity, Power, SpecificHeatCapacity, ThermalConductivity, VolumeRate},
        length::meter,
        mass_density::kilogram_per_cubic_meter,
        power::kilowatt,
        specific_heat_capacity::kilojoule_per_kilogram_kelvin,
        thermal_conductivity::watt_per_meter_kelvin,
        thermodynamic_temperature::degree_celsius,
        time::second,
        volume_rate::gallon_per_minute,
    };

    use std::f64::consts::PI;

    use uom::ConstZero;

    fn k_per_s(rate: TemperatureRate) -> f64 {
        use uom::si::temperature_interval::kelvin as delta_kelvin;
        (rate * Time::new::<second>(1.0)).get::<delta_kelvin>()
    }

    fn test_tank() -> StratifiedTank<3, 1, 1> {
        let fluid = Fluid {
            density: MassDensity::new::<kilogram_per_cubic_meter>(1000.0),
            specific_heat: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.0),
            thermal_conductivity: ThermalConductivity::ZERO,
        };

        let geometry = Geometry::VerticalCylinder {
            diameter: Length::new::<meter>((4.0 / PI).sqrt()),
            height: Length::new::<meter>(3.0),
        };

        StratifiedTank::new(
            fluid,
            geometry,
            Insulation::Adiabatic,
            [Location::tank_top()],
            [PortLocation {
                inlet: Location::tank_bottom(),
                outlet: Location::tank_top(),
            }],
        )
        .unwrap()
    }

    fn port_flow(gpm: f64, celsius: f64) -> PortFlow {
        PortFlow::new(
            VolumeRate::new::<gallon_per_minute>(gpm),
            ThermodynamicTemperature::new::<degree_celsius>(celsius),
        )
        .unwrap()
    }

    fn ambient(celsius: f64) -> Environment {
        let t = ThermodynamicTemperature::new::<degree_celsius>(celsius);
        Environment {
            bottom: t,
            side: t,
            top: t,
        }
    }

    #[test]
    fn model_call_delegates_to_evaluate() {
        let tank = test_tank();
        let t = ThermodynamicTemperature::new::<degree_celsius>(30.0);

        let input = StratifiedTankInput {
            temperatures: [t; 3],
            port_flows: [port_flow(0.0, 30.0)],
            aux_heat_flows: [AuxHeatFlow::None],
            environment: ambient(30.0),
        };

        let via_model = Model::call(&tank, &input).unwrap();
        let via_evaluate = tank.evaluate(&input);

        assert_eq!(via_model.temperatures, via_evaluate.temperatures);
        for i in 0..3 {
            assert_relative_eq!(
                k_per_s(via_model.derivatives[i]),
                k_per_s(via_evaluate.derivatives[i])
            );
        }
    }

    #[test]
    fn aux_heating_raises_temperature_over_time() {
        let tank = test_tank();
        let t0 = ThermodynamicTemperature::new::<degree_celsius>(20.0);

        // 20 kW heating at top node; V=1 m³, rho=1000 kg/m³, cp=4 kJ/(kg·K)
        // dT/dt = 20,000 / (1 * 1000 * 4000) = 0.005 K/s
        // After 100 s: ΔT = 0.5 K at top node
        let initial = StratifiedTankInput {
            temperatures: [t0; 3],
            port_flows: [port_flow(0.0, 20.0)],
            aux_heat_flows: [AuxHeatFlow::heating(Power::new::<kilowatt>(20.0)).unwrap()],
            environment: ambient(20.0),
        };

        let dt = Time::new::<second>(1.0);
        let solution =
            euler::solve_unobserved(&tank, &TankOdeProblem::<3, 1, 1>, initial, dt, 100).unwrap();

        let final_input = &solution.history.last().unwrap().input;
        let t_top = final_input.temperatures[2].get::<degree_celsius>();
        let t_bot = final_input.temperatures[0].get::<degree_celsius>();

        // Top node heated by 0.5 K; bottom two unchanged (adiabatic, no flow).
        assert_relative_eq!(t_top, 20.5, epsilon = 1e-10);
        assert_relative_eq!(t_bot, 20.0, epsilon = 1e-10);
    }

    #[test]
    fn at_equilibrium_all_derivatives_zero() {
        // At uniform temperature with zero flow and matching environment,
        // every derivative should be exactly zero.
        let tank = test_tank();
        let t = ThermodynamicTemperature::new::<degree_celsius>(20.0);

        let input = StratifiedTankInput {
            temperatures: [t; 3],
            port_flows: [port_flow(0.0, 20.0)],
            aux_heat_flows: [AuxHeatFlow::None],
            environment: ambient(20.0),
        };

        let out = tank.evaluate(&input);

        // Temperatures unchanged after buoyancy (already stable).
        assert!(out.temperatures.iter().all(|&nt| nt == t));

        // Every derivative should be zero.
        for deriv in out.derivatives {
            assert_relative_eq!(k_per_s(deriv), 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn aux_cooling_lowers_temperature_over_time() {
        // Symmetric counterpart to the heating test: 20 kW of cooling at the
        // top node should drop it by 0.5 K over 100 s.
        let tank = test_tank();
        let t0 = ThermodynamicTemperature::new::<degree_celsius>(50.0);

        let initial = StratifiedTankInput {
            temperatures: [t0; 3],
            port_flows: [port_flow(0.0, 50.0)],
            aux_heat_flows: [AuxHeatFlow::cooling(Power::new::<kilowatt>(20.0)).unwrap()],
            environment: ambient(50.0),
        };

        let dt = Time::new::<second>(1.0);
        let solution =
            euler::solve_unobserved(&tank, &TankOdeProblem::<3, 1, 1>, initial, dt, 100).unwrap();

        let final_input = &solution.history.last().unwrap().input;
        let t_top = final_input.temperatures[2].get::<degree_celsius>();
        let t_bot = final_input.temperatures[0].get::<degree_celsius>();

        assert_relative_eq!(t_top, 49.5, epsilon = 1e-10);
        assert_relative_eq!(t_bot, 50.0, epsilon = 1e-10);
    }

    #[test]
    fn instantaneous_aux_cooling_derivative_is_negative() {
        // Verify the sign of the derivative at t=0, not just the integrated result.
        let tank = test_tank();
        let t = ThermodynamicTemperature::new::<degree_celsius>(50.0);

        let input = StratifiedTankInput {
            temperatures: [t; 3],
            port_flows: [port_flow(0.0, 50.0)],
            aux_heat_flows: [AuxHeatFlow::cooling(Power::new::<kilowatt>(20.0)).unwrap()],
            environment: ambient(50.0),
        };

        let out = tank.evaluate(&input);

        // Q/C = −20 kW / (1 m³ × 1000 kg/m³ × 4 kJ/(kg·K)) = −0.005 K/s at node 2.
        assert_relative_eq!(k_per_s(out.derivatives[0]), 0.0);
        assert_relative_eq!(k_per_s(out.derivatives[1]), 0.0);
        assert_relative_eq!(k_per_s(out.derivatives[2]), -0.005);
    }

    #[test]
    fn tank_with_conduction_reaches_equilibrium() {
        // High conductivity → short time constant so the simulation reaches
        // equilibrium in a reasonable number of steps.
        //
        // Each node: V = 1 m³, ρ·cp = 4 MJ/(m³·K), k = 10 000 W/(m·K).
        // UA between adjacent nodes ≈ k / (0.5 h + 0.5 h) = 10 000 W/K.
        // τ = C / UA = 4 000 000 / 10 000 = 400 s.
        // After 50 000 s (5000 × 10 s): t/τ ≈ 125 → fully equilibrated.
        let fluid = Fluid {
            density: MassDensity::new::<kilogram_per_cubic_meter>(1000.0),
            specific_heat: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.0),
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(10_000.0),
        };
        let geometry = Geometry::VerticalCylinder {
            diameter: Length::new::<meter>((4.0 / PI).sqrt()),
            height: Length::new::<meter>(3.0),
        };

        let tank: StratifiedTank<3, 0, 0> =
            StratifiedTank::new(fluid, geometry, Insulation::Adiabatic, [], []).unwrap();

        let t_hot = ThermodynamicTemperature::new::<degree_celsius>(80.0);
        let t_cold = ThermodynamicTemperature::new::<degree_celsius>(20.0);
        let t_avg = ThermodynamicTemperature::new::<degree_celsius>(50.0);

        // Start thermally stable (cold at bottom, warm at top) so buoyancy
        // doesn't interfere — we're testing conduction-driven equilibration.
        let initial = StratifiedTankInput {
            temperatures: [t_cold, t_avg, t_hot],
            port_flows: [],
            aux_heat_flows: [],
            environment: Environment {
                bottom: t_avg,
                side: t_avg,
                top: t_avg,
            },
        };

        let dt = Time::new::<second>(10.0);
        let solution =
            euler::solve_unobserved(&tank, &TankOdeProblem::<3, 0, 0>, initial, dt, 5000).unwrap();

        let final_input = &solution.history.last().unwrap().input;

        // All nodes should have converged toward the mean temperature (50 °C).
        for temp in final_input.temperatures {
            assert_relative_eq!(temp.get::<degree_celsius>(), 50.0, epsilon = 1.0);
        }
    }
}
