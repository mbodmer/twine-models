use std::ops::Div;

use twine_core::StepIntegrable;
use uom::si::f64::{MassDensity, TemperatureInterval, ThermodynamicTemperature, Time};

/// The thermodynamic state of a fluid.
///
/// A `State<Fluid>` captures the thermodynamic state of a specific fluid,
/// including its temperature, density, and any fluid-specific data.
///
/// The `Fluid` type parameter can be a simple marker type,
/// such as [`Air`](crate::fluid::Air) or [`Water`](crate::fluid::Water),
/// or a structured type containing additional data, such as mixture composition
/// or particle concentration.
///
/// `State` is the primary input to capability-based thermodynamic models for
/// calculating pressure, enthalpy, entropy, and related quantities.
///
/// # Example
///
/// ```
/// use twine_models::support::thermo::{State, fluid::Air};
/// use uom::si::{
///     f64::{ThermodynamicTemperature, MassDensity},
///     thermodynamic_temperature::kelvin,
///     mass_density::kilogram_per_cubic_meter,
/// };
///
/// let state = State {
///     temperature: ThermodynamicTemperature::new::<kelvin>(300.0),
///     density: MassDensity::new::<kilogram_per_cubic_meter>(1.0),
///     fluid: Air,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State<Fluid> {
    pub temperature: ThermodynamicTemperature,
    pub density: MassDensity,
    pub fluid: Fluid,
}

impl<Fluid> State<Fluid> {
    /// Creates a new state with the given temperature, density, and fluid.
    #[must_use]
    pub fn new(temperature: ThermodynamicTemperature, density: MassDensity, fluid: Fluid) -> Self {
        Self {
            temperature,
            density,
            fluid,
        }
    }

    /// Returns a new state with the given temperature, keeping other fields unchanged.
    #[must_use]
    pub fn with_temperature(self, temperature: ThermodynamicTemperature) -> Self {
        Self {
            temperature,
            ..self
        }
    }

    /// Returns a new state with the given density, keeping other fields unchanged.
    #[must_use]
    pub fn with_density(self, density: MassDensity) -> Self {
        Self { density, ..self }
    }

    /// Returns a new state with the given fluid, keeping other fields unchanged.
    #[must_use]
    pub fn with_fluid(self, fluid: Fluid) -> Self {
        Self { fluid, ..self }
    }
}

/// The time derivative of a thermodynamic [`State`].
///
/// Parameterized over the fluid derivative type directly, keeping the struct
/// independent of the [`StepIntegrable`] trait.
/// The connection to `StepIntegrable` happens at the impl site on `State<Fluid>`,
/// where `StateDerivative<Fluid::Derivative>` becomes the associated type.
///
/// # Example
///
/// ```
/// use twine_models::support::thermo::{State, StateDerivative, fluid::Air};
/// use uom::si::{
///     f64::{MassDensity, TemperatureInterval, ThermodynamicTemperature, Time},
///     mass_density::kilogram_per_cubic_meter,
///     thermodynamic_temperature::kelvin,
///     temperature_interval::kelvin as interval_kelvin,
///     time::second,
/// };
///
/// let state = State {
///     temperature: ThermodynamicTemperature::new::<kelvin>(300.0),
///     density: MassDensity::new::<kilogram_per_cubic_meter>(1.2),
///     fluid: Air,
/// };
///
/// let dt = Time::new::<second>(1.0);
/// let deriv: StateDerivative<()> = StateDerivative {
///     temperature: TemperatureInterval::new::<interval_kelvin>(1.0) / dt,
///     density: MassDensity::new::<kilogram_per_cubic_meter>(0.1) / dt,
///     fluid: (),
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateDerivative<FluidDerivative> {
    /// Rate of change of temperature (K/s).
    pub temperature: <TemperatureInterval as Div<Time>>::Output,

    /// Rate of change of density (kg/m³/s).
    pub density: <MassDensity as Div<Time>>::Output,

    /// Fluid-specific derivative.
    pub fluid: FluidDerivative,
}

impl<Fluid> StepIntegrable<Time> for State<Fluid>
where
    Fluid: StepIntegrable<Time>,
{
    type Derivative = StateDerivative<Fluid::Derivative>;

    fn step(&self, derivative: Self::Derivative, delta: Time) -> Self {
        Self {
            temperature: self.temperature + derivative.temperature * delta,
            density: self.density + derivative.density * delta,
            fluid: self.fluid.step(derivative.fluid, delta),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use twine_core::StepIntegrable;
    use uom::si::{
        f64::{MassDensity, TemperatureInterval, ThermodynamicTemperature, Time},
        mass_density::kilogram_per_cubic_meter,
        temperature_interval::kelvin as interval_kelvin,
        thermodynamic_temperature::kelvin,
        time::second,
    };

    use crate::support::thermo::fluid::Air;

    #[test]
    fn step_advances_temperature_and_density() {
        let state = State {
            temperature: ThermodynamicTemperature::new::<kelvin>(300.0),
            density: MassDensity::new::<kilogram_per_cubic_meter>(1.2),
            fluid: Air,
        };

        let dt = Time::new::<second>(2.0);
        let deriv = StateDerivative {
            temperature: TemperatureInterval::new::<interval_kelvin>(3.0) / dt,
            density: MassDensity::new::<kilogram_per_cubic_meter>(0.4) / dt,
            fluid: (),
        };

        let next = state.step(deriv, dt);

        approx::assert_relative_eq!(
            next.temperature.get::<kelvin>(),
            303.0,
            max_relative = 1e-10,
        );
        approx::assert_relative_eq!(
            next.density.get::<kilogram_per_cubic_meter>(),
            1.6,
            max_relative = 1e-10,
        );
    }

    #[test]
    fn step_zero_derivative_is_identity() {
        let state = State {
            temperature: ThermodynamicTemperature::new::<kelvin>(290.0),
            density: MassDensity::new::<kilogram_per_cubic_meter>(1.0),
            fluid: Air,
        };

        let dt = Time::new::<second>(1.0);
        let deriv = StateDerivative {
            temperature: TemperatureInterval::new::<interval_kelvin>(0.0) / dt,
            density: MassDensity::new::<kilogram_per_cubic_meter>(0.0) / dt,
            fluid: (),
        };

        let next = state.step(deriv, dt);

        assert_eq!(next, state);
    }

    #[test]
    fn zst_fluid_markers_step_to_themselves() {
        use crate::support::thermo::fluid::{CarbonDioxide, Water};

        let dt = Time::new::<second>(1.0);

        assert_eq!(Air.step((), dt), Air);
        assert_eq!(Water.step((), dt), Water);
        assert_eq!(CarbonDioxide.step((), dt), CarbonDioxide);
    }
}
