use crate::support::thermo::State;

use super::ThermoModel;

/// Capability for constructing a [`State`] from a typed input.
///
/// A thermodynamic [`State`] includes a `Fluid` value. In Twine, `Fluid` is
/// allowed to carry *state-defining* information such as mixture composition,
/// salinity, or any other configuration needed to make the state well-defined.
///
/// `StateFrom<Input>` expresses, at compile time, which combinations of
/// inputs a model can use to construct a state.
/// If a model does not implement `StateFrom<Input>`, then that input is simply
/// not supported (no runtime "not implemented" errors).
///
/// ## Common input patterns
///
/// Inputs are intentionally represented as normal Rust types (often tuples).
/// Common patterns include:
/// - `(Fluid, ThermodynamicTemperature, Pressure)` (temperature + pressure)
/// - `(Fluid, Pressure, MassDensity)` (pressure + density)
/// - `(Fluid, Pressure, SpecificEnthalpy)` (pressure + enthalpy)
/// - `(Fluid, Pressure, SpecificEntropy)` (pressure + entropy)
/// - `(Fluid, ThermodynamicTemperature)` (e.g. for an incompressible liquid)
///
pub trait StateFrom<Input>: ThermoModel {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create a thermodynamic state from the provided input.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the state cannot be created from `input`.
    fn state_from(&self, input: Input) -> Result<State<Self::Fluid>, Self::Error>;
}

/// Blanket impl for borrowed models.
///
/// Any type that implements `StateFrom<Input>` also implements it when borrowed
/// as `&T`. This allows components that accept a thermo model by reference to
/// use the same trait bounds without requiring ownership.
impl<T, Input> StateFrom<Input> for &T
where
    T: StateFrom<Input>,
{
    type Error = T::Error;

    fn state_from(&self, input: Input) -> Result<State<Self::Fluid>, Self::Error> {
        T::state_from(self, input)
    }
}
