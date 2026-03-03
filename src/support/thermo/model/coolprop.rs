//! CoolProp-backed fluid property model.

// CoolProp uses C++ exceptions which require Emscripten's runtime.
// The `wasm32-unknown-unknown` target cannot provide this.
#[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
compile_error!(
    "CoolProp requires Emscripten for WASM builds. \
     Use target `wasm32-unknown-emscripten`, not `wasm32-unknown-unknown`."
);

mod error;
mod ffi;
mod wrapper;

use std::{
    marker::PhantomData,
    sync::{Mutex, MutexGuard},
};

use uom::si::{
    available_energy::joule_per_kilogram,
    f64::{MassDensity, MolarMass, Pressure, SpecificHeatCapacity, ThermodynamicTemperature},
    mass_density::kilogram_per_cubic_meter,
    molar_mass::kilogram_per_mole,
    pressure::pascal,
    specific_heat_capacity::joule_per_kilogram_kelvin,
    thermodynamic_temperature::kelvin,
};

use crate::support::thermo::{
    PropertyError, State,
    capability::{
        HasCp, HasCv, HasEnthalpy, HasEntropy, HasInternalEnergy, HasPressure, StateFrom,
        ThermoModel,
    },
};
use crate::support::units::{SpecificEnthalpy, SpecificEntropy, SpecificInternalEnergy};

use ffi::{InputPair, OutputParam};
use wrapper::AbstractState;

pub use error::CoolPropError;

/// Trait used to mark fluids as usable with the [`CoolProp`] model.
///
/// Implementors provide the backend and fluid identifiers needed to construct a
/// `CoolProp` `AbstractState`.
pub trait CoolPropFluid: Default + Send + Sync + 'static {
    const BACKEND: &'static str;
    const NAME: &'static str;
}

/// A fluid property model backed by `CoolProp`.
pub struct CoolProp<F: CoolPropFluid> {
    state: Mutex<AbstractState>,
    _f: PhantomData<F>,
}

impl<F: CoolPropFluid> ThermoModel for CoolProp<F> {
    type Fluid = F;
}

impl<F: CoolPropFluid> CoolProp<F> {
    /// Construct a new CoolProp-backed model instance.
    ///
    /// # Errors
    ///
    /// Returns [`CoolPropError`] if the underlying `AbstractState` cannot be
    /// created for the given `F::BACKEND` and `F::NAME`.
    pub fn new() -> Result<Self, CoolPropError> {
        let state = AbstractState::new(F::BACKEND, F::NAME)?;
        Ok(Self {
            state: Mutex::new(state),
            _f: PhantomData,
        })
    }

    /// Returns the molar mass of the fluid.
    ///
    /// # Errors
    ///
    /// Returns [`CoolPropError`] if the call fails.
    pub fn molar_mass(&self) -> Result<MolarMass, CoolPropError> {
        let abstract_state = self.state.lock()?;
        let molar_mass = abstract_state.keyed_output(OutputParam::MOLAR_MASS)?;
        Ok(MolarMass::new::<kilogram_per_mole>(molar_mass))
    }

    /// Locks the underlying `AbstractState` and updates it from `state`.
    fn lock_with_state(
        &self,
        state: &State<F>,
    ) -> Result<MutexGuard<'_, AbstractState>, CoolPropError> {
        let mut abstract_state = self.state.lock()?;
        abstract_state.update(
            InputPair::DMASS_T,
            state.density.get::<kilogram_per_cubic_meter>(),
            state.temperature.get::<kelvin>(),
        )?;
        Ok(abstract_state)
    }
}

impl<F: CoolPropFluid> HasPressure for CoolProp<F> {
    fn pressure(&self, state: &State<Self::Fluid>) -> Result<Pressure, PropertyError> {
        let abstract_state = self.lock_with_state(state)?;
        let pressure = abstract_state
            .keyed_output(OutputParam::P)
            .map_err(CoolPropError::from)?;
        Ok(Pressure::new::<pascal>(pressure))
    }
}

impl<F: CoolPropFluid> HasInternalEnergy for CoolProp<F> {
    fn internal_energy(
        &self,
        state: &State<Self::Fluid>,
    ) -> Result<SpecificInternalEnergy, PropertyError> {
        let abstract_state = self.lock_with_state(state)?;
        let internal_energy = abstract_state
            .keyed_output(OutputParam::UMASS)
            .map_err(CoolPropError::from)?;
        Ok(SpecificInternalEnergy::new::<joule_per_kilogram>(
            internal_energy,
        ))
    }
}

impl<F: CoolPropFluid> HasEnthalpy for CoolProp<F> {
    fn enthalpy(&self, state: &State<Self::Fluid>) -> Result<SpecificEnthalpy, PropertyError> {
        let abstract_state = self.lock_with_state(state)?;
        let enthalpy = abstract_state
            .keyed_output(OutputParam::HMASS)
            .map_err(CoolPropError::from)?;
        Ok(SpecificEnthalpy::new::<joule_per_kilogram>(enthalpy))
    }
}

impl<F: CoolPropFluid> HasEntropy for CoolProp<F> {
    fn entropy(&self, state: &State<Self::Fluid>) -> Result<SpecificEntropy, PropertyError> {
        let abstract_state = self.lock_with_state(state)?;
        let entropy = abstract_state
            .keyed_output(OutputParam::SMASS)
            .map_err(CoolPropError::from)?;
        Ok(SpecificEntropy::new::<joule_per_kilogram_kelvin>(entropy))
    }
}

impl<F: CoolPropFluid> HasCp for CoolProp<F> {
    fn cp(&self, state: &State<Self::Fluid>) -> Result<SpecificHeatCapacity, PropertyError> {
        let abstract_state = self.lock_with_state(state)?;
        let cp = abstract_state
            .keyed_output(OutputParam::CP_MASS)
            .map_err(CoolPropError::from)?;
        Ok(SpecificHeatCapacity::new::<joule_per_kilogram_kelvin>(cp))
    }
}

impl<F: CoolPropFluid> HasCv for CoolProp<F> {
    fn cv(&self, state: &State<Self::Fluid>) -> Result<SpecificHeatCapacity, PropertyError> {
        let abstract_state = self.lock_with_state(state)?;
        let cv = abstract_state
            .keyed_output(OutputParam::CV_MASS)
            .map_err(CoolPropError::from)?;
        Ok(SpecificHeatCapacity::new::<joule_per_kilogram_kelvin>(cv))
    }
}

impl<F: CoolPropFluid> StateFrom<(F, ThermodynamicTemperature, MassDensity)> for CoolProp<F> {
    type Error = CoolPropError;

    fn state_from(
        &self,
        (fluid, temperature, density): (F, ThermodynamicTemperature, MassDensity),
    ) -> Result<State<F>, Self::Error> {
        let mut abstract_state = self.state.lock()?;
        // Update CoolProp to validate the T-D state and surface invalid inputs early.
        abstract_state.update(
            InputPair::DMASS_T,
            density.get::<kilogram_per_cubic_meter>(),
            temperature.get::<kelvin>(),
        )?;

        Ok(State {
            temperature,
            density,
            fluid,
        })
    }
}

impl<F: CoolPropFluid> StateFrom<(F, ThermodynamicTemperature, Pressure)> for CoolProp<F> {
    type Error = CoolPropError;

    fn state_from(
        &self,
        (fluid, temperature, pressure): (F, ThermodynamicTemperature, Pressure),
    ) -> Result<State<F>, Self::Error> {
        let mut abstract_state = self.state.lock()?;
        abstract_state.update(
            InputPair::PT,
            pressure.get::<pascal>(),
            temperature.get::<kelvin>(),
        )?;

        let density = abstract_state.keyed_output(OutputParam::DMASS)?;

        Ok(State {
            temperature,
            density: MassDensity::new::<kilogram_per_cubic_meter>(density),
            fluid,
        })
    }
}

impl<F: CoolPropFluid> StateFrom<(F, Pressure, SpecificEnthalpy)> for CoolProp<F> {
    type Error = CoolPropError;

    fn state_from(
        &self,
        (fluid, pressure, enthalpy): (F, Pressure, SpecificEnthalpy),
    ) -> Result<State<F>, Self::Error> {
        let mut abstract_state = self.state.lock()?;
        abstract_state.update(
            InputPair::HMASS_P,
            enthalpy.get::<joule_per_kilogram>(),
            pressure.get::<pascal>(),
        )?;

        let temperature = abstract_state.keyed_output(OutputParam::T)?;
        let density = abstract_state.keyed_output(OutputParam::DMASS)?;

        Ok(State {
            temperature: ThermodynamicTemperature::new::<kelvin>(temperature),
            density: MassDensity::new::<kilogram_per_cubic_meter>(density),
            fluid,
        })
    }
}

impl<F: CoolPropFluid> StateFrom<(F, Pressure, SpecificEntropy)> for CoolProp<F> {
    type Error = CoolPropError;

    fn state_from(
        &self,
        (fluid, pressure, entropy): (F, Pressure, SpecificEntropy),
    ) -> Result<State<F>, Self::Error> {
        let mut abstract_state = self.state.lock()?;
        abstract_state.update(
            InputPair::PS_MASS,
            pressure.get::<pascal>(),
            entropy.get::<joule_per_kilogram_kelvin>(),
        )?;

        let temperature = abstract_state.keyed_output(OutputParam::T)?;
        let density = abstract_state.keyed_output(OutputParam::DMASS)?;

        Ok(State {
            temperature: ThermodynamicTemperature::new::<kelvin>(temperature),
            density: MassDensity::new::<kilogram_per_cubic_meter>(density),
            fluid,
        })
    }
}

impl<F: CoolPropFluid> StateFrom<(F, SpecificEnthalpy, SpecificEntropy)> for CoolProp<F> {
    type Error = CoolPropError;

    fn state_from(
        &self,
        (fluid, enthalpy, entropy): (F, SpecificEnthalpy, SpecificEntropy),
    ) -> Result<State<F>, Self::Error> {
        let mut abstract_state = self.state.lock()?;
        abstract_state.update(
            InputPair::HMASS_SMASS,
            enthalpy.get::<joule_per_kilogram>(),
            entropy.get::<joule_per_kilogram_kelvin>(),
        )?;

        let temperature = abstract_state.keyed_output(OutputParam::T)?;
        let density = abstract_state.keyed_output(OutputParam::DMASS)?;

        Ok(State {
            temperature: ThermodynamicTemperature::new::<kelvin>(temperature),
            density: MassDensity::new::<kilogram_per_cubic_meter>(density),
            fluid,
        })
    }
}

// Static assertion: `CoolProp<F>` must be `Send + Sync` for any `CoolPropFluid`.
// Thread safety is provided by `COOLPROP_LOCK` in `wrapper.rs`, which serializes
// all CoolProp FFI calls. The local `Mutex<AbstractState>` provides interior
// mutability and keeps update/query call pairs atomic.
// TODO: remove when CoolProp<F> gains a public method that exercises the
// Send + Sync bound (e.g., a parallel property evaluation API).
#[allow(dead_code)]
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check<F: CoolPropFluid>() {
        assert_send_sync::<CoolProp<F>>();
    }
};

// Exact `assert_eq!` on f64 is intentional in the spot-check tests —
// they verify bit-for-bit reproducibility across CoolProp builds.
#[allow(clippy::float_cmp)]
#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use uom::si::{
        available_energy::kilojoule_per_kilogram,
        f64::{MassDensity, ThermodynamicTemperature},
        mass_density::kilogram_per_cubic_meter,
        molar_mass::gram_per_mole,
        pressure::megapascal,
        specific_heat_capacity::{joule_per_kilogram_kelvin, kilojoule_per_kilogram_kelvin},
        thermodynamic_temperature::{degree_celsius, kelvin},
    };

    use crate::support::thermo::fluid::{CarbonDioxide, Water};

    fn co2_model() -> CoolProp<CarbonDioxide> {
        CoolProp::<CarbonDioxide>::new().unwrap()
    }

    fn co2_state() -> State<CarbonDioxide> {
        State::new(
            ThermodynamicTemperature::new::<degree_celsius>(42.0),
            MassDensity::new::<kilogram_per_cubic_meter>(670.0),
            CarbonDioxide,
        )
    }

    fn water_model() -> CoolProp<Water> {
        CoolProp::<Water>::new().unwrap()
    }

    fn water_state() -> State<Water> {
        State::new(
            ThermodynamicTemperature::new::<degree_celsius>(25.0),
            MassDensity::new::<kilogram_per_cubic_meter>(1000.0),
            Water,
        )
    }

    // ── Spot-check tests ──────────────────────────────────────────────
    //
    // Assert exact `f64` values for CO₂ at 42 °C / 670 kg/m³.
    // Running with both `coolprop-static` and `coolprop-dylib` verifies
    // the from-source build matches the prebuilt shared library at full
    // precision (not just within an epsilon).
    //
    // Reference values captured from CoolProp v7.2.0 (official macOS
    // AArch64 shared library via `coolprop-sys-macos-aarch64`).

    #[test]
    #[ignore = "offline verification — run manually when updating CoolProp"]
    fn co2_spot_check_pressure() {
        let model = co2_model();
        let state = co2_state();
        let value = model.pressure(&state).unwrap().get::<pascal>();
        assert_eq!(value, 1.133_616_265_282_128_8e7);
    }

    #[test]
    #[ignore = "offline verification — run manually when updating CoolProp"]
    fn co2_spot_check_internal_energy() {
        let model = co2_model();
        let state = co2_state();
        let value = model
            .internal_energy(&state)
            .unwrap()
            .get::<joule_per_kilogram>();
        assert_eq!(value, 2.909_564_765_862_165e5);
    }

    #[test]
    #[ignore = "offline verification — run manually when updating CoolProp"]
    fn co2_spot_check_enthalpy() {
        let model = co2_model();
        let state = co2_state();
        let value = model.enthalpy(&state).unwrap().get::<joule_per_kilogram>();
        assert_eq!(value, 3.078_761_223_366_96e5);
    }

    #[test]
    #[ignore = "offline verification — run manually when updating CoolProp"]
    fn co2_spot_check_entropy() {
        let model = co2_model();
        let state = co2_state();
        let value = model
            .entropy(&state)
            .unwrap()
            .get::<joule_per_kilogram_kelvin>();
        assert_eq!(value, 1.333_274_008_373_242_2e3);
    }

    #[test]
    #[ignore = "offline verification — run manually when updating CoolProp"]
    fn co2_spot_check_cp() {
        let model = co2_model();
        let state = co2_state();
        let value = model.cp(&state).unwrap().get::<joule_per_kilogram_kelvin>();
        assert_eq!(value, 4.125_049_199_079_285e3);
    }

    #[test]
    #[ignore = "offline verification — run manually when updating CoolProp"]
    fn co2_spot_check_cv() {
        let model = co2_model();
        let state = co2_state();
        let value = model.cv(&state).unwrap().get::<joule_per_kilogram_kelvin>();
        assert_eq!(value, 9.805_326_153_531_056e2);
    }

    #[test]
    fn co2_molar_mass_matches_expected() {
        let model = co2_model();
        let molar_mass = model.molar_mass().unwrap();
        assert_relative_eq!(molar_mass.get::<gram_per_mole>(), 44.0098);
    }

    #[test]
    fn co2_pressure_matches_expected() {
        let model = co2_model();
        let state = co2_state();
        let pressure = model.pressure(&state).unwrap();
        assert_relative_eq!(pressure.get::<megapascal>(), 11.3362, epsilon = 1e-4);
    }

    #[test]
    fn co2_internal_energy_matches_expected() {
        let model = co2_model();
        let state = co2_state();
        let internal_energy = model.internal_energy(&state).unwrap();
        assert_relative_eq!(
            internal_energy.get::<kilojoule_per_kilogram>(),
            290.9565,
            epsilon = 1e-4
        );
    }

    #[test]
    fn co2_enthalpy_matches_expected() {
        let model = co2_model();
        let state = co2_state();
        let enthalpy = model.enthalpy(&state).unwrap();
        assert_relative_eq!(
            enthalpy.get::<kilojoule_per_kilogram>(),
            307.8761,
            epsilon = 1e-4
        );
    }

    #[test]
    fn co2_entropy_matches_expected() {
        let model = co2_model();
        let state = co2_state();
        let entropy = model.entropy(&state).unwrap();
        assert_relative_eq!(
            entropy.get::<kilojoule_per_kilogram_kelvin>(),
            1.3333,
            epsilon = 1e-4
        );
    }

    #[test]
    fn co2_cp_matches_expected() {
        let model = co2_model();
        let state = co2_state();
        let cp = model.cp(&state).unwrap();
        assert_relative_eq!(
            cp.get::<kilojoule_per_kilogram_kelvin>(),
            4.125,
            epsilon = 1e-4
        );
    }

    #[test]
    fn co2_cv_matches_expected() {
        let model = co2_model();
        let state = co2_state();
        let cv = model.cv(&state).unwrap();
        assert_relative_eq!(
            cv.get::<joule_per_kilogram_kelvin>(),
            980.5326,
            epsilon = 1e-4
        );
    }

    #[test]
    fn co2_state_from_temperature_pressure_roundtrips_from_temperature_density() {
        let model = co2_model();

        let state = co2_state();
        let pressure = model.pressure(&state).unwrap();
        let roundtrip = model
            .state_from((CarbonDioxide, state.temperature, pressure))
            .unwrap();

        assert_relative_eq!(
            roundtrip.density.get::<kilogram_per_cubic_meter>(),
            state.density.get::<kilogram_per_cubic_meter>(),
            max_relative = 1e-9
        );
    }

    #[test]
    fn water_state_from_pressure_enthalpy_roundtrips_from_temperature_density() {
        let model = water_model();

        let state = water_state();
        let pressure = model.pressure(&state).unwrap();
        let enthalpy = model.enthalpy(&state).unwrap();
        let roundtrip = model.state_from((Water, pressure, enthalpy)).unwrap();

        assert_relative_eq!(
            roundtrip.temperature.get::<kelvin>(),
            state.temperature.get::<kelvin>(),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            roundtrip.density.get::<kilogram_per_cubic_meter>(),
            state.density.get::<kilogram_per_cubic_meter>(),
            max_relative = 1e-9
        );
    }

    #[test]
    fn water_state_from_pressure_entropy_roundtrips_from_temperature_density() {
        let model = water_model();

        let state = water_state();
        let pressure = model.pressure(&state).unwrap();
        let entropy = model.entropy(&state).unwrap();
        let roundtrip = model.state_from((Water, pressure, entropy)).unwrap();

        assert_relative_eq!(
            roundtrip.temperature.get::<kelvin>(),
            state.temperature.get::<kelvin>(),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            roundtrip.density.get::<kilogram_per_cubic_meter>(),
            state.density.get::<kilogram_per_cubic_meter>(),
            max_relative = 1e-9
        );
    }

    #[test]
    fn water_state_from_enthalpy_entropy_roundtrips_from_temperature_density() {
        let model = water_model();

        let state = water_state();
        let enthalpy = model.enthalpy(&state).unwrap();
        let entropy = model.entropy(&state).unwrap();
        let roundtrip = model.state_from((Water, enthalpy, entropy)).unwrap();

        assert_relative_eq!(
            roundtrip.temperature.get::<kelvin>(),
            state.temperature.get::<kelvin>(),
            max_relative = 1e-9
        );
        assert_relative_eq!(
            roundtrip.density.get::<kilogram_per_cubic_meter>(),
            state.density.get::<kilogram_per_cubic_meter>(),
            max_relative = 1e-9
        );
    }
}
