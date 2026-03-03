//! WASM smoke test for `CoolProp`.
//!
//! Verifies that `CoolProp` compiled to WASM via Emscripten produces the same
//! thermodynamic results as the native build.
//! Run with:
//! ```sh
//! cargo test --target wasm32-unknown-emscripten --features coolprop-static --tests
//! ```

#![cfg(all(target_arch = "wasm32", feature = "coolprop-static"))]

use approx::assert_relative_eq;
use uom::si::{
    f64::{MassDensity, Pressure, ThermodynamicTemperature},
    mass_density::kilogram_per_cubic_meter,
    pressure::megapascal,
    thermodynamic_temperature::degree_celsius,
};

use twine_models::support::thermo::{
    State,
    capability::{HasPressure, StateFrom},
    fluid::CarbonDioxide,
    model::CoolProp,
};

#[test]
fn co2_pressure_matches_native() {
    let model = CoolProp::<CarbonDioxide>::new().unwrap();
    let state = State::new(
        ThermodynamicTemperature::new::<degree_celsius>(42.0),
        MassDensity::new::<kilogram_per_cubic_meter>(670.0),
        CarbonDioxide,
    );
    let pressure = model.pressure(&state).unwrap();
    assert_relative_eq!(pressure.get::<megapascal>(), 11.3362, epsilon = 1e-4);
}

#[test]
fn co2_state_from_temperature_pressure_roundtrips() {
    let model = CoolProp::<CarbonDioxide>::new().unwrap();
    let state = State::new(
        ThermodynamicTemperature::new::<degree_celsius>(42.0),
        MassDensity::new::<kilogram_per_cubic_meter>(670.0),
        CarbonDioxide,
    );
    let pressure = model.pressure(&state).unwrap();

    let roundtrip: State<CarbonDioxide> = model
        .state_from((CarbonDioxide, state.temperature, pressure))
        .unwrap();

    assert_relative_eq!(
        roundtrip.density.get::<kilogram_per_cubic_meter>(),
        state.density.get::<kilogram_per_cubic_meter>(),
        max_relative = 1e-9
    );
}
