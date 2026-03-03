use twine_core::StepIntegrable;
use uom::si::{
    f64::{SpecificHeatCapacity, Time},
    specific_heat_capacity::joule_per_kilogram_kelvin,
};

use crate::support::thermo::model::perfect_gas::{PerfectGasFluid, PerfectGasParameters};
use crate::support::units::SpecificGasConstant;

#[cfg(feature = "coolprop-static")]
use crate::support::thermo::model::coolprop::CoolPropFluid;

/// Canonical identifier for carbon dioxide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CarbonDioxide;

impl PerfectGasFluid for CarbonDioxide {
    fn parameters() -> PerfectGasParameters {
        PerfectGasParameters::new(
            SpecificGasConstant::new::<joule_per_kilogram_kelvin>(188.92),
            SpecificHeatCapacity::new::<joule_per_kilogram_kelvin>(844.0),
        )
    }
}

impl StepIntegrable<Time> for CarbonDioxide {
    type Derivative = ();

    fn step(&self, (): (), _: Time) -> Self {
        *self
    }
}

#[cfg(feature = "coolprop-static")]
impl CoolPropFluid for CarbonDioxide {
    const BACKEND: &'static str = "HEOS";
    const NAME: &'static str = "CarbonDioxide";
}
