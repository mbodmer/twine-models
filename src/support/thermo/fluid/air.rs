use twine_core::StepIntegrable;
use uom::si::{
    f64::{SpecificHeatCapacity, Time},
    specific_heat_capacity::joule_per_kilogram_kelvin,
};

use crate::support::thermo::model::perfect_gas::{PerfectGasFluid, PerfectGasParameters};
use crate::support::units::SpecificGasConstant;

/// Canonical identifier for dry air.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Air;

impl PerfectGasFluid for Air {
    fn parameters() -> PerfectGasParameters {
        PerfectGasParameters::new(
            SpecificGasConstant::new::<joule_per_kilogram_kelvin>(287.053),
            SpecificHeatCapacity::new::<joule_per_kilogram_kelvin>(1005.0),
        )
    }
}

impl StepIntegrable<Time> for Air {
    type Derivative = ();

    fn step(&self, (): (), _: Time) -> Self {
        *self
    }
}
