use twine_core::StepIntegrable;
use uom::si::{
    f64::{MassDensity, SpecificHeatCapacity, Time},
    mass_density::kilogram_per_cubic_meter,
    specific_heat_capacity::kilojoule_per_kilogram_kelvin,
};

use crate::support::thermo::model::incompressible::{
    IncompressibleFluid, IncompressibleParameters,
};

/// Canonical identifier for water.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Water;

impl IncompressibleFluid for Water {
    fn parameters() -> IncompressibleParameters {
        IncompressibleParameters::new(
            SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.184),
            MassDensity::new::<kilogram_per_cubic_meter>(997.047),
        )
    }
}

impl StepIntegrable<Time> for Water {
    type Derivative = ();

    fn step(&self, (): (), _: Time) -> Self {
        *self
    }
}

#[cfg(feature = "coolprop")]
impl crate::support::thermo::model::coolprop::CoolPropFluid for Water {
    const BACKEND: &'static str = "HEOS";
    const NAME: &'static str = "Water";
}
