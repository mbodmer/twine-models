use uom::si::f64::ThermalConductance;

/// Options for specifying tank insulation.
///
/// The insulation setting determines what conductance value is applied between
/// the tank fluid and the surrounding environment on all exposed surfaces.
///
/// * `Adiabatic` — no heat transfer.
/// * `Conductive { bottom, side, top }` — non-zero conductance for each face.
///
/// Constructors such as [`Insulation::uniform`] and [`Insulation::conductive`]
/// make it easy to build the desired value.use uom::si::f64::ThermalConductance;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Insulation {
    /// The tank is perfectly insulated — no heat transfer to the environment.
    Adiabatic,

    /// Conductive heat loss to the environment. Each face may have its own UA value
    Conductive {
        bottom: ThermalConductance,
        side: ThermalConductance,
        top: ThermalConductance,
    },
}

impl Insulation {
    /// Construct an adiabatic setting (same as `Insulation::Adiabatic`).
    pub fn adiabatic() -> Self {
        Insulation::Adiabatic
    }

    /// Convenience for a uniform conductance on every face.
    pub fn uniform(ua: ThermalConductance) -> Self {
        Insulation::Conductive {
            bottom: ua,
            side: ua,
            top: ua,
        }
    }

    /// Full constructor when faces differ.
    pub fn conductive(
        bottom: ThermalConductance,
        side: ThermalConductance,
        top: ThermalConductance,
    ) -> Self {
        Insulation::Conductive { bottom, side, top }
    }
}
