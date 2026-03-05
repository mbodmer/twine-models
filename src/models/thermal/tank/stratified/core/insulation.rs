use uom::si::f64::HeatTransfer;

/// Options for specifying tank insulation.
///
/// The insulation setting determines the overall heat transfer coefficient
/// (U-value) applied between the tank fluid and the surrounding environment
/// on each exposed surface.
///
/// During tank construction, each U-value is multiplied by the corresponding
/// surface area to produce a thermal conductance (UA).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Insulation {
    /// The tank is perfectly insulated — no heat transfer to the environment.
    Adiabatic,

    /// Heat transfer to the environment defined by per-face U-values.
    ///
    /// Usually a loss, but could be a heat gain if the environment is warmer
    /// than the tank fluid.
    UValue {
        /// Heat transfer coefficient (U-value) of the bottom face to the environment.
        bottom: HeatTransfer,

        /// Heat transfer coefficient (U-value) of the side face to the environment.
        side: HeatTransfer,

        /// Heat transfer coefficient (U-value) of the top face to the environment.
        top: HeatTransfer,
    },
}

impl Insulation {
    /// Uniform heat transfer coefficient (U-value) to the environment on every face.
    #[must_use]
    pub fn uniform(u: HeatTransfer) -> Self {
        Self::UValue {
            bottom: u,
            side: u,
            top: u,
        }
    }

    /// Per-face heat transfer coefficient (U-value) to the environment.
    #[must_use]
    pub fn u_value(bottom: HeatTransfer, side: HeatTransfer, top: HeatTransfer) -> Self {
        Self::UValue { bottom, side, top }
    }
}
