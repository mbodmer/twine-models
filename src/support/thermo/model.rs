//! Thermodynamic property models.

pub mod incompressible;
pub mod perfect_gas;

pub(crate) mod ideal_gas_eos;

#[cfg(feature = "coolprop-static")]
#[cfg_attr(docsrs, doc(cfg(feature = "coolprop-static")))]
pub mod coolprop;

pub use incompressible::Incompressible;
pub use perfect_gas::PerfectGas;

#[cfg(feature = "coolprop-static")]
#[cfg_attr(docsrs, doc(cfg(feature = "coolprop-static")))]
pub use coolprop::CoolProp;
