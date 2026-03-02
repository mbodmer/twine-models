//! Capability traits used to query and construct thermodynamic states.
//!
//! All traits in this module have blanket impls for `&T`, so borrowed models
//! satisfy the same bounds as owned ones.

mod base;
mod properties;
mod state_from;

pub use base::ThermoModel;
pub use properties::*;
pub use state_from::StateFrom;
