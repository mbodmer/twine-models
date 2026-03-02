//! Models using the discretized heat exchanger approach.
//!
//! A discretized heat exchanger divides the flow into a linear series of
//! constant-property sub-exchangers so thermodynamic properties can vary
//! along a linear array of nodes, supporting real-fluid behavior.

pub(crate) mod core;
pub mod recuperator;

pub use recuperator::{
    Recuperator, RecuperatorConfig, RecuperatorError, RecuperatorInput, RecuperatorOutput,
};

pub use core::{HeatTransferRate, Inlets, MassFlows, MinDeltaT, PressureDrops};
