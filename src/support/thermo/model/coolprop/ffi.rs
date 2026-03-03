//! Raw FFI bindings to the CoolProp C API.
//!
//! This module declares the four `AbstractState_*` functions used from
//! `CoolPropLib.h`, plus typed constants for input pairs and output parameters.
//! Everything here is `unsafe` — use the wrapper layer above.

use std::os::raw::{c_char, c_double, c_long};

// ── Extern declarations ───────────────────────────────────────────────────────

unsafe extern "C" {
    /// Create a new `AbstractState` and return its integer handle.
    ///
    /// Returns `-1` on failure; check `errcode` and `message_buffer`.
    pub fn AbstractState_factory(
        backend: *const c_char,
        fluids: *const c_char,
        errcode: *mut c_long,
        message_buffer: *mut c_char,
        buffer_length: c_long,
    ) -> c_long;

    /// Release an `AbstractState` created by [`AbstractState_factory`].
    pub fn AbstractState_free(
        handle: c_long,
        errcode: *mut c_long,
        message_buffer: *mut c_char,
        buffer_length: c_long,
    );

    /// Update the thermodynamic state.
    pub fn AbstractState_update(
        handle: c_long,
        input_pair: c_long,
        value1: c_double,
        value2: c_double,
        errcode: *mut c_long,
        message_buffer: *mut c_char,
        buffer_length: c_long,
    );

    /// Query a single output parameter from the current state.
    ///
    /// Returns `f64::MAX` (HUGE_VAL equivalent) on failure; check `errcode`.
    pub fn AbstractState_keyed_output(
        handle: c_long,
        param: c_long,
        errcode: *mut c_long,
        message_buffer: *mut c_char,
        buffer_length: c_long,
    ) -> c_double;
}

// ── Input pairs ───────────────────────────────────────────────────────────────

/// An input-pair code for [`AbstractState_update`].
///
/// Values match the `CoolProp::input_pairs` C++ enum in `DataStructures.h`.
/// Only the pairs used in this crate are declared here; extend as needed.
///
/// A newtype over `c_long` (not `#[repr(i64)]`) so the value is correct on
/// both native (64-bit `c_long`) and WASM (32-bit `c_long`) targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputPair(c_long);

impl InputPair {
    /// Mass density (kg/m³) + temperature (K).
    pub const DMASS_T: Self = Self(10);

    /// Pressure (Pa) + temperature (K).
    pub const PT: Self = Self(9);

    /// Mass enthalpy (J/kg) + pressure (Pa).
    pub const HMASS_P: Self = Self(20);

    /// Pressure (Pa) + mass entropy (J/kg/K).
    pub const PS_MASS: Self = Self(22);

    /// Mass enthalpy (J/kg) + mass entropy (J/kg/K).
    pub const HMASS_SMASS: Self = Self(26);

    /// Returns the raw `c_long` value.
    pub const fn as_c_long(self) -> c_long {
        self.0
    }
}

// ── Output parameters ─────────────────────────────────────────────────────────

/// An output-parameter code for [`AbstractState_keyed_output`].
///
/// Values match the `CoolProp::parameters` C++ enum in `DataStructures.h`.
/// Only the parameters used in this crate are declared here; extend as needed.
///
/// A newtype over `c_long` (not `#[repr(i64)]`) so the value is correct on
/// both native (64-bit `c_long`) and WASM (32-bit `c_long`) targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputParam(c_long);

impl OutputParam {
    /// Molar mass (kg/mol) — a trivial (state-independent) property.
    pub const MOLAR_MASS: Self = Self(2);

    /// Temperature (K).
    pub const T: Self = Self(19);

    /// Pressure (Pa).
    pub const P: Self = Self(20);

    /// Mass-based density (kg/m³).
    pub const DMASS: Self = Self(39);

    /// Mass-based enthalpy (J/kg).
    pub const HMASS: Self = Self(40);

    /// Mass-based entropy (J/kg/K).
    pub const SMASS: Self = Self(41);

    /// Mass-based constant-pressure specific heat (J/kg/K).
    pub const CP_MASS: Self = Self(42);

    /// Mass-based constant-volume specific heat (J/kg/K).
    pub const CV_MASS: Self = Self(44);

    /// Mass-based internal energy (J/kg).
    pub const UMASS: Self = Self(45);

    /// Returns the raw `c_long` value.
    pub const fn as_c_long(self) -> c_long {
        self.0
    }
}
