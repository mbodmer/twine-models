//! Safe Rust wrapper around a `CoolProp` `AbstractState` handle.

use std::{
    ffi::{CString, NulError},
    os::raw::{c_char, c_long},
    sync::Mutex,
};

use super::ffi::{self, InputPair, OutputParam};

/// Size of the error message buffer passed to every `CoolProp` C API call.
const MSG_BUF_LEN: usize = 512;

/// Global mutex that serializes all `CoolProp` C API calls.
///
/// `CoolProp` does not document thread-safety guarantees for its C API.
/// Source inspection (v7.2.0) suggests per-handle operations are independent
/// — the handle manager has its own mutex, and `update`/`keyed_output` operate
/// on per-instance C++ objects — but the flash routines and Helmholtz backends
/// have not been audited for global mutable state.
/// This lock is conservative: it may be removable if a thorough audit or
/// upstream guarantee confirms per-handle thread safety.
///
/// The `CoolProp<F>` wrapper adds a second, per-instance
/// `Mutex<AbstractState>` to keep `update`/`keyed_output` pairs atomic.
/// That lock is needed regardless — it prevents interleaved calls on a single
/// handle.
static COOLPROP_LOCK: Mutex<()> = Mutex::new(());

/// A safe owner of a `CoolProp` `AbstractState` handle.
///
/// Calls `AbstractState_factory` on construction and `AbstractState_free` on drop.
/// All methods return `Result`, converting `CoolProp` error strings
/// into [`WrapperError`].
pub struct AbstractState {
    handle: c_long,
}

/// Errors returned by [`AbstractState`] methods.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WrapperError {
    /// A string argument contained an interior null byte.
    #[error("invalid C string argument: {0}")]
    InvalidCString(#[from] NulError),

    /// `CoolProp` reported a non-zero error code.
    #[error("{0}")]
    CoolProp(String),
}

impl AbstractState {
    /// Construct a new `AbstractState` for the given backend and fluid name.
    ///
    /// # Errors
    ///
    /// Returns [`WrapperError`] if either string contains an interior null or
    /// if `CoolProp` rejects the backend/fluid combination.
    pub fn new(backend: &str, fluid: &str) -> Result<Self, WrapperError> {
        let backend_c = CString::new(backend)?;
        let fluid_c = CString::new(fluid)?;

        let mut errcode: c_long = 0;
        let mut buf = [0u8; MSG_BUF_LEN];

        let _guard = COOLPROP_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // SAFETY: pointers are valid for the duration of the call; the global
        // lock ensures no concurrent CoolProp FFI call is in progress.
        let handle = unsafe {
            ffi::AbstractState_factory(
                backend_c.as_ptr(),
                fluid_c.as_ptr(),
                &raw mut errcode,
                buf.as_mut_ptr().cast::<c_char>(),
                c_long::try_from(MSG_BUF_LEN).expect("buffer length fits c_long"),
            )
        };

        if errcode != 0 {
            return Err(WrapperError::CoolProp(read_message(&buf)));
        }

        Ok(Self { handle })
    }

    /// Update the thermodynamic state from an input pair and two values.
    ///
    /// # Errors
    ///
    /// Returns [`WrapperError::CoolProp`] if `CoolProp` rejects the state.
    pub fn update(&mut self, pair: InputPair, v1: f64, v2: f64) -> Result<(), WrapperError> {
        let mut errcode: c_long = 0;
        let mut buf = [0u8; MSG_BUF_LEN];

        let _guard = COOLPROP_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // SAFETY: handle is valid; buffer is alive for the call duration;
        // the global lock prevents concurrent FFI calls.
        unsafe {
            ffi::AbstractState_update(
                self.handle,
                pair.as_c_long(),
                v1,
                v2,
                &raw mut errcode,
                buf.as_mut_ptr().cast::<c_char>(),
                c_long::try_from(MSG_BUF_LEN).expect("buffer length fits c_long"),
            );
        }

        if errcode != 0 {
            return Err(WrapperError::CoolProp(read_message(&buf)));
        }

        Ok(())
    }

    /// Query a single output parameter from the current state.
    ///
    /// Works for both state-dependent outputs and trivial (state-independent)
    /// outputs such as [`OutputParam::MOLAR_MASS`].
    ///
    /// # Errors
    ///
    /// Returns [`WrapperError::CoolProp`] if `CoolProp` signals an error.
    pub fn keyed_output(&self, param: OutputParam) -> Result<f64, WrapperError> {
        let mut errcode: c_long = 0;
        let mut buf = [0u8; MSG_BUF_LEN];

        let _guard = COOLPROP_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // SAFETY: handle is valid; buffer is alive for the call duration;
        // the global lock prevents concurrent FFI calls.
        let value = unsafe {
            ffi::AbstractState_keyed_output(
                self.handle,
                param.as_c_long(),
                &raw mut errcode,
                buf.as_mut_ptr().cast::<c_char>(),
                c_long::try_from(MSG_BUF_LEN).expect("buffer length fits c_long"),
            )
        };

        if errcode != 0 {
            return Err(WrapperError::CoolProp(read_message(&buf)));
        }

        Ok(value)
    }
}

impl Drop for AbstractState {
    fn drop(&mut self) {
        let mut errcode: c_long = 0;
        let mut buf = [0u8; MSG_BUF_LEN];

        let _guard = COOLPROP_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // SAFETY: handle is valid; the global lock prevents concurrent FFI
        // calls. Errors during drop are silently discarded — there is no clean
        // way to propagate them.
        unsafe {
            ffi::AbstractState_free(
                self.handle,
                &raw mut errcode,
                buf.as_mut_ptr().cast::<c_char>(),
                c_long::try_from(MSG_BUF_LEN).expect("buffer length fits c_long"),
            );
        }
    }
}

// SAFETY: `AbstractState` is a handle into CoolProp's global registry. All
// access is serialized through `COOLPROP_LOCK`, so it is safe to send across
// thread boundaries.
unsafe impl Send for AbstractState {}

/// Read a null-terminated C string from a byte buffer.
fn read_message(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}
