use std::sync::PoisonError;

use thiserror::Error;

use crate::support::thermo::PropertyError;

use super::wrapper::WrapperError;

/// Errors returned by the [`CoolProp`](super::CoolProp) model.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoolPropError {
    /// An error reported by the CoolProp C API.
    #[error("{0}")]
    CoolProp(String),

    /// The internal `AbstractState` mutex was poisoned.
    #[error("CoolProp abstract state mutex poisoned")]
    Poisoned,
}

impl From<WrapperError> for CoolPropError {
    fn from(e: WrapperError) -> Self {
        CoolPropError::CoolProp(e.to_string())
    }
}

impl<T> From<PoisonError<T>> for CoolPropError {
    fn from(_: PoisonError<T>) -> Self {
        CoolPropError::Poisoned
    }
}

impl From<CoolPropError> for PropertyError {
    fn from(error: CoolPropError) -> Self {
        match error {
            CoolPropError::CoolProp(message) => map_error_message(&message),
            CoolPropError::Poisoned => PropertyError::Calculation {
                context: "CoolProp abstract state mutex poisoned".to_string(),
            },
        }
    }
}

/// Maps a CoolProp error message to a [`PropertyError`] variant.
///
/// CoolProp errors are opaque strings with no structured error codes. This
/// function classifies them into [`PropertyError`] variants via substring
/// matching on a best-effort basis.
///
/// Unrecognised messages fall back to [`PropertyError::Calculation`],
/// preserving the original text for debugging.
fn map_error_message(message: &str) -> PropertyError {
    const UNDEFINED_MARKERS: &[&str] = &["not defined"];
    const OUT_OF_DOMAIN_MARKERS: &[&str] = &[
        "not in range",
        "out of range",
        "outside the range of validity",
        "must be in range",
        "must be between",
        "quality must be",
    ];
    const INVALID_STATE_MARKERS: &[&str] =
        &["not a valid number", "invalid state", "invalid number"];

    let lowered = message.to_lowercase();
    let context = message.to_string();

    if contains_any(&lowered, UNDEFINED_MARKERS) {
        PropertyError::Undefined { context }
    } else if contains_any(&lowered, OUT_OF_DOMAIN_MARKERS) {
        PropertyError::OutOfDomain { context }
    } else if contains_any(&lowered, INVALID_STATE_MARKERS) {
        PropertyError::InvalidState { context }
    } else {
        PropertyError::Calculation { context }
    }
}

/// Returns `true` if any needle is found in the haystack.
fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_error_message_returns_undefined() {
        let error = map_error_message("inputs are not defined for this region");
        assert!(matches!(error, PropertyError::Undefined { .. }));
    }

    #[test]
    fn map_error_message_returns_out_of_domain() {
        let error = map_error_message("input 3.14 is out of range");
        assert!(matches!(error, PropertyError::OutOfDomain { .. }));
    }

    #[test]
    fn map_error_message_returns_invalid_state() {
        let error = map_error_message("p is not a valid number");
        assert!(matches!(error, PropertyError::InvalidState { .. }));
    }

    #[test]
    fn map_error_message_returns_calculation_by_default() {
        let error = map_error_message("some other failure");
        assert!(matches!(error, PropertyError::Calculation { .. }));
    }
}
