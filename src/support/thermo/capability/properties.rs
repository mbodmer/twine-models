use uom::si::f64::{Pressure, SpecificHeatCapacity};

use crate::support::thermo::{PropertyError, State};
use crate::support::units::{SpecificEnthalpy, SpecificEntropy, SpecificInternalEnergy};

use super::ThermoModel;

pub trait HasPressure: ThermoModel {
    /// Returns the pressure for the given state.
    ///
    /// # Errors
    ///
    /// Returns [`PropertyError`] if the pressure cannot be calculated.
    fn pressure(&self, state: &State<Self::Fluid>) -> Result<Pressure, PropertyError>;
}

pub trait HasInternalEnergy: ThermoModel {
    /// Returns the specific internal energy for the given state.
    ///
    /// # Errors
    ///
    /// Returns [`PropertyError`] if the internal energy cannot be calculated.
    fn internal_energy(
        &self,
        state: &State<Self::Fluid>,
    ) -> Result<SpecificInternalEnergy, PropertyError>;
}

pub trait HasEnthalpy: ThermoModel {
    /// Returns the specific enthalpy for the given state.
    ///
    /// # Errors
    ///
    /// Returns [`PropertyError`] if the enthalpy cannot be calculated.
    fn enthalpy(&self, state: &State<Self::Fluid>) -> Result<SpecificEnthalpy, PropertyError>;
}

pub trait HasEntropy: ThermoModel {
    /// Returns the specific entropy for the given state.
    ///
    /// The computation depends on the model's assumptions. For example,
    /// ideal gas models include a pressure term while incompressible
    /// liquid models do not.
    ///
    /// # Errors
    ///
    /// Returns [`PropertyError`] if the entropy cannot be calculated.
    fn entropy(&self, state: &State<Self::Fluid>) -> Result<SpecificEntropy, PropertyError>;
}

pub trait HasCp: ThermoModel {
    /// Returns the specific heat capacity at constant pressure for the given state.
    ///
    /// # Errors
    ///
    /// Returns [`PropertyError`] if `cp` cannot be calculated.
    fn cp(&self, state: &State<Self::Fluid>) -> Result<SpecificHeatCapacity, PropertyError>;
}

pub trait HasCv: ThermoModel {
    /// Returns the specific heat capacity at constant volume for the given state.
    ///
    /// # Errors
    ///
    /// Returns [`PropertyError`] if `cv` cannot be calculated.
    fn cv(&self, state: &State<Self::Fluid>) -> Result<SpecificHeatCapacity, PropertyError>;
}

impl<T: HasPressure> HasPressure for &T {
    fn pressure(&self, state: &State<Self::Fluid>) -> Result<Pressure, PropertyError> {
        T::pressure(self, state)
    }
}

impl<T: HasInternalEnergy> HasInternalEnergy for &T {
    fn internal_energy(
        &self,
        state: &State<Self::Fluid>,
    ) -> Result<SpecificInternalEnergy, PropertyError> {
        T::internal_energy(self, state)
    }
}

impl<T: HasEnthalpy> HasEnthalpy for &T {
    fn enthalpy(&self, state: &State<Self::Fluid>) -> Result<SpecificEnthalpy, PropertyError> {
        T::enthalpy(self, state)
    }
}

impl<T: HasEntropy> HasEntropy for &T {
    fn entropy(&self, state: &State<Self::Fluid>) -> Result<SpecificEntropy, PropertyError> {
        T::entropy(self, state)
    }
}

impl<T: HasCp> HasCp for &T {
    fn cp(&self, state: &State<Self::Fluid>) -> Result<SpecificHeatCapacity, PropertyError> {
        T::cp(self, state)
    }
}

impl<T: HasCv> HasCv for &T {
    fn cv(&self, state: &State<Self::Fluid>) -> Result<SpecificHeatCapacity, PropertyError> {
        T::cv(self, state)
    }
}
