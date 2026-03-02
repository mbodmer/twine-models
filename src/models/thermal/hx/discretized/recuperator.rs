use std::{error::Error as StdError, marker::PhantomData};

use thiserror::Error;
use twine_core::Model;
use uom::si::f64::{TemperatureInterval, ThermalConductance};

use crate::{
    models::thermal::hx::discretized::core::{
        DiscretizedHx, DiscretizedHxThermoModel, GivenUaConfig, GivenUaError, GivenUaResults,
        Inlets, Known, MassFlows, MinDeltaT, PressureDrops,
    },
    support::{hx::arrangement::CounterFlow, thermo::State},
};

/// A single-fluid counterflow heat exchanger model for heat recovery.
///
/// `Recuperator` implements [`Model`] for use in cycle simulations where heat
/// is transferred between two streams of the same working fluid, typically a
/// hot stream leaving a turbine and a cold stream leaving a compressor.
///
/// # How it works
///
/// The heat exchanger is discretized into segments (sub heat exchangers) so
/// that thermodynamic properties can vary along the flow path, capturing
/// real-fluid behavior. The number of segments controls the fidelity of
/// this discretization.
///
/// Given a target overall thermal conductance (UA), the solver iterates on
/// the outlet temperature to match it. The achieved UA and outlet states
/// are returned.
///
/// # Streams
///
/// Streams are labeled **top** and **bottom**, referring to their position
/// in a schematic layout. The top stream flows left to right; the bottom
/// stream flows right to left (counterflow). Either stream can be the hot
/// or cold side depending on operating conditions.
///
/// # Segments
///
/// The `segments` parameter controls how many constant-property sub heat
/// exchangers the flow is divided into. More segments improve accuracy for
/// fluids with properties that vary significantly with temperature, at the
/// cost of additional computation.
///
/// Supported values: 1, 5, 10, 20, 50.
///
/// Use 1 segment for quick estimates or analytical verification (reduces to
/// the classical ε-NTU result). Use 10-20 for typical engineering accuracy.
/// Use 50 for convergence studies.
///
/// # Example
///
/// ```
/// use twine_core::Model;
/// use twine_models::{
///     models::thermal::hx::discretized::{
///         Recuperator, RecuperatorConfig, RecuperatorInput,
///         Inlets, MassFlows, PressureDrops,
///     },
///     support::thermo::{
///         State,
///         fluid::Air,
///         model::PerfectGas,
///     },
/// };
/// use uom::si::{
///     f64::{MassDensity, MassRate, Pressure, ThermalConductance, ThermodynamicTemperature},
///     mass_density::kilogram_per_cubic_meter,
///     mass_rate::kilogram_per_second,
///     pressure::kilopascal,
///     thermal_conductance::watt_per_kelvin,
///     thermodynamic_temperature::kelvin,
/// };
///
/// // Build a perfect-gas thermo model for air.
/// let thermo = PerfectGas::<Air>::new().unwrap();
///
/// // Construct the recuperator: 10 segments, default solver tolerances.
/// let recuperator = Recuperator::new(&thermo, 10, RecuperatorConfig::default()).unwrap();
///
/// // Define operating conditions.
/// let hot_inlet = State::new(
///     ThermodynamicTemperature::new::<kelvin>(600.0),
///     MassDensity::new::<kilogram_per_cubic_meter>(1.0),
///     Air,
/// );
/// let cold_inlet = State::new(
///     ThermodynamicTemperature::new::<kelvin>(400.0),
///     MassDensity::new::<kilogram_per_cubic_meter>(2.0),
///     Air,
/// );
///
/// let result = recuperator.call(&RecuperatorInput {
///     inlets: Inlets { top: cold_inlet, bottom: hot_inlet },
///     mass_flows: MassFlows::new_unchecked(
///         MassRate::new::<kilogram_per_second>(1.0),
///         MassRate::new::<kilogram_per_second>(1.0),
///     ),
///     pressure_drops: PressureDrops::zero(),
///     ua: ThermalConductance::new::<watt_per_kelvin>(500.0),
/// }).unwrap();
///
/// // The cold side (top) is heated; the hot side (bottom) is cooled.
/// assert!(result.top_outlet.temperature > cold_inlet.temperature);
/// assert!(result.bottom_outlet.temperature < hot_inlet.temperature);
/// ```
#[derive(Debug, Clone)]
pub struct Recuperator<Fluid, Thermo> {
    thermo: Thermo,
    segments: usize,
    config: RecuperatorConfig,
    _fluid: PhantomData<Fluid>,
}

/// Recuperator solver configuration.
#[derive(Debug, Clone, Copy)]
pub struct RecuperatorConfig {
    /// Relative tolerance on UA (dimensionless).
    ///
    /// Convergence is reached when
    /// `|achieved_ua - target_ua| / target_ua < ua_rel_tol`.
    pub ua_rel_tol: f64,

    /// Absolute tolerance on the temperature search variable.
    pub temp_abs_tol: TemperatureInterval,

    /// Maximum number of solver iterations.
    pub max_iters: usize,
}

impl Default for RecuperatorConfig {
    fn default() -> Self {
        Self {
            ua_rel_tol: 1e-6,
            temp_abs_tol: TemperatureInterval::new::<uom::si::temperature_interval::kelvin>(1e-6),
            max_iters: 100,
        }
    }
}

/// Recuperator inputs.
#[derive(Debug, Clone)]
pub struct RecuperatorInput<Fluid> {
    /// Inlet states for top and bottom streams.
    pub inlets: Inlets<Fluid, Fluid>,

    /// Mass flow rates for top and bottom streams (strictly positive).
    pub mass_flows: MassFlows,

    /// Pressure drops for top and bottom streams (non-negative).
    pub pressure_drops: PressureDrops,

    /// Target overall thermal conductance.
    pub ua: ThermalConductance,
}

/// Recuperator outputs.
#[derive(Debug, Clone)]
pub struct RecuperatorOutput<Fluid> {
    /// Top stream outlet state.
    pub top_outlet: State<Fluid>,

    /// Bottom stream outlet state.
    pub bottom_outlet: State<Fluid>,

    /// Heat transfer rate.
    pub q_dot: super::HeatTransferRate,

    /// Achieved overall thermal conductance.
    pub ua: ThermalConductance,

    /// Minimum hot-to-cold temperature difference and its location.
    pub min_delta_t: MinDeltaT,

    /// Number of solver iterations.
    pub iterations: usize,
}

/// Errors from recuperator construction and solving.
#[derive(Debug, Error)]
pub enum RecuperatorError {
    /// The segment count is not supported.
    #[error("unsupported segment count {0}; supported values are 1, 5, 10, 20, 50")]
    UnsupportedSegments(usize),

    /// The solver failed to converge.
    #[error("recuperator solver failed to converge: {message}")]
    Convergence {
        /// Human-readable details from the convergence failure.
        message: String,

        /// Iteration count when available.
        iterations: Option<usize>,
    },

    /// The target UA is negative.
    #[error("target UA must be non-negative, got {0:?}")]
    NegativeUa(ThermalConductance),

    /// A thermodynamic model operation failed.
    ///
    /// This failure can be from property evaluation or state construction.
    #[error("thermodynamic model failed: {context}")]
    ThermoModelFailed {
        /// Operation context for the thermodynamic model failure.
        context: String,

        /// Underlying thermodynamic model error.
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl<Fluid, Thermo> Recuperator<Fluid, Thermo> {
    /// Creates a recuperator model.
    ///
    /// # Errors
    ///
    /// Returns [`RecuperatorError::UnsupportedSegments`] if `segments` is not in
    /// `{1, 5, 10, 20, 50}`.
    pub fn new(
        thermo: Thermo,
        segments: usize,
        config: RecuperatorConfig,
    ) -> Result<Self, RecuperatorError> {
        if !matches!(segments, 1 | 5 | 10 | 20 | 50) {
            return Err(RecuperatorError::UnsupportedSegments(segments));
        }

        Ok(Self {
            thermo,
            segments,
            config,
            _fluid: PhantomData,
        })
    }

    fn solve<const N: usize>(
        &self,
        input: &RecuperatorInput<Fluid>,
    ) -> Result<RecuperatorOutput<Fluid>, RecuperatorError>
    where
        Fluid: Clone,
        Thermo: DiscretizedHxThermoModel<Fluid>,
    {
        let known = Known {
            inlets: input.inlets.clone(),
            m_dot: input.mass_flows,
            dp: input.pressure_drops,
        };

        let ua_abs_tol = input.ua * self.config.ua_rel_tol.abs();
        let given_ua_config = GivenUaConfig {
            max_iters: self.config.max_iters,
            temp_tol: self.config.temp_abs_tol,
            ua_tol: ua_abs_tol,
        };

        let given_ua_results = DiscretizedHx::<CounterFlow, N>::given_ua_same(
            &known,
            input.ua,
            given_ua_config,
            &self.thermo,
        )
        .map_err(RecuperatorError::from)?;

        Ok(Self::to_output(given_ua_results))
    }

    fn to_output<const N: usize>(
        given_ua_results: GivenUaResults<Fluid, Fluid, N>,
    ) -> RecuperatorOutput<Fluid>
    where
        Fluid: Clone,
    {
        let results = given_ua_results.results;

        RecuperatorOutput {
            top_outlet: results.top[N - 1].clone(),
            bottom_outlet: results.bottom[0].clone(),
            q_dot: results.q_dot,
            ua: results.ua,
            min_delta_t: results.min_delta_t,
            iterations: given_ua_results.iterations,
        }
    }
}

impl<Fluid, Thermo> Model for Recuperator<Fluid, Thermo>
where
    Fluid: Clone,
    Thermo: DiscretizedHxThermoModel<Fluid>,
{
    type Input = RecuperatorInput<Fluid>;
    type Output = RecuperatorOutput<Fluid>;
    type Error = RecuperatorError;

    fn call(&self, input: &Self::Input) -> Result<Self::Output, Self::Error> {
        match self.segments {
            1 => self.solve::<2>(input),
            5 => self.solve::<6>(input),
            10 => self.solve::<11>(input),
            20 => self.solve::<21>(input),
            50 => self.solve::<51>(input),
            _ => unreachable!("validated at construction"),
        }
    }
}

impl From<GivenUaError> for RecuperatorError {
    fn from(value: GivenUaError) -> Self {
        match value {
            GivenUaError::NegativeUa(ua) => Self::NegativeUa(ua),
            GivenUaError::Solve(error) => Self::ThermoModelFailed {
                context: "discretized heat exchanger solve".to_owned(),
                source: Box::new(error),
            },
            GivenUaError::Bisection(error) => Self::Convergence {
                message: error.to_string(),
                iterations: None,
            },
            GivenUaError::MaxIters { iters, .. } => Self::Convergence {
                message: "iteration limit reached".to_owned(),
                iterations: Some(iters),
            },
        }
    }
}
