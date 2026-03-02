//! Stratified hot water tank simulation.
//!
//! Simulates 5 days of residential hot water tank operation with:
//! - 20-node stratified tank (~120 gallons)
//! - Daily draw schedule (morning and evening draws)
//! - Setpoint thermostat controlling a 6 kW heating element
//! - Forward Euler integration at 1-minute time steps
//! - Interactive plot of temperature traces and operational data
//!
//! Run with:
//!
//! ```sh
//! cargo run --example stratified_tank --release
//! ```

use std::{convert::Infallible, error::Error};

use jiff::{
    SignedDuration,
    civil::{DateTime, Time as CivilTime},
};
use twine_core::{DerivativeOf, Model, OdeProblem};
use twine_models::{
    models::thermal::tank::stratified::{
        AuxHeatFlow, Environment, Fluid, Geometry, Insulation, Location, PortFlow, PortLocation,
        StratifiedTank, StratifiedTankInput, StratifiedTankOutput, TankDerivative, TankState,
    },
    support::{
        control::{
            SwitchState,
            thermostat::setpoint::{Deadband, SetpointThermostatInput, heating},
        },
        schedule::step_schedule::{Step, StepSchedule},
    },
};
use twine_observers::{PlotObserver, ShowConfig};
use twine_solvers::transient::euler;
use uom::si::{
    f64::{
        Length, MassDensity, Power, SpecificHeatCapacity, TemperatureInterval, ThermalConductivity,
        ThermodynamicTemperature, Time, VolumeRate,
    },
    length::meter,
    mass_density::kilogram_per_cubic_meter,
    power::kilowatt,
    specific_heat_capacity::kilojoule_per_kilogram_kelvin,
    temperature_interval::degree_celsius as delta_celsius,
    thermal_conductivity::watt_per_meter_kelvin,
    thermodynamic_temperature::degree_celsius,
    time::{minute, second},
    volume_rate::gallon_per_minute,
};

/// Number of tank nodes (bottom = 0, top = N-1).
const NODES: usize = 20;

/// Days to simulate.
const DAYS: usize = 5;

// ---------------------------------------------------------------------------
// Input type
// ---------------------------------------------------------------------------

/// Full model input at each simulation step.
///
/// Wraps the tank's own input and carries the simulation clock and element
/// state alongside it. The tank knows nothing about time or thermostats —
/// those live here.
#[derive(Clone)]
struct SimInput {
    tank: StratifiedTankInput<NODES, 1, 1>,
    /// Current simulation datetime (civil, no timezone).
    time: DateTime,
    /// Current on/off state of the heating element.
    element_state: SwitchState,
}

// ---------------------------------------------------------------------------
// Scenario (model + problem combined)
// ---------------------------------------------------------------------------

/// Scenario configuration: everything fixed for the duration of the run.
struct TankScenario {
    tank: StratifiedTank<NODES, 1, 1>,
    draw_schedule: StepSchedule<CivilTime, VolumeRate>,
    /// Index of the node containing the heating element.
    element_node: usize,
    /// Rated power of the heating element.
    element_power: Power,
    setpoint: ThermodynamicTemperature,
    deadband: Deadband,
    ground_temperature: ThermodynamicTemperature,
    room_temperature: ThermodynamicTemperature,
}

/// The model delegates directly to the inner tank.
impl Model for TankScenario {
    type Input = SimInput;
    type Output = StratifiedTankOutput<NODES>;
    type Error = Infallible;

    fn call(&self, input: &SimInput) -> Result<StratifiedTankOutput<NODES>, Infallible> {
        Ok(self.tank.evaluate(&input.tank))
    }
}

/// The ODE problem manages state extraction, time tracking, and thermostat control.
impl OdeProblem for TankScenario {
    type Input = SimInput;
    type Output = StratifiedTankOutput<NODES>;
    type Delta = Time;
    type State = TankState<NODES>;
    type Error = Infallible;

    fn state(&self, input: &SimInput) -> Result<TankState<NODES>, Infallible> {
        Ok(TankState {
            temperatures: input.tank.temperatures,
        })
    }

    fn derivative(
        &self,
        _input: &SimInput,
        output: &StratifiedTankOutput<NODES>,
    ) -> Result<DerivativeOf<TankState<NODES>, Time>, Infallible> {
        Ok(TankDerivative {
            rates: output.derivatives,
        })
    }

    /// Advances simulation time, looks up the current draw, and carries the
    /// element state forward.
    ///
    /// The element state is preserved from the base input; `finalize_step`
    /// updates it after evaluating the thermostat.
    fn build_input(
        &self,
        base: &SimInput,
        state: &TankState<NODES>,
        delta: &Time,
    ) -> Result<SimInput, Infallible> {
        let delta_secs = delta.get::<second>();
        let new_time = base
            .time
            .checked_add(SignedDuration::from_secs_f64(delta_secs))
            .unwrap_or(base.time);

        // Look up the draw rate for this time of day; default to zero.
        let draw_rate = self
            .draw_schedule
            .value_at(&new_time.time())
            .copied()
            .unwrap_or(VolumeRate::new::<gallon_per_minute>(0.0));

        let aux = element_heat_flow(base.element_state, self.element_power);

        Ok(SimInput {
            tank: StratifiedTankInput {
                temperatures: state.temperatures,
                port_flows: [PortFlow::new(draw_rate, self.ground_temperature).unwrap()],
                aux_heat_flows: [aux],
                environment: Environment {
                    bottom: self.room_temperature,
                    side: self.room_temperature,
                    top: self.room_temperature,
                },
            },
            time: new_time,
            element_state: base.element_state,
        })
    }

    /// Runs the thermostat against the previous step's stabilized temperatures
    /// and updates the element state and aux heat flow for the next step.
    ///
    /// Using the post-buoyancy temperatures from the model output (rather than
    /// the raw temperatures in the input) is physically correct — the sensor
    /// sees the actual mixed temperature, not the integration guess.
    fn finalize_step(
        &self,
        mut next_input: SimInput,
        _prev_input: &SimInput,
        prev_output: &StratifiedTankOutput<NODES>,
        _delta: &Time,
    ) -> Result<SimInput, Infallible> {
        let element_temp = prev_output.temperatures[self.element_node];
        let new_state = heating(SetpointThermostatInput {
            state: next_input.element_state,
            temperature: element_temp,
            setpoint: self.setpoint,
            deadband: self.deadband,
        });

        next_input.element_state = new_state;
        next_input.tank.aux_heat_flows[0] = element_heat_flow(new_state, self.element_power);

        Ok(next_input)
    }
}

/// Returns the `AuxHeatFlow` corresponding to the element's switch state.
fn element_heat_flow(state: SwitchState, power: Power) -> AuxHeatFlow {
    match state {
        SwitchState::Off => AuxHeatFlow::None,
        SwitchState::On => AuxHeatFlow::heating(power).unwrap(),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn build_scenario() -> Result<TankScenario, Box<dyn Error>> {
    let tank = StratifiedTank::new::<NODES>(
        Fluid {
            density: MassDensity::new::<kilogram_per_cubic_meter>(990.0),
            specific_heat: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.18),
            thermal_conductivity: ThermalConductivity::new::<watt_per_meter_kelvin>(0.6),
        },
        Geometry::VerticalCylinder {
            diameter: Length::new::<meter>(0.6),
            height: Length::new::<meter>(1.6),
        },
        Insulation::Adiabatic,
        [Location::point_in_node(12)],
        [PortLocation {
            inlet: Location::tank_bottom(),
            outlet: Location::tank_top(),
        }],
    )?;

    // Daily draw schedule — morning gentle draw, evening burst draw.
    let draw_schedule = StepSchedule::new([
        Step::new(
            CivilTime::constant(9, 0, 0, 0)..CivilTime::constant(10, 30, 0, 0),
            VolumeRate::new::<gallon_per_minute>(0.2),
        )?,
        Step::new(
            CivilTime::constant(18, 0, 0, 0)..CivilTime::constant(18, 15, 0, 0),
            VolumeRate::new::<gallon_per_minute>(2.5),
        )?,
    ])?;

    Ok(TankScenario {
        tank,
        draw_schedule,
        element_node: 12,
        element_power: Power::new::<kilowatt>(6.0),
        setpoint: ThermodynamicTemperature::new::<degree_celsius>(50.0),
        deadband: Deadband::new(TemperatureInterval::new::<delta_celsius>(10.0))?,
        ground_temperature: ThermodynamicTemperature::new::<degree_celsius>(10.0),
        room_temperature: ThermodynamicTemperature::new::<degree_celsius>(20.0),
    })
}

/// Initial conditions: tank cold, element off, no draw.
fn initial_input(scenario: &TankScenario) -> SimInput {
    let t_init = ThermodynamicTemperature::new::<degree_celsius>(20.0);
    SimInput {
        tank: StratifiedTankInput {
            temperatures: [t_init; NODES],
            port_flows: [PortFlow::new(
                VolumeRate::new::<gallon_per_minute>(0.0),
                scenario.ground_temperature,
            )
            .unwrap()],
            aux_heat_flows: [AuxHeatFlow::None],
            environment: Environment {
                bottom: scenario.room_temperature,
                side: scenario.room_temperature,
                top: scenario.room_temperature,
            },
        },
        time: DateTime::default(),
        element_state: SwitchState::Off,
    }
}

fn run(scenario: &TankScenario) -> Result<(), Box<dyn Error>> {
    let initial = initial_input(scenario);
    let dt = Time::new::<minute>(1.0);
    let steps = DAYS * 24 * 60;

    let mut obs = PlotObserver::<6>::new([
        "Ground water (°C)",
        "Tank top (°C)",
        "Tank middle (°C)",
        "Tank bottom (°C)",
        "Draw (GPM)",
        "Element (kW)",
    ]);

    euler::solve(
        scenario,
        scenario,
        initial,
        dt,
        steps,
        |event: &euler::Event<SimInput, StratifiedTankOutput<NODES>>| {
            #[allow(clippy::cast_precision_loss)]
            let t_hours = event.step as f64 / 60.0;

            let input = &event.snapshot.input;
            let output = &event.snapshot.output;

            // Ground water temperature is a fixed scenario parameter.
            let t_ground = scenario.ground_temperature.get::<degree_celsius>();

            // Top, middle (node 10), and bottom node temperatures.
            let t_top = output.temperatures[NODES - 1].get::<degree_celsius>();
            let t_mid = output.temperatures[NODES / 2].get::<degree_celsius>();
            let t_bot = output.temperatures[0].get::<degree_celsius>();

            // Draw from the port flow (rate set in build_input).
            let draw_gpm = input.tank.port_flows[0].rate().get::<gallon_per_minute>();

            // Element power: 0 when off, rated kW when on.
            let element_kw = match input.element_state {
                SwitchState::Off => 0.0,
                SwitchState::On => scenario.element_power.get::<kilowatt>(),
            };

            obs.record(
                t_hours,
                [
                    Some(t_ground),
                    Some(t_top),
                    Some(t_mid),
                    Some(t_bot),
                    Some(draw_gpm),
                    Some(element_kw),
                ],
            );

            None
        },
    )?;

    obs.show(
        ShowConfig::new()
            .title("Stratified Tank — 5-Day Simulation")
            .legend(),
    )?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let scenario = build_scenario()?;
    run(&scenario)
}
