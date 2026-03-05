mod aux_heat_flow;
mod buoyancy;
mod energy_balance;
mod environment;
mod fluid;
mod geometry;
mod insulation;
mod location;
mod mass_balance;
mod node;
mod port_flow;

use std::{array, ops::Div, ops::Mul};

use thiserror::Error;
use uom::{
    ConstZero,
    si::f64::{
        HeatCapacity, HeatTransfer, Ratio, TemperatureInterval, ThermalConductance,
        ThermalConductivity, ThermodynamicTemperature, Time, Volume, VolumeRate,
    },
};

use geometry::NodeGeometry;
use node::{Adjacent, Node};

pub use aux_heat_flow::{AuxHeatFlow, ValidatedPower};
pub use environment::Environment;
pub use fluid::Fluid;
pub use geometry::Geometry;
pub use insulation::Insulation;
pub use location::{Location, PortLocation};
pub use port_flow::PortFlow;

/// Rate of change of temperature (K/s in SI).
///
/// Uses `TemperatureInterval / Time` rather than `ThermodynamicTemperature / Time`
/// so that the type aligns with standard uom arithmetic (e.g., `Power / HeatCapacity`).
pub type TemperatureRate = <TemperatureInterval as Div<Time>>::Output;

type InverseHeatCapacity = <Ratio as Div<HeatCapacity>>::Output;
type InverseVolume = <Ratio as Div<Volume>>::Output;
type TemperatureFlow = <VolumeRate as Mul<TemperatureInterval>>::Output;

/// Errors that can occur when creating or using a [`StratifiedTank`].
#[derive(Debug, Error)]
pub enum StratifiedTankError {
    /// The tank geometry is invalid.
    #[error("geometry is invalid: {0}")]
    Geometry(String),

    /// An auxiliary heat source location is invalid.
    #[error("aux[{index}] location is invalid: {context}")]
    AuxLocation {
        /// Index of the auxiliary heat source.
        index: usize,
        /// Description of the validation failure.
        context: String,
    },

    /// A port inlet location is invalid.
    #[error("port[{index}] inlet location is invalid: {context}")]
    PortInletLocation {
        /// Index of the port pair.
        index: usize,
        /// Description of the validation failure.
        context: String,
    },

    /// A port outlet location is invalid.
    #[error("port[{index}] outlet location is invalid: {context}")]
    PortOutletLocation {
        /// Index of the port pair.
        index: usize,
        /// Description of the validation failure.
        context: String,
    },

    /// A port flow rate is negative, non-finite, or NaN.
    #[error("flow rate must be non-negative and finite, got {0:?}")]
    NegativePortFlowRate(VolumeRate),

    /// An inlet temperature is non-finite or NaN.
    #[error("inlet temperature must be finite, got {0:?}")]
    InvalidInletTemperature(ThermodynamicTemperature),

    /// An auxiliary power value is not strictly positive.
    #[error("auxiliary power must be strictly positive, got {0:?}")]
    NonPositiveAuxPower(uom::si::f64::Power),
}

/// A stratified thermal energy storage tank.
///
/// Represents a fixed-geometry tank with `N` fully mixed vertical nodes.
/// Supports energy exchange through `P` port pairs and `Q` auxiliary heat
/// sources.
///
/// A **port pair** models a real-world connection between the tank and an
/// external hydraulic circuit:
/// - One end returns fluid to the tank at a known temperature (inlet).
/// - The other end draws fluid out at the same volumetric rate (outlet).
///
/// This pairing maintains mass balance in the tank. The outlet temperature
/// comes from the node(s) where the outflow is taken.
///
/// Port and auxiliary locations are fixed when the tank is created; each may
/// apply to a single node or be distributed across multiple nodes.
///
/// Generic over:
/// - `N`: number of nodes (0 is a sentinel used during construction)
/// - `P`: number of port pairs
/// - `Q`: number of auxiliary heat sources
#[derive(Debug)]
pub struct StratifiedTank<const N: usize, const P: usize, const Q: usize> {
    nodes: [Node<P, Q>; N],
    vols: [Volume; N],
}

/// Input to the stratified tank model.
///
/// Runtime values that change at each evaluation step. The tank geometry,
/// port locations, and auxiliary locations are fixed at construction.
#[derive(Debug, Clone, Copy)]
pub struct StratifiedTankInput<const N: usize, const P: usize, const Q: usize> {
    /// Node temperatures from bottom (index 0) to top (index N-1).
    ///
    /// Values need not be thermally stable; the model mixes unstable nodes
    /// before computing energy balances.
    pub temperatures: [ThermodynamicTemperature; N],

    /// Flow rate and inlet temperature for each port pair.
    pub port_flows: [PortFlow; P],

    /// Heat input or extraction for each auxiliary source.
    pub aux_heat_flows: [AuxHeatFlow; Q],

    /// Ambient temperatures surrounding the tank.
    pub environment: Environment,
}

/// Output from the stratified tank model.
#[derive(Debug, Clone, Copy)]
pub struct StratifiedTankOutput<const N: usize> {
    /// Thermally stable node temperatures from bottom to top.
    ///
    /// Guaranteed stable: no node is warmer than any node above it.
    pub temperatures: [ThermodynamicTemperature; N],

    /// Time derivatives of temperature for each node (K/s).
    ///
    /// Includes the effects of fluid flow, auxiliary heat, conduction to
    /// neighboring nodes, and environmental heat transfer.
    pub derivatives: [TemperatureRate; N],
}

impl<const P: usize, const Q: usize> StratifiedTank<0, P, Q> {
    /// Creates a stratified tank with `N` nodes.
    ///
    /// Specify the number of nodes with the const generic parameter:
    ///
    /// ```ignore
    /// let tank = StratifiedTank::new::<20>(fluid, geometry, insulation, aux, ports)?;
    /// ```
    ///
    /// `P` (port pairs) and `Q` (auxiliary sources) are inferred from the
    /// lengths of `port_locations` and `aux_locations`.
    ///
    /// # Errors
    ///
    /// Returns a [`StratifiedTankError`] if geometry or location validation fails.
    pub fn new<const N: usize>(
        fluid: Fluid,
        geometry: Geometry,
        insulation: Insulation,
        aux_locations: [Location; Q],
        port_locations: [PortLocation; P],
    ) -> Result<StratifiedTank<N, P, Q>, StratifiedTankError> {
        let node_geometries = geometry
            .into_node_geometries::<N>()
            .map_err(StratifiedTankError::Geometry)?;

        let heights = node_geometries.map(|n| n.height);

        // Compute per-node auxiliary weights: aux_weight_by_node[node][aux].
        let mut aux_weight_by_node = [[0.0; Q]; N];
        for (index, loc) in aux_locations.iter().enumerate() {
            let weights = loc
                .into_weights(&heights)
                .map_err(|context| StratifiedTankError::AuxLocation { index, context })?;
            for node_idx in 0..N {
                aux_weight_by_node[node_idx][index] = weights[node_idx];
            }
        }

        // Compute per-node port weights: inlet/outlet_weight_by_node[node][port].
        let mut inlet_weight_by_node = [[0.0; P]; N];
        let mut outlet_weight_by_node = [[0.0; P]; N];
        for (index, port_loc) in port_locations.iter().enumerate() {
            let inlet_weights = port_loc
                .inlet
                .into_weights(&heights)
                .map_err(|context| StratifiedTankError::PortInletLocation { index, context })?;
            let outlet_weights = port_loc
                .outlet
                .into_weights(&heights)
                .map_err(|context| StratifiedTankError::PortOutletLocation { index, context })?;
            for node_idx in 0..N {
                inlet_weight_by_node[node_idx][index] = inlet_weights[node_idx];
                outlet_weight_by_node[node_idx][index] = outlet_weights[node_idx];
            }
        }

        let nodes = array::from_fn(|i| {
            let node = node_geometries[i];

            let ua = node_ua(
                i,
                N,
                fluid.thermal_conductivity,
                insulation,
                &node_geometries,
            );

            Node {
                inv_volume: node.volume.recip(),
                inv_heat_capacity: (node.volume * fluid.density * fluid.specific_heat).recip(),
                ua,
                aux_heat_weights: aux_weight_by_node[i],
                port_inlet_weights: inlet_weight_by_node[i],
                port_outlet_weights: outlet_weight_by_node[i],
            }
        });

        Ok(StratifiedTank {
            nodes,
            vols: node_geometries.map(|n| n.volume),
        })
    }
}

impl<const N: usize, const P: usize, const Q: usize> StratifiedTank<N, P, Q> {
    /// Evaluates the tank's thermal response at a single point in time.
    ///
    /// Enforces thermal stability via buoyancy mixing, then applies mass and
    /// energy balances to compute per-node temperature derivatives.
    #[must_use]
    pub fn evaluate(&self, input: &StratifiedTankInput<N, P, Q>) -> StratifiedTankOutput<N> {
        let StratifiedTankInput {
            temperatures: t_guess,
            port_flows,
            aux_heat_flows,
            environment,
        } = input;

        // Stabilize node temperatures via buoyancy mixing.
        let mut temperatures = *t_guess;
        buoyancy::stabilize(&mut temperatures, &self.vols);

        // Compute node-to-node flow rates.
        let flow_rates: [VolumeRate; P] = port_flows.map(|pf| pf.rate());
        let upward_flows = mass_balance::compute_upward_flows(
            &flow_rates,
            &self.nodes.map(|n| n.port_inlet_weights),
            &self.nodes.map(|n| n.port_outlet_weights),
        );

        // Compute the total dT/dt for each node.
        let derivatives = array::from_fn(|i| {
            self.deriv_from_flows(i, &temperatures, &upward_flows, port_flows)
                + self.deriv_from_aux(i, aux_heat_flows)
                + self.deriv_from_conduction(i, &temperatures, environment)
        });

        StratifiedTankOutput {
            temperatures,
            derivatives,
        }
    }

    /// Computes node `i`'s `dT/dt` due to fluid flows.
    fn deriv_from_flows(
        &self,
        i: usize,
        temperatures: &[ThermodynamicTemperature; N],
        upward_flows: &[VolumeRate; N],
        port_flows: &[PortFlow; P],
    ) -> TemperatureRate {
        let node = self.nodes[i];

        // Optional flow from the node below (upward into node i).
        let flow_from_below = if i > 0 && upward_flows[i - 1] > VolumeRate::ZERO {
            Some((upward_flows[i - 1], temperatures[i - 1]))
        } else {
            None
        };

        // Optional flow from the node above (downward into node i).
        let flow_from_above = if i < N - 1 && upward_flows[i] < VolumeRate::ZERO {
            Some((-upward_flows[i], temperatures[i + 1]))
        } else {
            None
        };

        let inflows = port_flows
            .iter()
            .zip(node.port_inlet_weights)
            .map(|(pf, weight)| (pf.rate() * weight, pf.inlet_temperature))
            .chain(flow_from_below)
            .chain(flow_from_above);

        energy_balance::derivative_from_fluid_flows(temperatures[i], node.inv_volume, inflows)
    }

    /// Computes node `i`'s `dT/dt` due to auxiliary heat sources.
    fn deriv_from_aux(&self, i: usize, aux_heat_flows: &[AuxHeatFlow; Q]) -> TemperatureRate {
        let node = self.nodes[i];

        let heat_flows = aux_heat_flows
            .iter()
            .zip(node.aux_heat_weights)
            .map(|(q_dot, weight)| q_dot.signed() * weight);

        energy_balance::derivative_from_heat_flows(node.inv_heat_capacity, heat_flows)
    }

    /// Computes node `i`'s `dT/dt` due to conduction to its surroundings.
    fn deriv_from_conduction(
        &self,
        i: usize,
        temperatures: &[ThermodynamicTemperature; N],
        env: &Environment,
    ) -> TemperatureRate {
        let node = self.nodes[i];

        let t_bottom = if i == 0 {
            env.bottom
        } else {
            temperatures[i - 1]
        };

        let t_top = if i == N - 1 {
            env.top
        } else {
            temperatures[i + 1]
        };

        energy_balance::derivative_from_conduction(
            temperatures[i],
            Adjacent {
                bottom: t_bottom,
                side: env.side,
                top: t_top,
            },
            node.ua,
            node.inv_heat_capacity,
        )
    }
}

/// Overall conductance (UA) between two adjacent, well-mixed nodes.
///
/// The interface is modeled as two thermal resistances in series:
/// ```text
/// R_total = R_below + R_above
///         = (0.5·h_below) / (k·A_below_top)
///         + (0.5·h_above) / (k·A_above_bottom)
/// UA = 1 / R_total
/// ```
fn ua_between_nodes(
    k: ThermalConductivity,
    below: NodeGeometry,
    above: NodeGeometry,
) -> ThermalConductance {
    let r_below = 0.5 * below.height / (k * below.area.top);
    let r_above = 0.5 * above.height / (k * above.area.bottom);
    (r_below + r_above).recip()
}

/// Computes the UA values (bottom, side, top) for node `i`.
fn node_ua<const N: usize>(
    i: usize,
    n: usize,
    k: ThermalConductivity,
    insulation: Insulation,
    node_geometries: &[NodeGeometry; N],
) -> Adjacent<ThermalConductance> {
    let node = node_geometries[i];

    let (u_bottom, u_side, u_top) = match insulation {
        Insulation::Adiabatic => (HeatTransfer::ZERO, HeatTransfer::ZERO, HeatTransfer::ZERO),
        Insulation::UValue { bottom, side, top } => (bottom, side, top),
    };

    Adjacent {
        bottom: if i == 0 {
            u_bottom * node.area.bottom
        } else {
            ua_between_nodes(k, node_geometries[i - 1], node)
        },
        side: u_side * node.area.side,
        top: if i == n - 1 {
            u_top * node.area.top
        } else {
            ua_between_nodes(k, node, node_geometries[i + 1])
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::PI;

    use approx::assert_relative_eq;
    use uom::si::{
        f64::{
            HeatTransfer, Length, MassDensity, Power, SpecificHeatCapacity, ThermalConductivity,
            VolumeRate,
        },
        heat_transfer::watt_per_square_meter_kelvin,
        length::meter,
        mass_density::kilogram_per_cubic_meter,
        power::kilowatt,
        specific_heat_capacity::kilojoule_per_kilogram_kelvin,
        thermodynamic_temperature::degree_celsius,
        volume_rate::gallon_per_minute,
    };

    // Test tank:
    // - vertical cylinder
    // - 3 nodes, each V=1 m³ and UA=0
    // - 1 port: inlet 100% to node 0, outlet 100% from node 2
    // - 1 aux: 100% applied to node 2
    fn test_tank() -> StratifiedTank<3, 1, 1> {
        let fluid = Fluid {
            density: MassDensity::new::<kilogram_per_cubic_meter>(1000.0),
            specific_heat: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.0),
            thermal_conductivity: ThermalConductivity::ZERO,
        };

        let geometry = Geometry::VerticalCylinder {
            diameter: Length::new::<meter>((4.0 / PI).sqrt()),
            height: Length::new::<meter>(3.0),
        };

        let insulation = Insulation::Adiabatic;
        let aux_locations = [Location::tank_top()];
        let port_locations = [PortLocation {
            inlet: Location::tank_bottom(),
            outlet: Location::tank_top(),
        }];

        StratifiedTank::new(fluid, geometry, insulation, aux_locations, port_locations).unwrap()
    }

    fn port_flow(gpm: f64, celsius: f64) -> PortFlow {
        PortFlow::new(
            VolumeRate::new::<gallon_per_minute>(gpm),
            ThermodynamicTemperature::new::<degree_celsius>(celsius),
        )
        .unwrap()
    }

    fn zero_port_flows<const P: usize>() -> [PortFlow; P] {
        [port_flow(0.0, 25.0); P]
    }

    fn k_per_s(rate: TemperatureRate) -> f64 {
        use uom::si::{temperature_interval::kelvin as delta_kelvin, time::second};
        (rate * Time::new::<second>(1.0)).get::<delta_kelvin>()
    }

    #[test]
    fn at_equilibrium_all_zero_derivatives() {
        let tank = test_tank();
        let t = ThermodynamicTemperature::new::<degree_celsius>(20.0);

        let input = StratifiedTankInput {
            temperatures: [t; 3],
            port_flows: zero_port_flows(),
            aux_heat_flows: [AuxHeatFlow::None],
            environment: Environment {
                bottom: t,
                side: t,
                top: t,
            },
        };

        let out = tank.evaluate(&input);

        assert!(out.temperatures.iter().all(|&node_t| node_t == t));
        for deriv in out.derivatives {
            assert_relative_eq!(k_per_s(deriv), 0.0);
        }
    }

    #[test]
    fn aux_only_heats_target_node() {
        let tank = test_tank();
        let t = ThermodynamicTemperature::new::<degree_celsius>(50.0);

        let input = StratifiedTankInput {
            temperatures: [t; 3],
            port_flows: zero_port_flows(),
            aux_heat_flows: [AuxHeatFlow::heating(Power::new::<kilowatt>(20.0)).unwrap()],
            environment: Environment {
                bottom: t,
                side: t,
                top: t,
            },
        };

        let out = tank.evaluate(&input);

        // Q/C = Q / (V * rho * cp)
        //     = 20 kW / (1 m³ * 1,000 kg/m³ * 4 kJ/(kg·K))
        //     = 20,000 / 4,000,000 = 0.005 K/s at node 2
        assert_relative_eq!(k_per_s(out.derivatives[0]), 0.0);
        assert_relative_eq!(k_per_s(out.derivatives[1]), 0.0);
        assert_relative_eq!(k_per_s(out.derivatives[2]), 0.005);
    }

    #[test]
    fn insulation_type_affects_thermal_response() {
        let fluid = Fluid {
            density: MassDensity::new::<kilogram_per_cubic_meter>(1000.0),
            specific_heat: SpecificHeatCapacity::new::<kilojoule_per_kilogram_kelvin>(4.0),
            thermal_conductivity: ThermalConductivity::ZERO,
        };

        let geometry = Geometry::VerticalCylinder {
            diameter: Length::new::<meter>((4.0 / PI).sqrt()),
            height: Length::new::<meter>(3.0),
        };

        let aux_locations = [Location::tank_top()];
        let port_locations = [PortLocation {
            inlet: Location::tank_bottom(),
            outlet: Location::tank_top(),
        }];

        // Create two identical tanks with different insulation
        // Specify overall heat transfer coefficients (U-values) in W/(m²·K).
        // Geometry: vertical cylinder with diameter = sqrt(4/π), height = 3.0
        // For 3 nodes (each 1m height, so areas are: bottom = 1 m², side ≈ 3.545 m², top = 1 m²)
        //
        // These U-values, when multiplied by surface areas, produce UA values:
        //   - u_bottom = 2000 W/(m²·K) → UA = 2000 * 1 = 2000 W/K (bottom/top surfaces)
        //   - u_side ≈ 564 W/(m²·K) → UA = 564 * 3.545 ≈ 2000 W/K (side surfaces)
        let u_bottom = HeatTransfer::new::<watt_per_square_meter_kelvin>(2000.0);
        let u_side =
            HeatTransfer::new::<watt_per_square_meter_kelvin>(2000.0 / (PI * (4.0 / PI).sqrt()));
        let u_top = HeatTransfer::new::<watt_per_square_meter_kelvin>(2000.0);

        let insulation_adiabatic = Insulation::Adiabatic;
        let insulation_u_value = Insulation::u_value(u_bottom, u_side, u_top);

        let tank_adiabatic = StratifiedTank::new(
            fluid,
            geometry.clone(),
            insulation_adiabatic,
            aux_locations,
            port_locations,
        )
        .unwrap();
        let tank_u_value = StratifiedTank::new(
            fluid,
            geometry.clone(),
            insulation_u_value,
            aux_locations,
            port_locations,
        )
        .unwrap();

        // Setup identical inputs with environmental temperature gradient
        let t_hot = ThermodynamicTemperature::new::<degree_celsius>(50.0);
        let t_cold = ThermodynamicTemperature::new::<degree_celsius>(10.0);

        let input = StratifiedTankInput {
            temperatures: [t_hot; 3],
            port_flows: zero_port_flows(),
            aux_heat_flows: [AuxHeatFlow::None],
            environment: Environment {
                bottom: t_cold,
                side: t_cold,
                top: t_cold,
            },
        };

        // Evaluate both tanks
        let out_adiabatic = tank_adiabatic.evaluate(&input);
        let out_u_value = tank_u_value.evaluate(&input);

        // Adiabatic tank should have zero derivatives (no heat loss)
        for deriv in out_adiabatic.derivatives {
            assert_relative_eq!(k_per_s(deriv), 0.0, max_relative = 1e-10);
        }

        // U-value tank: heat loss through external insulation
        // With k=0 for fluid, internal node-to-node conduction is zero.
        // Only external surfaces conduct to environment.
        // dT/dt = UA_ext * ΔT / (V * ρ * cp)
        // Heat capacity = V * ρ * cp = 1 m³ * 1000 kg/m³ * 4 kJ/(kg·K) = 4000 kJ/K = 4e6 J/K
        // ΔT = 50°C - 10°C = 40 K
        //
        // Node 0 (bottom): UA_external = 2000 W/K (bottom) + 2000 W/K (side) = 4000 W/K
        //   dT/dt = 4000 * 40 / 4e6 = -0.04 K/s
        // Node 1 (middle): UA_external = 2000 W/K (side only)
        //   dT/dt = 2000 * 40 / 4e6 = -0.02 K/s
        // Node 2 (top): UA_external = 2000 W/K (side) + 2000 W/K (top) = 4000 W/K
        //   dT/dt = 4000 * 40 / 4e6 = -0.04 K/s
        assert_relative_eq!(k_per_s(out_u_value.derivatives[0]), -0.04);
        assert_relative_eq!(k_per_s(out_u_value.derivatives[1]), -0.02);
        assert_relative_eq!(k_per_s(out_u_value.derivatives[2]), -0.04);
    }
}
