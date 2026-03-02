pub trait ThermoModel {
    type Fluid;
}

impl<T: ThermoModel> ThermoModel for &T {
    type Fluid = T::Fluid;
}
