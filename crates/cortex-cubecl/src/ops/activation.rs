use crate::{CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement};
use cortex_backend::ops::ActivationOps;

impl<R, F, I, BT> ActivationOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
}
