use core::marker::PhantomData;

/// Simply transfers tensors between backends via the underlying [tensor data](cortex_backend::TensorData).
pub struct ByteBridge<Backends> {
    backends: PhantomData<Backends>,
}
