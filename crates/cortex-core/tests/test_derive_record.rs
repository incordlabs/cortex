use cortex_core as cortex;
use cortex_core::record::Record;

use cortex_tensor::Tensor;
use cortex_tensor::backend::Backend;

// It compiles
#[derive(Record)]
pub struct TestWithBackendRecord<B: Backend> {
    tensor: Tensor<B, 2>,
}

// It compiles
#[derive(Record)]
pub struct TestWithoutBackendRecord {
    _tensor: usize,
}
