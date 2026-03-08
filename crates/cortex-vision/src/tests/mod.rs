use std::path::PathBuf;

use cortex_tensor::{Shape, Tensor, TensorData, backend::Backend};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};

mod connected_components;
mod morphology;

#[macro_export]
macro_rules! testgen_all {
    () => {
        use cortex_tensor::{Bool, Float, Int};

        pub type TestTensor<const D: usize> = cortex_tensor::Tensor<TestBackend, D>;
        pub type TestTensorInt<const D: usize> = cortex_tensor::Tensor<TestBackend, D, Int>;
        pub type TestTensorBool<const D: usize> = cortex_tensor::Tensor<TestBackend, D, Bool>;

        pub mod vision {
            pub use super::*;

            pub type IntType = <TestBackend as cortex_tensor::backend::Backend>::IntElem;

            cortex_vision::testgen_connected_components!();
            cortex_vision::testgen_morphology!();
        }
    };
}
