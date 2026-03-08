// Re-export
use super::FloatElemType;

// Default
#[cfg(feature = "ndarray")]
pub type TestBackend = cortex_ndarray::NdArray<FloatElemType>;

#[cfg(feature = "tch")]
pub type TestBackend = cortex_tch::LibTorch<FloatElemType>;

#[cfg(feature = "cuda")]
pub type TestBackend = cortex_cuda::Cuda<FloatElemType, super::IntElemType>;

#[cfg(feature = "rocm")]
pub type TestBackend = cortex_rocm::Rocm<FloatElemType, super::IntElemType>;

#[cfg(feature = "wgpu")]
pub type TestBackend = cortex_wgpu::Wgpu<FloatElemType, super::IntElemType>;

#[cfg(feature = "cpu")]
pub type TestBackend = cortex_cpu::Cpu<FloatElemType, super::IntElemType>;

#[cfg(feature = "router")]
pub type TestBackend = cortex_router::BackendRouter<
    cortex_router::DirectByteChannel<(cortex_ndarray::NdArray, cortex_wgpu::Wgpu)>,
>;

/// Collection of types used across tests
#[allow(unused)]
pub mod prelude {
    pub use cortex_autodiff::Autodiff;
    pub use cortex_tensor::Tensor;

    use super::*;
    pub type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = Tensor<TestBackend, D, cortex_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<TestBackend, D, cortex_tensor::Bool>;

    pub type FloatElem = cortex_tensor::ops::FloatElem<TestBackend>;
    pub type IntElem = cortex_tensor::ops::IntElem<TestBackend>;

    pub type TestAutodiffBackend = Autodiff<TestBackend>;
    pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;
}

#[allow(unused)]
pub use prelude::*;
