#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "256"]

//! Cortex optimizers.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Optimizer module.
pub mod optim;
pub use optim::*;

/// Gradient clipping module.
pub mod grad_clipping;

/// Gradient scaler for mixed precision (AMP) training.
#[cfg(feature = "std")]
pub mod grad_scaler;

/// Learning rate scheduler module.
#[cfg(feature = "std")]
pub mod lr_scheduler;

/// Type alias for the learning rate.
///
/// LearningRate also implements [learning rate scheduler](crate::lr_scheduler::LrScheduler) so it
/// can be used for constant learning rate.
pub type LearningRate = f64; // We could potentially change the type.

/// Backend for test cases
#[cfg(all(
    test,
    not(feature = "test-tch"),
    not(feature = "test-wgpu"),
    not(feature = "test-cuda"),
    not(feature = "test-rocm")
))]
pub type TestBackend = cortex_ndarray::NdArray<f32>;

#[cfg(all(test, feature = "test-tch"))]
/// Backend for test cases
pub type TestBackend = cortex_tch::LibTorch<f32>;

#[cfg(all(test, feature = "test-wgpu"))]
/// Backend for test cases
pub type TestBackend = cortex_wgpu::Wgpu;

#[cfg(all(test, feature = "test-cuda"))]
/// Backend for test cases
pub type TestBackend = cortex_cuda::Cuda;

#[cfg(all(test, feature = "test-rocm"))]
/// Backend for test cases
pub type TestBackend = cortex_rocm::Rocm;

/// Backend for autodiff test cases
#[cfg(test)]
pub type TestAutodiffBackend = cortex_autodiff::Autodiff<TestBackend>;

#[cfg(all(test, feature = "test-memory-checks"))]
mod tests {
    cortex_fusion::memory_checks!();
}
