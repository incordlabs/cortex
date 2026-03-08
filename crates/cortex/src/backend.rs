#[cfg(feature = "ndarray")]
pub use cortex_ndarray as ndarray;

#[cfg(feature = "ndarray")]
pub use ndarray::NdArray;

#[cfg(feature = "autodiff")]
pub use cortex_autodiff as autodiff;

#[cfg(feature = "remote")]
pub use cortex_remote as remote;
#[cfg(feature = "remote")]
pub use cortex_remote::RemoteBackend;

#[cfg(feature = "autodiff")]
pub use cortex_autodiff::Autodiff;

#[cfg(feature = "wgpu")]
pub use cortex_wgpu as wgpu;

#[cfg(feature = "wgpu")]
pub use cortex_wgpu::Wgpu;

#[cfg(feature = "webgpu")]
pub use cortex_wgpu::WebGpu;

#[cfg(feature = "vulkan")]
pub use cortex_wgpu::Vulkan;

#[cfg(feature = "metal")]
pub use cortex_wgpu::Metal;

#[cfg(feature = "cuda")]
pub use cortex_cuda as cuda;

#[cfg(feature = "cuda")]
pub use cortex_cuda::Cuda;

#[cfg(feature = "candle")]
pub use cortex_candle as candle;

#[cfg(feature = "candle")]
pub use cortex_candle::Candle;

#[cfg(feature = "rocm")]
pub use cortex_rocm as rocm;

#[cfg(feature = "rocm")]
pub use cortex_rocm::Rocm;

#[cfg(feature = "tch")]
pub use cortex_tch as libtorch;

#[cfg(feature = "tch")]
pub use cortex_tch::LibTorch;

#[cfg(feature = "router")]
pub use cortex_router::Router;

#[cfg(feature = "router")]
pub use cortex_router as router;

#[cfg(feature = "ir")]
pub use cortex_ir as ir;

#[cfg(feature = "collective")]
pub use cortex_collective as collective;
#[cfg(feature = "cpu")]
pub use cortex_cpu as cpu;

#[cfg(feature = "cpu")]
pub use cortex_cpu::Cpu;
