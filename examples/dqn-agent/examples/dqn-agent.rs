#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use cortex::backend::{
        Autodiff,
        ndarray::{NdArray, NdArrayDevice},
    };
    use dqn_agent::training;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<Autodiff<NdArray>>(device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use cortex::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use dqn_agent::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        training::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
mod wgpu {
    use cortex::backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    };
    use dqn_agent::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use cortex::backend::{Autodiff, Cuda};
    use dqn_agent::training;

    pub fn run() {
        let device = Default::default();
        training::run::<Autodiff<Cuda>>(device);
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use cortex::backend::{Autodiff, Rocm};
    use dqn_agent::training;

    pub fn run() {
        let device = Default::default();
        training::run::<Autodiff<Rocm>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use cortex::backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    };
    use dqn_agent::training;

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        training::run::<Autodiff<LibTorch>>(device);
    }
}

#[cfg(feature = "remote")]
mod remote {
    use cortex::backend::{Autodiff, RemoteBackend};
    use dqn_agent::training;

    pub fn run() {
        training::run::<Autodiff<RemoteBackend>>(Default::default());
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(any(feature = "wgpu", feature = "metal", feature = "vulkan"))]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "remote")]
    remote::run();
}
