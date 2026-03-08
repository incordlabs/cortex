# Cortex CUDA Backend

[Cortex](https://github.com/qora-protocol/cortex) CUDA backend

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-cuda.svg)](https://crates.io/crates/cortex-cuda)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/qora-protocol/cortex-cuda/blob/master/README.md)

This crate provides a CUDA backend for [Cortex](https://github.com/qora-protocol/cortex) using the
[cubecl](https://github.com/qora-protocol/cubecl.git) and [cudarc](https://github.com/coreylowman/cudarc.git)
crates.

## Usage Example

```rust
#[cfg(feature = "cuda")]
mod cuda {
    use cortex_autodiff::Autodiff;
    use cortex_cuda::{Cuda, CudaDevice};
    use mnist::training;

    pub fn run() {
        let device = CudaDevice::default();
        training::run::<Autodiff<Cuda<f32, i32>>>(device);
    }
}
```

## Dependencies

Requires CUDA 12.x to be installed and on the `PATH`.