<div align="center">


---

**Cortex is a next generation Tensor Library and Deep Learning Framework that doesn't compromise on
<br /> flexibility, efficiency and portability.**

<br/>
</div>

<div align="left">

Cortex is both a tensor library and a deep learning framework optimized for numerical computing, model
inference and model training. Cortex leverages Rust to perform optimizations normally only available
in static-graph frameworks, offering optimal speed without impacting flexibility.

## Backend


Cortex strives to be as fast as possible on as many hardwares as possible, with robust
implementations. We believe this flexibility is crucial for modern needs where you may train your
models in the cloud, then deploy on customer hardwares, which vary from user to user.

</div>

### Supported Backends

Most backends support all operating systems, so we don't mention them in the tables below.

**GPU Backends:**

|         | CUDA | ROCm | Metal | Vulkan | WebGPU | LibTorch |
| ------- | ---- | ---- | ----- | ------ | ------ | -------- |
| Nvidia  | ☑️   | -    | -     | ☑️     | ☑️     | ☑️       |
| AMD     | -    | ☑️   | -     | ☑️     | ☑️     | ☑️       |
| Apple   | -    | -    | ☑️    | -      | ☑️     | ☑️       |
| Intel   | -    | -    | -     | ☑️     | ☑️     | -        |
| Qualcom | -    | -    | -     | ☑️     | ☑️     | -        |
| Wasm    | -    | -    | -     | -      | ☑️     | -        |

**CPU Backends:**

|        | Cpu (CubeCL) | NdArray | LibTorch |
| ------ | ------------ | ------- | -------- |
| X86    | ☑️           | ☑️      | ☑️       |
| Arm    | ☑️           | ☑️      | ☑️       |
| Wasm   | -            | ☑️      | -        |
| no-std | -            | ☑️      | -        |

<br />

Compared to other frameworks, Cortex has a very different approach to supporting many backends. By
design, most code is generic over the Backend trait, which allows us to build Cortex with swappable
backends. This makes composing backend possible, augmenting them with additional functionalities
such as autodifferentiation and automatic kernel fusion.

<details>
<summary>
Autodiff: Backend decorator that brings backpropagation to any backend 🔄
</summary>
<br />

Contrary to the aforementioned backends, Autodiff is actually a backend _decorator_. This means that
it cannot exist by itself; it must encapsulate another backend.

The simple act of wrapping a base backend with Autodiff transparently equips it with
autodifferentiation support, making it possible to call backward on your model.

```rust
use Cortex::backend::{Autodiff, Wgpu};
use Cortex::tensor::{Distribution, Tensor};

fn main() {
    type Backend = Autodiff<Wgpu>;

    let device = Default::default();

    let x: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device);
    let y: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default, &device).require_grad();

    let tmp = x.clone() + y.clone();
    let tmp = tmp.matmul(x);
    let tmp = tmp.exp();

    let grads = tmp.backward();
    let y_grad = y.grad(&grads).unwrap();
    println!("{y_grad}");
}
```

Of note, it is impossible to make the mistake of calling backward on a model that runs on a backend
that does not support autodiff (for inference), as this method is only offered by an Autodiff
backend.

See the [Autodiff Backend README](./crates/Cortex-autodiff/README.md) for more details.

</details>

<details>
<summary>
Fusion: Backend decorator that brings kernel fusion to all first-party backends
</summary>
<br />

This backend decorator enhances a backend with kernel fusion, provided that the inner backend
supports it. Note that you can compose this backend with other backend decorators such as Autodiff.
All first-party accelerated backends (like WGPU and CUDA) use Fusion by default (`Cortex/fusion`
feature flag), so you typically don't need to apply it manually.

```rust
#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = Cortex_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;
```

Of note, we plan to implement automatic gradient checkpointing based on compute bound and memory
bound operations, which will work gracefully with the fusion backend to make your code run even
faster during training, see [this issue](https://github.com/qora-protocol/cortex/issues/936).

See the [Fusion Backend README](./crates/Cortex-fusion/README.md) for more details.

</details>

<details>
<summary>
Router (Beta): Backend decorator that composes multiple backends into a single one
</summary>
<br />

That backend simplifies hardware operability, if for instance you want to execute some operations on
the CPU and other operations on the GPU.

```rust
use Cortex::tensor::{Distribution, Tensor};
use Cortex::backend::{
    NdArray, Router, Wgpu, ndarray::NdArrayDevice, router::duo::MultiDevice, wgpu::WgpuDevice,
};

fn main() {
    type Backend = Router<(Wgpu, NdArray)>;

    let device_0 = MultiDevice::B1(WgpuDevice::DiscreteGpu(0));
    let device_1 = MultiDevice::B2(NdArrayDevice::Cpu);

    let tensor_gpu =
        Tensor::<Backend, 2>::random([3, 3], Cortex::tensor::Distribution::Default, &device_0);
    let tensor_cpu =
        Tensor::<Backend, 2>::random([3, 3], Cortex::tensor::Distribution::Default, &device_1);
}

```

</details>

<details>
<summary>
Remote (Beta): Backend decorator for remote backend execution, useful for distributed computations
</summary>
<br />

That backend has two parts, one client and one server. The client sends tensor operations over the
network to a remote compute backend. You can use any first-party backend as server in a single line
of code:

```rust
fn main_server() {
    // Start a server on port 3000.
    Cortex::server::start::<Cortex::backend::Cuda>(Default::default(), 3000);
}

fn main_client() {
    // Create a client that communicate with the server on port 3000.
    use Cortex::backend::{Autodiff, RemoteBackend};

    type Backend = Autodiff<RemoteDevice>;

    let device = RemoteDevice::new("ws://localhost:3000");
    let tensor_gpu =
        Tensor::<Backend, 2>::random([3, 3], Distribution::Default, &device);
}

```

</details>

<br />

## Training & Inference


The whole deep learning workflow is made easy with Cortex, as you can monitor your training progress
with an ergonomic dashboard, and run inference everywhere from embedded devices to large GPU
clusters.

Cortex was built from the ground up with training and inference in mind. It's also worth noting how
Cortex, in comparison to frameworks like PyTorch, simplifies the transition from training to
deployment, eliminating the need for code changes.

</div>

<div align="center">

<img width="1278" height="754" alt="cortex-train-tui" src="https://github.com/user-attachments/assets/954b8ff8-36a1-4c9a-a817-9e11cb954e98" />

**Click on the following sections to expand 👇**

<details>
<summary>
Training Dashboard 📈
</summary>
<br />

As you can see in the previous video (click on the picture!), a new terminal UI dashboard based on
the [Ratatui](https://github.com/ratatui-org/ratatui) crate allows users to follow their training
with ease without having to connect to any external application.

You can visualize your training and validation metrics updating in real-time and analyze the
lifelong progression or recent history of any registered metrics using only the arrow keys. Break
from the training loop without crashing, allowing potential checkpoints to be fully written or
important pieces of code to complete without interruption 🛡

</details>

<details>
<summary>
Importing PyTorch or Safetensors Models 🚚
</summary>
<br />

You can load weights from PyTorch or Safetensors formats directly into your Cortex-defined models.
This makes it easy to reuse existing models while benefiting from Cortex's performance and deployment
features.
</details>

<details>
<summary>
Inference in the Browser 🌐
</summary>
<br />

> ⚠️ **Warning** When using one of the `wgpu` backends, you may encounter compilation errors related
> to recursive type evaluation. This is due to complex type nesting within the `wgpu` dependency
> chain. To resolve this issue, add the following line at the top of your `main.rs` or `lib.rs`
> file:
>
> ```rust
> #![recursion_limit = "256"]
> ```
>
> The default recursion limit (128) is often just below the required depth (typically 130-150) due
> to deeply nested associated types and trait bounds.

## Getting Started

Just heard of Cortex? You are at the right place! Just continue reading this section and we hope you
can get on board really quickly.

</div>

<details>
<summary>
The Cortex Book 🔥
</summary>
<br />

To begin working effectively with Cortex, it is crucial to understand its key components and
philosophy. This is why we highly recommend new users to read the first sections of
The Cortex Book 🔥. It provides detailed examples and explanations
covering every facet of the framework, including building blocks like tensors, modules, and
optimizers, all the way to advanced usage, like coding your own GPU kernels.

> The project is constantly evolving, and we try as much as possible to keep the book up to date
> with new additions. However, we might miss some details sometimes, so if you see something weird,
> let us know! We also gladly accept Pull Requests 😄

</details>

<details>
<summary>
Examples 🙏
</summary>
<br />

Let's start with a code snippet that shows how intuitive the framework is to use! In the following,
we declare a neural network module with some parameters along with its forward pass.

```rust
use Cortex::nn;
use Cortex::module::Module;
use Cortex::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: nn::Linear<B>,
    linear_outer: nn::Linear<B>,
    dropout: nn::Dropout,
    gelu: nn::Gelu,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

For more practical insights, you can clone the repository and run any of them directly on your
computer!

</details>

<details>
<summary>
Pre-trained Models 🤖
</summary>
<br />

Don't see the model you want? Don't hesitate to open an issue, and we may prioritize it. Built a
model using Cortex and want to share it? You can also open a Pull Request and add your model under the
community section!

</details>

<details>
<summary>
Why use Rust for Deep Learning? 🦀
</summary>
<br />

Deep Learning is a special form of software where you need very high level abstractions as well as
extremely fast execution time. Rust is the perfect candidate for that use case since it provides
zero-cost abstractions to easily create neural network modules, and fine-grained control over memory
to optimize every detail.

It's important that a framework be easy to use at a high level so that its users can focus on
innovating in the AI field. However, since running models relies so heavily on computations,
performance can't be neglected.

To this day, the mainstream solution to this problem has been to offer APIs in Python, but rely on
bindings to low-level languages such as C/C++. This reduces portability, increases complexity and
creates frictions between researchers and engineers. We feel like Rust's approach to abstractions
makes it versatile enough to tackle this two languages dichotomy.

Rust also comes with the Cargo package manager, which makes it incredibly easy to build, test, and
deploy from any environment, which is usually a pain in Python.

Although Rust has the reputation of being a difficult language at first, we strongly believe it
leads to more reliable, bug-free solutions built faster (after some practice 😅)!

</details>

<br />

> **Deprecation Note**<br />Since `0.14.0`, the internal structure for tensor data has changed. The
> previous `Data` struct was deprecated and officially removed since `0.17.0` in favor of the new
> `TensorData` struct, which allows for more flexibility by storing the underlying data as bytes and
> keeping the data type as a field. If you are using `Data` in your code, make sure to switch to
> `TensorData`.

<!-- >
> In the event that you are trying to load a model record saved in a previous version, make sure to
> enable the `record-backward-compat` feature using a previous version of Cortex (<=0.16.0). Otherwise,
> the record won't be deserialized correctly and you will get an error message (which will also point
> you to the backward compatible feature flag). The backward compatibility was maintained for
> deserialization (loading), so as soon as you have saved the record again it will be saved according
> to the new structure and you will be able to upgrade to this version. Please note that binary formats
> are not backward compatible. Thus, you will need to load your record in a previous version and save it
> to another of the self-describing record formats before using a compatible version (as described) with the
> `record-backward-compat` feature flag. -->

<details id="deprecation">
<summary>
Loading Model Records From Previous Versions ⚠️
</summary>
<br />

In the event that you are trying to load a model record saved in a version older than `0.14.0`, make
sure to use a compatible version (`0.14`, `0.15` or `0.16`) with the `record-backward-compat`
feature flag.

```
features = [..., "record-backward-compat"]
```

Otherwise, the record won't be deserialized correctly and you will get an error message. This error
will also point you to the backward compatible feature flag.

The backward compatibility was maintained for deserialization when loading records. Therefore, as
soon as you have saved the record again it will be saved according to the new structure and you can
upgrade back to the current version

Please note that binary formats are not backward compatible. Thus, you will need to load your record
in a previous version and save it in any of the other self-describing record format (e.g., using the
`NamedMpkFileRecorder`) before using a compatible version (as described) with the
`record-backward-compat` feature flag.

</details>

**Contributing**

Before contributing, please take a moment to review our
[code of conduct](https://github.com/blockmandev/Cortex/tree/main/CODE-OF-CONDUCT.md). It's also highly
recommended to read the
[architecture overview](https://github.com/blockmandev/Cortex/tree/main/contributor-book/src/project-architecture),
which explains some of our architectural decisions. Refer to our
[contributing guide](/CONTRIBUTING.md) for more details.

## Status

Cortex is currently in active development, and there will be breaking changes. While any resulting
issues are likely to be easy to fix, there are no guarantees at this stage.

## License

Cortex is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.

</div>
