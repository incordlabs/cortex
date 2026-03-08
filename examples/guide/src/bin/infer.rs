#![recursion_limit = "131"]
use cortex::{backend::WebGpu, data::dataset::Dataset};
use guide::inference;

fn main() {
    type MyBackend = WebGpu<f32, i32>;

    let device = cortex::backend::wgpu::WgpuDevice::default();

    // All the training artifacts are saved in this directory
    let artifact_dir = "/tmp/guide";

    // Infer the model
    inference::infer::<MyBackend>(
        artifact_dir,
        device,
        cortex::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
