use cortex::optim::decay::WeightDecayConfig;
use text_generation::{DbPediaDataset, training::ExperimentConfig};

#[cfg(feature = "f16")]
type Elem = cortex::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = cortex::backend::Autodiff<cortex::backend::LibTorch<Elem>>;

fn main() {
    let config = ExperimentConfig::new(
        cortex::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        cortex::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    text_generation::training::train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            cortex::tensor::Device::<Backend>::Mps
        } else {
            cortex::tensor::Device::<Backend>::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
