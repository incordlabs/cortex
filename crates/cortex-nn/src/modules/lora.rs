use cortex_core as cortex;

use crate::{Dropout, DropoutConfig, Linear, LinearConfig};
use cortex::config::Config;
use cortex::module::{Content, DisplaySettings, Module, ModuleDisplay, Param, ParamId};
use cortex::tensor::{Tensor, backend::Backend};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Configuration to create a [LoRA Linear](LoraLinear) layer using the
/// [init function](LoraLinearConfig::init).
///
/// LoRA (Low-Rank Adaptation) decomposes weight updates into two low-rank matrices,
/// dramatically reducing the number of trainable parameters for fine-tuning.
///
/// Instead of updating `W` directly, LoRA learns `W + (A @ B) * scaling` where
/// A has shape `[d_input, rank]` and B has shape `[rank, d_output]`.
///
/// Reference: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
#[derive(Config, Debug)]
pub struct LoraLinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// The LoRA rank (typically 4-64). Lower = fewer params, higher = more capacity.
    pub rank: usize,
    /// The LoRA scaling factor. Default: same as rank.
    #[config(default = 0.0)]
    pub alpha: f64,
    /// Dropout applied to the LoRA path. Default: 0.0 (no dropout).
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Whether the base linear layer has a bias. Default: true.
    #[config(default = true)]
    pub bias: bool,
}

/// A Linear layer with LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
///
/// The base `linear` weights are frozen (not trained). Only the low-rank matrices
/// `lora_a` and `lora_b` receive gradients during training.
///
/// Forward: `output = linear(x) + dropout(x @ lora_a @ lora_b) * scaling`
///
/// # Params
///
/// - `linear`: Frozen base [`Linear`] layer.
/// - `lora_a`: Trainable down-projection `[d_input, rank]`.
/// - `lora_b`: Trainable up-projection `[rank, d_output]`, initialized to zeros.
///
/// Should be created with [LoraLinearConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LoraLinear<B: Backend> {
    /// The frozen base linear layer.
    pub linear: Linear<B>,
    /// LoRA down-projection matrix A of shape `[d_input, rank]`.
    pub lora_a: Param<Tensor<B, 2>>,
    /// LoRA up-projection matrix B of shape `[rank, d_output]`.
    pub lora_b: Param<Tensor<B, 2>>,
    /// Dropout applied to the LoRA path.
    pub dropout: Dropout,
    /// The LoRA scaling factor: `alpha / rank`.
    pub scaling: f64,
    /// The LoRA rank.
    pub rank: usize,
}

impl<B: Backend> ModuleDisplay for LoraLinear<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _rank] = self.lora_a.shape().dims();
        let [_rank2, d_output] = self.lora_b.shape().dims();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("rank", &self.rank)
            .add("scaling", &self.scaling)
            .optional()
    }
}

impl LoraLinearConfig {
    /// Initialize a new [LoRA Linear](LoraLinear) module.
    ///
    /// The base linear weights are initialized with Kaiming uniform and immediately frozen.
    /// `lora_a` is initialized with Kaiming uniform, `lora_b` is initialized to zeros.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LoraLinear<B> {
        let alpha = if self.alpha == 0.0 {
            self.rank as f64
        } else {
            self.alpha
        };
        let scaling = alpha / self.rank as f64;

        // Base linear layer — will be frozen
        let linear = LinearConfig::new(self.d_input, self.d_output)
            .with_bias(self.bias)
            .init(device);
        let linear = linear.no_grad();

        // LoRA A: Kaiming uniform initialization (same as Linear default)
        let lora_a = cortex::module::Initializer::KaimingUniform {
            gain: 1.0 / num_traits::Float::sqrt(3.0),
            fan_out_only: false,
        }
        .init_with(
            [self.d_input, self.rank],
            Some(self.d_input),
            Some(self.rank),
            device,
        );

        // LoRA B: Zero initialization (ensures LoRA output is 0 at start of training)
        let lora_b = Tensor::zeros([self.rank, self.d_output], device);

        // lora_b needs require_grad for training
        let lora_b = Param::initialized(ParamId::new(), lora_b);

        LoraLinear {
            linear,
            lora_a,
            lora_b,
            dropout: DropoutConfig::new(self.dropout).init(),
            scaling,
            rank: self.rank,
        }
    }
}

impl<B: Backend> LoraLinear<B> {
    /// Wrap an existing trained [`Linear`] layer with LoRA adapters.
    ///
    /// The base linear weights are frozen. New trainable LoRA matrices are initialized
    /// (A with Kaiming, B with zeros).
    pub fn from_linear(
        linear: Linear<B>,
        rank: usize,
        alpha: f64,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let [d_input, _] = linear.weight.shape().dims::<2>();
        let d_output = if let Some(ref bias) = linear.bias {
            bias.shape().dims::<1>()[0]
        } else {
            linear.weight.shape().dims::<2>()[1]
        };

        let scaling = alpha / rank as f64;

        // Freeze the base linear
        let linear = linear.no_grad();

        let lora_a = cortex::module::Initializer::KaimingUniform {
            gain: 1.0 / num_traits::Float::sqrt(3.0),
            fan_out_only: false,
        }
        .init_with([d_input, rank], Some(d_input), Some(rank), device);

        let lora_b = Tensor::zeros([rank, d_output], device);
        let lora_b = Param::initialized(ParamId::new(), lora_b);

        LoraLinear {
            linear,
            lora_a,
            lora_b,
            dropout: DropoutConfig::new(dropout).init(),
            scaling,
            rank,
        }
    }

    /// Merge the LoRA adapters into the base linear weight and return a standard [`Linear`].
    ///
    /// After merging: `W' = W + (A @ B) * scaling`
    ///
    /// This produces a single `Linear` that is equivalent to the `LoraLinear` forward pass
    /// (when dropout is disabled), but without the overhead of the extra matrix multiplications.
    pub fn merge(self) -> Linear<B> {
        let lora_weight = self
            .lora_a
            .val()
            .matmul(self.lora_b.val())
            .mul_scalar(self.scaling as f32);

        let merged_weight = self.linear.weight.val() + lora_weight;

        Linear {
            weight: Param::initialized(ParamId::new(), merged_weight),
            bias: self.linear.bias,
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// `output = linear(x) + dropout(x @ lora_a @ lora_b) * scaling`
    ///
    /// # Shapes
    ///
    /// - input: `[..., d_input]`
    /// - output: `[..., d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let base_output = self.linear.forward(input.clone());

        // LoRA path: x @ A @ B * scaling
        let lora_output = input
            .matmul(self.lora_a.val().unsqueeze())
            .matmul(self.lora_b.val().unsqueeze());
        let lora_output = self.dropout.forward(lora_output);

        base_output + lora_output.mul_scalar(self.scaling as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use cortex::tensor::{Distribution, Tolerance, ops::FloatElem};

    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_lora_output_shape() {
        let device = Default::default();
        let lora = LoraLinearConfig::new(64, 32, 8).init::<TestBackend>(&device);

        let input = Tensor::random([2, 10, 64], Distribution::Default, &device);
        let output = lora.forward(input);

        assert_eq!(output.dims(), [2, 10, 32]);
    }

    #[test]
    fn test_lora_initial_output_equals_base() {
        // Since lora_b is initialized to zeros, LoRA output should be same as base linear
        let device = Default::default();
        let lora = LoraLinearConfig::new(64, 32, 8).init::<TestBackend>(&device);

        let input: Tensor<TestBackend, 3> =
            Tensor::random([2, 10, 64], Distribution::Default, &device);
        let lora_output = lora.forward(input.clone());
        let base_output = lora.linear.forward(input);

        lora_output
            .to_data()
            .assert_approx_eq::<FT>(&base_output.to_data(), Tolerance::default());
    }

    #[test]
    fn test_lora_merge_equivalence() {
        // After merge, the merged Linear should produce the same output
        let device = Default::default();
        let mut lora = LoraLinearConfig::new(64, 32, 4)
            .with_alpha(8.0)
            .init::<TestBackend>(&device);

        // Simulate some "training" by replacing lora_b with non-zero values
        let new_b = Tensor::random([4, 32], Distribution::Default, &device);
        lora.lora_b = Param::initialized(ParamId::new(), new_b);

        let input: Tensor<TestBackend, 3> =
            Tensor::random([2, 10, 64], Distribution::Default, &device);
        let lora_output = lora.forward(input.clone());

        let merged = lora.merge();
        let merged_output = merged.forward(input);

        merged_output
            .to_data()
            .assert_approx_eq::<FT>(&lora_output.to_data(), Tolerance::permissive());
    }

    #[test]
    fn test_lora_from_linear() {
        let device = Default::default();
        let linear = LinearConfig::new(64, 32).init::<TestBackend>(&device);
        let lora = LoraLinear::from_linear(linear, 8, 8.0, 0.0, &device);

        let input = Tensor::random([2, 10, 64], Distribution::Default, &device);
        let output = lora.forward(input);

        assert_eq!(output.dims(), [2, 10, 32]);
    }

    #[test]
    fn test_lora_param_count() {
        let device = Default::default();
        let lora = LoraLinearConfig::new(1024, 1024, 8).init::<TestBackend>(&device);

        // LoRA params: A = 1024*8 = 8192, B = 8*1024 = 8192, total = 16384
        // Base params: W = 1024*1024 = 1048576 + bias = 1024
        // LoRA is ~1.5% of base params
        let [a_rows, a_cols] = lora.lora_a.shape().dims();
        let [b_rows, b_cols] = lora.lora_b.shape().dims();

        assert_eq!(a_rows, 1024);
        assert_eq!(a_cols, 8);
        assert_eq!(b_rows, 8);
        assert_eq!(b_cols, 1024);
    }

    #[test]
    fn display() {
        let device = Default::default();
        let lora = LoraLinearConfig::new(64, 32, 8).init::<TestBackend>(&device);
        let s = alloc::format!("{lora}");
        assert!(s.contains("rank: 8"));
    }
}
