use cortex_core as cortex;

use cortex::config::Config;

/// Configuration to create a [GradScaler](GradScaler) for mixed precision training.
///
/// The GradScaler dynamically scales the loss to prevent gradient underflow when training
/// with float16. It multiplies the loss by a large scale factor before the backward pass,
/// then divides gradients by the same factor before the optimizer step.
///
/// If inf/NaN gradients are detected (indicating overflow), the optimizer step is skipped
/// and the scale factor is reduced. If training proceeds without overflow for several
/// consecutive steps, the scale factor is increased to use more of the float16 dynamic range.
///
/// # Example
///
/// ```ignore
/// let mut scaler = GradScalerConfig::new().init();
///
/// // Training loop:
/// let loss = model.forward(input);
/// let scaled_loss = scaler.scale_loss(loss);
/// let grads = scaled_loss.backward();
/// let mut grads_params = GradientsParams::from_grads(grads, &model);
///
/// // Returns true if step is valid (no inf/nan), false if step should be skipped
/// if scaler.unscale_and_check::<B>(&mut grads_params) {
///     model = optimizer.step(lr, model, grads_params);
/// }
/// scaler.update();
/// ```
#[derive(Config, Debug)]
pub struct GradScalerConfig {
    /// The initial loss scale factor. Default: 65536.0 (2^16).
    #[config(default = 65536.0)]
    pub init_scale: f64,
    /// Factor to increase scale when no overflow is detected. Default: 2.0.
    #[config(default = 2.0)]
    pub growth_factor: f64,
    /// Factor to decrease scale when overflow is detected. Default: 0.5.
    #[config(default = 0.5)]
    pub backoff_factor: f64,
    /// Number of consecutive non-overflow steps before increasing scale. Default: 2000.
    #[config(default = 2000)]
    pub growth_interval: usize,
}

impl GradScalerConfig {
    /// Initialize a new [GradScaler].
    pub fn init(&self) -> GradScaler {
        GradScaler {
            scale: self.init_scale,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
            growth_tracker: 0,
            found_inf_last: false,
        }
    }
}

/// Dynamic loss scaler for mixed precision (AMP) training.
///
/// Prevents gradient underflow when training with float16 by scaling the loss before
/// the backward pass and unscaling gradients before the optimizer step.
///
/// The scale factor is adjusted dynamically:
/// - **Decreased** when inf/NaN is detected in gradients (overflow)
/// - **Increased** after `growth_interval` consecutive clean steps (to use more dynamic range)
///
/// See [GradScalerConfig] for configuration options.
#[derive(Debug, Clone)]
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    growth_tracker: usize,
    found_inf_last: bool,
}

impl GradScaler {
    /// Scale the loss tensor before calling `.backward()`.
    ///
    /// Multiplies the loss by the current scale factor.
    ///
    /// # Shapes
    ///
    /// - input: `[...]` (any shape, typically a scalar loss)
    /// - output: `[...]` (same shape as input)
    pub fn scale_loss<B: cortex::tensor::backend::Backend, const D: usize>(
        &self,
        loss: cortex::tensor::Tensor<B, D>,
    ) -> cortex::tensor::Tensor<B, D> {
        loss.mul_scalar(self.scale as f32)
    }

    /// Unscale gradients and check for inf/NaN.
    ///
    /// This divides all gradient tensors by the scale factor and checks if any
    /// contain inf or NaN values (indicating overflow during the backward pass).
    ///
    /// Returns `true` if gradients are valid and the optimizer step should proceed.
    /// Returns `false` if inf/NaN was found and the step should be skipped.
    ///
    /// Call [`update`](GradScaler::update) after this to adjust the scale factor.
    pub fn unscale_and_check<B: cortex::tensor::backend::Backend>(
        &mut self,
        grads: &mut crate::GradientsParams,
    ) -> bool {
        // Unscale: divide all gradients by the scale factor
        let inv_scale = 1.0 / self.scale as f32;
        grads.scale_all::<B>(inv_scale);

        // Check for inf/nan
        let found_inf = grads.has_inf_or_nan::<B>();
        self.found_inf_last = found_inf;
        !found_inf
    }

    /// Update the scale factor based on the last step's inf/NaN status.
    ///
    /// Must be called after [`unscale_and_check`](GradScaler::unscale_and_check).
    ///
    /// - If overflow was detected: scale is reduced by `backoff_factor`, growth tracker resets
    /// - If no overflow for `growth_interval` steps: scale is increased by `growth_factor`
    pub fn update(&mut self) {
        if self.found_inf_last {
            self.scale *= self.backoff_factor;
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
            }
        }
    }

    /// Get the current loss scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Whether the last step found inf/NaN in gradients.
    pub fn found_inf(&self) -> bool {
        self.found_inf_last
    }

    /// Returns whether the scaler is effectively enabled (scale > 0).
    pub fn is_enabled(&self) -> bool {
        self.scale > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let scaler = GradScalerConfig::new().init();
        assert_eq!(scaler.scale(), 65536.0);
        assert!(!scaler.found_inf());
    }

    #[test]
    fn test_custom_config() {
        let scaler = GradScalerConfig::new()
            .with_init_scale(1024.0)
            .with_growth_factor(4.0)
            .with_backoff_factor(0.25)
            .with_growth_interval(100)
            .init();
        assert_eq!(scaler.scale(), 1024.0);
    }

    #[test]
    fn test_scale_loss() {
        let scaler = GradScalerConfig::new()
            .with_init_scale(1024.0)
            .init();

        let device = Default::default();
        let loss =
            cortex::tensor::Tensor::<crate::TestBackend, 1>::from_floats([2.0], &device);
        let scaled = scaler.scale_loss(loss);
        let data = scaled.into_data();
        let value: f32 = data.iter::<f32>().next().unwrap();
        assert!((value - 2048.0).abs() < 1e-3);
    }

    #[test]
    fn test_backoff_on_inf() {
        let mut scaler = GradScalerConfig::new()
            .with_init_scale(1024.0)
            .with_backoff_factor(0.5)
            .init();

        // Simulate finding inf
        scaler.found_inf_last = true;
        scaler.update();

        assert_eq!(scaler.scale(), 512.0);
        assert_eq!(scaler.growth_tracker, 0);
    }

    #[test]
    fn test_growth_after_interval() {
        let mut scaler = GradScalerConfig::new()
            .with_init_scale(1024.0)
            .with_growth_factor(2.0)
            .with_growth_interval(3)
            .init();

        // 3 clean steps should trigger growth
        for _ in 0..3 {
            scaler.found_inf_last = false;
            scaler.update();
        }

        assert_eq!(scaler.scale(), 2048.0);
    }

    #[test]
    fn test_growth_resets_on_inf() {
        let mut scaler = GradScalerConfig::new()
            .with_init_scale(1024.0)
            .with_growth_interval(3)
            .init();

        // 2 clean steps
        scaler.found_inf_last = false;
        scaler.update();
        scaler.found_inf_last = false;
        scaler.update();
        assert_eq!(scaler.growth_tracker, 2);

        // Then inf — resets tracker and halves scale
        scaler.found_inf_last = true;
        scaler.update();
        assert_eq!(scaler.growth_tracker, 0);
        assert_eq!(scaler.scale(), 512.0);
    }
}
