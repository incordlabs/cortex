use cortex_core as cortex;

use cortex::module::Module;
use cortex::tensor::Tensor;
use cortex::tensor::backend::Backend;

/// Applies the softsign function element-wise
/// See also [softsign](cortex::tensor::activation::softsign)
#[derive(Module, Clone, Debug, Default)]
pub struct Softsign;

impl Softsign {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        cortex::tensor::activation::softsign(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Softsign::new();

        assert_eq!(alloc::format!("{layer}"), "Softsign");
    }
}
