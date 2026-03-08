use cortex_core as cortex;

#[cfg(feature = "collective")]
use cortex_collective::{CollectiveError, PeerId, ReduceOperation, all_reduce};

use cortex::{
    Tensor,
    tensor::{
        backend::{AutodiffBackend, Backend},
        container::TensorContainer,
    },
};

use cortex::module::{AutodiffModule, ParamId};

use super::visitor::{GradientsParamsChangeDevice, GradientsParamsConverter};

/// Data type that contains gradients for parameters.
#[derive(Default, Debug)]
pub struct GradientsParams {
    container: TensorContainer<ParamId>,
}

impl GradientsParams {
    /// Creates a new [GradientsParams](GradientsParams).
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    ///
    /// Note: This consumes the gradients. See ['from_module'] to extract gradients only for
    ///  a specific module.
    pub fn from_grads<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads = grads;
        Self::from_module(&mut grads, module)
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    pub fn from_module<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, None);
        module.visit(&mut visitor);
        grads_params
    }

    /// Extract tensor gradients for the given [module](AutodiffModule) and given parameters.
    pub fn from_params<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
        params: &[ParamId],
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor =
            GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, Some(params.to_vec()));
        module.visit(&mut visitor);
        grads_params
    }

    /// Get the gradients for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// You should use [remove](GradientsParams::remove) if you want to get the gradients
    /// only one time.
    pub fn get<B, const D: usize>(&self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.get(&id).map(Tensor::from_primitive)
    }

    /// Remove the gradients for the given [parameter id](ParamId).
    pub fn remove<B, const D: usize>(&mut self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.remove(&id).map(Tensor::from_primitive)
    }

    /// Register a gradients tensor for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given [parameter id](ParamId), it will be replaced.
    pub fn register<B, const D: usize>(&mut self, id: ParamId, value: Tensor<B, D>)
    where
        B: Backend,
    {
        self.container.register(id, value.into_primitive())
    }

    /// Get the IDs of all registered gradient tensors.
    pub fn ids(&self) -> alloc::vec::Vec<&ParamId> {
        self.container.ids()
    }

    /// The number of gradients tensors registered.
    pub fn len(&self) -> usize {
        self.container.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Multiply all gradient tensors by a scalar factor.
    ///
    /// Useful for gradient unscaling in mixed precision training (divide by the loss scale).
    #[cfg(feature = "std")]
    pub fn scale_all<B: Backend>(&mut self, factor: f32) {
        let ids: alloc::vec::Vec<ParamId> =
            self.container.ids().into_iter().copied().collect();

        for id in ids {
            let Some(grad) = self.container.remove::<B>(&id) else {
                continue;
            };

            let scaled = match grad {
                cortex::tensor::TensorPrimitive::Float(t) => {
                    cortex::tensor::TensorPrimitive::Float(B::float_mul_scalar(t, factor.into()))
                }
                other => other,
            };

            self.container.register::<B>(id, scaled);
        }
    }

    /// Check if any gradient tensor contains inf or NaN values.
    ///
    /// Returns `true` if any gradient has inf or NaN. Used by [`GradScaler`](crate::grad_scaler::GradScaler)
    /// to detect overflow during mixed precision training.
    #[cfg(feature = "std")]
    pub fn has_inf_or_nan<B: Backend>(&self) -> bool {
        for id in self.container.ids() {
            let Some(grad) = self.container.get::<B>(id) else {
                continue;
            };

            match grad {
                cortex::tensor::TensorPrimitive::Float(t) => {
                    let is_nan = B::float_is_nan(t.clone());
                    let is_inf = B::float_is_inf(t);
                    let any_nan = B::bool_any(is_nan);
                    let any_inf = B::bool_any(is_inf);
                    let any_bad = B::bool_or(any_nan, any_inf);

                    // Read the scalar value synchronously
                    let data = cortex::tensor::Tensor::<B, 1, cortex::tensor::Bool>::from_primitive(
                        any_bad,
                    )
                    .into_data();

                    if data.iter::<bool>().next().unwrap_or(false) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Change the device of each tensor gradients registered for the given [module](AutodiffModule).
    pub fn to_device<B: AutodiffBackend, M: AutodiffModule<B>>(
        mut self,
        device: &B::Device,
        module: &M,
    ) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M, B>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }

    /// Syncs the gradient params with the other peers in the collective.
    #[cfg(feature = "collective")]
    pub fn all_reduce<B: Backend>(
        mut self,
        peer_id: PeerId,
        op: ReduceOperation,
    ) -> Result<Self, CollectiveError> {
        let mut ids = self
            .container
            .ids()
            .into_iter()
            .copied()
            .collect::<Vec<ParamId>>();
        // This is crucial, since the all-reduce operations need to happen in the same order for the same parameters on all nodes!
        ids.sort();

        for id in ids {
            let Some(grad) = self.container.remove::<B>(&id) else {
                todo!()
            };

            let grad = match grad {
                cortex::tensor::TensorPrimitive::Float(grad) => {
                    let grad = all_reduce::<B>(peer_id, grad, op)?;
                    cortex::tensor::TensorPrimitive::Float(grad)
                }
                cortex::tensor::TensorPrimitive::QFloat(_grad) => {
                    unimplemented!("quantized all-reduce unimplemented")
                }
            };

            self.container.register::<B>(id, grad);
        }

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use cortex::module::{Module, list_param_ids};
    use cortex::tensor::{Distribution, backend::Backend};
    use cortex_nn::{Linear, LinearConfig};

    #[test]
    fn test_convert_grads() {
        let device = Default::default();
        let layer_1 = layer::<TestAutodiffBackend>(&device);
        let mut layer_2 = layer_1.clone();
        layer_2 = layer_2.fork(&device);
        let loss_1 = layer_1.forward(random_tensor(&device));
        let loss_2 = layer_2.forward(random_tensor(&device));
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer_1);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer_2);

        let param_ids_1 = list_param_ids(&layer_1);
        let param_ids_2 = list_param_ids(&layer_2);

        assert_eq!(param_ids_1, param_ids_2);
        assert_eq!(grads_1.len(), param_ids_1.len());
        assert_eq!(grads_2.len(), param_ids_2.len());
    }

    fn layer<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::random([2, 20], Distribution::Default, device)
    }
}
