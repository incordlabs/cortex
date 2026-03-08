use crate::{BackendRouter, RunnerChannel};
use cortex_backend::ops::ActivationOps;

impl<R: RunnerChannel> ActivationOps<Self> for BackendRouter<R> {}
