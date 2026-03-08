use cortex_backend::ops::TransactionOps;

use crate::{BackendRouter, RunnerChannel};

impl<R: RunnerChannel> TransactionOps<Self> for BackendRouter<R> {}
