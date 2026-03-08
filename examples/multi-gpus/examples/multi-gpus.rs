fn main() {
    #[cfg(feature = "cuda")]
    multi_gpus::run::<cortex::backend::Cuda>();
    #[cfg(feature = "rocm")]
    multi_gpus::run::<cortex::backend::Rocm>();
    #[cfg(feature = "tch-gpu")]
    multi_gpus::run::<cortex::backend::LibTorch>();
}
