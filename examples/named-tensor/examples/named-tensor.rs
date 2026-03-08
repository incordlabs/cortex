fn main() {
    named_tensor::run::<cortex::backend::ndarray::NdArray<f32>>(&Default::default());
}
