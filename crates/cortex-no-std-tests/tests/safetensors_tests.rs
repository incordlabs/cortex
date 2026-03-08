extern crate alloc;

#[test]
fn test_safetensors_no_std() {
    use cortex_ndarray::NdArray;
    use cortex_no_std_tests::safetensors;
    type Backend = NdArray<f32>;
    let device = Default::default();

    // Run all SafeTensors tests
    safetensors::run_all_tests::<Backend>(&device);
}
