use super::*;
use cortex_tensor::{Element, Shape};

#[test]
fn should_support_int_dtype() {
    let tensor = TestTensorInt::<2>::from([[0, -1, 2], [3, 4, -5]]).into_primitive();

    assert_eq!(
        cortex_tensor::TensorMetadata::shape(&tensor),
        Shape::new([2, 3])
    );
    assert_eq!(
        cortex_tensor::TensorMetadata::dtype(&tensor),
        IntElem::dtype() // default int elem type
    );
}
