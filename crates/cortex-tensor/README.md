# Cortex Tensor

> [Cortex](https://github.com/qora-protocol/cortex) Tensor Library

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-tensor.svg)](https://crates.io/crates/cortex-tensor)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/qora-protocol/cortex-tensor/blob/master/README.md)

This library provides the core abstractions required to run tensor operations with Cortex.

`Tensor`s are generic over the backend to allow users to perform operations using different
`Backend` implementations. Cortex's tensors also support auto-differentiation thanks to the
`AutodiffBackend` trait.
