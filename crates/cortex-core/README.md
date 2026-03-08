# Cortex Core

This crate should be used with [cortex](https://github.com/qora-protocol/cortex). It contains the core
traits and components for building and training deep learning models with Cortex.

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-core.svg)](https://crates.io/crates/cortex-core)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/qora-protocol/cortex-core/blob/master/README.md)

## Feature Flags

This crate can be used without the standard library (`#![no_std]`) with `alloc` by disabling the
default `std` feature.

- `std` - enables the standard library. Enabled by default.
- `experimental-named-tensor` - enables experimental named tensor.
