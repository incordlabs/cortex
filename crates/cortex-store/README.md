# Cortex Store

> Advanced model storage and serialization for the Cortex deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-store.svg)](https://crates.io/crates/cortex-store)
[![Documentation](https://docs.rs/cortex-store/badge.svg)](https://docs.rs/cortex-store)

A comprehensive storage library for Cortex that enables efficient model serialization, cross-framework
interoperability, and advanced tensor management.

> **Migrating from cortex-import?** See the [Migration Guide](MIGRATION.md) for help moving from
> `PyTorchFileRecorder`/`SafetensorsFileRecorder` to the new Store API.

## Features

- **Burnpack Format** - Native Cortex format with CBOR metadata, memory-mapped loading, ParamId
  persistence for stateful training, and no-std support
- **SafeTensors Format** - Industry-standard format for secure and efficient tensor serialization
- **PyTorch Support** - Direct loading of PyTorch .pth/.pt files with automatic weight
  transformation
- **Zero-Copy Loading** - Memory-mapped files and lazy tensor materialization for optimal
  performance
- **Flexible Filtering** - Load/save specific model subsets with regex, exact paths, or custom
  predicates
- **Tensor Remapping** - Rename tensors during load/save for framework compatibility
- **No-std Support** - Burnpack and SafeTensors formats available in embedded and WASM environments

## Quick Start

```rust
use cortex_store::{ModuleSnapshot, PytorchStore, SafetensorsStore, BurnpackStore};

// Load from PyTorch
let mut store = PytorchStore::from_file("model.pt");
model.load_from(&mut store)?;

// Load from SafeTensors (with PyTorch adapter)
let mut store = SafetensorsStore::from_file("model.safetensors")
    .with_from_adapter(PyTorchToCortexAdapter);
model.load_from(&mut store)?;

// Save to Burnpack
let mut store = BurnpackStore::from_file("model.bpk");
model.save_into(&mut store)?;
```

## Documentation

For comprehensive documentation including:

- Exporting weights from PyTorch
- Loading weights into Cortex models
- Saving models to various formats
- Advanced features (filtering, remapping, partial loading, zero-copy)
- API reference and troubleshooting

See the **[Cortex Book - Model Weights](https://qoranet.com/book/import/model-weights.html)** chapter.

## Running Benchmarks

```bash
# Generate model files (one-time setup)
uv run benches/generate_unified_models.py

# Run loading benchmarks
cargo bench --bench unified_loading

# Run saving benchmarks
cargo bench --bench unified_saving

# With specific backend
cargo bench --bench unified_loading --features metal
```

## License

This project is dual-licensed under MIT and Apache-2.0.
