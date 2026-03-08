# Cortex Dataset

> [Cortex](https://github.com/qora-protocol/cortex) dataset library

[![Current Crates.io Version](https://img.shields.io/crates/v/cortex-dataset.svg)](https://crates.io/crates/cortex-dataset)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/qora-protocol/cortex-dataset/blob/master/README.md)

The Cortex Dataset library is designed to streamline your machine learning (ML) data pipeline creation
process. It offers a variety of dataset implementations, transformation functions, and data sources.

## Feature Flags

- `audio` - enables audio dataset (SpeechCommandsDataset). Run the following example to try it out:

  ```shell
  cargo run --example speech_commands --features audio
  ```
