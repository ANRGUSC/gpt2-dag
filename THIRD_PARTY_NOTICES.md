# Third-Party Notices

This repository depends on and interfaces with third-party open-source software.

## Hugging Face Transformers (GPT-2 implementation)

- Project: `huggingface/transformers`
- Upstream URL: https://github.com/huggingface/transformers
- License: Apache License 2.0
- SPDX: `Apache-2.0`
- Dependency declaration: `transformers>=5.1.0` (see `pyproject.toml`)
- Referenced module paths:
  - `transformers.models.gpt2.modeling_gpt2`
  - `transformers.cache_utils`
  - `transformers.modeling_outputs`

Usage in this repository:

- Runtime imports and extension of Hugging Face GPT-2 classes/functions are used for DAG-structured execution and profiling.
- Source-level references to Hugging Face GPT-2 implementation are used for grounding methodology documentation.

Any redistributed copies or modifications of upstream source files must retain applicable upstream copyright and license notices.
