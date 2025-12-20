# Architecture

This document provides a detailed overview of the StegoBenchy codebase architecture.

## High-Level Overview

StegoBenchy is organized into modular layers:

- **Models (`src/models`)**: Wrappers around Ollama / HF models, including reasoning-specific models and finetuning utilities.
- **Datasets (`src/datasets`)**: Synthetic task generators for steganography, paraphrasing, robustness, and monitoring.
- **Pipelines (`src/pipelines`)**: Experiment runners for standard benchmarks, encoded reasoning, and reward hacking.
- **Eval (`src/eval`)**: Metrics for stego detection, reasoning accuracy, and entropy.
- **Interp (`src/interp`)**: Interpretability tools (probing, SAE analysis, causal methods).
- **Safety (`src/safety`)**: Safety case and control-agenda monitoring.
- **Viz (`src/viz`)**: Plotting and dashboards used by the demo.
- **Demo (`demo/app.py`)**: Streamlit UI that wires everything together.

## Data Flow

Data / control flow for a typical experiment:

1. **Dataset generation** → `src/datasets/*`
2. **Model selection / initialization** → `src/models/*`
3. **Pipeline execution** (generation + metrics) → `src/pipelines/*` + `src/eval/*`
4. **Optional finetuning / RL** → `src/models/finetune.py`, `src/models/advanced_finetune.py`
5. **Interpretability + safety analysis** → `src/interp/*`, `src/safety/*`
6. **Visualization + demo** → `src/viz/*`, `demo/app.py`

## Module Details

### Models (`src/models`)

#### `ollama_wrapper.py`
- `OllamaModel`: Thin wrapper around the `ollama` Python client.
- Provides `generate()` and `chat()` methods with options for `max_tokens`, `temperature`, `top_p`, etc.
- Used everywhere the app needs plain text generation from an Ollama model.

#### `reasoning_models.py`
- `ReasoningModel(OllamaModel)`: Adds support for reasoning-style models (e.g., DeepSeek-R1, GPT-OSS, QwQ-like models).
- Features:
  - Model registry for reasoning models and capabilities.
  - `generate_with_reasoning()`: Formats prompts for chain-of-thought and splits reasoning vs answer.
  - `generate_encoded_reasoning()`: Encourages the model to encode hidden information in its reasoning.
- `get_reasoning_model()`: Convenience factory.

#### `finetune.py`
- `finetune_sft()`: Vanilla SFT using TRL + HF models.
- `setup_ppo_trainer()`: Pre-configures a PPO trainer for RL experiments (generic, not stego-specific).

#### `advanced_finetune.py`
- `EmergentBehaviorTrainer`: Higher-level finetuner for encoded reasoning experiments.
- `rl_finetune_for_reward_hacking()`: Skeleton RL loop for exploring reward hacking.
- `RewardHackingStudy`: Utilities to create reward-hacking-centric datasets and analyze outputs for hacking patterns.

### Datasets (`src/datasets`)

#### `coin_flip.py`
- `generate_coin_flip_dataset()`: Basic coin-flip prompts with labels `heads` / `tails`.
- `generate_coin_flip_with_context()`: Adds few-shot contextual examples and metadata.

#### `paraphrase.py`
- `generate_paraphrase_dataset()`: Uses Ollama to paraphrase base texts.
- `generate_encoded_paraphrase_dataset()`: Asks the model to paraphrase while encoding a specific hidden string.

#### `advanced_datasets.py`
- `generate_robust_paraphrase_dataset()`: Multi-style paraphrasing for robustness.
- `generate_stego_cover_dataset()`: Cover-text tasks (story/email/article/review) that must encode hidden info.
- `generate_robustness_evaluation_dataset()`: Adds perturbations (typos, noise, capitalization) to prompts.
- `generate_monitoring_robustness_dataset()`: Scenarios designed for safety / control-agenda monitoring.

### Pipelines (`src/pipelines`)

#### `experiment_pipeline.py`
- `run_experiment()`: Core loop – for each dataset sample:
  - Calls `model.generate()` on `sample['prompt']`.
  - Runs `eval_metrics(output, label)`.
  - Logs results (optionally to W&B) and returns a list of per-sample dicts.
- `run_rl_experiment()`: Lightweight RL-style loop for sampling outputs and applying a scalar `reward_fn`.

#### `advanced_pipelines.py`
- `RewardHackingPipeline`:
  - Runs many prompt → output → reward iterations.
  - Classifies behavior into coarse patterns (e.g., `always_agree`, `minimal_response`).
  - Detects indicators of reward hacking (reward consistency, pattern repetition, etc.).
- `EncodedReasoningPipeline`:
  - Uses a `ReasoningModel` to generate encoded reasoning for `(prompt, hidden_info)` pairs.
  - Computes success rate and a coarse encoding-quality score.

### Evaluation (`src/eval`)

#### `metrics.py`
- `stego_detection_rate()`: Checks for forbidden tokens (e.g., `heads`, `tails`) and counts them.
- `reasoning_accuracy()`: Heuristics for whether the model's reasoning matches the label (direct or encoded).
- `hidden_info_entropy()`: Shannon entropy of the output token distribution (proxy for obfuscation).
- `compute_all_metrics()`: Combines all of the above into a single metrics dict + `combined_score`.

### Interpretability (`src/interp`)

#### `probes.py`
- `probe_encoded_reasoning()`: Uses TransformerLens `HookedTransformer` to capture activations and basic stats.
- `setup_probe()`: Placeholder for a learned linear/MLP probe.
- `ablation_study()`: Residual-stream ablation per layer.

#### `sae_analysis.py`
- `SAEAnalyzer`: Skeleton for SAE-based feature analysis:
  - `analyze_features()`: Captures layer activations and reports top-K latent directions + sparsity.
  - `find_encoded_features()`: Heuristic for features strongly associated with hidden info.

#### `causal_analysis.py`
- `CausalAnalyzer` implementing:
  - `direct_activation_substitution()` (DAS-style swap of activations between prompts).
  - `melbo_analysis()` (coarse entropy-based latent exploration).
  - `latent_adversarial_training()` (LAT-style perturbation loop).
  - `causal_trace_analysis()`: Per-layer ablation and effect measurement.

### Safety (`src/safety`)

#### `monitoring.py`
- `SafetyLevel` + `SafetyCase`: Small schema for describing safety cases.
- `ControlAgendaMonitor`:
  - Regex + keyword based checks for unsafe language (deception, bypassing safety, goal misgeneralization).
  - `monitor_reasoning()`: Per-text assessment and risk score.
  - `monitor_chain_of_thought()`: Step-wise CoT assessment with escalation detection.
  - `get_safety_report()`: Human-readable report string.

### Visualization (`src/viz`)

#### `plots.py`
- `plot_metrics_over_time()`: Line plots of metrics across samples.
- `plot_stego_success_rate()`: Histogram of success vs failure.
- `plot_entropy_distribution()`: Entropy histogram.
- `create_experiment_dashboard()`: 2x2 Plotly dashboard aggregating the above.

### Demo (`demo/app.py`)

Streamlit UI with tabs:
- **Single Prompt**: Manual prompt → output, quick metrics.
- **Coin Flip Experiment**: Runs `generate_coin_flip_dataset` + `run_experiment` and shows table + metrics.
- **Batch Evaluation**: Filters and inspects individual examples.
- **Visualizations**: Renders Plotly dashboard + histograms.
- **Safety Monitoring**: Front-end for `ControlAgendaMonitor`.
- **Advanced Pipelines**: UI for reward hacking and encoded reasoning experiments + stego cover dataset preview.

## Architecture Diagram

```text
           +-------------------+
           |   demo/app.py     |
           |  (Streamlit UI)   |
           +----------+--------+
                      |
                      v
        +-------------+--------------+
        |         Pipelines          |
        |  (src/pipelines/*)         |
        +------+------+--------------+
               |      |
      uses     |      | uses
               v      v
      +--------+--+  +------------------+
      |  Models   |  |     Datasets     |
      | src/models|  |  src/datasets    |
      +-----+-----+  +---------+--------+
            |                  |
            v                  v
      +-----+-----+      +-----+---------+
      |   Ollama  |      |   Metrics     |
      |  (runtime)|      | src/eval      |
      +-----------+      +------+--------+
                                  |
                                  v
                 +----------------+----------------+
                 | Interp & Safety (src/interp,    |
                 | src/safety) + Viz (src/viz)     |
                 +---------------------------------+
```

This shows the main dependencies: the **demo** talks to **pipelines**, which orchestrate **models**, **datasets**, and **eval**. Interpretability, safety, and visualization sit alongside to analyze and display what the pipelines produce.

