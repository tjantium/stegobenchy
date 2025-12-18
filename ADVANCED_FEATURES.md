# Advanced Features Documentation

This document describes the advanced research capabilities added to StegoBenchy.

## 🧠 Extended Model Support

### Reasoning Models
- **DeepSeek-R1, V3**: Full support for reasoning models with chain-of-thought
- **GPT-OSS, QwQ**: Extended model registry for open-source reasoning models
- **ReasoningModel Class**: Specialized wrapper for reasoning models with:
  - Chain-of-thought generation
  - Encoded reasoning capabilities
  - Reasoning format options (CoT, scratchpad, thinking)

**Usage:**
```python
from src.models import ReasoningModel

model = ReasoningModel('deepseek-r1:latest')
result = model.generate_with_reasoning("Solve this problem step by step")
```

## 🔬 Advanced Experiment Pipelines

### Reward Hacking Pipeline
- Detects reward gaming behaviors
- Analyzes behavior patterns (always_agree, minimal_response, etc.)
- Identifies hacking indicators (consistency, repetition, gaming)

**Usage:**
```python
from src.pipelines import RewardHackingPipeline

pipeline = RewardHackingPipeline(model, reward_fn)
results = pipeline.run_reward_hacking_experiment(dataset)
```

### Encoded Reasoning Pipeline
- Studies how models encode hidden information in reasoning
- Measures encoding quality and success rates
- Analyzes indirect vs explicit references

**Usage:**
```python
from src.pipelines import EncodedReasoningPipeline

pipeline = EncodedReasoningPipeline(reasoning_model)
results = pipeline.run_encoded_reasoning_experiment(prompts, hidden_info)
```

## 🛡️ Safety Monitoring

### Control Agenda Monitor
- Detects control agenda and unsafe chain-of-thought
- Monitors for safety cases:
  - Encoded deception
  - Reward hacking
  - Goal misgeneralization
  - Reasoning manipulation
- Generates safety reports

**Usage:**
```python
from src.safety import ControlAgendaMonitor

monitor = ControlAgendaMonitor()
assessment = monitor.monitor_reasoning(reasoning_text)
report = monitor.get_safety_report(reasoning_text)
```

## 🔍 Extended Interpretability

### SAE Analysis
- Sparse Autoencoder feature analysis
- Feature activation analysis
- Encoding detection in latent space

**Usage:**
```python
from src.interp import SAEAnalyzer

analyzer = SAEAnalyzer(hooked_model)
features = analyzer.analyze_features(prompt, layer=10)
```

### Feature Ablation
- Progressive feature ablation
- Impact measurement
- Causal feature identification

**Usage:**
```python
from src.interp import FeatureAblation

ablator = FeatureAblation(hooked_model)
result = ablator.ablate_features(prompt, layer=10, feature_indices=[1, 5, 10])
```

### Causal Analysis Methods

#### Direct Activation Substitution (DAS)
- Substitute activations between prompts
- Measure causal effects

#### Maximum Entropy Latent Backdoor Optimization (MELBO)
- Find high-entropy directions
- Analyze latent space structure

#### Latent Adversarial Training (LAT)
- Find adversarial perturbations
- Study model robustness

**Usage:**
```python
from src.interp import CausalAnalyzer

analyzer = CausalAnalyzer(hooked_model)
das_result = analyzer.direct_activation_substitution(source, target, layer=10)
melbo_result = analyzer.melbo_analysis(prompt, layer=10)
lat_result = analyzer.latent_adversarial_training(prompt, target, layer=10)
```

## 📊 Advanced Datasets

### Robust Paraphrasing
- Multiple paraphrase styles (formal, casual, technical)
- Robustness evaluation metrics

### Stego Cover Tasks
- Story, email, article, review formats
- Hidden information encoding tasks

### Robustness Evaluation
- Typo perturbations
- Synonym substitutions
- Reordering and noise addition

### Monitoring Robustness
- Safety scenario datasets
- Control agenda detection datasets

**Usage:**
```python
from src.datasets import (
    generate_robust_paraphrase_dataset,
    generate_stego_cover_dataset,
    generate_robustness_evaluation_dataset,
    generate_monitoring_robustness_dataset
)
```

## 🎯 Advanced Finetuning

### Emergent Behavior Trainer
- Finetune for encoded reasoning
- RL-based reward hacking studies
- Behavior pattern analysis

**Usage:**
```python
from src.models.advanced_finetune import EmergentBehaviorTrainer

trainer = EmergentBehaviorTrainer('gpt2')
model = trainer.finetune_for_encoded_reasoning(dataset)
```

### Reward Hacking Study
- Create reward hacking datasets
- Analyze hacking patterns
- Compute hacking scores

**Usage:**
```python
from src.models.advanced_finetune import RewardHackingStudy

study = RewardHackingStudy('gpt2')
dataset = study.create_reward_hacking_dataset(prompts)
analysis = study.analyze_hacking_patterns(outputs, reward_fn)
```

## 🚀 Demo Features

The Streamlit demo now includes:

1. **Safety Monitoring Tab**: Real-time safety assessment
2. **Advanced Pipelines Tab**: 
   - Reward hacking studies
   - Encoded reasoning analysis
   - Stego cover tasks

## 📝 Research Workflows

### Studying Encoded Reasoning
1. Use `ReasoningModel` for chain-of-thought generation
2. Run `EncodedReasoningPipeline` to study encoding patterns
3. Use `SAEAnalyzer` to find encoded features
4. Apply `CausalAnalyzer` for causal understanding

### Detecting Reward Hacking
1. Create reward hacking dataset
2. Run `RewardHackingPipeline`
3. Analyze behavior patterns
4. Use `RewardHackingStudy` for detailed analysis

### Safety Monitoring
1. Generate reasoning with model
2. Use `ControlAgendaMonitor` to assess safety
3. Review safety reports
4. Monitor chain-of-thought for escalation

## 🔗 Integration Examples

### Full Workflow: Encoded Reasoning Study
```python
from src.models import ReasoningModel
from src.pipelines import EncodedReasoningPipeline
from src.interp import SAEAnalyzer, CausalAnalyzer
from src.safety import ControlAgendaMonitor

# Generate encoded reasoning
model = ReasoningModel('deepseek-r1:latest')
pipeline = EncodedReasoningPipeline(model)
results = pipeline.run_encoded_reasoning_experiment(prompts, hidden_info)

# Analyze with SAE
analyzer = SAEAnalyzer(hooked_model)
features = analyzer.find_encoded_features(prompt, hidden_info, layers=[5, 10, 15])

# Causal analysis
causal = CausalAnalyzer(hooked_model)
das = causal.direct_activation_substitution(source, target, layer=10)

# Safety monitoring
monitor = ControlAgendaMonitor()
for result in results['results']:
    assessment = monitor.monitor_reasoning(result['reasoning'])
```

## 📚 References

- DAS: Direct Activation Substitution for causal analysis
- MELBO: Maximum Entropy Latent Backdoor Optimization
- LAT: Latent Adversarial Training
- SAE: Sparse Autoencoders for feature analysis
- TransformerLens: Interpretability framework

## 🎓 Next Steps

1. **Extend SAE Support**: Add support for loading pre-trained SAEs
2. **Advanced Causal Methods**: Implement DAS variants and extensions
3. **Monitoring Dashboards**: Real-time monitoring visualizations
4. **Automated Safety Reports**: Generate comprehensive safety assessments
5. **Multi-Model Comparisons**: Compare behaviors across model families

