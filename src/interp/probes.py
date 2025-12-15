"""Probing tools using TransformerLens for interpretability."""

from typing import Dict, Any, Optional, List
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name


def probe_encoded_reasoning(
    model_name: str,
    prompt: str,
    layers: Optional[List[int]] = None,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Probe model activations for encoded reasoning patterns.
    
    Note: This requires a HuggingFace model compatible with TransformerLens.
    For Ollama models, you may need to export to HF format first.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input prompt text
        layers: List of layer indices to probe (None = all layers)
        use_cache: Whether to use model cache
        
    Returns:
        Dictionary with activation patterns and insights
    """
    try:
        # Load model with TransformerLens
        hooked_model = HookedTransformer.from_pretrained(
            model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Tokenize
        tokens = hooked_model.to_tokens(prompt)
        
        # Run forward pass and collect activations
        activations = {}
        
        def activation_hook(name, activation, hook):
            activations[name] = activation.detach().cpu()
        
        # Register hooks for specified layers or all layers
        if layers is None:
            layers = list(range(hooked_model.cfg.n_layers))
        
        for layer in layers:
            hook_name = get_act_name("resid_post", layer)
            hooked_model.add_hook(hook_name, activation_hook)
        
        # Run model
        with torch.no_grad():
            logits = hooked_model(prompt)
        
        # Compute statistics
        activation_stats = {}
        for name, act in activations.items():
            activation_stats[name] = {
                'mean': act.mean().item(),
                'std': act.std().item(),
                'max': act.max().item(),
                'min': act.min().item(),
                'shape': list(act.shape)
            }
        
        return {
            'activations': activations,
            'activation_stats': activation_stats,
            'logits': logits.detach().cpu() if isinstance(logits, torch.Tensor) else logits,
            'tokens': tokens,
            'layers_probed': layers
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'note': 'TransformerLens requires HuggingFace models. For Ollama, export to HF format first.'
        }


def setup_probe(
    model_name: str,
    probe_type: str = 'linear'
) -> Any:
    """
    Set up a probing classifier for detecting encoded information.
    
    Args:
        model_name: Model name
        probe_type: Type of probe ('linear', 'mlp', etc.)
        
    Returns:
        Probe model instance
    """
    # Placeholder for probe setup
    # In practice, this would train a linear/MLP probe on activations
    # to predict hidden information
    
    return {
        'model_name': model_name,
        'probe_type': probe_type,
        'status': 'not_implemented',
        'note': 'Probe training requires labeled activation data'
    }


def ablation_study(
    model_name: str,
    prompt: str,
    layers_to_ablate: List[int],
    ablation_type: str = 'zero'
) -> Dict[str, Any]:
    """
    Perform ablation study by zeroing out or modifying specific layer activations.
    
    Args:
        model_name: HuggingFace model name
        prompt: Input prompt
        layers_to_ablate: List of layer indices to ablate
        ablation_type: Type of ablation ('zero', 'mean', 'random')
        
    Returns:
        Dictionary with ablation results
    """
    try:
        hooked_model = HookedTransformer.from_pretrained(
            model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        def ablation_hook(name, activation, hook):
            if ablation_type == 'zero':
                return torch.zeros_like(activation)
            elif ablation_type == 'mean':
                return torch.full_like(activation, activation.mean())
            elif ablation_type == 'random':
                return torch.randn_like(activation)
            return activation
        
        # Register hooks
        for layer in layers_to_ablate:
            hook_name = get_act_name("resid_post", layer)
            hooked_model.add_hook(hook_name, ablation_hook)
        
        # Run model
        with torch.no_grad():
            logits_ablated = hooked_model(prompt)
        
        # Compare with baseline
        hooked_model_baseline = HookedTransformer.from_pretrained(model_name)
        with torch.no_grad():
            logits_baseline = hooked_model_baseline(prompt)
        
        return {
            'logits_ablated': logits_ablated,
            'logits_baseline': logits_baseline,
            'layers_ablated': layers_to_ablate,
            'ablation_type': ablation_type
        }
    
    except Exception as e:
        return {'error': str(e)}

