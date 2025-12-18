"""Causal analysis methods: DAS, MELBO, LAT."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name


class CausalAnalyzer:
    """Causal analysis using various methods."""
    
    def __init__(self, model: HookedTransformer):
        """
        Initialize causal analyzer.
        
        Args:
            model: HookedTransformer model
        """
        self.model = model
    
    def direct_activation_substitution(
        self,
        source_prompt: str,
        target_prompt: str,
        layer: int,
        intervention_type: str = 'resid'
    ) -> Dict[str, Any]:
        """
        Direct Activation Substitution (DAS) method.
        
        Args:
            source_prompt: Source prompt for activations
            target_prompt: Target prompt to intervene on
            layer: Layer to perform intervention
            intervention_type: Type of activation to substitute
            
        Returns:
            Dictionary with DAS results
        """
        # Get source activations
        source_activations = {}
        
        def source_hook(name, activation, hook):
            source_activations[name] = activation.detach().clone()
        
        hook_name = get_act_name(intervention_type, layer)
        self.model.add_hook(hook_name, source_hook)
        
        with torch.no_grad():
            _ = self.model(source_prompt)
        
        # Remove hook and add intervention hook
        self.model.remove_all_hook_fns()
        
        # Get baseline target output
        with torch.no_grad():
            baseline_logits = self.model(target_prompt)
        
        # Intervention hook
        def intervention_hook(name, activation, hook):
            if name in source_activations:
                return source_activations[name]
            return activation
        
        self.model.add_hook(hook_name, intervention_hook)
        
        with torch.no_grad():
            intervened_logits = self.model(target_prompt)
        
        # Compute effect
        effect = torch.abs(baseline_logits - intervened_logits).mean().item()
        
        return {
            'layer': layer,
            'intervention_type': intervention_type,
            'effect': effect,
            'baseline_logits': baseline_logits.detach().cpu(),
            'intervened_logits': intervened_logits.detach().cpu()
        }
    
    def melbo_analysis(
        self,
        prompt: str,
        layer: int,
        num_samples: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Maximum Entropy Latent Backdoor Optimization (MELBO) analysis.
        
        Args:
            prompt: Input prompt
            layer: Layer to analyze
            num_samples: Number of samples for optimization
            temperature: Temperature for sampling
            
        Returns:
            Dictionary with MELBO results
        """
        # Get activations
        activations = {}
        
        def activation_hook(name, activation, hook):
            activations[name] = activation.detach()
        
        hook_name = get_act_name("resid_post", layer)
        self.model.add_hook(hook_name, activation_hook)
        
        with torch.no_grad():
            logits = self.model(prompt)
        
        # MELBO: Find directions that maximize entropy while preserving output
        # (Simplified implementation)
        if activations:
            act = activations[hook_name]
            
            # Compute entropy of activations
            act_flat = act.flatten()
            hist, _ = np.histogram(act_flat.numpy(), bins=50)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            
            # Find high-entropy directions (simplified)
            # In practice, this would involve optimization
            high_entropy_directions = torch.randn(act.shape[-1], num_samples)
            
            return {
                'layer': layer,
                'entropy': entropy,
                'num_samples': num_samples,
                'high_entropy_directions': high_entropy_directions.shape,
                'activation_statistics': {
                    'mean': act.mean().item(),
                    'std': act.std().item(),
                    'entropy': entropy
                }
            }
        
        return {}
    
    def latent_adversarial_training(
        self,
        prompt: str,
        target_output: str,
        layer: int,
        num_iterations: int = 10,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """
        Latent Adversarial Training (LAT) analysis.
        
        Args:
            prompt: Input prompt
            target_output: Target output to match
            layer: Layer to perturb
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary with LAT results
        """
        # Get baseline
        with torch.no_grad():
            baseline_logits = self.model(prompt)
        
        # LAT: Find adversarial perturbations in latent space
        hook_name = get_act_name("resid_post", layer)
        
        # Initialize perturbation
        perturbation = torch.zeros(
            (1, self.model.cfg.n_ctx, self.model.cfg.d_model),
            requires_grad=True
        )
        
        # Optimization loop (simplified)
        losses = []
        for i in range(num_iterations):
            def lat_hook(name, activation, hook):
                return activation + perturbation
            
            self.model.add_hook(hook_name, lat_hook)
            
            with torch.no_grad():
                perturbed_logits = self.model(prompt)
            
            # Compute loss (simplified - would use target_output)
            loss = torch.abs(baseline_logits - perturbed_logits).mean()
            losses.append(loss.item())
            
            # Update perturbation (simplified gradient step)
            with torch.no_grad():
                perturbation -= learning_rate * torch.randn_like(perturbation) * loss.item()
            
            self.model.remove_all_hook_fns()
        
        return {
            'layer': layer,
            'num_iterations': num_iterations,
            'losses': losses,
            'final_perturbation_norm': perturbation.norm().item(),
            'converged': losses[-1] < losses[0] * 0.5 if len(losses) > 1 else False
        }
    
    def causal_trace_analysis(
        self,
        prompt: str,
        layers: List[int],
        intervention_type: str = 'zero'
    ) -> Dict[str, Any]:
        """
        Causal trace analysis across multiple layers.
        
        Args:
            prompt: Input prompt
            layers: Layers to trace
            intervention_type: Type of intervention
            
        Returns:
            Dictionary with causal trace results
        """
        # Get baseline
        with torch.no_grad():
            baseline_logits = self.model(prompt)
        
        effects = {}
        
        for layer in layers:
            # Ablate layer
            hook_name = get_act_name("resid_post", layer)
            
            def ablation_hook(name, activation, hook):
                if intervention_type == 'zero':
                    return torch.zeros_like(activation)
                elif intervention_type == 'mean':
                    return torch.full_like(activation, activation.mean())
                return activation
            
            self.model.add_hook(hook_name, ablation_hook)
            
            with torch.no_grad():
                ablated_logits = self.model(prompt)
            
            effect = torch.abs(baseline_logits - ablated_logits).mean().item()
            effects[layer] = effect
            
            self.model.remove_all_hook_fns()
        
        return {
            'layers': layers,
            'effects': effects,
            'max_effect_layer': max(effects, key=effects.get) if effects else None,
            'baseline_logits': baseline_logits.detach().cpu()
        }

