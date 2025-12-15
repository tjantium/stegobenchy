"""Ollama model wrapper for LLM inference."""

import ollama
from typing import Optional, Dict, Any


class OllamaModel:
    """Wrapper class for Ollama model inference."""
    
    def __init__(self, model_name: str = 'llama3:8b'):
        """
        Initialize Ollama model.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3:8b', 'phi3:mini')
        """
        self.model = model_name
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    **kwargs
                }
            )
            return response.get('response', '')
        except Exception as e:
            raise RuntimeError(f"Error generating with Ollama model {self.model}: {str(e)}")
    
    def chat(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Response dictionary with 'message' and 'response' keys
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    **kwargs
                }
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Error in chat with Ollama model {self.model}: {str(e)}")

