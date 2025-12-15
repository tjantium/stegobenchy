"""Tests for models module."""

import pytest
from unittest.mock import Mock, patch
from src.models import OllamaModel


class TestOllamaModel:
    """Test cases for OllamaModel."""
    
    def test_init(self):
        """Test model initialization."""
        model = OllamaModel('llama3:8b')
        assert model.model == 'llama3:8b'
    
    def test_init_default(self):
        """Test default model initialization."""
        model = OllamaModel()
        assert model.model == 'llama3:8b'
    
    @patch('src.models.ollama_wrapper.ollama.generate')
    def test_generate(self, mock_generate):
        """Test text generation."""
        mock_generate.return_value = {'response': 'Hello, world!'}
        
        model = OllamaModel('llama3:8b')
        output = model.generate("Hello")
        
        assert output == 'Hello, world!'
        mock_generate.assert_called_once()
    
    @patch('src.models.ollama_wrapper.ollama.generate')
    def test_generate_with_params(self, mock_generate):
        """Test generation with custom parameters."""
        mock_generate.return_value = {'response': 'Test output'}
        
        model = OllamaModel('llama3:8b')
        output = model.generate("Test", max_tokens=256, temperature=0.9)
        
        assert output == 'Test output'
        call_args = mock_generate.call_args
        assert call_args[1]['options']['num_predict'] == 256
        assert call_args[1]['options']['temperature'] == 0.9
    
    @patch('src.models.ollama_wrapper.ollama.generate')
    def test_generate_error_handling(self, mock_generate):
        """Test error handling in generation."""
        mock_generate.side_effect = Exception("Connection error")
        
        model = OllamaModel('llama3:8b')
        
        with pytest.raises(RuntimeError) as exc_info:
            model.generate("Test")
        
        assert "Error generating" in str(exc_info.value)
    
    @patch('src.models.ollama_wrapper.ollama.chat')
    def test_chat(self, mock_chat):
        """Test chat completion."""
        mock_chat.return_value = {
            'message': {'content': 'Hello!'},
            'response': 'Hello!'
        }
        
        model = OllamaModel('llama3:8b')
        messages = [{'role': 'user', 'content': 'Hi'}]
        response = model.chat(messages)
        
        assert 'message' in response
        mock_chat.assert_called_once()

