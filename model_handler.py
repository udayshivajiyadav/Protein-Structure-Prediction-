"""
ESMFold Model Handler
====================

This module handles loading and inference with the ESMFold model,
including direct model loading (not using pipelines) and proper
attention weight extraction.
"""

import streamlit as st
import torch
import numpy as np
import time
import warnings
from typing import Tuple, List, Optional, Dict, Any
from transformers import EsmTokenizer, EsmForProteinFolding
from config import MODEL_NAME, ERROR_MESSAGES, PROGRESS_STEPS

warnings.filterwarnings('ignore')


class ESMFoldHandler:
    """Handles ESMFold model loading, inference, and attention extraction."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.attention_hooks = []
    
    def load_model(self) -> Tuple[bool, str]:
        """
        Load ESMFold model and tokenizer directly (not using pipeline).
        
        Returns:
            Tuple[bool, str]: (success, device_used)
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Determine device
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_str)
            
            status_text.text(f"🔄 Loading ESMFold tokenizer...")
            progress_bar.progress(PROGRESS_STEPS['model_loading']['tokenizer'])
            
            # Load tokenizer
            self.tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
            
            status_text.text(f"🔄 Loading ESMFold model on {device_str.upper()}...")
            progress_bar.progress(PROGRESS_STEPS['model_loading']['model'])
            
            # Load model
            self.model = EsmForProteinFolding.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            status_text.text("🔧 Configuring model for attention extraction...")
            progress_bar.progress(PROGRESS_STEPS['model_loading']['config'])
            
            # Configure model to output attentions
            self.model.esm.config.output_attentions = True
            
            status_text.text("🧪 Testing model inference...")
            progress_bar.progress(PROGRESS_STEPS['model_loading']['test'])
            
            # Test with short sequence
            test_success = self._test_model_inference()
            if not test_success:
                raise RuntimeError("Model test inference failed")
            
            progress_bar.progress(PROGRESS_STEPS['model_loading']['complete'])
            status_text.text("✅ Model loaded successfully!")
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True, device_str
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in device_str and any(keyword in error_msg for keyword in ["out of memory", "out of bounds", "assert"]):
                st.warning("⚠️ GPU error detected, attempting CPU fallback...")
                return self._fallback_to_cpu(progress_bar, status_text)
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Model loading failed: {str(e)}")
                return False, None
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Unexpected error loading model: {str(e)}")
            return False, None
    
    def _fallback_to_cpu(self, progress_bar, status_text) -> Tuple[bool, str]:
        """Fallback to CPU if GPU loading fails."""
        try:
            status_text.text("🔄 Loading model on CPU...")
            progress_bar.progress(50)
            
            self.device = torch.device("cpu")
            
            # Reload model on CPU
            self.model = EsmForProteinFolding.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model.esm.config.output_attentions = True
            
            progress_bar.progress(90)
            
            # Test CPU inference
            test_success = self._test_model_inference()
            if not test_success:
                raise RuntimeError("CPU fallback test failed")
            
            progress_bar.progress(100)
            status_text.text("✅ Model loaded on CPU!")
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True, "cpu"
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"CPU fallback failed: {str(e)}")
            return False, None
    
    def _test_model_inference(self) -> bool:
        """Test model with a short sequence."""
        try:
            test_sequence = "MQIFVKTLTG"
            with torch.no_grad():
                tokens = self.tokenizer(
                    test_sequence,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                # Run forward pass
                outputs = self.model.esmfold(**tokens)
                return True
                
        except Exception as e:
            st.error(f"Model test failed: {str(e)}")
            return False
    
    def predict_structure(self, sequence: str) -> Tuple[str, List[torch.Tensor]]:
        """
        Predict protein structure and extract attention weights.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            Tuple[str, List[torch.Tensor]]: PDB string and attention weights
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                # Tokenize sequence
                tokens = self.tokenizer(
                    sequence,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                # Run inference
                outputs = self.model.esmfold(**tokens)
                
                # Extract PDB string
                pdb_string = self.model.output_to_pdb(outputs)[0]
                
                # Extract attention weights
                attention_weights = []
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    for attn in outputs.attentions:
                        if attn is not None:
                            # Move to CPU and detach
                            attn_cpu = attn.detach().cpu()
                            attention_weights.append(attn_cpu)
                
                return pdb_string, attention_weights
                
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "GPU out of memory. Try using a shorter sequence or restart with CPU mode."
            )
        except RuntimeError as e:
            if "CUDA" in str(e):
                raise RuntimeError(
                    f"CUDA error during prediction: {str(e)}. Try restarting with CPU mode."
                )
            else:
                raise RuntimeError(f"Model prediction failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during prediction: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "device": str(self.device),
            "model_name": MODEL_NAME,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "attention_layers": getattr(self.model.esm.config, 'num_hidden_layers', 'Unknown'),
            "attention_heads": getattr(self.model.esm.config, 'num_attention_heads', 'Unknown')
        }
    
    def clear_memory(self):
        """Clear model from memory to free up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global instance for caching
@st.cache_resource
def get_model_handler() -> ESMFoldHandler:
    """Get cached model handler instance."""
    return ESMFoldHandler()


# Convenience functions
def load_model() -> Tuple[bool, str]:
    """Load the ESMFold model."""
    handler = get_model_handler()
    return handler.load_model()


def predict_structure(sequence: str) -> Tuple[str, List[torch.Tensor]]:
    """Predict structure for a given sequence."""
    handler = get_model_handler()
    return handler.predict_structure(sequence)


def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    handler = get_model_handler()
    return handler.get_model_info()