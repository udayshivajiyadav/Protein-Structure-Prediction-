"""
Attention Processing Module
===========================

This module processes transformer attention weights from ESMFold,
aggregates them across layers and heads, and prepares them for
visualization and analysis.
"""

import streamlit as st
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from config import DEFAULT_THRESHOLD_PERCENTILE, MAX_ATTENTION_LAYERS


class AttentionProcessor:
    """Processes and analyzes transformer attention weights."""
    
    def __init__(self):
        pass
    
    def process_attention(
        self, 
        attention_weights: List[torch.Tensor], 
        threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE
    ) -> Optional[np.ndarray]:
        """
        Process attention weights from ESMFold transformer layers.
        
        Args:
            attention_weights: List of attention tensors from different layers
            threshold_percentile: Percentile threshold for top interactions
            
        Returns:
            Aggregated attention matrix or None if processing fails
        """
        if not attention_weights:
            st.warning("⚠️ No attention weights provided")
            return None
        
        try:
            # Validate and process each attention tensor
            valid_attentions = []
            
            for i, attn in enumerate(attention_weights):
                if attn is None:
                    continue
                
                # Ensure tensor is on CPU
                if attn.device.type != 'cpu':
                    attn = attn.cpu()
                
                # Validate tensor shape (should be 4D: [batch, heads, seq_len, seq_len])
                if len(attn.shape) != 4:
                    st.warning(f"⚠️ Unexpected attention shape at layer {i}: {attn.shape}")
                    continue
                
                # Average across attention heads (dimension 1)
                try:
                    attn_averaged = attn.mean(dim=1)  # [batch, seq_len, seq_len]
                    
                    # Remove batch dimension
                    if attn_averaged.shape[0] == 1:
                        attn_averaged = attn_averaged.squeeze(0)  # [seq_len, seq_len]
                    
                    # Validate final shape
                    if len(attn_averaged.shape) != 2:
                        st.warning(f"⚠️ Invalid processed attention shape at layer {i}: {attn_averaged.shape}")
                        continue
                    
                    valid_attentions.append(attn_averaged)
                    
                except Exception as e:
                    st.warning(f"⚠️ Error processing attention at layer {i}: {str(e)}")
                    continue
            
            if not valid_attentions:
                st.warning("⚠️ No valid attention weights found")
                return None
            
            # Ensure all tensors have the same shape
            shapes = [attn.shape for attn in valid_attentions]
            unique_shapes = list(set(shapes))
            
            if len(unique_shapes) > 1:
                st.warning(f"⚠️ Inconsistent attention shapes: {unique_shapes}")
                # Use the most common shape
                from collections import Counter
                most_common_shape = Counter(shapes).most_common(1)[0][0]
                valid_attentions = [attn for attn in valid_attentions if attn.shape == most_common_shape]
                
                if not valid_attentions:
                    return None
            
            # Stack and aggregate across layers
            try:
                stacked_attention = torch.stack(valid_attentions)  # [layers, seq_len, seq_len]
                aggregated_attention = stacked_attention.mean(dim=0)  # [seq_len, seq_len]
            except Exception as e:
                st.error(f"Error stacking attention tensors: {str(e)}")
                return None
            
            # Convert to numpy for easier manipulation
            attention_matrix = aggregated_attention.numpy()
            
            # Remove special tokens (typically first and last positions for ESM models)
            if attention_matrix.shape[0] > 2:
                attention_matrix = attention_matrix[1:-1, 1:-1]
            
            # Validate final matrix
            if attention_matrix.shape[0] == 0 or attention_matrix.shape[1] == 0:
                st.warning("⚠️ Empty attention matrix after token removal")
                return None
            
            # Apply threshold to highlight top interactions
            if threshold_percentile > 0 and threshold_percentile < 100:
                non_zero_values = attention_matrix[attention_matrix > 0]
                if len(non_zero_values) > 0:
                    threshold = np.percentile(non_zero_values, threshold_percentile)
                    attention_matrix = np.where(attention_matrix >= threshold, attention_matrix, 0)
            
            return attention_matrix
            
        except Exception as e:
            st.error(f"Error processing attention weights: {str(e)}")
            return None
    
    def get_attention_statistics(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics for the attention matrix.
        
        Args:
            attention_matrix: Processed attention matrix
            
        Returns:
            Dictionary of attention statistics
        """
        if attention_matrix is None or attention_matrix.size == 0:
            return {}
        
        try:
            # Basic statistics
            total_interactions = np.count_nonzero(attention_matrix)
            max_attention = np.max(attention_matrix)
            min_attention = np.min(attention_matrix[attention_matrix > 0]) if total_interactions > 0 else 0
            mean_attention = np.mean(attention_matrix[attention_matrix > 0]) if total_interactions > 0 else 0
            std_attention = np.std(attention_matrix[attention_matrix > 0]) if total_interactions > 0 else 0
            
            # Sparsity
            total_possible = attention_matrix.shape[0] * attention_matrix.shape[1]
            sparsity = 1 - (total_interactions / total_possible)
            
            # Per-residue statistics
            residue_attention_sum = attention_matrix.sum(axis=1)
            residue_attention_max = attention_matrix.max(axis=1)
            residue_connections = (attention_matrix > 0).sum(axis=1)
            
            return {
                'total_interactions': int(total_interactions),
                'max_attention': float(max_attention),
                'min_attention': float(min_attention),
                'mean_attention': float(mean_attention),
                'std_attention': float(std_attention),
                'sparsity': float(sparsity),
                'matrix_shape': attention_matrix.shape,
                'residue_stats': {
                    'attention_sum': residue_attention_sum,
                    'attention_max': residue_attention_max,
                    'num_connections': residue_connections
                }
            }
            
        except Exception as e:
            st.error(f"Error calculating attention statistics: {str(e)}")
            return {}
    
    def get_top_interactions(
        self, 
        attention_matrix: np.ndarray, 
        top_k: int = 20
    ) -> List[tuple]:
        """
        Get the top-k strongest interactions from the attention matrix.
        
        Args:
            attention_matrix: Processed attention matrix
            top_k: Number of top interactions to return
            
        Returns:
            List of tuples (i, j, weight) for top interactions
        """
        if attention_matrix is None or attention_matrix.size == 0:
            return []
        
        try:
            # Get all non-zero interactions
            interactions = []
            for i in range(attention_matrix.shape[0]):
                for j in range(i + 1, attention_matrix.shape[1]):  # Avoid duplicates and self-loops
                    weight = attention_matrix[i, j]
                    if weight > 0:
                        interactions.append((i, j, weight))
            
            if not interactions:
                return []
            
            # Sort by weight and take top-k
            interactions.sort(key=lambda x: x[2], reverse=True)
            return interactions[:min(top_k, len(interactions))]
            
        except Exception as e:
            st.error(f"Error getting top interactions: {str(e)}")
            return []
    
    def get_attention_features(self, attention_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract attention-based features for each residue.
        
        Args:
            attention_matrix: Processed attention matrix
            
        Returns:
            Dictionary of attention features
        """
        if attention_matrix is None or attention_matrix.size == 0:
            return {}
        
        try:
            features = {
                'attention_sum': attention_matrix.sum(axis=1),
                'attention_max': attention_matrix.max(axis=1),
                'attention_mean': attention_matrix.mean(axis=1),
                'num_connections': (attention_matrix > 0).sum(axis=1),
                'attention_var': attention_matrix.var(axis=1)
            }
            
            return features
            
        except Exception as e:
            st.error(f"Error extracting attention features: {str(e)}")
            return {}


# Global instance
processor = AttentionProcessor()

def process_attention(
    attention_weights: List[torch.Tensor], 
    threshold_percentile: float = DEFAULT_THRESHOLD_PERCENTILE
) -> Optional[np.ndarray]:
    """Convenience function for processing attention weights."""
    return processor.process_attention(attention_weights, threshold_percentile)

def get_attention_statistics(attention_matrix: np.ndarray) -> Dict[str, Any]:
    """Convenience function for attention statistics."""
    return processor.get_attention_statistics(attention_matrix)

def get_top_interactions(attention_matrix: np.ndarray, top_k: int = 20) -> List[tuple]:
    """Convenience function for getting top interactions."""
    return processor.get_top_interactions(attention_matrix, top_k)

def get_attention_features(attention_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Convenience function for getting attention features."""
    return processor.get_attention_features(attention_matrix)
                