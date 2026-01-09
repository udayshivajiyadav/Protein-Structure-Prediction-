"""
Structure Visualization Module
==============================

This module handles all visualization components including 3D structure
rendering, attention heatmaps, interaction networks, and statistical plots.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from typing import List, Optional, Tuple
from config import (
    HEATMAP_WIDTH, HEATMAP_HEIGHT, NETWORK_WIDTH, NETWORK_HEIGHT,
    STRUCTURE_WIDTH, STRUCTURE_HEIGHT, ATTENTION_COLORSCALE, NODE_COLORS
)

# Try to import py3dmol
try:
    import py3dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False


class StructureVisualizer:
    """Handles all visualization components for the protein structure app."""
    
    def __init__(self):
        self.py3dmol_available = PY3DMOL_AVAILABLE
    
    def visualize_3d_structure(self, pdb_string: str) -> Optional[str]:
        """
        Create 3D protein structure visualization using py3dmol.
        
        Args:
            pdb_string: PDB format structure data
            
        Returns:
            HTML string for embedding or None if unavailable
        """
        if not self.py3dmol_available:
            return None
        
        try:
            # Create py3dmol view
            view = py3dmol.view(width=STRUCTURE_WIDTH, height=STRUCTURE_HEIGHT)
            view.addModel(pdb_string, 'pdb')
            
            # Enhanced styling
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.addStyle({'stick': {'radius': 0.15, 'colorscheme': 'default'}})
            view.setBackgroundColor('white')
            view.zoomTo()
            
            # Add interactive controls
            view.spin(False)  # Disable auto-spinning for better control
            
            return view._make_html()
            
        except Exception as e:
            st.error(f"Error creating 3D visualization: {str(e)}")
            return None
    
    def create_attention_heatmap(
        self, 
        attention_matrix: np.ndarray, 
        sequence: str
    ) -> Optional[go.Figure]:
        """
        Create interactive attention heatmap using Plotly.
        
        Args:
            attention_matrix: Processed attention matrix
            sequence: Amino acid sequence for labels
            
        Returns:
            Plotly Figure object or None if error
        """
        try:
            # Create residue labels
            labels = [f"{aa}{i+1}" for i, aa in enumerate(sequence)]
            
            # Ensure matrix dimensions match sequence length
            expected_len = len(sequence)
            if attention_matrix.shape[0] != expected_len:
                st.warning(f"⚠️ Matrix size mismatch. Adjusting to fit sequence length.")
                min_len = min(attention_matrix.shape[0], expected_len)
                attention_matrix = attention_matrix[:min_len, :min_len]
                labels = labels[:min_len]
            
            # Create hover text with interaction strength categories
            hover_text = np.empty_like(attention_matrix, dtype=object)
            non_zero_values = attention_matrix[attention_matrix > 0]
            
            if len(non_zero_values) > 0:
                strong_threshold = np.percentile(non_zero_values, 75)
                medium_threshold = np.percentile(non_zero_values, 50)
                
                for i in range(attention_matrix.shape[0]):
                    for j in range(attention_matrix.shape[1]):
                        value = attention_matrix[i, j]
                        if value > strong_threshold:
                            strength = "Strong"
                        elif value > medium_threshold:
                            strength = "Medium"
                        elif value > 0:
                            strength = "Weak"
                        else:
                            strength = "None"
                        hover_text[i, j] = strength
            else:
                hover_text.fill("None")
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=attention_matrix,
                x=labels,
                y=labels,
                colorscale=ATTENTION_COLORSCALE,
                hoveronclick=True,
                hovertemplate=(
                    '<b>Source:</b> %{y}<br>'
                    '<b>Target:</b> %{x}<br>'
                    '<b>Attention:</b> %{z:.4f}<br>'
                    '<b>Strength:</b> %{text}<br>'
                    '<extra></extra>'
                ),
                text=hover_text,
                showscale=True,
                colorbar=dict(
                    title="Attention Weight",
                    titleside="right",
                    thickness=20
                )
            ))
            
            fig.update_layout(
                title={
                    'text': 'Attention Weight Heatmap - Residue Interactions',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title='Target Residue',
                yaxis_title='Source Residue',
                width=HEATMAP_WIDTH,
                height=HEATMAP_HEIGHT,
                font=dict(size=10),
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating attention heatmap: {str(e)}")
            return None
    
    def create_interaction_network(
        self,
        attention_matrix: np.ndarray,
        sequence: str,
        top_k: int = 20
    ) -> Optional[go.Figure]:
        """
        Create network graph of top residue interactions.
        
        Args:
            attention_matrix: Processed attention matrix
            sequence: Amino acid sequence
            top_k: Number of top interactions to display
            
        Returns:
            Plotly Figure object or None if error
        """
        try:
            # Get all non-zero interactions
            interactions = []
            for i in range(attention_matrix.shape[0]):
                for j in range(i + 1, attention_matrix.shape[1]):
                    weight = attention_matrix[i, j]
                    if weight > 0:
                        interactions.append((i, j, weight))
            
            if not interactions:
                st.warning("⚠️ No significant interactions found for network visualization")
                return None
            
            # Sort and take top-k
            interactions.sort(key=lambda x: x[2], reverse=True)
            effective_k = min(top_k, len(interactions))
            top_interactions = interactions[:effective_k]
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add all residues as nodes
            for i, aa in enumerate(sequence):
                G.add_node(i, residue=aa, label=f"{aa}{i+1}")
            
            # Add top interactions as edges
            edge_weights = []
            for i, j, weight in top_interactions:
                G.add_edge(i, j, weight=weight)
                edge_weights.append(weight)
            
            # Generate layout
            try:
                pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
            except:
                pos = nx.circular_layout(G)  # Fallback
            
            # Prepare node information
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = [f"{sequence[node]}{node+1}" for node in G.nodes()]
            
            # Color and size nodes based on connectivity
            node_colors = []
            node_sizes = []
            node_hover = []
            
            for node in G.nodes():
                degree = G.degree(node)
                if degree > 0:
                    node_colors.append(NODE_COLORS['connected'])
                    node_sizes.append(max(20, min(50, 20 + degree * 5)))
                else:
                    node_colors.append(NODE_COLORS['isolated'])
                    node_sizes.append(15)
                
                node_hover.append(f"Residue: {sequence[node]}{node+1}<br>Connections: {degree}")
            
            # Create edges
            edge_x = []
            edge_y = []
            edge_weights_normalized = []
            
            if edge_weights:
                max_weight = max(edge_weights)
                min_weight = min(edge_weights)
                weight_range = max_weight - min_weight if max_weight > min_weight else 1
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Normalize edge weight for line width
                weight = G[edge[0]][edge[1]]['weight']
                if edge_weights:
                    normalized_weight = 1 + 4 * ((weight - min_weight) / weight_range)
                else:
                    normalized_weight = 2
                edge_weights_normalized.extend([normalized_weight, normalized_weight, None])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(125,125,125,0.6)'),
                hoverinfo='none',
                mode='lines',
                name='Interactions',
                showlegend=False
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="middle center",
                hovertext=node_hover,
                hoverinfo='text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='black'),
                    opacity=0.8
                ),
                name='Residues',
                showlegend=False
            ))
            
            fig.update_layout(
                title={
                    'text': f'Top {effective_k} Residue Interactions Network',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=NETWORK_WIDTH,
                height=NETWORK_HEIGHT,
                plot_bgcolor='white',
                annotations=[
                    dict(
                        text=f"Node size indicates connectivity<br>Showing top {effective_k} interactions",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        xanchor="left", yanchor="top",
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                ]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating interaction network: {str(e)}")
            return None
    
    def create_attention_distribution_plot(self, attention_matrix: np.ndarray) -> Optional[go.Figure]:
        """
        Create histogram of attention weight distribution.
        
        Args:
            attention_matrix: Processed attention matrix
            
        Returns:
            Plotly Figure object or None if error
        """
        try:
            attention_values = attention_matrix[attention_matrix > 0]
            
            if len(attention_values) == 0:
                st.warning("⚠️ No positive attention values found")
                return None
            
            fig = go.Figure(data=[go.Histogram(
                x=attention_values,
                nbinsx=30,
                marker_color='skyblue',
                opacity=0.7,
                marker_line_color='black',
                marker_line_width=1
            )])
            
            fig.update_layout(
                title={
                    'text': 'Distribution of Attention Weights',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Attention Weight',
                yaxis_title='Frequency',
                showlegend=False,
                plot_bgcolor='white'
            )
            
            # Add statistics as annotations
            mean_val = np.mean(attention_values)
            median_val = np.median(attention_values)
            
            fig.add_annotation(
                text=f"Mean: {mean_val:.4f}<br>Median: {median_val:.4f}",
                x=0.98, y=0.98,
                xref="paper", yref="paper",
                xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating distribution plot: {str(e)}")
            return None


# Global instance
visualizer = StructureVisualizer()

def visualize_3d_structure(pdb_string: str) -> Optional[str]:
    """Convenience function for 3D structure visualization."""
    return visualizer.visualize_3d_structure(pdb_string)

def create_attention_heatmap(attention_matrix: np.ndarray, sequence: str) -> Optional[go.Figure]:
    """Convenience function for attention heatmap."""
    return visualizer.create_attention_heatmap(attention_matrix, sequence)

def create_interaction_network(attention_matrix: np.ndarray, sequence: str, top_k: int = 20) -> Optional[go.Figure]:
    """Convenience function for interaction network."""
    return visualizer.create_interaction_network(attention_matrix, sequence, top_k)

def create_attention_distribution_plot(attention_matrix: np.ndarray) -> Optional[go.Figure]:
    """Convenience function for attention distribution plot."""
    return visualizer.create_attention_distribution_plot(attention_matrix)

def is_py3dmol_available() -> bool:
    """Check if py3dmol is available."""
    return PY3DMOL_AVAILABLE