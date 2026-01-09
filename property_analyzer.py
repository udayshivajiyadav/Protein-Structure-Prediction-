"""
Biochemical Property Analyzer
=============================

This module analyzes biochemical properties of protein sequences
and correlates them with attention patterns from the transformer model.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy.stats import pearsonr
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from config import (
    HYDROPHOBICITY_SCALE, CHARGE_SCALE, HELIX_PROPENSITY_SCALE,
    MIN_CORRELATION_POINTS
)


class PropertyAnalyzer:
    """Analyzes biochemical properties and their correlations with attention."""
    
    def __init__(self):
        self.hydrophobicity_scale = HYDROPHOBICITY_SCALE
        self.charge_scale = CHARGE_SCALE
        self.helix_propensity_scale = HELIX_PROPENSITY_SCALE
    
    def get_residue_properties(self, sequence: str) -> Dict[str, List[float]]:
        """
        Extract biochemical properties for each residue.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of property arrays
        """
        try:
            properties = {
                'position': list(range(len(sequence))),
                'residue': list(sequence),
                'hydrophobicity': [self.hydrophobicity_scale.get(aa, 0.0) for aa in sequence],
                'charge': [self.charge_scale.get(aa, 0.0) for aa in sequence],
                'helix_propensity': [self.helix_propensity_scale.get(aa, 1.0) for aa in sequence]
            }
            
            return properties
            
        except Exception as e:
            st.error(f"Error calculating residue properties: {str(e)}")
            return {}
    
    def get_sequence_analysis(self, sequence: str) -> Dict[str, Any]:
        """
        Get comprehensive sequence analysis using BioPython.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of sequence analysis results
        """
        analysis_results = {}
        
        try:
            # Basic composition
            aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
            most_common = max(aa_counts, key=aa_counts.get) if aa_counts else 'N/A'
            
            analysis_results.update({
                'length': len(sequence),
                'unique_residues': len(set(sequence)),
                'most_common_residue': most_common,
                'most_common_count': aa_counts.get(most_common, 0),
                'composition': aa_counts
            })
            
            # BioPython analysis
            try:
                bio_analysis = ProteinAnalysis(sequence)
                
                analysis_results.update({
                    'molecular_weight': bio_analysis.molecular_weight(),
                    'isoelectric_point': bio_analysis.isoelectric_point(),
                    'aromaticity': bio_analysis.aromaticity(),
                    'instability_index': bio_analysis.instability_index(),
                    'gravy_score': bio_analysis.gravy()
                })
                
                # Secondary structure fractions
                sec_struct = bio_analysis.secondary_structure_fraction()
                analysis_results.update({
                    'helix_fraction': sec_struct[0],
                    'turn_fraction': sec_struct[1],
                    'sheet_fraction': sec_struct[2]
                })
                
            except Exception as bio_e:
                st.warning(f"⚠️ Could not complete BioPython analysis: {str(bio_e)}")
                analysis_results['bio_analysis_error'] = str(bio_e)
            
        except Exception as e:
            st.error(f"Error in sequence analysis: {str(e)}")
            analysis_results['analysis_error'] = str(e)
        
        return analysis_results
    
    def correlate_properties_with_attention(
        self, 
        attention_matrix: np.ndarray, 
        sequence: str
    ) -> pd.DataFrame:
        """
        Correlate residue properties with attention patterns.
        
        Args:
            attention_matrix: Processed attention matrix
            sequence: Amino acid sequence
            
        Returns:
            DataFrame with correlation results
        """
        try:
            # Get residue properties
            properties = self.get_residue_properties(sequence)
            
            # Calculate attention-based features
            attention_features = {
                'attention_sum': attention_matrix.sum(axis=1),
                'attention_max': attention_matrix.max(axis=1),
                'attention_mean': attention_matrix.mean(axis=1),
                'num_connections': (attention_matrix > 0).sum(axis=1),
                'attention_var': attention_matrix.var(axis=1)
            }
            
            correlations = []
            
            # Calculate correlations between properties and attention features
            for prop_name in ['hydrophobicity', 'charge', 'helix_propensity']:
                if prop_name not in properties:
                    continue
                
                prop_values = properties[prop_name]
                
                for feat_name, feat_values in attention_features.items():
                    try:
                        # Ensure same length
                        min_len = min(len(prop_values), len(feat_values))
                        
                        if min_len < MIN_CORRELATION_POINTS:
                            continue
                        
                        prop_vals = np.array(prop_values[:min_len])
                        feat_vals = np.array(feat_values[:min_len])
                        
                        # Skip if no variance
                        if np.var(prop_vals) == 0 or np.var(feat_vals) == 0:
                            continue
                        
                        # Calculate Pearson correlation
                        corr_coeff, p_value = pearsonr(prop_vals, feat_vals)
                        
                        # Handle NaN results
                        if np.isnan(corr_coeff) or np.isnan(p_value):
                            continue
                        
                        # Determine significance and effect size
                        significance = 'Significant (p < 0.05)' if p_value < 0.05 else 'Not Significant (p ≥ 0.05)'
                        
                        if abs(corr_coeff) > 0.5:
                            effect_size = 'Large'
                        elif abs(corr_coeff) > 0.3:
                            effect_size = 'Medium'
                        else:
                            effect_size = 'Small'
                        
                        correlations.append({
                            'Property': prop_name.replace('_', ' ').title(),
                            'Attention_Feature': feat_name.replace('_', ' ').title(),
                            'Correlation': round(corr_coeff, 4),
                            'P_Value': round(p_value, 6),
                            'Significance': significance,
                            'Effect_Size': effect_size
                        })
                        
                    except Exception as corr_e:
                        st.warning(f"Could not calculate correlation for {prop_name} vs {feat_name}: {str(corr_e)}")
                        continue
            
            if not correlations:
                st.warning("⚠️ Could not calculate meaningful correlations")
                return pd.DataFrame()
            
            return pd.DataFrame(correlations)
            
        except Exception as e:
            st.error(f"Error in property correlation analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_property_summary(self, sequence: str) -> Dict[str, Any]:
        """
        Get summary statistics for sequence properties.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of property summaries
        """
        try:
            properties = self.get_residue_properties(sequence)
            summary = {}
            
            for prop_name in ['hydrophobicity', 'charge', 'helix_propensity']:
                if prop_name in properties:
                    values = properties[prop_name]
                    summary[prop_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
            
            return summary
            
        except Exception as e:
            st.error(f"Error calculating property summary: {str(e)}")
            return {}
    
    def get_composition_dataframe(self, sequence: str) -> pd.DataFrame:
        """
        Get amino acid composition as a DataFrame.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            DataFrame with amino acid composition
        """
        try:
            aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
            aa_df = pd.DataFrame(list(aa_counts.items()), columns=['Amino_Acid', 'Count'])
            aa_df['Percentage'] = (aa_df['Count'] / len(sequence) * 100).round(1)
            aa_df = aa_df.sort_values('Count', ascending=False)
            
            # Add property information
            aa_df['Hydrophobicity'] = aa_df['Amino_Acid'].map(self.hydrophobicity_scale)
            aa_df['Charge'] = aa_df['Amino_Acid'].map(self.charge_scale)
            aa_df['Helix_Propensity'] = aa_df['Amino_Acid'].map(self.helix_propensity_scale)
            
            return aa_df
            
        except Exception as e:
            st.error(f"Error creating composition DataFrame: {str(e)}")
            return pd.DataFrame()


# Global instance
analyzer = PropertyAnalyzer()

def get_residue_properties(sequence: str) -> Dict[str, List[float]]:
    """Convenience function for getting residue properties."""
    return analyzer.get_residue_properties(sequence)

def get_sequence_analysis(sequence: str) -> Dict[str, Any]:
    """Convenience function for sequence analysis."""
    return analyzer.get_sequence_analysis(sequence)

def correlate_properties_with_attention(attention_matrix: np.ndarray, sequence: str) -> pd.DataFrame:
    """Convenience function for property-attention correlation."""
    return analyzer.correlate_properties_with_attention(attention_matrix, sequence)

def get_property_summary(sequence: str) -> Dict[str, Any]:
    """Convenience function for property summary."""
    return analyzer.get_property_summary(sequence)

def get_composition_dataframe(sequence: str) -> pd.DataFrame:
    """Convenience function for composition DataFrame."""
    return analyzer.get_composition_dataframe(sequence)