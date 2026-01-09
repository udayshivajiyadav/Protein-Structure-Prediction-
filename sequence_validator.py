"""
Sequence Validation Module
==========================

This module handles input validation, cleaning, and preprocessing
of protein sequences for structure prediction.
"""

import streamlit as st
from typing import List, Tuple
import re
from config import (
    MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH_WARNING,
    STANDARD_AA, VALID_AA, ERROR_MESSAGES
)


class SequenceValidator:
    """Handles protein sequence validation and preprocessing."""
    
    def __init__(self):
        self.standard_aa = STANDARD_AA
        self.valid_aa = VALID_AA
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean and preprocess input sequence.
        
        Args:
            sequence (str): Raw input sequence
            
        Returns:
            str: Cleaned sequence
        """
        if not sequence or not isinstance(sequence, str):
            return ""
        
        # Remove FASTA headers and clean whitespace
        lines = []
        for line in sequence.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('>'):  # Skip FASTA headers and empty lines
                lines.append(line)
        
        # Join all lines and convert to uppercase
        cleaned = ''.join(lines).upper()
        
        # Remove any remaining whitespace or special characters except valid AAs
        cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYXBZJU]', '', cleaned)
        
        return cleaned
    
    def validate_sequence(self, sequence: str) -> str:
        """
        Validate and clean protein sequence.
        
        Args:
            sequence (str): Input amino acid sequence
            
        Returns:
            str: Cleaned and validated sequence
            
        Raises:
            ValueError: If sequence is invalid
        """
        # Clean the sequence first
        sequence = self.clean_sequence(sequence)
        
        if not sequence:
            raise ValueError(ERROR_MESSAGES['empty_sequence'])
        
        # Check for invalid characters
        invalid_chars = set(sequence) - self.valid_aa
        if invalid_chars:
            raise ValueError(ERROR_MESSAGES['invalid_chars'].format(
                chars=', '.join(sorted(invalid_chars))
            ))
        
        # Warn about non-standard amino acids
        non_standard = set(sequence) - self.standard_aa
        if non_standard:
            st.warning(
                f"⚠️ Non-standard amino acids detected: {', '.join(sorted(non_standard))}. "
                "Results may be less reliable."
            )
        
        # Check length constraints
        if len(sequence) < MIN_SEQUENCE_LENGTH:
            raise ValueError(ERROR_MESSAGES['too_short'])
        
        if len(sequence) > MAX_SEQUENCE_LENGTH_WARNING:
            st.warning(
                f"⚠️ Long sequence detected ({len(sequence)} residues). "
                "Processing may be slow and memory-intensive."
            )
        
        if len(sequence) > 1000:  # Hard limit for stability
            raise ValueError(ERROR_MESSAGES['too_long'])
        
        return sequence
    
    def get_sequence_info(self, sequence: str) -> dict:
        """
        Get basic information about the sequence.
        
        Args:
            sequence (str): Validated amino acid sequence
            
        Returns:
            dict: Sequence information
        """
        aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
        most_common = max(aa_counts, key=aa_counts.get) if aa_counts else 'N/A'
        
        return {
            'length': len(sequence),
            'unique_residues': len(set(sequence)),
            'most_common': most_common,
            'most_common_count': aa_counts.get(most_common, 0),
            'composition': aa_counts
        }
    
    def format_sequence_for_display(self, sequence: str, line_length: int = 60) -> str:
        """
        Format sequence for display with line breaks.
        
        Args:
            sequence (str): Amino acid sequence
            line_length (int): Characters per line
            
        Returns:
            str: Formatted sequence
        """
        if not sequence:
            return ""
        
        lines = []
        for i in range(0, len(sequence), line_length):
            lines.append(sequence[i:i + line_length])
        
        return '\n'.join(lines)
    
    def detect_fasta_format(self, text: str) -> bool:
        """
        Detect if input text is in FASTA format.
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if FASTA format detected
        """
        lines = text.strip().split('\n')
        return any(line.startswith('>') for line in lines)
    
    def extract_fasta_sequences(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract sequences from FASTA format text.
        
        Args:
            text (str): FASTA format text
            
        Returns:
            List[Tuple[str, str]]: List of (header, sequence) tuples
        """
        sequences = []
        current_header = ""
        current_sequence = []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header and current_sequence:
                    sequences.append((current_header, ''.join(current_sequence)))
                
                # Start new sequence
                current_header = line[1:]  # Remove '>' character
                current_sequence = []
            elif line and not line.startswith('>'):
                current_sequence.append(line.upper())
        
        # Don't forget the last sequence
        if current_header and current_sequence:
            sequences.append((current_header, ''.join(current_sequence)))
        
        return sequences


# Convenience functions for direct use
validator = SequenceValidator()

def validate_sequence(sequence: str) -> str:
    """Convenience function for sequence validation."""
    return validator.validate_sequence(sequence)

def get_sequence_info(sequence: str) -> dict:
    """Convenience function for sequence information."""
    return validator.get_sequence_info(sequence)

def clean_sequence(sequence: str) -> str:
    """Convenience function for sequence cleaning."""
    return validator.clean_sequence(sequence)