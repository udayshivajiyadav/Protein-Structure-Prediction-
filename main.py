"""
Main Streamlit Application
==========================

This is the main entry point for the Explainable Protein Structure Prediction app.
It orchestrates all the other modules and provides the user interface.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
from typing import Dict, Any

# Import our modules
from config import (
    UI_CONFIG, EXAMPLE_SEQUENCES, DEFAULT_THRESHOLD_PERCENTILE, 
    DEFAULT_TOP_K_INTERACTIONS, ERROR_MESSAGES, HELP_TEXT
)
from sequence_validator import validate_sequence, get_sequence_info
from model_handler import load_model, predict_structure, get_model_info
from attention_processor import process_attention, get_attention_statistics
from property_analyzer import (
    get_sequence_analysis, correlate_properties_with_attention, 
    get_composition_dataframe
)
from structure_visualizer import (
    visualize_3d_structure, create_attention_heatmap, 
    create_interaction_network, create_attention_distribution_plot,
    is_py3dmol_available
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'sequence_input' not in st.session_state:
        st.session_state.sequence_input = ""
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=UI_CONFIG['page_title'],
        page_icon=UI_CONFIG['page_icon'],
        layout=UI_CONFIG['layout'],
        initial_sidebar_state=UI_CONFIG['sidebar_state']
    )


def display_header():
    """Display the main header and description."""
    st.title("🧬 Explainable Protein Structure Prediction with ESMFold")
    st.markdown("""
    Predict protein 3D structures using Meta's ESMFold and visualize transformer attention weights 
    to understand which residue interactions the model considers important for structure prediction.
    """)


def display_sidebar() -> Dict[str, Any]:
    """
    Display sidebar with input controls and return user settings.
    
    Returns:
        Dictionary of user input settings
    """
    with st.sidebar:
        st.header("🔬 Input Parameters")
        
        # Sequence input section
        st.subheader("Protein Sequence")
        sequence_input = st.text_area(
            "Enter amino acid sequence:",
            value=st.session_state.sequence_input,
            height=150,
            help=HELP_TEXT['sequence_input'],
            placeholder="Example: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        )
        
        # Update session state
        st.session_state.sequence_input = sequence_input
        
        # Example sequences
        st.subheader("📚 Example Sequences")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔸 Ubiquitin (76 aa)", use_container_width=True):
                st.session_state.sequence_input = EXAMPLE_SEQUENCES['ubiquitin']
                st.rerun()
        
        with col2:
            if st.button("🔸 Small Peptide (20 aa)", use_container_width=True):
                st.session_state.sequence_input = EXAMPLE_SEQUENCES['small_peptide']
                st.rerun()
        
        # Additional examples
        if st.button("🔸 Insulin A Chain (21 aa)", use_container_width=True):
            st.session_state.sequence_input = EXAMPLE_SEQUENCES['insulin_a_chain']
            st.rerun()
        
        if st.button("🔸 Lysozyme Fragment (43 aa)", use_container_width=True):
            st.session_state.sequence_input = EXAMPLE_SEQUENCES['lysozyme_fragment']
            st.rerun()
        
        # Analysis options
        st.subheader("⚙️ Analysis Options")
        
        enable_explainability = st.toggle(
            "Enable Explainable AI",
            value=True,
            help=HELP_TEXT['explainability']
        )
        
        threshold_percentile = st.slider(
            "Attention Threshold (%)",
            min_value=70,
            max_value=99,
            value=DEFAULT_THRESHOLD_PERCENTILE,
            help=HELP_TEXT['threshold']
        )
        
        top_k_interactions = st.slider(
            "Top Interactions to Show",
            min_value=5,
            max_value=50,
            value=DEFAULT_TOP_K_INTERACTIONS,
            help=HELP_TEXT['top_k']
        )
        
        # Model information
        if st.session_state.model_loaded:
            st.subheader("🤖 Model Status")
            model_info = get_model_info()
            if model_info.get('loaded', False):
                st.success(f"✅ Model loaded on {model_info['device'].upper()}")
                
                with st.expander("Model Details"):
                    st.write(f"**Parameters**: {model_info.get('parameters', 'Unknown'):,}")
                    st.write(f"**Attention Layers**: {model_info.get('attention_layers', 'Unknown')}")
                    st.write(f"**Attention Heads**: {model_info.get('attention_heads', 'Unknown')}")
        
        # Prediction button
        st.markdown("---")
        predict_button = st.button(
            "🔬 Predict Structure",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.model_loaded
        )
    
    return {
        'sequence_input': sequence_input,
        'enable_explainability': enable_explainability,
        'threshold_percentile': threshold_percentile,
        'top_k_interactions': top_k_interactions,
        'predict_button': predict_button
    }


def handle_model_loading():
    """Handle model loading with proper error handling."""
    if not st.session_state.model_loaded:
        with st.status("Loading ESMFold model...", expanded=True) as status:
            st.write("🔄 Initializing model components...")
            
            success, device = load_model()
            
            if success:
                st.session_state.model_loaded = True
                st.write(f"✅ Model loaded successfully on {device.upper()}")
                status.update(label="Model ready!", state="complete")
            else:
                st.error("❌ Failed to load ESMFold model.")
                status.update(label="Model loading failed!", state="error")
                st.stop()


def handle_prediction(sequence_input: str, settings: Dict[str, Any]):
    """
    Handle structure prediction and store results.
    
    Args:
        sequence_input: Raw sequence input from user
        settings: User settings from sidebar
    """
    try:
        # Clear previous results
        st.session_state.prediction_results = None
        
        # Validate sequence
        with st.status("Validating sequence...", expanded=True) as status:
            st.write("🔍 Checking sequence format and constraints...")
            sequence = validate_sequence(sequence_input)
            seq_info = get_sequence_info(sequence)
            st.write(f"✅ Valid sequence with {seq_info['length']} residues")
            status.update(label="Sequence validated!", state="complete")
        
        # Predict structure
        with st.status("Predicting protein structure...", expanded=True) as status:
            st.write("🧠 Running ESMFold inference...")
            st.write("⏱️ This may take several minutes for long sequences...")
            
            start_time = time.time()
            pdb_string, attention_weights = predict_structure(sequence)
            prediction_time = time.time() - start_time
            
            st.write(f"✅ Structure predicted in {prediction_time:.1f} seconds")
            status.update(label="Structure prediction complete!", state="complete")
        
        # Store results in session state
        st.session_state.prediction_results = {
            'sequence': sequence,
            'sequence_info': seq_info,
            'pdb_string': pdb_string,
            'attention_weights': attention_weights,
            'prediction_time': prediction_time,
            'settings': settings
        }
        
    except ValueError as e:
        st.error(f"❌ **Input Error**: {str(e)}")
    except Exception as e:
        st.error(f"❌ **Prediction Error**: {str(e)}")
        st.error("💡 **Troubleshooting tips:**")
        st.write("• Try a shorter sequence (< 200 residues)")
        st.write("• Check available memory (GPU/CPU)")
        st.write("• Restart the application if issues persist")


def display_structure_results(results: Dict[str, Any]):
    """Display structure prediction results."""
    sequence = results['sequence']
    seq_info = results['sequence_info']
    pdb_string = results['pdb_string']
    prediction_time = results['prediction_time']
    
    st.success(f"🎉 Structure prediction completed in {prediction_time:.1f} seconds!")
    
    # Create main result columns
    col1, col2 = st.columns([1.4, 0.6])
    
    with col1:
        st.subheader("🏗️ 3D Protein Structure")
        
        # 3D visualization
        if is_py3dmol_available():
            try:
                with st.spinner("Rendering 3D structure..."):
                    structure_html = visualize_3d_structure(pdb_string)
                
                if structure_html:
                    components.html(structure_html, height=500)
                else:
                    st.info("3D visualization unavailable. You can download the PDB file below.")
            except Exception as e:
                st.error(f"Error rendering 3D structure: {str(e)}")
                st.info("3D visualization failed. PDB file is still available for download.")
        else:
            st.info("💡 Install py3dmol for 3D visualization: `pip install py3dmol`")
            with st.expander("📄 View PDB Structure (first 30 lines)"):
                st.code("\n".join(pdb_string.split('\n')[:30]), language="text")
        
        # Download section
        st.markdown("### 📁 Downloads")
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            st.download_button(
                label="📁 Download PDB File",
                data=pdb_string,
                file_name=f"predicted_structure_{seq_info['length']}aa.pdb",
                mime="text/plain",
                use_container_width=True
            )
        
        with download_col2:
            report = generate_prediction_report(results)
            st.download_button(
                label="📊 Download Report",
                data=report,
                file_name=f"prediction_report_{seq_info['length']}aa.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col2:
        st.subheader("📊 Sequence Analysis")
        
        # Basic metrics
        st.metric("Sequence Length", f"{seq_info['length']} residues")
        st.metric("Unique Residues", seq_info['unique_residues'])
        st.metric("Most Common", f"{seq_info['most_common']} ({seq_info['most_common_count']})")
        
        # Detailed sequence analysis
        try:
            analysis = get_sequence_analysis(sequence)
            
            if 'molecular_weight' in analysis:
                st.metric("Molecular Weight", f"{analysis['molecular_weight']:.1f} Da")
            if 'isoelectric_point' in analysis:
                st.metric("Isoelectric Point", f"{analysis['isoelectric_point']:.2f}")
            
            # Additional properties in expander
            if any(key in analysis for key in ['aromaticity', 'instability_index', 'gravy_score']):
                with st.expander("🔬 Additional Properties"):
                    for prop, value in analysis.items():
                        if prop in ['aromaticity', 'instability_index', 'gravy_score']:
                            st.write(f"**{prop.replace('_', ' ').title()}**: {value:.3f}")
                        elif prop.endswith('_fraction'):
                            st.write(f"**{prop.replace('_', ' ').title()}**: {value:.3f}")
            
        except Exception as e:
            st.info("Some sequence properties could not be computed")
        
        # Amino acid composition
        with st.expander("📈 Amino Acid Composition"):
            try:
                comp_df = get_composition_dataframe(sequence)
                if not comp_df.empty:
                    st.dataframe(comp_df[['Amino_Acid', 'Count', 'Percentage']], 
                               use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not generate composition table: {str(e)}")


def display_explainability_results(results: Dict[str, Any]):
    """Display explainable AI analysis results."""
    sequence = results['sequence']
    attention_weights = results['attention_weights']
    settings = results['settings']
    threshold_percentile = settings['threshold_percentile']
    top_k_interactions = settings['top_k_interactions']
    
    if not attention_weights:
        st.warning("⚠️ No attention weights captured from the model.")
        st.info("This may be due to model architecture differences or configuration issues.")
        return
    
    st.divider()
    st.header("🔍 Explainable AI Analysis")
    st.markdown("Understanding which residue interactions the transformer model considers important for structure prediction.")
    
    with st.status("Processing attention weights...", expanded=True) as status:
        st.write("🧮 Aggregating attention across layers and heads...")
        attention_matrix = process_attention(attention_weights, threshold_percentile)
        
        if attention_matrix is not None:
            st.write("📊 Calculating attention statistics...")
            attention_stats = get_attention_statistics(attention_matrix)
            st.write("✅ Attention processing complete")
            status.update(label="Attention analysis ready!", state="complete")
        else:
            st.write("❌ Attention processing failed")
            status.update(label="Attention processing failed!", state="error")
            return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Attention Heatmap", 
        "🕸️ Interaction Network", 
        "📊 Property Correlations",
        "📈 Summary Statistics"
    ])
    
    with tab1:
        st.subheader("Attention Weight Heatmap")
        st.markdown(f"""
        This heatmap shows which residue pairs the model considers important for structure prediction.
        Only the top {threshold_percentile}% of interactions are displayed for clarity.
        """)
        
        try:
            heatmap_fig = create_attention_heatmap(attention_matrix, sequence)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.error("Could not generate attention heatmap")
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
    
    with tab2:
        st.subheader("Top Residue Interactions Network")
        st.markdown(f"""
        Network graph showing the top {top_k_interactions} strongest residue interactions.
        Node size indicates connectivity (number of interactions).
        """)
        
        try:
            network_fig = create_interaction_network(attention_matrix, sequence, top_k_interactions)
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)
            else:
                st.warning("Could not generate interaction network")
        except Exception as e:
            st.error(f"Error creating network graph: {str(e)}")
    
    with tab3:
        st.subheader("Biochemical Property Correlations")
        st.markdown("""
        Analysis of how attention patterns correlate with residue biochemical properties.
        Significant correlations (p < 0.05) suggest the model uses these properties for predictions.
        """)
        
        try:
            correlation_df = correlate_properties_with_attention(attention_matrix, sequence)
            
            if not correlation_df.empty:
                # Display correlation table
                st.dataframe(correlation_df, use_container_width=True, hide_index=True)
                
                # Highlight significant correlations
                significant_corrs = correlation_df[correlation_df['P_Value'] < 0.05]
                if not significant_corrs.empty:
                    st.success(f"Found {len(significant_corrs)} statistically significant correlations!")
                    
                    # Show top correlations
                    top_corrs = significant_corrs.nlargest(3, 'Correlation')
                    if not top_corrs.empty:
                        st.markdown("**🔝 Strongest Correlations:**")
                        for _, row in top_corrs.iterrows():
                            correlation_type = "positive" if row['Correlation'] > 0 else "negative"
                            st.write(
                                f"• **{row['Property']}** ↔ **{row['Attention_Feature']}**: "
                                f"r = {row['Correlation']:.3f} ({correlation_type}, p = {row['P_Value']:.4f})"
                            )
                else:
                    st.info("No statistically significant correlations found (p ≥ 0.05)")
                
                # Download correlation results
                csv_data = correlation_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Correlation Results",
                    data=csv_data,
                    file_name=f"correlation_analysis_{len(sequence)}aa.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Could not perform correlation analysis")
                
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
    
    with tab4:
        st.subheader("Attention Summary Statistics")
        
        if attention_stats:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Interactions", attention_stats.get('total_interactions', 0))
            with col2:
                st.metric("Max Attention", f"{attention_stats.get('max_attention', 0):.4f}")
            with col3:
                st.metric("Mean Attention", f"{attention_stats.get('mean_attention', 0):.4f}")
            with col4:
                st.metric("Sparsity", f"{attention_stats.get('sparsity', 0):.2%}")
            
            # Attention distribution plot
            st.markdown("**📊 Attention Weight Distribution**")
            try:
                dist_fig = create_attention_distribution_plot(attention_matrix)
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating distribution plot: {str(e)}")
            
            # Top residues by attention
            if 'residue_stats' in attention_stats:
                st.markdown("**🏆 Most Attended Residues**")
                residue_attention = attention_stats['residue_stats']['attention_sum']
                top_residues = [(i, sequence[i], score) for i, score in enumerate(residue_attention)]
                top_residues.sort(key=lambda x: x[2], reverse=True)
                
                top_residues_df = pd.DataFrame(
                    top_residues[:10], 
                    columns=['Position', 'Residue', 'Total_Attention']
                )
                top_residues_df['Position'] = top_residues_df['Position'] + 1  # 1-indexed
                top_residues_df['Total_Attention'] = top_residues_df['Total_Attention'].round(4)
                
                st.dataframe(top_residues_df, use_container_width=True, hide_index=True)


def generate_prediction_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive prediction report."""
    sequence = results['sequence']
    seq_info = results['sequence_info']
    prediction_time = results['prediction_time']
    settings = results['settings']
    
    model_info = get_model_info()
    device = model_info.get('device', 'Unknown')
    
    report = f"""Protein Structure Prediction Report
{'='*50}

SEQUENCE INFORMATION
{'='*20}
Length: {seq_info['length']} residues
Unique Residues: {seq_info['unique_residues']}
Most Common: {seq_info['most_common']} ({seq_info['most_common_count']} occurrences)

PREDICTION DETAILS
{'='*18}
Model: ESMFold (facebook/esmfold_v1)
Device: {device.upper()}
Prediction Time: {prediction_time:.1f} seconds
Explainable AI: {'Enabled' if settings['enable_explainability'] else 'Disabled'}
Attention Threshold: {settings['threshold_percentile']}%
Top Interactions Shown: {settings['top_k_interactions']}

SEQUENCE
{'='*8}
{sequence}

ANALYSIS RESULTS
{'='*16}
The predicted 3D structure is provided in the accompanying PDB file.
{'Attention analysis and biochemical correlations are included.' if settings['enable_explainability'] else 'For attention analysis, enable Explainable AI option.'}

IMPORTANT NOTES
{'='*15}
• This prediction is AI-generated and should be validated experimentally
• ESMFold is a research tool and may not capture all aspects of protein folding
• For critical applications (drug design, etc.), experimental validation is essential
• Results are most reliable for sequences between 50-400 residues

Generated by: Explainable Protein Structure Prediction App
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report


def display_help_section():
    """Display help and information section."""
    with st.expander("ℹ️ About This Application", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔬 How It Works
            
            1. **Direct Model Loading**: Uses ESMFold model with direct tokenization
            2. **Structure Prediction**: Predicts 3D coordinates from sequence
            3. **Attention Extraction**: Captures transformer attention during inference
            4. **Explainability**: Visualizes attention patterns and correlations
            5. **Analysis**: Statistical analysis of biochemical property relationships
            
            ### ✨ Key Features
            
            - **Enhanced Model Loading**: Direct ESMFold integration (no pipeline issues)
            - **Interactive 3D Visualization**: py3dmol integration
            - **Attention Analysis**: Multi-layer attention aggregation
            - **Statistical Correlations**: Property-attention relationships
            - **Professional Reports**: Downloadable results and analysis
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Limitations & Best Practices
            
            - **Sequence Length**: Optimal for 10-400 residues
            - **Memory Requirements**: Long sequences need more GPU/CPU memory
            - **Prediction Quality**: Best for single-domain proteins
            - **Validation**: Always validate predictions experimentally
            - **Interpretation**: Attention ≠ physical interactions
            
            ### 🔧 Troubleshooting
            
            - **GPU Errors**: App automatically falls back to CPU
            - **Memory Issues**: Try shorter sequences or restart
            - **Import Errors**: Check virtual environment activation
            - **Slow Performance**: Use GPU when available
            
            ### 📚 Citation
            
            **ESMFold**: Lin et al. "Language models of protein sequences at the scale 
            of evolution enable accurate structure prediction" *Science* (2023)
            """)


def main():
    """Main application function."""
    # Setup
    setup_page()
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Handle model loading
    handle_model_loading()
    
    # Display sidebar and get user input
    settings = display_sidebar()
    
    # Handle prediction
    if settings['predict_button'] and settings['sequence_input'].strip():
        handle_prediction(settings['sequence_input'], settings)
    elif settings['predict_button'] and not settings['sequence_input'].strip():
        st.warning("⚠️ Please enter a protein sequence to predict its structure.")
    
    # Display results if available
    if st.session_state.prediction_results:
        results = st.session_state.prediction_results
        
        # Display structure results
        display_structure_results(results)
        
        # Display explainability results if enabled
        if results['settings']['enable_explainability']:
            display_explainability_results(results)
    
    # Display help section
    display_help_section()
    
    # Footer
    st.divider()
    st.markdown(
        "🧬 **Explainable Protein Structure Prediction** | "
        "Built with Streamlit, ESMFold, and ❤️ for science | "
        "**Modular Architecture v2.0**"
    )


if __name__ == "__main__":
    main()