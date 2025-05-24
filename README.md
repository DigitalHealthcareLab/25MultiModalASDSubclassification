# Multimodal Clustering of ASD Using fMRI and Gaze Features

**Authors**: Chanyoung Ko  
**Date**: 2025-04-08  
**Python Version**: 3.10.16  
**PyTorch Version**: 1.12.1+cu116  
**PyG Version**: 2.4.0  

---

## Objective

This project aims to identify biologically meaningful subtypes of Autism Spectrum Disorder (ASD) through a novel multimodal approach:

1. **Use joint attention metrics as a bridge** between neurobiological (fMRI) and behavioral (gaze) data modalities
2. **Extract meaningful features from both modalities** using Regularized Canonical Correlation Analysis (RCCA)
3. **Identify robust subtypes** through hierarchical clustering on integrated features
4. **Characterize subtypes** by clinical, behavioral, and potentially genetic measures

---

## Methodology

Our approach involves a multi-step process:

1. **RCCA Analysis**:
   - First RCCA: Gaze features → Joint attention metrics (IJA, RJA-low, RJA-high)
   - Second RCCA: rsfMRI connectivity metrics → Joint attention metrics

2. **Feature Integration**:
   - Combine canonical variates from both modalities that relate to the same joint attention types
   - Normalize and standardize the integrated feature space

3. **Subtype Identification**:
   - Apply hierarchical clustering with cosine similarity
   - Determine optimal number of clusters using multiple validity metrics
   - Enhance robustness through resampling and consensus clustering

4. **Subtype Characterization**:
   - Analyze clinical and behavioral profiles across identified subtypes
   - Map back to original feature space for neurobiological interpretation

---

## Project Structure

- **code/**: Source code for data processing, RCCA analysis, and clustering
- **data/**: 
  - Processed fMRI connectivity features
  - Processed gaze features from head pose estimation
  - Joint attention metrics
- **results/**: 
  - RCCA results for both modalities
  - Clustering results and subtype characterization
  - Visualization of identified subtypes

---

## Data Sources

### 1. fMRI Data
- **SNU Dataset**: Resting-state functional connectivity data
- **Atlas**: Power atlas for ROI definition
- **Connectivity Metrics**: Full correlation matrices

### 2. Gaze Data
- **SNU Dataset**: Head pose estimates from video recordings
- **Features**: Various metrics extracted from head pose time series

### 3. Behavioral Data
- **Joint Attention Metrics**:
  - Initiating Joint Attention (IJA)
  - Responding to Joint Attention - Low complexity (RJA-low)
  - Responding to Joint Attention - High complexity (RJA-high)

---

## Novelty and Advantages

1. Using joint attention as a "bridge" between modalities ensures that extracted features are functionally relevant to social behavior
2. The resulting multimodal clusters represent biologically meaningful subtypes that incorporate both brain function and behavioral manifestations
3. This approach may identify novel phenotypes with distinct genetic underpinnings
4. The two-step process (using joint attention as a bridge, then integrating features) represents a methodological innovation in ASD subtyping
