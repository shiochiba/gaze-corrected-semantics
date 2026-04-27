# Gaze-Corrected Visual-Semantic Encoding Models
This repository implements voxelwise encoding models for naturalistic movie-watching fMRI data. It compares standard visual-semantic feature spaces with gaze-corrected feature spaces derived from eye-tracking data.

## Project Goal
To evaluate whether incorporating eye-tracking information into visual-semantic feature spaces improves prediction of cortical activity in voxelwise encoding models.

## Key Idea
Standard voxelwise encoding models assume uniform attention to salient objects across subjects.
This project introduces a gaze-corrected feature space in which object features are weighted by individual fixation patterns.

## Repository Structure
- `notebooks/` — analysis pipelines and model comparisons  
- `src/` — utility functions for feature construction and encoding models  
- `data/` — object label mappings used for feature space construction  

## Methods Overview
1. All objects were manually labeled in every 3rd frame of the movie stimuli   
2. Standard one-hot visual-semantic feature space was constructed  
3. Eye-tracking data was used to weight features corresponding to fixated objects  
4. Voxelwise encoding models were trained using kernel ridge regression  
5. Models were evaluated using cross-validated R²

## Results Summary  
- Gaze-corrected features improve prediction in select cortical regions, strongest in the subject with variable fixation  
- Minimal differences are observed in the subject with consistent fixation patterns    

## Notes
Completed in the Gallant Lab at UC Berkeley as part of undergraduate research in computational neuroscience under PhD-level supervision.
