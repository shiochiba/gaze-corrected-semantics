# Gaze-Corrected Visual-Semantic Encoding Models
This repository implements voxelwise encoding models for naturalistic movie-watching fMRI data. It compares standard visual-semantic feature spaces with gaze-corrected feature spaces derived from eye-tracking data.

## Project Goal
To evaluate whether incorporating eye-tracking information into visual-semantic feature spaces improves prediction of cortical activity in voxelwise encoding models.

## Key Idea
Standard voxelwise encoding models assume that all subjects attend to the same salient objects in a scene.
This project introduces a gaze-corrected feature space, where object features are weighted based on individual fixation patterns, providing a subject-specific representation of visual input.

## Repository Structure
- `notebooks/` — analysis pipelines and model comparisons  
- `src/` — utility functions for feature construction and encoding models  
- `data/` — object label mappings used for feature space construction  

## Methods Overview
1. Manual object annotations were extracted from movie frames  
2. Standard one-hot visual-semantic feature space was constructed  
3. Eye-tracking data was used to weight features corresponding to fixated objects  
4. Voxelwise encoding models were trained using kernel ridge regression  
5. Models were evaluated using cross-validated R²

## Results Summary
- Both models produce structured cortical prediction maps  
- Gaze-corrected features improve prediction in select cortical regions  
- Improvements are stronger in subjects with variable fixation behavior  
- Minimal differences are observed in subjects with consistent fixation patterns  

## Notes
Completed in the Gallant Lab at UC Berkeley as part of undergraduate research in computational neuroscience under PhD-level supervision.