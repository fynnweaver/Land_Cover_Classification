# Land_Cover_Classification
AISC workshop learning project tackling ML classification using remote sensing (Sentinal) data and the NRCAN 2015 Canada land cover classification dataset. This project was done in collaboration with [Leah Lourenco](https://github.com/MudSnail).

Main coding workflow occurs in `ModelWorkflow.ipynb` and contains EDA, feature selection and modeling steps. Custom functions used for this project can all be found in the `custom_functions.py` file. The outcomes of different model iterations can be found in the folder `evaluation`. The legend for the different model versions and pre-processing steps can be found in `evaluation_legend.md`. Individual experiments with image processing can be found in `ImageExperiments.ipynb`. `Ensemble.ipynb` contains work attempting to combine different shallow ML models for a more robust predictor.

We achieved a final result of 43.66% balanced accuracy (58.05% accuracy) on a completely unknown test extent. See visualization of the prediction versus the true land cover classifications below.

![Final_Prediction](https://github.com/fynnweaver/Land_Cover_Classification/blob/main/evaluation/demo/Combo/14_13_comparison.png)
