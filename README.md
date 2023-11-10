# Learn from the patient, not the doctor: Predicting downstream outcomes versus specialist labels

| **[Installation](#installation)**
| **[Usage](#usage)**
| **[Citation](#citation)**

Many high-stakes machine learning applications, including in healthcare, hiring, and lending, occur in settings where a human specialist makes a decision about an individual: for example, a doctor decides which patients to test for the disease. For a subset of individuals, we then observe downstream outcomes: for example, whether patients test positive.
Practitioners often train machine learning models to stand in for specialist judgment in these settings. Here, we compare two widely-used strategies for doing so. The first strategy seeks to predict specialist judgment (e.g., predict which patients the doctor will test); we term this learning from the doctor. The second strategy seeks to predict downstream outcomes on the subset of individuals who have them (e.g., predict test results among the tested patients); we term this strategy learning from the patient. Both these strategies have fundamental limitations: learning from the doctor will seek to imitate any flaws in specialist judgment, and learning from the patient trains only on the biased subset of observations with downstream outcomes. It remains unclear whether the two strategies yield similar results in real-world settings and when each approach is preferable. We first show on three real-world datasets that these two approaches yield very different results: they do not correlate well with each other, and they also vary in how well they achieve other desiderata like predicting long-term outcomes or achieving equity. We then show via theoretical argument and empirical comparison that training a model to learn from downstream outcomes is generally preferable to training a model to imitate specialist judgment. Finally, we characterize two situations that produce exceptions to this rule: when the predictor of downstream outcomes has very little variance, and when it cannot be well-estimated. Our results provide insight into a design choice that practitioners constantly confront but rarely analyze; propose an evaluative framework that is widely applicable, including to more complex estimators; and suggest that collecting downstream outcomes is essential, though it is often not done.


<!-- ## Acknowledgements -->



## Installation

Set up and activate the Python environment by executing

```
conda env create -f environment.yml
conda activate predicting_outcomes
```

<!-- SLURM system can be used to run jobs. An example script for submitting SLURM job is given in ```./scripts/combined_sbatch.sub```.
In the scripts folder, customize the script ```init_env.sh``` for your environment and path. This path is then referenced in ```./scripts/combined_sbatch.sub``` . -->


## Usage

AnalysisFuncs.py contains the functions used for pre-processing, model training and analysis

The following Pipeline is used:
Dataset: Stop & Frisk 
PreProcess.ipynb - preprocess the dataset
Calibrate.ipynb - 
Estimators include 
Analysis_v2.ipynb - 
SecondStageTrain.ipynb -
SecondStageAnalysis.ipynb - 

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@article{rastogi2023,
  author = {Rastogi,Richa and Meister,Michela and Obermeyer,UC and Kleinberg,Jon and Koh,Pang Wei and Pierson,Emma},
  title = {Learn from the patient, not the doctor:Predicting downstream outcomes versus specialist labels. Working Paper},
  journal = {Working Paper},
  year = {2023}
}

```

## Feedback
