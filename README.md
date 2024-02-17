# Learn from the patient, not the doctor: Predicting downstream outcomes versus specialist labels

| **[Installation](#installation)**
| **[Usage](#usage)**
| **[Citation](#citation)**

Many high-stakes machine learning applications, including in healthcare, hiring, and lending, occur in settings where a human specialist makes a decision about an individual: for example, a doctor decides which patients to test for a disease. For a subset of individuals, we then observe downstream outcomes: for example, whether patients test positive. Practitioners often train models to stand in for specialist judgment in these settings. Here, we compare two widely-used strategies for doing so. The first strategy seeks to predict specialist judgment (e.g., predict which patients the doctor will test); we term this <em>learning from the doctor</em>. The second strategy seeks to predict downstream outcomes on the subset of individuals who have them (e.g., predict test results among the tested patients); we term this strategy <em>learning from the patient</em>. Both these strategies have fundamental limitations: learning from the doctor will seek to imitate any flaws in specialist judgment, and learning from the patient trains only on the biased subset of observations with downstream outcomes. It remains unclear whether the two strategies yield similar results in real-world settings and when each approach is preferable. We first show on three real-world datasets that these two approaches yield very different results. We then show via theoretical argument and empirical comparison that training a model to learn from downstream outcomes is generally preferable to training a model to imitate specialist judgment. Finally, we characterize two situations that produce exceptions to this rule: when the predictor of downstream outcomes has very little variance, and when it cannot be well-estimated. Our results provide insight into a design choice that practitioners constantly confront but rarely analyze; propose an evaluative framework that is widely applicable, including to more complex estimators; and suggest that collecting downstream outcomes is essential, though it is often not done.

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

The following Pipeline is used: <br>
Dataset: Stop & Frisk <br>
PreProcess.ipynb - preprocess the dataset and extract features as well as labels (Y, Y|T=1, Y=1,T=1 etc.) <br>
Calibrate.ipynb - train and calibrate the estimators <br>
Estimators include P(T=1|X), P(Y=1|T=1,X), P(Y=1,T=1|X), IPW/importance weighting estimator, Hard Pseudo labels estimator <br>
Analysis_v2.ipynb - Evaluation of each of the estimators with various desiderata such as correlation among the estimators, agreement with the target regression, agreement with predicting long term outcomes or achieving equity <br>
SecondStageTrain.ipynb - Second stage training and calibration of all the estimators. Second Stage dataset creation is referred to as semi-synthetic dataset in section 5.1 of the paper <br>
SecondStageAnalysis.ipynb - Evaluation of each of the estimators with the semi synthetic dataset <br>

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@article{rastogi2023,
  author = {Rastogi,Richa and Meister,Michela and Obermeyer,UC and Kleinberg,Jon and Koh,Pang Wei and Pierson,Emma},
  title = {Learn from the patient, not the doctor:Predicting downstream outcomes versus specialist labels.},
  booktitle = {Working Paper},
  year = {2023},
  url={https://github.com/RichRast/Predicting_downstream_outcomes_vs_specialist_labels}
}

```

## Feedback
For any questions/feedback regarding this repo, please contact [here](rr568@cornell.edu)