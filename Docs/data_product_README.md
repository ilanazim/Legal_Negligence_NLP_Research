## Data Product README

### Purpose

This README is meant to be a reference guide for using our code to reproduce our results. This README is assuming you have read our report and have a working understanding of the project.

### Pre-Requisites

The programming language used for the entire project is Python 3. The following external packages need to be installed in order to run all code

**Data Manipulation Packages**
- Pandas*
- numpy*

**Machine Learning Packages**
- scikit-learn*
- XGBoost (Optional)

**Visualization Packages**
- matplotlib*
- Jupyter Notebook/Lab* (Optional)
- altair
- seaborn

*packages that are included if you install Python via the Anaconda distribution.

### Description

All code required to reproduce our visualizations can be found in `visualize.ipynb`. To open the notebook, you must have Jupyter Notebook or Jupyter Lab installed. The notebook has each cell producing a different type of chart. The charts will change depending on the model that is used to gather the results
 
All code required to reproduce our results can be found in `key_functions.py`. Functions in this file are required by `visualize.ipynb`. The file is organized into several functions - some of which are meant to be helper functions to others. The main pipeline most users will use is described below.

We have also included a saved copy of our best trained models in pickle format if you wish to skip the training part. The relevant files are named, `damage_model.pkl`, `damage_vectorizer.pkl`, `damage_annotations.pkl`, `cn_model.pkl`, `cn_vectorizer.pkl`, `cn_annotations.pkl`. The following code snippet may be used to read in the pickle files:

```
with open('damage_model.pkl', 'rb') as file:
    dmg_model = pickle.load(file)
```

##### Train the classifiers on the same training data

The relevant function to train the damage classifier is `train_classifier(...)` where the function for contributory negligence is `train_CN_classifier(...)`. Both functions will take the path to the training data as the first argument and the classifier to use as the second. The functions will return the model, the vectorizer, and the gold annotations used to train it. The following parameters were used to train our best model. Warning: If you change any parameters (such as the classifier) it is possible the code may crash at some point or produce different results.

`import key_functions as kf`

Damage Classifier

(Note: Adjust path to annotations as necessary)

`dmg_model, dmg_vectorizer, annotations = kf.train_classifier('annotations.txt', clf = LogisticRegression(C = 1, penalty = 'l2', solver = 'newton-cg', max_iter = 1000, random_state=42), context_length = 6, fit_model=True)`

CN Classifier

`cn_model, cn_vectorizer, cn_annotations = kf.train_CN_classifier('annotations.txt', clf = LogisticRegression(C = 1, penalty = 'l1', class_weight = 'balanced', solver = 'liblinear', max_iter = 10000, random_state = 42), context_length = 6) `

##### Run Classifier on Unseen Data

Next, run the classifier model on the rest of the dataset. The dataset is organized into 85 different text files that are titled `P1.txt`, `P2.txt`, ... `P85.txt`. The following code snippet gives an example of how to provide the path to the function `parse_BCJ` along with the classifier models. Make sure to adjust the path to the data as necessary. The function has many parameters that you can use to filter different types of results

```
# Setup path variables
path_to_data = '../data/Lexis Cases txt header/'
file_prefix = 'P'
file_suffix = '.txt'
file_identifiers = range(1, 86) # Range from 1 to 85

clf_results = []
for file_number in file_identifiers:
    print('## Processing ' + path_to_data + file_prefix + str(file_number) + file_suffix + ' ##', end='\r')
    clf_results.extend(kf.parse_BCJ(path_to_data + file_prefix + str(file_number) + file_suffix, damage_model = dmg_model, damage_vectorizer = dmg_vectorizer, annotated_damages = annotations, cn_model = cn_model, cn_vectorizer = cn_vectorizer, annotated_cn = cn_annotations, min_predict_proba = 0.5, dmg_context_length = 6, cn_context_length = 2))
```
    
    
### Function Hierarchy

Below we have included the function hierarchy. Many functions are meant to be helper functions and not used on their own. The large squares with the blue background are the main functions that a user should be calling. The green and red boxes are both helper functions; red typically being even smaller helper functions compared to the green. Follow the colour coded lines to determien which functions call which. 

![Function hierarchy drawing](./Imgs/functions.png)
