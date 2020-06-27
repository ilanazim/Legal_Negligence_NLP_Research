## Data Product README

### Purpose

This README is meant to be a reference guide for using our code to reproduce our results. This README is assuming you have read our report and have a working understanding of the project.

### Files

[key_functions.py](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/key_functions.py) - Contains majority of codebase. Includes all functions needed for rule-based & classification based information extraction. Also includes code to train classifiers and evaluate. See graphic at bottom of file for visual of how functions are inter-linked.

[visualize.ipynb](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/visualize.ipynb) - Contains all code used to create visualizations of our results.

[DOCX to TXT format.ipynb](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/DOCX%20to%20TXT%20format.ipynb) - Contains code to convert LexisNexus cases from .DOCX to .TXT

[Project Code Samples.ipynb](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/Project%20Code%20Samples.ipynb) - Contains example of full code pipeline described below

[parsing_script.py](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/parsing_script.py) - Simple script that can open DOCX files, run classifier over the data, and output a spreadsheet. This is the simplest but least flexible approach to using our system.

[Ablation_Study.ipynb](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/Ablation_Study.ipynb) - Contains code used for ablation study to test which features are useful.

### Pre-Requisites

The programming language used for the entire project is Python 3. The following external packages need to be installed in order to run our code

**Data Manipulation Packages**
- Pandas*
- numpy*
- docx

**Machine Learning Packages**
- scikit-learn*
- XGBoost (Optional)

**Visualization Packages**

Note: All visualization packages are optional. These packages are not needed to run the core prediction system.
- matplotlib*
- Jupyter Notebook/Lab*
- altair
- seaborn

*packages that are already included if you install Python via the Anaconda distribution.

### Description

All code is located in the `/code/` folder. Note that moving code out of this directory and running it may not work because of the import statements across files only working in the same directory. The main file needed to reproduce our visualizations can be found in `visualize.ipynb`. To open the notebook, you must have Jupyter Notebook or Jupyter Lab installed. The notebook has each cell producing a different type of chart. The charts will change depending on the model that is used to gather the results
 
All code required to reproduce our results can be found in `key_functions.py`. The file is organized into several functions - some of which are meant to be helper functions to others. The main pipelines users can choose to use are described below and included in `Project Code Samples.ipynb`. If you wish to avoid any coding you may use `parsing_script.py` which will take a DOCX file or a directory containing DOCX files and output a spreadsheet containing the results - more details on how to run this can be found below.

### Simple Pipeline

We have included a `parsing_script.py` which handles the bulk of work required to take a DOCX case from LexisNexus and turn the predictions into a spreadsheet format. This approach is the simplest & easiest to follow. However, there is very limited ability to customize or change any parameters. There is no ability to train new models. If you wish to play with different models, parameters, or datasets refer to the full code pipeline below.

Before running `parsing_script.py`, all non-optional package requirements must be fulfilled. 

To run the script, you must open a terminal or command line window and navigate to the `/code/` folder. Basic information on how to navigate with the command line can be found [here](https://riptutorial.com/cmd/example/8646/navigating-in-cmd) for Windows and [here](https://www.macworld.com/article/2042378/master-the-command-line-navigating-files-and-folders.html) for Mac/Linux. Then type the command `python3 parsing_script.py` to run the script. You will be prompted for 3 paths, the paths can be relative or absolute.

1. The path to the saved models.

We have included our best saved models in the `/models/` directory. If you are running the script from the `/code/` folder you can supply a relative path as: `../models/` and then hit the return key. If you wish to use the rule based classifier then leave this blank and simply hit the return key.

2. The path to the DOCX data.

This can be a DOCX file or a DOCX directory. If it is a directory the script will try to analyze every file that ends in ".DOCX". We have included the data that was used for our analysis in `/data/Lexis Cases docx/`. If you are running the script from the `/code/` folder you can supply a relative path as `../data/Lexis Cases docx/` and then hit the return key.

3. The save path for the spreadsheet

You can choose where the .csv file will be saved. If you wish to save it to the `/data/` folder you can supply a relative path as `../data/my_results.csv`.

After entering the paths the script will read the DOCX files, run a classifier over them, and save the results into a .csv format which can be opened in any spreadsheet program.

### Full Code Pipeline

Note: This code can be found in the `Project Code Samples.ipynb` Jupyter Notebook as a starting point.

##### Warnings

While running the code there may be some warning statements printed out.

1. "Didnt find any tags in ..."

This warning occurs when training a classifier. Some cases may only contain in-text damage tags and some cases may only contain percentage tags. The warning exists to notify the user that it could not find any in-text annotations from these cases and therefore will not learn anything from the cases. This warning can be ignored unless you believe there should be tags.

2. "The least populated class in y has only 2 members"

This warning occurs when training a classifier. To evalaute our model during training we use K-fold cross validation which can be thought of as splitting the data into "K" train sets and test sets. The default value of K is set to 10. Some in-text annotation types are very rare and we are being warned that the specific tag type is not appearing in every "K-fold". This can be ignored as the tag type is very rare and of low importance to us.

3. Version Warnings

If you are reading in our saved models there may be a warning suggesting that the item was saved with a different version of the package. If you see results that are very different due to the model not being read in correctly you have two options. The first option is to downgrade your version of the package to match the version mentioned in the warning. The second option is to retrain the model rather than using the pre-saved models.

##### Pre-saved Models

We have included a saved copy of our best trained models in pickle format if you wish to skip the training part. The saved models are located under `/models/`. The relevant files are named, `damage_model.pkl`, `damage_vectorizer.pkl`, `damage_annotations.pkl`, `cn_model.pkl`, `cn_vectorizer.pkl`, `cn_annotations.pkl`. The following code snippet may be used to read in any of the pickle files:

```
import key_functions as kf
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

with open('../models/damage_model.pkl', 'rb') as file:
    dmg_model = pickle.load(file)
    
with open('../models/damage_vectorizer.pkl', 'rb') as file:
    dmg_vectorizer = pickle.load(file)
    
with open('../models/damage_annotations.pkl', 'rb') as file:
    dmg_annotations = pickle.load(file)
    
with open('../models/cn_model.pkl', 'rb') as file:
    cn_model = pickle.load(file)
    
with open('../models/cn_vectorizer.pkl', 'rb') as file:
    cn_vectorizer = pickle.load(file)
    
with open('../models/cn_annotations.pkl', 'rb') as file:
    cn_annotations = pickle.load(file)
```
##### Converting cases from .DOCX to .TXT

To convert the LexisNexus cases from .DOCX form to .TXT, use the `DOCX to TXT converter.ipynb` in the `/code/` folder. The only things the user needs to change is the path to where the documents are located. The DOCX data is located in `/data/Lexis Cases docx/`. We were unable to compress the file due to it being over 100MB as a single file. Follow instructions in the notebook for more information.

##### Train the classifiers on the same training data

The relevant function to train the damage classifier is `train_classifier(...)` where the function for contributory negligence is `train_CN_classifier(...)`. Both functions will take the path to the training data as the first argument and the classifier to use as the second. The functions will return the model, the vectorizer, and the gold annotations used to train it. The following parameters were used to train our best model. Warning: If you change any parameters (such as the classifier) it will produce different results. If the classifier does not implement `predict_proba`, some functionality will not work and may result in a crash.

```
# Damage Classifier
# (Note: Annotations are located in folder `/data/annotations/`)

dmg_model, dmg_vectorizer, annotations = kf.train_classifier('../data/annotations/final_annotations.txt', clf = LogisticRegression(C = 1, penalty = 'l2', solver = 'newton-cg', max_iter = 1000, random_state=42), context_length = 6, fit_model=True)

# CN Classifier

cn_model, cn_vectorizer, cn_annotations = kf.train_CN_classifier('../data/annotations/final_annotations.txt', clf = LogisticRegression(C = 1, penalty = 'l1', class_weight = 'balanced', solver = 'liblinear', max_iter = 10000, random_state = 42), context_length = 6)
```

##### Run Classifier on Unseen Data

Next, run the classifier model on the rest of the dataset. The dataset is organized into 85 different text files that are titled `P1.txt`, `P2.txt`, ... `P85.txt`. The data is included in the `/data/` folder but must be unzipped from `Lexis Cases txt.zip`. The following code snippet gives an example of how to provide the path to the function `parse_BCJ` along with the classifier models. Make sure to adjust the path to the data as necessary. The function has many parameters that you can use to filter different types of results. **Note: You can run the rule-based approach by not supplying any damage or CN model to `parse_BCJ`**

```
# Setup path variables
path_to_data = '../data/Lexis Cases txt/'
file_prefix = 'P'
file_suffix = '.txt'
file_identifiers = range(1, 86) # Range from 1 to 85

clf_results = []
for file_number in file_identifiers:
    print('## Processing ' + path_to_data + file_prefix + str(file_number) + file_suffix + ' ##', end='\r')
    with open(path_to_data + file_prefix + str(file_number) + file_suffix) as file:
       document_data = file.read()
    clf_results.extend(kf.parse_BCJ(document_data, damage_model = dmg_model, damage_vectorizer = dmg_vectorizer, annotated_damages = dmg_annotations, cn_model = cn_model, cn_vectorizer = cn_vectorizer, annotated_cn = cn_annotations, min_predict_proba = 0.5, dmg_context_length = 6, cn_context_length = 2))
```

##### Convert data into a dataframe or CSV & evaluation

If you are running the model over new data - you will be unable to evaluate unless you create a gold set. The gold set used to evaluate our model is included in the `data/` folder. Before evaluating, the data must be converted into a dataframe. The function `convert_cases_to_DF` takes in a list of cases and returns them in dataframe format. Dataframe format is easier to work with and can easily be saved as a CSV. See example below of converting the results achieved in the last code block into a dataframe and running an evaluation

```
df = kf.convert_cases_to_DF(clf_results)

gold_df = pd.read_csv('../data/gold_annotations.csv')
gold_df.dropna(how = 'all', inplace=True) 

kf.evaluate(df, gold_df)
```

The dataframe can be saved as a CSV (which can be manipulated with any spreadsheet program such as Excel)

`df.to_csv('../data/my_data.csv', index = False)`
    
    
##### Function Hierarchy

Below we have included the function hierarchy of functions appearing in `key_functions.py`. Many functions are meant to be helper functions and not used on their own. The large squares with the blue background are the main functions that a user should be calling. The green and red boxes are both helper functions; red typically being even smaller helper functions compared to the green. Follow the colour coded lines to determien which functions call which. 

![Function hierarchy drawing](./Imgs/functions.png)
