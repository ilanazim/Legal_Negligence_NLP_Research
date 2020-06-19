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

### Code Description

All code required to reproduce our visualizations can be found in `visualize.ipynb`. To open the notebook, you must have Jupyter Notebook or Jupyter Lab installed. The notebook has each cell producing a different type of chart. The charts will change depending on the model that is used to gather the results
 
All code required to reproduce our results can be found in `key_functions.py`. Functions in this file are required by `visualize.ipynb`. The file is organized into several functions - some of which are meant to be helper functions to others. The main pipeline most users will use is described below.

##### Train the classifiers on the same training data

The relevant function to train the damage classifier is `train_classifier(...)` where the function for contributory negligence is `train_CN_classifier(...)`. Both functions will take the path to the training data as the first argument and the classifier to use as the second. The functions will return the model, the vectorizer, and the gold annotations used to train it. The following parameters were used to train our best model. Warning: If you change any parameters (such as the classifier) it is possible the code may crash at some point or produce different results.

`import key_functions as kf`

Damage Classifier

`dmg_model, dmg_vectorizer, annotations = kf.train_classifier('../data/annotations/all_annotations_CN_headers.txt', clf = LogisticRegression(C = 1, penalty='l2', solver = 'newton-cg', max_iter = 1000, random_state=42), context_length = 6, fit_model=True)`

CN Classifier

` TODO IM UNSURE ABOUT WHAT THE BEST PARAMS ARE RIGHT NOW `

##### Run Classifier on Unseen Data

Next, run the classifier model on the rest of the dataset. The dataset is organized into 85 different text files that are titled `P1.txt`, `P2.txt`, ... `P85.txt`. The following code snippet gives an example of how to provide the path to the function `rule_based_parse_BCJ` along with the classifier models. The function has many parameters that you can use to filter different types of results

```
# Setup path variables
path_to_data = '../../data/Lexis Cases txt header/'
file_prefix = 'P'
file_suffix = '.txt'
file_identifiers = range(1, 86) # Range from 1 to 85

clf_results = []
for file_number in file_identifiers:
    print('## Processing ' + path_to_data + file_prefix + str(file_number) + file_suffix + ' ##', end='\r')
    clf_results.extend(kf.rule_based_parse_BCJ(path_to_data + file_prefix + str(file_number) + file_suffix, damage_model = dmg_model, damage_vectorizer = dmg_vectorizer, annotated_damages = annotations, cn_model = cn_model, cn_vectorizer = cn_vec, annotated_cn = cn_annotations, min_predict_proba = 0.5))
```
    
    
### Function Hierarchy

Since many functions are helper functions that were not meant to be called on their own we have included a comprehensive list of which functions rely on which. 

* rule_based_parse_BCJ
    * filter_unwanted_cases
    * rule_based_multiple_defendants_parse
    * plaintiff_wins
    * predict
        * extract_features
            * clean_money_amount
        * extract_CN_features
            * get_wordnet_pos
    * assign_classification_damages
    * rule_based_damage_extraction
        * get_matching_text
        * assign_damage_to_category
            * match_contains_words
            * is_best_score
        * clean_money_amount
    * get_percent_reduction_and_contributory_negligence_success
        * get_context_and_float
        * conditions_for_extracted_value
        * contributory_negligence_successful_fun
        * summary_tokenize
    * assign_classification_CN
    * plaintiff_wins
   
* train_classifier
    * filter_unwanted_cases
    * summary_tokenize
    * extract_features
        * clean_money_amount
    *
    
* train_CN_classifier
    * filter_unwanted_cases
    * summary_tokenize
    * extract_CN_features
        * get_wordnet_pos
        
* rule_based_convert_cases_to_DF
* evaluate
        
