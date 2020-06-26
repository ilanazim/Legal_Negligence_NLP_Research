print('Loading...')
import os
import pickle
import pandas as pd
import key_functions as kf
import docx_functions

print('\n####')
print('This script will analyze case reports and outputs a spreadsheet with information about the case')
print('Only supports cases in .DOCX format pulled from LexisNexus')
print('Erroneous behaviour will occur if using different case formats')
print('To exit script close the application or press CTRL+C')
print('####\n')

# Get input about directories from user
# Saved Models Directory
model_path = input('Enter path to saved statistical models (Leave blank to use rule based classification): ')
if model_path != '' and not os.path.isdir(model_path):
    print('ERROR: Invalid Directory')
    exit(0)
if model_path != '' and model_path[-1] != '/':
    model_path += '/'

# DOCX Directory
in_path = input('Enter path to DOCX case file: ')
is_dir = False
if os.path.isdir(in_path):
    print(in_path, 'is a directory. Will read all .DOCX files in', in_path)
    is_dir = True
    if in_path[-1] != '/': # Add slash in case does not exist at the end
        in_path += '/'
elif os.path.isfile(in_path):
    print('Set to analyze', in_path)
else:
    print('ERROR: Invalid directory')
    exit(0)

# Output directory
out_path = input('Enter output path for spreadsheet: ')


# Read in models
dmg_model = None
dmg_vectorizer = None
cn_model = None
cn_vectorizer = None

try:
    if model_path != '':
        print('Loading saved models . . .')
        with open(model_path + 'damage_model.pkl', 'rb') as file:
            dmg_model = pickle.load(file)
        with open(model_path + 'damage_vectorizer.pkl', 'rb') as file:
            dmg_vectorizer = pickle.load(file)
        with open(model_path + 'cn_model.pkl', 'rb') as file:
            cn_model = pickle.load(file)
        with open(model_path + 'cn_vectorizer.pkl', 'rb') as file:
            cn_vectorizer = pickle.load(file)
except:
    print('Warning: There was a problem loading statistical models')
    print('Be sure you supplied the correct path and have not changed')
    print('the file names.')
    print('Using rule based classifier instead')

    dmg_model = None
    dmg_vectorizer = None
    cn_model = None
    cn_vectorizer = None


# Add all file paths to a single list
if is_dir:
    files = []
    for file in os.listdir(in_path):
        if os.path.isfile(in_path + file) and file[-5:].lower() == '.docx':
            files.append(in_path + file)
else:
    files = [in_path]

# Iterate over every .DOCX file and analyze
print('Analyzing documents . . .')
case_data = []
for file_path in sorted(files):
    print('> Analyzing', file_path)
    file_data = docx_functions.convert_docx_to_txt(file_path)
    case_data.extend(kf.parse_BCJ(file_data, damage_model = dmg_model, damage_vectorizer = dmg_vectorizer,
                 cn_model = cn_model, cn_vectorizer = cn_vectorizer, dmg_context_length = 2,
                 cn_context_length = 5))

# Convert to dataframe and save as a CSV file
print('Saving data . . .')
case_data_df = kf.convert_cases_to_DF(case_data)
case_data_df.to_csv(out_path, index=False)

print('DONE!')
