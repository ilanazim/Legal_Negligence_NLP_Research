# British Columbia Judicial Decisions Analysis

![alt text](./Docs/Imgs/law.jpg)

Negligence law is “an area of tort law that deals with the breach of duty to take care and it involves harm caused by carelessness, not intentional harm.”1 As of today there is currently no reliable data regarding the amount of compensation a court awards to injured people in British Columbia. When a negligence case wins, the court awards what are called ‘damages’, a financial amount the injured person is paid from the person who injured them. The damages depend on the category of the harm to the defendant such as punitive damages or aggravated damages.
In this project we are trying to analyze B.C. court negligence cases from the past 20 years and extract some information such as whether the damage awards have gone up during these past 20 years. There are some challenges with these information extraction processes such as the fact that they were not written in a cohesive format, the damages mentioned in the cases may or may not get awarded to the defendant, or that the data is not annotated. We are aiming to use a combination of rule based methods such as pattern matching and classification algorithms such as tree based models to extract the desired information. 

------------
## Directory:

### [Code Folder ](https://github.ubc.ca/nikihm/Capstone-project/tree/master/code)

#### [Experimental-notebooks: ](https://github.ubc.ca/nikihm/Capstone-project/tree/master/code/experimental-notebooks) Series of .ipynb notebooks for each persons experiment with the project

#### [DOCX to TXT format.ipynb: ](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/DOCX%20to%20TXT%20format.ipynb): Notebook to convert DOCX formated files to .TXT formated files

#### [Project Code Samples.ipynb: ](https://github.ubc.ca/nikihm/Capstone-project/blob/master/code/Project%20Code%20Samples.ipynb) Notebook with examples on how to run entire prediction system 

An example of a case information extraction:

{**'case_title'**: 'Madill v. Sithivong, [2010] B.C.J. No. 2603',
 **'year'**: '2010',
 **'registry'**: 'Chilliwack',
 **'judge'**: 'N.E. Morrison J.',
 **'decision_length'**: '213',
 **'multiple_defendants'**: 'Y',
 **'contributory_negligence_raised'**: True,
 **'written_decision'**: True,
 **'Plaintiff_wins'**: True}

### [Data Folder ](https://github.ubc.ca/nikihm/Capstone-project/tree/master/data)

Includes original case data as well as zip folder of the case data converted into a .TXT format. Contains a folder for the annotations that were used in the project.

#### [final_annotations.txt:](https://github.ubc.ca/nikihm/Capstone-project/blob/master/data/annotations/final_annotations.txt) In text annotation using XML tags

An example of in text annotation:

Plaintiff's was awarded woth non-pecuniary damages of \<damage type = 'non pecuniary'\>\$75,000\</damage\> 

#### [gold_annotations.csv:](https://github.ubc.ca/nikihm/Capstone-project/blob/master/data/annotations/gold_annotations.csv) High-level annotation of all gold cases

### [Docs: ](https://github.ubc.ca/nikihm/Capstone-project/tree/master/Docs)

#### [Project-plan.pdf: ](https://github.ubc.ca/nikihm/Capstone-project/blob/master/Docs/Project-Plan.pdf) British Columbia Judicial Decisions Analysis Project plan in pdf format

#### [Project-plan.md: ](https://github.ubc.ca/nikihm/Capstone-project/blob/master/Docs/project-plan.md) British Columbia Judicial Decisions Analysis Project plan in markdown format

#### [Teamwork_contract: ](https://github.ubc.ca/nikihm/Capstone-project/blob/master/Docs/Team_contract.md) Contrat between group members, working hours and terms and conditions 

### Data Product

#### [Data Product README: ](https://github.ubc.ca/nikihm/Capstone-project/blob/master/Docs/data_product_README.md) Guide on how to run code to reproduce results found in report
