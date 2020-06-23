from collections import defaultdict, Counter
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import math
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords
from itertools import chain
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

def parse_BCJ(path, damage_model = None, damage_vectorizer = None, annotated_damages = None, cn_model = None, cn_vectorizer = None, annotated_cn = None, min_predict_proba = 0.5, high_precision_mode = False, include_no_damage_cases = True, dmg_context_length = 5, cn_context_length = 5):
    '''Given file path (text file) of negligence cases, finds static 
    information within the case (information that can be pattern matched)
    Expects a B.C.J. case format (British Columbia Judgments)
    
    The following fields are currently implemented:
    - Case Title
    - Judge Name
    - Registry
    - Year
    - Decision Length (in paragraphs)
    - Damages
    - Multiple Defendants
    - Plaintiff Wins
    
    Arguments: 
    doc (String): The case in text format following the form used in the DOCX to TXT notebook
    [Optional] damage_model (sklearn model) - Used for damage classification. If not supplied uses rule based
    [Optional] damage_vectorizer (DictVectorizer) Used for damage classification. If not supplied uses rule based
    [Optional] annotated_damages (dict) The results of cross-validation classification for annotated damages, mapping case title to the predicted damages, only if using damage classifier.
    [Optional] cn_model (sklearn model) - Used for contributory negligence percent classification. If not supplied uses rule based
    [Optional] cn_vectorizer (DictVectorizer) Used for contributory negligence percent classification. If not supplied uses rule based
    [Optional] annotated_cn (dict) The results of cross-validation classification for annotated percent values, mapping case title to the predicted damages, only if using damage classifier.
    [Optional] min_predict_proba (float) - When classifying, sets the min confidence a tag must have to be assigned
    [Optional] high_precision_mode (bool) - When classifying, if true, will never sum or use "sub-" tags. Only high precision predictions.
    [Optional] include_no_damage_cases (bool) - If true, will include cases where no damages are awarded. If false, will only return cases where it found damages.
    [Optional] dmg_context_length (int) - The number of words around a damage to use as context if using a classifier approach
    [Optional] cn_context_length (int) - The number of words around a percent to use as context if using a classifier approach

    Returns: case_parsed_data (list) of case_dict (Dictionary): List of Dictionaries with rule based parsable fields filled in
    '''
    with open(path, encoding='utf-8') as document:
        document_data = document.read()
        
    document_data = document_data.split('End of Document\n') # Always split on 'End of Document\n'
    case_parsed_data = []
    for i in range(len(document_data)):
        case_dict = dict() 
        case = document_data[i]
        case = case.strip() # Make sure to strip!
        if len(case) == 0: # Skip empty lines
            continue
        
        lines = case.split('\n')
        if len(lines) < 2:
            print(case)
        case_title = lines[0]
        case_type = lines[1]

        # If the case should be included in our analysis...
        if filter_unwanted_cases(case, case_title, case_type):
            # ----
            # Fields that can be found via pattern matching
            # ----

            if re.search('contributory negligence', case, re.IGNORECASE):
                case_dict['contributory_negligence_raised'] = True
            else:
                case_dict['contributory_negligence_raised'] = False
            decision_len = re.search(r'\(([0-9]+) paras\.?\)', case) # e.g.) (100 paras.)
            registry = re.search(r'(Registry|Registries): ?([A-Za-z0-9 ]+)', case) # e.g.) Registry: Vancouver
            case_dict['written_decision'] = True if int(decision_len.group(1)) > 1 else False
            if registry:
                registry = registry.group(2).strip()
            else:
                registry = re.search(r'([A-Za-z ]+) Registry No.', case) # Alt form e.g.) Vancouver Registory No. XXX
                if registry:
                    registry = registry.group(1).strip()
                else:
                    registry = re.search(r'([A-Za-z ]+) No. S[0-9]*', case)
                    if registry:
                        registry = registry.group(1).strip()
                    else:
                        print('WARNING: Registry could not be found (This shouldn\'t occur!)')

            case_dict['decision_length'] = decision_len.group(1)
            case_dict['registry'] = registry

            # ----
            # Fields that are always in the same place
            # ----

            judge_name = lines[4].strip()
            case_title = lines[0].strip()
            year = re.search(r'20[0-2][0-9]', case_title) # Year in case title. Limit regex to be from 2000 to 2029
            if year:
                year = year.group(0)
            else:
                # Rare case: Sometimes the title is too long. Rely on Heard date instead.
                year = re.search(r'Heard:.* ([2][0][0-2][0-9])', case)
                if year:
                    year = year.group(1)
                else:
                    print('WARNING: Year not found')
            case_dict['case_title'] = case_title
            case_dict['year'] = year
            case_dict['judge'] = judge_name

            # ----
            # Fields that require some parsing
            # ----

            case_dict['multiple_defendants'] = rule_based_multiple_defendants_parse(case)
            case_dict['plaintiff_wins'] = plaintiff_wins(case)

            # ----
            # Damages
            # ----
                        
            if case_dict['plaintiff_wins'] == True:
                # Only use classifier if model, vectorizer, and already annotated cases not supplied
                if damage_model and damage_vectorizer and annotated_damages:
                    if case_title in annotated_damages:
                        case_dict['damages'] = annotated_damages[case_title]
                    else:
                        predictions = predict(case, damage_model, damage_vectorizer, context_length = dmg_context_length)
                        case_dict['damages'] = assign_classification_damages(predictions, min_predict_proba = min_predict_proba, high_precision_mode = high_precision_mode)
                else:
                    case_dict['damages'] = rule_based_damage_extraction(case)  
            else:
                case_dict['damages'] = None

            # ----
            # Contributory Negligence
            # ----
            
            # find a way to not use rule based to get CN successful
            percent_reduction, contributory_negligence_successful = get_percent_reduction_and_contributory_negligence_success(case_dict, case)

            if cn_model and cn_vectorizer and annotated_cn:
                percent_reduction_clf = None
                if case_title in annotated_cn:
                    if contributory_negligence_successful:
                        case_dict['percent_reduction'] = annotated_cn[case_title]
                    else:
                        case_dict['percent_reduction'] = percent_reduction
                else:
                    predictions = predict(case, cn_model, cn_vectorizer, category='cn', context_length = cn_context_length)
                    percent_reduction_clf = assign_classification_CN(predictions) 
                if percent_reduction_clf and contributory_negligence_successful:
                    case_dict['percent_reduction'] = percent_reduction_clf
                else:
                    case_dict['percent_reduction'] = percent_reduction*0.01 if percent_reduction != None else None
            else: 
                case_dict['percent_reduction'] = percent_reduction*0.01 if percent_reduction != None else None
            case_dict['contributory_negligence_successful'] = contributory_negligence_successful
             
        
        if include_no_damage_cases:
            if case_dict != dict(): 
                case_parsed_data.append(case_dict)
        else:  
            if case_dict != dict() and case_dict['damages'] != None: 
                case_parsed_data.append(case_dict)
    return case_parsed_data

def rule_based_multiple_defendants_parse(doc):
    ''' Helper function for parse_BCJ
    
    Given a case. Uses regex/pattern-matching to determine whether we have multiple defendants.
    For the most part the logic relies on whether the langauge used implies plurality or not.
    
    Arguments: doc (String): The case in text format following the form used in the DOCX to TXT notebook
    Returns: response (True, False, or 'UNK')
    '''
    assert len(doc) > 0

    # Case 1)
    # Traditional/most common. Of form "Between A, B, C, Plaintiff(s), X, Y, Z Defendant(s)"
    # Will also allow "IN THE MATTER OF ... Plaintiff .... Defendant..."
    # Can successfully cover ~98% of data
    regex_between_plaintiff_claimant = re.search(r'([Between|IN THE MATTER OF].*([P|p]laintiff[s]?|[C|c]laimant[s]?|[A|a]ppellant[s]?|[P|p]etitioner[s]?|[R|r]espondent[s]?).*([D|d]efendant[s]?|[R|r]espondent[s]?|[A|a]pplicant[s]?).*\n)', doc)
    
    # Match found
    if regex_between_plaintiff_claimant:
        text = regex_between_plaintiff_claimant.group(0).lower()
        if 'defendants' in text or 'respondents' in text or 'applicants' in text: # Defendant/respondent same thing.
            return True
        elif 'defendant' in text or 'respondent' in text or 'applicant' in text:
            return False
    
    # If not found, try other less common cases
    else:
        # Case 2)
        # Sometimes it does not mention the name of the second item. (Defendent/Respondent)
        # We can estimate if there are multiple based on the number of "," in the line (Covers all cases in initial data)
        regex_missing_defendent = re.search(r'(Between.*([P|p]laintiff[s]?|[C|c]laimant[s]?|[A|a]ppellant[s]?|[P|p]etitioner[s]?).*\n)', doc)
        if regex_missing_defendent:
            text = regex_missing_defendent.group(0).lower()
            if len(text.split(',')) > 5:
                return True
            else:
                return False
            
        else:
            print('Multiple defendants: Unknown! Unable to regex match')
            return 'UNK'
        
def rule_based_damage_extraction(doc, min_score = 0.9, max_match_len_split = 10):
    '''Helper function for parse_BCJ
    
    Given a case, attempts to extract damages using regex patterns
    
    Arguments: doc (String): The case in text format following the form used in the DOCX to TXT notebook
    min_score (float): The minimum paragraph score to consider having a valid $ number
                       Paragraph has score 1 if its the last paragraph
                       Paragraph has score 0 if its the first paragraph
    max_match_len_split (int): The max amount of items that can appear in a regex match after splitting (no. words)
    
    Returns: damages (Dict): Contains any found damages
    
    '''
    damages = defaultdict(float)
    repetition_detection = defaultdict(set) # Used to stem adding repeated values
    no_paras = re.search(r'\(([0-9|,]+) paras?\.?\)', doc).group(1) # Get number of paragraphs
    pattern = r'([.]?)(?=\n[0-9]{1,%s}[\xa0|\s| ]{2})'%len(no_paras) # Used to split into paras
    paras_split = re.split(pattern, doc)
    money_patt = r'\$[0-9|,]+' # Used to get all paragraphs with a money amount
    scored_paras = [] # Score paragraphs based on where they appear in the document
                      # Score of 0.0 would be the first paragraph. Score of 1.0 would be the last paragraph
        
    for i, paragraph in enumerate(paras_split):
        if re.search(money_patt, paragraph):
            scored_paras.append((i / len(paras_split), paragraph)) # (score, paragraph). Score formula: i/no_paras
            
    scored_paras = sorted(scored_paras, key=lambda x:x[0])[::-1] # Store from last paragraph to first
    if len(scored_paras) == 0:
        return None
    if scored_paras[0][0] < min_score: #If highest scored paragraph is less than minimum score.
        return None
    
    # Rule based dmg extraction REGEX patterns
    regex_damages = r'[\w|-]* ?(?:damage|loss|capacity|cost).+?\$? ?[0-9][0-9|,|.]+[0-9]'
    regex_damages_2 = r'[^:] \$? ?[0-9][0-9|,|.]+[0-9] (?:for|representing)?[ \w\-+]+damages?'
    regex_damages_3 = r'[^:] \$? ?[0-9][0-9|,|.]+[0-9] (?:for|representing)?[ \w\-+]+damages?(?:(?:for|representing)?.*?[;.\n])'
    regex_future_care_loss = r'(?:future|past|in[-| ]?trust|award).*?(?:loss|costs?|income|care)?.*?\$? ?[0-9][0-9|,|.]+[0-9]'
    regex_for_cost_of = r'\$? ?[0-9][0-9|,|.]+[0-9][\w ]*? cost .*?\.'

    # Keywords to look in match for categorization
    general_damage_keywords = [('general',), ('future', 'income', 'loss'), ('future', 'income'), ('future', 'wage', 'loss'), ('future', 'earning'), ('!past', 'earning', 'capacity'), ('future', 'capacity'), ('future', 'earning'), ('!past', 'loss', 'opportunity'), ('!past', 'loss', 'housekeep'), ('ei', 'benefit')]
    special_damage_keywords = [('special',), ('trust',), ('past', 'income', 'loss'), ('past', 'wage'), ('past', 'earning'), ('past', 'income'), ('earning', 'capacity')]
    aggravated_damage_keywords = [('aggravated',)]
    non_pecuniary_damage_keywords = [('non', 'pecuniary')]
    punitive_damage_keywords = [('punitive',)]
    future_care_damage_keywords = [('future', 'care'), ('future', 'cost')]
    
    patterns = [regex_damages, regex_damages_2, regex_damages_3, regex_future_care_loss, regex_for_cost_of]
    banned_words = ['seek', 'claim', 'propose', 'range', ' v. '] # Skip paragraphs containing these
    counter_words = ['summary', 'dismissed'] # Unless these are mentioned. 
                                             # example) "Special damage is $5k. But claims for aggravated are 'dismissed'" 
    
    # Get money amounts from the text
    total = None
    matches = []
    summary_matches = []
    for i, scored_para in enumerate(scored_paras):
        text = scored_para[1]
        score = scored_para[0]
        
        if score > min_score:
            if any(item.startswith('summary') for item in text.lower().split()[:4]) or any(item.startswith('conclusion') for item in text.lower().split()[:4]):
                text_matches = get_matching_text(patterns, text, max_match_len_split)
                for t_m in text_matches:
                    summary_matches.append((score, t_m))
            elif i+1 < len(scored_paras) and (any(item.startswith('summary') for item in scored_paras[i+1][1].lower().split()[-4:]) or any(item.startswith('conclusion') for item in scored_paras[i+1][1].lower().split()[-4:])):
                text_matches = get_matching_text(patterns, text, max_match_len_split)
                for t_m in text_matches:
                    summary_matches.append((score, t_m))
            else:
                skip = False # Skip paras with banned words
                for banned_word in banned_words: 
                    if banned_word in text:
                        skip = True       
                for counter_word in counter_words:
                    if counter_word in text:
                        skip = False
                if skip:
                    continue

                text_matches = get_matching_text(patterns, text, max_match_len_split)
                for t_m in text_matches:
                    matches.append((score, t_m))
        
    # Only keep matches from the summary if a summary was found. If not keep all matches.
    if len(summary_matches) > 0: 
        matches = summary_matches

    # Extract $ value from paragraphs with money in them. Determine correct column
    regex_number_extraction = r' ?[0-9][0-9|,|.]+[0-9]'
    for score, match in matches:
        skip = False # Banned words should not appear in final matches
        for banned_word in banned_words: 
            if banned_word in match:    
                skip = True
                break
        if skip:
            continue
        
        amount = re.findall(regex_number_extraction, match, re.IGNORECASE)
        extracted_value = clean_money_amount(amount)
        if extracted_value is None: # Make sure we are able to extract a value
            continue
            
        value_mapped = False # If we mapped the value into a damage category - stop trying to map into other categories
        value_mapped = assign_damage_to_category(extracted_value, general_damage_keywords, match, score, matches, 'General', damages, repetition_detection, repetition_key = ('general',))
        if not value_mapped:
            value_mapped = assign_damage_to_category(extracted_value, special_damage_keywords, match, score, matches, 'Special', damages, repetition_detection, repetition_key = ('special',))
        if not value_mapped:
            value_mapped = assign_damage_to_category(extracted_value, non_pecuniary_damage_keywords, match, score, matches, 'Non Pecuniary', damages, repetition_detection, repetition_key = ('non','pecuniary'))
        if not value_mapped:
            value_mapped = assign_damage_to_category(extracted_value, aggravated_damage_keywords, match, score, matches, 'Aggravated', damages, repetition_detection, repetition_key = ('aggravated',))
        if not value_mapped:
            value_mapped = assign_damage_to_category(extracted_value, punitive_damage_keywords, match, score, matches, 'Punitive', damages, repetition_detection, repetition_key = ('punitive',))
        if not value_mapped:
            value_mapped = assign_damage_to_category(extracted_value, future_care_damage_keywords, match, score, matches, 'Future Care', damages, repetition_detection) 
        if not value_mapped: # Last attempt: Only use "total amounts" if nothing else was found
            total_keywords = [('total',), ('sum',), ('award',)]
            for keywords in total_keywords:
                if match_contains_words(match.lower(), keywords):
                    if is_best_score(score, matches, keywords):
                        if extracted_value not in repetition_detection[('total',)]:
                            damages['Pecuniary Total'] = damages['Special'] + damages['General'] + damages['Punitive'] + damages['Aggravated'] + damages['Future Care']
                            damages['Total'] = damages['Pecuniary Total'] + damages['Non Pecuniary']
                            if damages['Total'] == 0:
                                total = extracted_value
                                repetition_detection[('total',)].add(extracted_value)
                        
    damages['Pecuniary Total'] = damages['Special'] + damages['General'] + damages['Punitive'] + damages['Aggravated'] + damages['Future Care']
    damages['Total'] = damages['Pecuniary Total'] + damages['Non Pecuniary']
    
    if damages['Total'] == 0 and total is not None: # Only use the "total" if we couldnt find anything else!
        damages['Total'] = total
        damages['General'] = total
        
    columns = ['Total', 'Pecuniary Total', 'Non Pecuniary', 'Special', 'General', 'Punitive', 'Aggravated', 'Future Care']
    for c in columns:
        damages[c] = None if damages[c] == 0 else damages[c]
        if damages[c] != None:
            assert damages[c] > 0
    
    return damages

def assign_damage_to_category(damage, damage_keywords, match, match_score, matches, damage_type, damage_dict, repetition_dict, repetition_key = None):
    '''Helper function for rule based damage extraction.
    
    Adds damage to dictionary based on given parameters so long as it is the
    highest scoring match & doesn't appear in the repetition dictionary
    
    Argumets:
    damage (float) - The damage amount in the match
    damage_keywords (list) - Keywords that may appear in match
    match (string) - The match string itself
    matches (list) - All matches. Used to determine if we found the best match
    damage_dict (dict) - Dictionary storing all damages
                       - Will be modified in place
    repetition_dict (dict) - Dictionary storing repeated values
                           - Will be modified in place
    (Optional) repetition_key (Tuple) - If not none, will use this key to store repetitions. Else will use matching keyword
    
    Returns:
    value_belongs (Boolean) - True if the value belongs in the given keyword category. False otherwise
    '''
    match = match.lower()
    value_belongs = False
    
    for keywords in damage_keywords:
        if match_contains_words(match, keywords):
            value_belongs = True
            if is_best_score(match_score, matches, keywords):
                if damage not in repetition_dict[repetition_key if repetition_key else keywords]:
                    damage_dict[damage_type] += damage
                    repetition_dict[repetition_key if repetition_key else keywords].add(damage)
            break
    
    return value_belongs

def clean_money_amount(money_regex_match):
    '''Helper function to extract float from a money regex match
    
    Arguments:
    money_regex_match (Regex.findall object) - Match of $ amount
    
    Returns:
    None if a bad match
    extracted_value (float) - The money amount in float form
    '''
    # If our regex contains more than 1 or 0 money values. We cannot use the match.
    if len(money_regex_match) > 1:
        return None
    if len(money_regex_match) == 0:
        print('Error: No Money in match!', match)
        return None

    extracted_value = None
    amount = money_regex_match[0].replace(',' , '')
    amount = amount.replace(' ' , '')
    # Deals with money at end of sentence. example) ... for '5,000.00.' -> '5000.00'
    if amount[-1] == '.': 
        amount = amount[:-1]
    # Deals with quantities such as $2.5 million
    if 'million' in amount or amount[-1] == 'm':
        amount = str(float(re.findall('[0-9|\.]+', amount)[0])*10e6)
    # Deals with a rare typo in some cases. example) 50.000.00 -> 50000.00
    if amount.count('.') > 1: 
        dot_count = amount.count('.')
        changes_made = 0
        new_amount = ''
        for letter in amount:
            if letter == '.' and changes_made != dot_count-1:
                changes_made += 1
            else:
                new_amount += letter
        amount = new_amount
    extracted_value = float(amount)
    return extracted_value

def get_matching_text(patterns, text, max_match_len_split):
    '''Helper function for rule based damage extraction.
    
    Given a set of regex; pulls out all matching text
    
    Arguments:
    patterns (list) - List of regex patterns in string format
    text (string) - Text to search for matches in
    
    Returns:
    matches (list) - List containing all matches in text format
    '''

    matches = []
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            if 'and' not in match:
                if len(match.split()) <= max_match_len_split:
                    matches.append(match)
                    
    return matches

def is_best_score(score, matches, keywords):
    '''Helper function for rule based damage extraction.
    
    Given a set of regex matches, determine if the score is the highest score out of all matches for the given keywords
    Score is from 0 - 1; describes where in the paragraph the match was found
    Score is 1 if the match came from the final paragraph
    Score is 0 if the match came from the first paragraph
    
    Arguments:
    score (float) - The score of the item you're inspecting
    matches (list) - List of matches where each element is of form (score, match text)
    keywords (tuple) - All words that should appear in the match
    
    Returns: True or False
    
    '''
    best_score = score
    
    for score, match in matches:
        if all(word in match.lower() for word in keywords):
            if score > best_score:
                return False
            
    return True

def match_contains_words(match, words):
    '''Helper function for rule based damage extraction.
    
    Given some text. Find if the words are all present in the text.
    If word begins with '!' the word cannot appear in the text, acts as a negation. 
    Can handle mix/matching of both types.
    
    Example: ('!good', 'day') would match any string with the word "day" present and "good" NOT present.
    
    Arguments:
    match (String) - The text to look for words in
    words (list) - List of words to check for. If word begins with ! (i.e. '!past'), then the word cannot appear in it
    
    Returns:
    True if all words are present (or not present if using !)
    False otherwise
    
    '''
    pos_words = []
    neg_words = []
    for word in words:
        if word.startswith('!'):
            neg_words.append(word[1:])
        else:
            pos_words.append(word)
            
    if all(word in match for word in pos_words):
        if all(word not in match for word in neg_words):
            return True
        
    return False

def filter_unwanted_cases(case, case_title, case_type):
    '''Given a case, its title & type, determines whether the case
    is relevant or not for our analysis
    
    Arguments:
    case (string) - Case data in string form
    case_title (string) - Case title (line 1 of case)
    case_type (string) - Case type (line 2 of case)
    
    Returns:
    boolean - True if case should be analyzed. False if it should be skipped.
    '''
    
    if 'R. v.' in case_title or '(Re)' in case_title: # Skip crown cases, Skip (Re) cases
        return False
    
    # Third party cases
    if 'parties' in case.lower() and 'third party procedure' in case.lower():
        return False
    
    # Disposition without trial cases
    if 'disposition without trial' in case.lower() and 'civil procedure' in case.lower():
        return False
    
    # Applications & Motions brought to the judge
    if 'civil procedure' in case.lower() and 'applications and motions' in case.lower():
        return False
    else:
        if 'applications and motions' in case.lower() and 'practice' in case.lower():
            return False
        
    # Right to a jury case
    if 'right to a jury' in case.lower() and 'juries and jury trials' in case.lower():
        return False
    
    # Adding/Subbing Parties case
    if 'adding or substituting parties' in case.lower():
        return False

    # Skip client/solicitor cases (not same as plaintiff/defendant)
    regex_client_solicitor = re.search(r'(Between.*([C|c]lient[s]?).*([S|s]olicitor[s]?|[L|l]awyer[s]?))', case)
    if regex_client_solicitor:
        return False

    regex_solicitor_client = re.search(r'(Between.*([L|l]awyer[s]?|[S|s]olicitor[s]?).*([C|c]lient[s]?))', case)
    if regex_solicitor_client:
        return False

    # In some rare cases we have 'IN THE MATTER OF ..' (rather than 'Between ...') .. but it is following by the normal
    # plaintiff/defendant dynamic. Only skip cases if there is no mention of the following terms
    # (Can be cleaned up in future)
    key_words = ['appellant', 'respondent', 'claimant', 'petitioner', 'plaintiff', 'defendant',
    'appellants', 'respondents', 'claimants', 'petitioners', 'plaintiffs', 'defendants']
    regex_in_matter_of = re.search(r'IN THE MATTER OF .*\n\([0-9]+ paras.\)', case)
    if regex_in_matter_of:
        remove = True
        for key in key_words:
            if key in regex_in_matter_of.group(0).lower().strip():
                remove = False

        if remove:
            return False

    if 'British Columbia Judgments' in case_type:
        return True
    
    return False


def paragraph_tokenize(case):
    ''' Takes string input of the entire document (case) and returns list of lists of paragraphs in the document.
    
    Arguments: 
    case (str) - string of single legal case
    
    Returns:
    case_data(list) - list of of numbrered paragraphs in the document where the first item is the case_title'''
    
    case_data = []
    lines = case.split('\n')
    if not 'British Columbia Judgments' in lines[1]:
        return
    case_data.append(lines[0])
    decision_length = re.search(r'\(([0-9|,]+) paras?\.?\)', case).group(1)

    # split paragraphs on newline, paragraph number, two spaces
    pattern = r'.?(?=\n[0-9]{1,%s}[\xa0]{2})'%len(decision_length)
    paras_split = re.split(pattern, case)

    paras = []
    for para in paras_split:   
        # make sure the paragraph starts with the correct characters
        para_start = re.match(r'^\n([0-9]{1,%s})[\xa0]{2}'%len(decision_length), para)
        if para_start:
            paras.append(para)
    case_data.extend(paras)
    return case_data

def summary_tokenize(case):
    ''' String of Entire Document and returns the document summary and HELD section.

    Arguments: 
    case (str) - string of single legal case

    Return: Tuple: summary (str)- summary and HELD section of case (str)
                    summart start idx (int)
                    summary end idx (int)'''
    
    # split paragraphs on newline, paragraph number, two spaces
    summary = re.search(r'\([0-9]{1,3} paras\.\)\ncase summary\n((.*\n+?)+)(?=HELD|(Statutes, Regulations Rules Cited:)|(Counsel\n))', case, re.IGNORECASE)
    if summary:
        return summary.group(1), summary.span(1)[0], summary.span(1)[1]
    else:
        return None, None, None # Must return 3 items

def get_context_and_float(value, text, context_length = 8, plaintiff_name = 'Plaintiff', defendant_name = 'Defendant'):
    '''Given a string value found in a body of text, 
    return its context, and its float equivalent.
    -----------------
    Arguments:
    value - percent match found in text
    text - string value where matches were extracted from, eg paragraph or summary (str)
    context_length - the length of context around each quantity to return
    
    Returns:
    value_context - string of context around value (str)
    extracted_value - string quantity value extracted to its float equivalent'''
    
    
    # get context for monetary/percent values 
    context = ''
    amount = re.findall(r'[0-9]+[0-9|,]*(?:\.[0-9]+)?', value)
    extracted_value = clean_money_amount(amount) #use helper function to get float of dollar/percent value
    if not extracted_value:
        print('\nERROR: cant convert string\n', amount)
        return context, None
    # get indices of last instance of value in text - tokenize like this for values of type 'per cent and percent'
    start_idx = text.rfind(value)
    if start_idx == -1:
        print('ERROR: value not in text')
    end_idx = start_idx + len(value)
    tokens = text[:start_idx].split() + [value] + text[end_idx:].split()
    
    # get indices of quantity value in text
    loc = [i for i, token in enumerate(tokens) if value in token] 
    
    # if the quantity is in the text, choose context of last mention of value
    if len(loc) > 0:
        loc = loc[-1] 
        if loc - context_length >= 0 and loc + context_length < len(tokens):
            context = " ".join(tokens[loc - context_length:loc + context_length + 1])
        elif loc - context_length < 0 and loc + context_length < len(tokens):
            beg = abs(loc -context_length)
            context = " ".join(tokens[loc-context_length + beg:loc + context_length + 1])
        elif loc - context_length > 0 and loc + context_length > len(tokens): 
            context = " ".join(tokens[loc - context_length:len(tokens)])

    return context.lower(), extracted_value

def conditions_for_extracted_value(context, extracted_value, keywords, plaintiff_split, defendant_split, entities):
    ''' Given the context surrounding an extracted value (percent), keywords relevant to contributory negligence (ie liability, approtion, fault, etc), 
    a list of the Plaintiffs names (ie John Doe), a list of the defendants names, and a combined list of entities(ie plaintiff, john, doe, defendant):
    Return: the modifed extracted value (float)
    ------------
    Arguments:
    context: (str)
    extracted_value: (float) found in context
    keywords, plaintiff_split, defendant_split, entities: (list) of strings
    ------------
    Example:
    context = 'the defendant is responsible for 30% of damages'
    extracted_value = 30.0
    keywords = ['fault', 'liable', 'liability', 'apportion', 'contributor', 'recover', 'responsible']
    plaintiff_split = ['john', 'doe']
    defendant_split = ['jane', 'smith']
    entities = ['plaintiff', 'defendant', 'john', 'jane', 'doe', 'smith']
    conditions_for_extracted_value(context, extracted_value, 
                        keywords, plaintiff_split, defendant_split) = 70.0
    '''
    # conditions for keeping extracted_value and updating extracted_value
    # skip extracted_values with contexts lacking keywords/entities
    if extracted_value == 100 or extracted_value == 0 or extracted_value < 10:
        return
    if not any(token in context for token in keywords + entities) or context == '' or any('costs' == token for token in context.split()) or ('interest' in context and 'rate' in context.split()):
        return 
    if 'recover' in context and any(word in context for word in plaintiff_split + ['plaintiff']):
        extracted_value = 100 - extracted_value
    if any(word1 in context and word2 in context for word1 in defendant_split + ['defendant'] for word2 in ['liable', 'responsible', 'fault', 'against']):
        extracted_value = 100 - extracted_value
    return extracted_value

def contributory_negligence_successful_fun(context, keywords):
    '''Given text containing percent reduction and a list of keywords to check for,
    confirm presence of keywords and return whether or not contributory negligence was successful
    --------------
    Arguments:
    context (str)
    keywords(list)

    Returns: True or None (bool)'''
    if any(word in context for word in keywords):
        if 'plaintiff' or 'damages' or 'defendant' in context:
            contributory_negligence_successful = True
            return contributory_negligence_successful
    return

def get_percent_reduction_and_contributory_negligence_success(case_dict, case, min_score = 0.6):
    paragraphs = paragraph_tokenize(case)
    case_title = case_dict['case_title']
    assert paragraphs[0] == case_title
    
    # default value for contributory negligence success is FALSE
    contributory_negligence_successful = False
    percent_pattern = r'([0-9][0-9|\.]*(?:%|\sper\s?cent))'
    
    # entities and keywords used to filter percent values
    keywords = ['against', 'reduce', 'liability', 'liable', 'contributor', 'fault', 'apportion', 'recover', 'responsible']
    # extract plaintiff and defendant name for use in %reduction conditions
    plaintiff_defendant_pattern = r'([A-Za-z|-|\.]+(:? \(.*\))?)+ v\. ([A-Za-z|-]+)+' # group 1 is plaintiff group 2 is defendant
    if re.search(plaintiff_defendant_pattern, case_title):
        plaitiff_defendant = re.search(plaintiff_defendant_pattern, case_title).groups() # tuple (plaintiff, defendant)
    else:
        plaitiff_defendant = ('Plaintiff', 'Defendant')
    plaintiff_split = [word.lower() for word in plaitiff_defendant[0].split()]
    defendant_split = [word.lower() for word in plaitiff_defendant[-1].split()]
    entities = ['defendant', 'plaintiff'] + plaintiff_split + defendant_split 

    if case_dict['contributory_negligence_raised'] and case_dict['plaintiff_wins']:
        percent_reduction = None
        best_percent = None
        best_score = 0
        for j, paragraph in enumerate(paragraphs[1:]):
            score = float((j+1)/int(case_dict['decision_length']))
            paragraph = paragraph.lower()
            if not score >= min_score: ## min score not existant in bcj parser
                continue

            percent_mentioned = re.findall(percent_pattern, paragraph, re.IGNORECASE)
            extracted_value_tie_breaker = Counter()
            if len(percent_mentioned) > 0:
                for percent in percent_mentioned:
                    context, extracted_value = get_context_and_float(percent, paragraph)
                    # conditions for keeping extracted_value and updating extracted_value
                    # skip extracted_values with contexts lacking keywords/entities
                    if context == '':
                        continue
                    extracted_value = conditions_for_extracted_value(context, extracted_value, keywords, plaintiff_split, defendant_split, entities)
                    if not extracted_value:
                        continue
                        
                    extracted_value_tie_breaker.update([extracted_value])
                
                    # conditions for contributory negligence successful
                    if not contributory_negligence_successful and extracted_value:
                        contributory_negligence_successful = contributory_negligence_successful_fun(context, keywords)

                    # matches patter "PERCENT against plaintiff"
                    if ('against' in context or 'fault' in context) and any(plaintiff_word in context for plaintiff_word in plaintiff_split+['plaintiff']):
                        best_percent = extracted_value
                        best_score = score
                        break                    
                    
                    # choose most common percent mentioned in highest scoring paragraph
                    if extracted_value_tie_breaker != Counter():
                        if score > best_score:
                            best_score = score
                            best_percent = extracted_value_tie_breaker.most_common(1)[0][0]

             # if no percent found, check for equal apportionment
            else:
                equal_apportionment = re.findall(r'.{20} (?:liability|fault) [a-zA-Z]{1,3} apportione?d? equally .{20}', paragraph)
                if len(equal_apportionment) > 0:
                    if contributory_negligence_successful_fun(equal_apportionment[0], keywords):
                        best_percent = 50.0
                        contributory_negligence_successful = True
        
        if not contributory_negligence_successful:
            # no CN percents found in paragraphs - time to check summary - same process
            summary, summary_start_idx, summary_end_idx = summary_tokenize(case)
            if summary:
                summary = summary.lower()
                percent_mentioned = re.findall(percent_pattern, summary, re.IGNORECASE)
                extracted_value_tie_breaker = Counter()
                if len(percent_mentioned) > 0:
                    for percent in percent_mentioned:
                        context, extracted_value = get_context_and_float(percent, summary)
                        # conditions for keeping extracted_value and updating extracted_value
                        # skip extracted_values with contexts lacking keywords/entities
                        extracted_value = conditions_for_extracted_value(context, extracted_value, keywords, plaintiff_split, defendant_split, entities)
                        if not extracted_value:
                            continue
                        extracted_value_tie_breaker.update([extracted_value])
                                                   
                        # conditions for contributory negligence successful
                        if not contributory_negligence_successful and extracted_value:
                            contributory_negligence_successful = contributory_negligence_successful_fun(context, keywords) 
                            
                        # matches patter "PERCENT against plaintiff"
                        if ('against' in context or 'fault' in context) and any(plaintiff_word in context for plaintiff_word in plaintiff_split+['plaintiff']):
                            best_percent = extracted_value
                            best_score = score
                            break 
                        # choose most common percent mentioned in summary
                        if extracted_value_tie_breaker != Counter():
                            best_percent = extracted_value_tie_breaker.most_common(1)[0][0]

               # if no percent found, check for equal apportionment
                else:
                    equal_apportionment = re.findall(r'.{20} (?:liability|fault) [a-zA-Z]{1,3} apportione?d? equally .{20}', summary)
                    if len(equal_apportionment) > 0:
                        if contributory_negligence_successful_fun(equal_apportionment[0], keywords):
                            best_percent = 50.0
                            contributory_negligence_successful = True
        if contributory_negligence_successful:
            percent_reduction = best_percent
    else:
        percent_reduction = None
 
    return percent_reduction, contributory_negligence_successful

def train_classifier(path, clf = MultinomialNB(), context_length = 5, min_para_score = 0, min_predict_proba = 0.5, high_precision_mode = False, fit_model = False):
    '''Trains a damages classifier based on the given training data path
    
    Arguments:
    path (String) - Path to .txt containing training data
    clf (sklearn classifier instance) - untrained sklearn classifier, ie MultinomialNB()
    [Optional] context_length (int) - Number of words around each value to include in features. Default 5.
    [Optional] min_para_score (float) - Value must have this min para score to be used as a feature. Score ranges from 0 to 1. Default 0
    [Optional] min_predict_proba (float) - Classifier must be this confident in order to assign the damage. Default 0.5
    [Optional] high_precision_mode (boolean) - If true, will not add up any "sub-" values in order to maintain better precision
    [Optional] fit_model (boolean) - if True it fits the model if false just returns the X, y, vectorizer
    
    Returns:
    model (sklearn model) - Trained model
    vectorizer (sklearn DictVectorizer) - fit-transformed vectorizer
    case_damages (dict) - Dictionarry mapping annotated damages to their cross-validated predictions

    '''
    tag_extractor = re.compile('''<damage type ?= ?['"](.*?)['"]> ?(\$?.*?) ?<\/damage>''')
    CN_tag_extractor = re.compile('''<percentage type ?= ?['"](.*?)['"]> ?(\$?.*?) ?<\/percentage>''')
    stop_words = set(stopwords.words('english'))
    
    with open(path, encoding='utf-8') as document:
        document_data = document.read()
        
    document_data = document_data.split('End of Document\n') # Always split on 'End of Document\n'
    
    examples_per_case = [] # Each element contains all examples in a case
    answers_per_case = [] # Each element contains all answers in a case 
    case_titles = [] #list of case titles visited
    num_cases = len(document_data)
    
    for i in range(len(document_data)):
        print('Reading training data and extracting features...', i / num_cases * 100, '%', end='\r')
        case = document_data[i]
        case = case.strip() # Make sure to strip!
        if len(case) == 0: # Skip empty lines
            continue
        
        lines = case.split('\n')
        case_title = lines[0]
        case_type = lines[1]
        
        case_examples = []
        case_answers = []
        if filter_unwanted_cases(case, case_title, case_type):  
            summary = summary_tokenize(case)
            # lower case and remove stopwords
            case = ' '.join([word for word in case.lower().split() if word not in stop_words])
            summary, summary_start_idx, summary_end_idx = summary_tokenize(case)
            
            matches = tag_extractor.finditer(case) # Extract all <damage ...>$x</damage> tags used for training
            for match in matches:
                features, answer = extract_features(match, case, tag_extractor, CN_tag_extractor, context_length = context_length, purpose='train')
                # if value is found in case summary, replace start_idx_ratio with 1
                if summary:
                    if match.start() >= summary_start_idx and match.end() <= summary_end_idx and answer != 'other':
                        features['start_idx_ratio'] = 1
                    
                case_examples.append(features)
                case_answers.append(answer)
                
        if len(case_examples) > 0 and len(case_answers) > 0:
            examples_per_case.append(case_examples)
            answers_per_case.append(case_answers)
            case_titles.append(case_title)
        else:
            print('Didnt find any tags in', case_title)
                    
    print('\nVectorizing...')    
    vectorizer = DictVectorizer()
    feats = list(chain.from_iterable(examples_per_case)) # Puts it into one big list

    # Delete the "value" and "float" feature. Easier to do this way
    # because we use the cleaned money amount elsewhere
    values = [feat['float'] for feat in feats]
    value_locations = [feat['start_idx_ratio'] for feat in feats]

    # Delete value/float feature to discourage overfitting
    for feat in feats:
        del feat['value']
        del feat['float']

    X = vectorizer.fit_transform(feats)
    y = list(chain.from_iterable(answers_per_case))
    
    print('Tag Distribution')
    dist = Counter(y)
    print(dist)

    if not fit_model:
        return X, y, vectorizer

    y_pred = cross_val_predict(clf, X, y, cv = 10) 
    y_prob = cross_val_predict(clf, X, y, cv = 10, method='predict_proba')
    
    values_per_case = [len(vals) for vals in examples_per_case] #number of tagged values in each case
    
    # for all cases in our annotations, get separated value, prediction, location, prob 
    prediction_features = list(zip(values, y_pred, value_locations, y_prob))
    prediction_feats_per_case = []
    number_visited = 0
    for i in range(len(values_per_case)):
        if i == 0:
            prediction_feats_per_case.append(prediction_features[:values_per_case[i]]) 
        elif i < len(values_per_case)-1:
            prediction_feats_per_case.append(prediction_features[number_visited:number_visited + values_per_case[i]])
        else:
            prediction_feats_per_case.append(prediction_features[-values_per_case[i]:])
        number_visited += values_per_case[i]
        assert len(prediction_feats_per_case[i]) == values_per_case[i]
    assert sum([len(feats) for feats in prediction_feats_per_case]) == sum(values_per_case)
    assert prediction_feats_per_case[-1][-1] == prediction_features[-1]
    assert len(case_titles) == len(prediction_feats_per_case)
    
    # assign damages to case
    case_damages = defaultdict(dict)
    for i in range(len(prediction_feats_per_case)):
        case_preds = prediction_feats_per_case[i]
        damages = assign_classification_damages(case_preds, min_score = min_para_score, min_predict_proba = min_predict_proba, high_precision_mode = high_precision_mode)
        case_damages[case_titles[i]].update(damages)

    print('Cross validation evaluation...')
    print(classification_report(y, y_pred))
    # print('Scores (F1-MACRO):', np.mean(cross_val_score(clf, X, y, cv = 5, scoring = 'f1_macro')))
    # print('Scores (F1-MICRO):', np.mean(cross_val_score(clf, X, y, cv = 5, scoring = 'f1_micro')))
    # print('Scores (F1-WEIGHTED):', np.mean(cross_val_score(clf, X, y, cv = 5, scoring = 'f1_weighted')))
    
    print('Training final model...')
    clf.fit(X, y)
    return clf, vectorizer, case_damages

def extract_features(match, case, dmg_pattern, cn_pattern = None, context_length = 4, purpose = 'train'):
    '''Given a match will return the features associated with the specific example
    Extracts the examples by finding the damage annotation tags
    in the form <damage type = "TYPE">$5000</damage>
    
    Arguments:
    match (Match Object) - Match object with the type as group 1 and value as group 2 if purpose = train, otherwise match group 0 is the value
    case (str) - The case data in string format
    pattern (str, regex pattern) - The regex pattern being used to find damages.
                                      Used to remove the tags in features using context around value.
    cn_pattern (str, regex pattern) - The regex pattern being used to find percentages.
    [Optional] context_length (int) - The number of words to use around the value for context
    [Optional] purpose (str) - Default is 'train', used to determine pattern type
    
    
    Returns:
    features (dict) - Dictionary containing each feature for the current match
    damage_type (str) or None - The type of damage associated with the value if purpose = 'train'
    '''
    features = dict()
    if purpose == 'train':
        damage_type = match.group(1)
        damage_value = match.group(2)
    else:
        damage_type = None
        damage_value = match.group(0)
        
    start_idx = match.start()
    end_idx = match.end()

    # Detect all <header> </header> tags
    header_regex = re.compile('''<header>(.*?)</header>''')
    headers = header_regex.finditer(case)
    current_heading = ''
    # Look for the one that is closest to the start idx (but occurs before it!)
    for header in headers:
        if header.end() < start_idx:
            current_heading = header.group(1).lower()

    feature_heading = dict(Counter(current_heading.split()))
    feature_heading = {k+'@Heading': v for k, v in feature_heading.items()}
    features.update(feature_heading)
    
    # Get 10 + Context Length on each side 
    # Used to get rid of damage tags within context around our match
    # We want to avoid getting half a damage tag else it wont be removed
    # So we get more than we need.
    start_tokenized = ' '.join(case[:start_idx].split()[-context_length-30:])
    end_tokenized = ' '.join(case[end_idx:].split()[:context_length+30])

    if purpose == 'train':
        if cn_pattern is None:
            print('Error: Didnt include percentage regex')
            return None
        # Remove damage tags AND percentage tags in context around match
        start_matches_dmg = dmg_pattern.finditer(start_tokenized)
        for s_dmg in start_matches_dmg:
            start_tokenized = start_tokenized.replace(s_dmg.group(0), s_dmg.group(2))
        start_matches_cn = cn_pattern.finditer(start_tokenized)
        for s_cn in start_matches_cn:
            start_tokenized = start_tokenized.replace(s_cn.group(0), s_cn.group(2))

        end_matches_dmg = dmg_pattern.finditer(end_tokenized)
        for e_dmg in end_matches_dmg:
            end_tokenized = end_tokenized.replace(e_dmg.group(0), e_dmg.group(2))
        end_matches_cn = cn_pattern.finditer(end_tokenized)
        for e_cn in end_matches_cn:
            end_tokenized = end_tokenized.replace(e_cn.group(0), e_cn.group(2))

    # Remove header tags from text. No need to regex these because they are always the same
    start_tokenized = start_tokenized.replace('<header>', '')
    start_tokenized = start_tokenized.replace('</header>', '')

    end_tokenized = end_tokenized.replace('<header>', '')
    end_tokenized = end_tokenized.replace('</header>', '')

    # Reconstruct sentence
    start_tokenized = start_tokenized.split()[-context_length:]
    end_tokenized = end_tokenized.split()[:context_length]
    #tokens = ' '.join(start_tokenized) + " " + damage_value + " " + ' '.join(end_tokenized)

    # Grab only the sentence value is in - rather than the entire context length
    start_tok_sent = sent_tokenize(' '.join(start_tokenized))
    start_tokenized = start_tok_sent[-1] if len(start_tok_sent) >= 1 else ''
    end_tok_sent = sent_tokenize(' '.join(end_tokenized))
    end_tokenized = end_tok_sent[0] if len(end_tok_sent) >= 1 else ''
    tokens = start_tokenized + " " + damage_value + " " + end_tokenized    

    value_start_idx = len(start_tokenized.split()) # Location of value in relation to sentence (token level)
    if len(damage_value.split()) > 1: # Deals with problems like '2 million' (where value is multiple tokens)
        value_end_idx = value_start_idx + len(damage_value.split()) - 1
    else:
        value_end_idx = value_start_idx
    tokens = tokens.split()
    
    # Features: value, float, location, BOW_before, BOW_after, BOW, token before and after
    start_boundary = value_start_idx - context_length if value_start_idx - context_length >= 0 else 0
    end_boundary = value_end_idx + context_length + 1 if value_end_idx + context_length + 1 < len(tokens) else len(tokens)
    context = " ".join(tokens[start_boundary: value_start_idx] + tokens[value_end_idx + 1:end_boundary])
    before = tokens[start_boundary : value_start_idx]
    after = tokens[value_end_idx + 1 : end_boundary]
    
    features['value'] = damage_value
    features['float'] = clean_money_amount([damage_value.strip('$')])
    features['start_idx_ratio'] = match.start()/len(case)

    assert features['float'] >= 0
    assert features['start_idx_ratio'] >= 0

    # Money "bins"
    if features['float'] < 1000:
        features['range'] = '< 1000'
    elif features['float'] < 25000:
        features['range'] = '1000 - 25000'
    elif features['float'] < 100000:
        features['range'] = '25000 - 100000'
    elif features['float'] < 500000:
        features['range'] = '100000 - 500000'
    else:
        features['range'] = '500000+'
    
    # next and previous word
    if len(before) > 0:
        features['prev word'] = before[-1]
    else:
        features['prev word'] = ''
    if len(after) > 0:
        features['next word'] = after[0]
    else:
        features['next word'] = ''

    # Bigrams
    if len(before) >= 2:
        features['prev bigram'] = before[-2] + ' ' + before[-1]
    elif len(before) == 1:
        features['prev bigram'] = before[-1]
    else:
        features['prev bigram'] = ''
    

    if len(after) >= 2:
        features['next bigram'] = after[0] + ' ' + after[1]
    elif len(after) == 1:
        features['next bigram'] = after[0]
    else:
        features['next bigram'] = ''

    # Trigrams
    if len(before) >= 3:
        features['prev trigram'] = before[-3] + ' ' + before[-2] + ' ' + before[-1]
    elif len(before) == 2:
        features['prev trigram'] = features['prev bigram']
    elif len(before) == 1:
        features['prev trigram'] = features['prev word']
    else:
        features['prev trigram'] = ''

    if len(after) >= 3:
        features['next trigram'] = after[0] + ' ' + after[1] + ' ' + after[2]
    elif len(after) == 2:
        features['next trigram'] = features['next bigram']
    elif len(after) == 1:
        features['next trigram'] = features['next word']
    else:
        features['next trigram'] = ''

    # BOW features
    features_bow_b = dict(Counter(before))
    features_bow_b = {k+'@Before': v for k, v in features_bow_b.items()}

    features.update(features_bow_b)
    bow = Counter(context.split())
    features.update(bow)


    return features, damage_type

def predict(case, clf, vectorizer, category='damages', context_length = 5):
    '''Given a legal negligence case (str), a trained classifier, and a fit_transformed DictVectorizer(), 
    Return a list of tuples of (value, prediction, value_location), where value_location is the ratio of the 
    character start index 
    ----------------------
    Arguments:
    case (String): legal negligence case 
    clf (Sklearn classifier instance): trained classifier with .fit method
    vectorizer (Sklearn DictVectorizer()): fit_transformed vectorizer 
    category (String): type of prediction being made, default 'damages' 
    ----------------------
    Returns: list of tuples or an empty list if no matches in the case
    Example: 
    case = 'I award $5,000 in punitive damages.'
    predict(case, clf, vectorizer)
    > [(MONEY, TYPE, LOCATION, PROBABILITY)]
    > [($5,000, 'punitive', 0.023, 0.95)]'''
    
    stop_words = set(stopwords.words('english'))
    if category == 'damages':
        value_extractor = re.compile('''\$ ?[1-9]+[0-9|,|\.]+''')
    else:
        value_extractor = re.compile('''([0-9]+[0-9]?(\.?[0-9])?(?:%|\sper\s?cent))''') #if theres a decimal, require number after

    case_examples = []
    value_locations = []
    values = []

    # Remove stopwords, lowercase, and update summary idx
    case = ' '.join([word for word in case.lower().split() if word not in stop_words])
    summary, summary_start_idx, summary_end_idx = summary_tokenize(case)
    matches = value_extractor.finditer(case) # Extract all <damage ...>$x</damage> tags used for training

    for match in matches:
        # extract features per match found
        if category == 'damages':
            features, _ = extract_features(match, case, value_extractor, purpose = 'predict', context_length = context_length)
        else:
            features, _ = extract_CN_features(match, case, value_extractor, purpose = 'predict', context_length = context_length)

        if summary:
            if match.start() >= summary_start_idx and match.end() <= summar_end_idx:
                features['start_idx_ratio'] = 1

        case_examples.append(features)
        value_locations.append(features['start_idx_ratio'])
        values.append(features['float'])
        
    # if money values found in the case, predict type
    if len(case_examples) > 0:
        X_test = vectorizer.transform(case_examples)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        return list(zip(values, y_pred, value_locations, y_prob))
    else:
        return []
    
def assign_classification_damages(predictions, min_score = 0, min_predict_proba = 0.5, high_precision_mode = False):
    '''Helper function for rule based BCJ
    Handles assigning predictions into final damage amounts
    
    Arguments:
    predictions (tuple returned from predict function)
    min_score (float) - If a prediction appears before this point in the case it is discarded
    min_predict_proba (float) - The minimum confidence a tag has before it can be assigned
    high_precision_mode (boolean) - If true, will not add any "sub-" categories to maintain higher precision
    
    Returns:
    damages (defaultdict(float)) - Damages with values filled in based on predictions
    '''
    
    damages = defaultdict(float)
    temporary_damages = defaultdict(list)
    for value, prediction_type, ratio, predict_proba in predictions:
        if ratio < min_score:
            continue
        
        if max(predict_proba) > min_predict_proba:
            temporary_damages[prediction_type].append(value)


    # High Precision Mode: Only assigns damage if the final damage tag is found
    # Does not add up and "sub-total" values
    # Only adds things like "In-Trust" and "Special" so long as both tags are found
    if high_precision_mode:
        damages['Pecuniary Total'] = None
        damages['Future Care'] = temporary_damages['future care'][0] if len(temporary_damages['future care']) != 0 \
                                 else None
        damages['Future Wage Loss'] = temporary_damages['future wage loss'][0] if len(temporary_damages['future wage loss']) != 0 \
                                      else None 
        damages['General'] = temporary_damages['general'][0] if len(temporary_damages['general']) != 0 \
                             else None
        damages['In Trust'] = temporary_damages['in trust'][0] if len(temporary_damages['in trust']) != 0 \
                             else None
        damages['Non Pecuniary'] = temporary_damages['non pecuniary'][0] if len(temporary_damages['non pecuniary']) != 0 \
                             else None
        damages['Past Wage Loss'] = temporary_damages['past wage loss'][0] if len(temporary_damages['past wage loss']) != 0 \
                             else None
        damages['Punitive'] = temporary_damages['punitive'][0] if len(temporary_damages['punitive']) != 0 \
                             else None
        damages['Special'] = temporary_damages['special'][0] if len(temporary_damages['special']) != 0 \
                             else None
        damages['Aggravated'] = temporary_damages['aggravated'][0] if len(temporary_damages['aggravated']) != 0 \
                             else None
        damages['Total'] = temporary_damages['total'][0] if len(temporary_damages['total']) != 0 \
                             else None
        
        # Only sum damages that actually had the tag itself predicted
        if damages['Future Wage Loss']:
            if damages['General']:
                damages['General'] += damages['Future Wage Loss']
            else:
                damages['General'] = damages['Future Wage Loss']
        if damages['Past Wage Loss']:
            if damages['Special']:
                damages['Special'] += damages['Past Wage Loss']
            else:
                damages['Special'] = damages['Past Wage Loss']
        if damages['In Trust']:
            if damages['Special']:
                damages['Special'] += damages['In Trust']
            else:
                damages['Special'] = damages['In Trust']
        if damages['Special']:
            damages['Pecuniary Total'] = damages['Special']
        if damages['General']:
            if damages['Pecuniary Total']:
                damages['Pecuniary Total'] += damages['General']
            else:
                damages['Pecuniary Total'] = damages['General']
        if damages['Future Care']:
            if damages['Pecuniary Total']:
                damages['Pecuniary Total'] += damages['Future Care']
            else:
                damages['Pecuniary Total'] = damages['Future Care']
    
    # "Low" precision mode
    # Adds up the sub- categories
    else:
        damages['Future Care'] = temporary_damages['future care'][-1] if len(temporary_damages['future care']) != 0 \
                                 else sum(temporary_damages['sub-future care'])
        damages['Future Wage Loss'] = temporary_damages['future wage loss'][-1] if len(temporary_damages['future wage loss']) != 0 \
                                      else sum(temporary_damages['sub-future wage loss'])
        damages['General'] = temporary_damages['general'][-1] if len(temporary_damages['general']) != 0 \
                             else sum(temporary_damages['sub-general'])
        damages['In Trust'] = temporary_damages['in trust'][-1] if len(temporary_damages['in trust']) != 0 \
                             else sum(temporary_damages['sub-in trust'])
        damages['Non Pecuniary'] = temporary_damages['non pecuniary'][-1] if len(temporary_damages['non pecuniary']) != 0 \
                             else sum(temporary_damages['sub-non pecuniary'])
        damages['Past Wage Loss'] = temporary_damages['past wage loss'][-1] if len(temporary_damages['past wage loss']) != 0 \
                             else sum(temporary_damages['sub-past wage loss'])
        damages['Punitive'] = temporary_damages['punitive'][-1] if len(temporary_damages['punitive']) != 0 \
                             else sum(temporary_damages['sub-punitive'])
        damages['Special'] = temporary_damages['special'][-1] if len(temporary_damages['special']) != 0 \
                             else sum(temporary_damages['sub-special'])
        damages['Aggravated'] = temporary_damages['aggravated'][-1] if len(temporary_damages['aggravated']) != 0 \
                             else sum(temporary_damages['sub-saggravated'])
        damages['Total'] = temporary_damages['total'][-1] if len(temporary_damages['total']) != 0 \
                             else sum(temporary_damages['sub-total'])
        damages['General'] += damages['Future Wage Loss']
        damages['Special'] += damages['Past Wage Loss'] + damages['In Trust']
        damages['Pecuniary Total'] = damages['Special'] + damages['General'] + damages['Future Care']
    
        # Only sum up damages if we weren't able to properly extract it from the text.
        if damages['Total'] is None:
            damages['Total'] = damages['Pecuniary Total'] + damages['Non Pecuniary'] + damages['Aggravated']
    
    columns = ['Total', 'Pecuniary Total', 'Non Pecuniary', 'Special', 'General', 'Punitive', 'Aggravated', 'Future Care']
    for c in columns:
        damages[c] = None if damages[c] == 0 else damages[c]
        if damages[c] != None:
            assert damages[c] >= 0
    
    return damages     

def rule_based_convert_cases_to_DF(cases):
    '''Given a list of parsed cases returns a dataframe

    Arguments:
    cases (list of dict): Each item in list is an entire parsed case in dictionary form

    Returns:
    df (DataFrame): pandas dataframe containing case information
    '''

    lists = defaultdict(list)    
    for case in cases:
        lists['Case Name'].append(case['case_title'])
        lists['Year'].append(case['year'])
        lists['Total Damage'].append(case['damages']['Total'] if case['damages'] != None else None)
        lists['Total Pecuniary'].append(case['damages']['Pecuniary Total'] if case['damages'] != None else None)
        lists['Non Pecuniary'].append(case['damages']['Non Pecuniary'] if case['damages'] != None else None)
        lists['General'].append(case['damages']['General'] if case['damages'] != None else None)
        lists['Special'].append(case['damages']['Special'] if case['damages'] != None else None)
        lists['Punitive'].append(case['damages']['Punitive'] if case['damages'] != None else None)
        lists['Aggravated'].append(case['damages']['Aggravated'] if case['damages'] != None else None)
        lists['Future Care'].append(case['damages']['Future Care'] if case['damages'] != None else None)
        lists['Judge Name'].append(case['judge'])
        lists['Decision Length'].append(case['decision_length'])
        lists['Multiple defendants?'].append(case['multiple_defendants'])
        lists['Plaintiff Wins?'].append(case['plaintiff_wins'])
        lists['Contributory Negligence Raised'].append(case['contributory_negligence_raised'])
        lists['Written Decision?'].append(case['written_decision'])
        lists['Registry'].append(case['registry'])
        lists['Contributory Negligence Successful'].append(case['contributory_negligence_successful'])
        lists['Percent Reduction'].append(case['percent_reduction'])
        
    df = pd.DataFrame()
    yes_no_columns = ['Written Decision?', 'Multiple defendants?', 'Contributory Negligence Raised', 'Contributory Negligence Successful', 'Plaintiff Wins?']
    for key in lists.keys():
        if key in yes_no_columns:
            new_list = []
            for value in lists[key]:
                if value == True:
                    new_list.append('Y')
                elif value == False:
                    new_list.append('N')
                else:
                    new_list.append(value)
            df[key] = new_list
        else:
            df[key] = lists[key]
    
    return df

def evaluate(dev_data, gold_data, subset=None, focus_column=None):
    '''Evaluates the results against a gold standard set
    
    Arguments:
    dev_data (dataframe) - Dataframe containing results from rule based parse BCJ
    gold_data (dataframe) - Dataframe containing manually annotated data
    [Optional] subset (list/string) - Specific columns to evaluate
    [Optional] focus_column (string) - If set, will drop rows when the focus column is empty
                                       Helpful for evaluating how a specific col performs when it makes a prediction

    Returns: None
    '''
    
    print('#### Evaluation ####')
    
    # Use case name as 'primary key'
    dev_case_names = list(dev_data['Case Name'])
    gold_case_names = list(gold_data['Case Name'])
    
    # Filter data to only use overlapping items
    gold_data = gold_data[gold_data['Case Name'].isin(dev_case_names)]
    dev_data = dev_data[dev_data['Case Name'].isin(gold_case_names)]
    assert gold_data.shape[0] == dev_data.shape[0]

    # If we are focussing on a specific column. We drop results when it is NaN
    # Useful because our model has high precision, low recall. Sometimes want
    # to evaluate how model did when it actually made a prediction (its precision)
    if focus_column:
        dev_data.dropna(subset = [focus_column], inplace=True)
    
    # Mapping from our variable names to Lachlan's column names
    column_mapping = {'Decision Length': 'Decision Length: paragraphs)',
                      'Total Damage': '$ Damages total before contributory negligence',
                      'Non Pecuniary': '$ Non-Pecuniary Damages', 
                      'Total Pecuniary': '$ Pecuniary Damages Total',
                      'Special': '$ Special damages Pecuniary (ie. any expenses already incurred)',
                      'Future Care': 'Future Care Costs (General Damages)',
                      'General': '$ General Damages',
                      'Punitive': '$ Punitive Damages',
                      'Aggravated': '$Aggravated Damages',
                     'Contributory Negligence Successful':'Contributory Negligence Successful?',
                     'Percent Reduction':'% Reduction as a result of contributory negligence'}
    dev_data.rename(columns = column_mapping, inplace = True)
     
    if subset is None: # Use all columns if no subset specified
        subset = dev_data.columns
        
    for column in dev_data.columns:
        if column in gold_data.columns:
            if column in subset:
                errors = [] # Store how wrong our prediction is.
                empty_correct = 0
                non_empty_correct = 0
                total_empty = 0
                total_non_empty = 0
                for case_name in list(dev_data['Case Name']):
                    dev_value = list(dev_data[dev_data['Case Name'] == case_name][column])[0]
                    gold_value = list(gold_data[gold_data['Case Name'] == case_name][column])[0]

                    # Convert string to float if possible
                    try:
                        gold_value = float(gold_value)
                    except:
                        pass

                    try:
                        dev_value = float(dev_value)
                    except:
                        pass
                    # Set values to 'None' if they're a NaN float value
                    dev_value = None if isinstance(dev_value, float) and math.isnan(dev_value) else dev_value
                    gold_value = None if isinstance(gold_value, float) and math.isnan(gold_value) else gold_value
                    # Lowercase values if they're a string
                    dev_value = dev_value.lower().strip() if isinstance(dev_value, str) else dev_value
                    gold_value = gold_value.lower().strip() if isinstance(gold_value, str) else gold_value

                    if gold_value is None:
                        total_empty += 1
                        if dev_value is None:
                            empty_correct += 1
                        elif isinstance(dev_value, float):
                            errors.append(dev_value)
                    else:
                        total_non_empty += 1
                        if isinstance(dev_value, float) and isinstance(gold_value, float):
                            if math.isclose(dev_value, gold_value, abs_tol=1): # Tolerance within 1
                                non_empty_correct += 1
                            else:
                                errors.append(dev_value - gold_value)
                        elif dev_value == gold_value:
                            non_empty_correct += 1
                        
                print('-------')
                print('COLUMN:', column)
                if len(errors) > 0:
                    print('Average distance from correct answer: $' + str(np.mean(errors)))
                if total_empty != 0:
                    print('Empty field accuracy:', empty_correct / total_empty * 100, '%', empty_correct, '/', total_empty)
                if total_non_empty != 0:
                    print('Filled field accuracy:', non_empty_correct / total_non_empty * 100, '%', non_empty_correct, '/', total_non_empty)
                print('Overall accuracy:', (empty_correct+non_empty_correct) / (total_non_empty+total_empty) * 100, '%', (empty_correct+non_empty_correct), '/', (total_non_empty+total_empty))

def plaintiff_wins(case):
    '''Determines whether plaintiff has won the case using a rule based approach

    Arguments:
    case (String) - Case in string format

    Returns: True, False, or "OpenCase"
    '''
    
    plaintiff_dict = {}
    lines = case.strip().split("\n")
    name = lines[0]        
    # regex search for keyword HELD in cases, which determines if case was allowed or dismissed
    HELD = re.search(r'HELD(.+)?', case)
    if HELD:
        matched = HELD.group(0)  
        # regex searching for words such as liablity, liable, negligance, negligant, convicted, convict in matched
        liable = re.search(r'(l|L)iab(.+)?.+|(neglige(.+)?)|(convict(.+)?)', matched)
        # regex searching fot dissmiss/dissmissed/adjourned, negative in matched
        dismiss = re.search(r'(dismiss(.+)?.+)|(adjourned.+?)|(negative(.+)?)', matched)
        # regex searching for damage/Damage/fault/faulty
        damage = re.search(r'(D|d)amage(.+)?.+|(fault(.+)?)', matched)
        if "allowed" in matched or "favour" in matched or "awarded" in matched or "granted" in matched or "accepted" in matched or "entitled" in matched or "guilty" in matched or liable or damage:
            return True

        elif dismiss:
            return False

    else:
        if case and name not in plaintiff_dict :

            last_paras = lines[-5]+" "+lines[-4]+" "+lines[-3]+" "+lines[-2]
            #regex searches for pattern of award ... plaintiff ...
            awarded =  re.search(r'award(.+)?.+?(plaintiff(.+)?)?', last_paras)
            #regex searches for pattern of plaintiff/defendant/applicant....entitled/have...costs
            entiteled = re.search(r'(plaintiff|defendant.?|applicant)(.+)?(entitle(.)?(.+)?|have).+?cost(.+)?', last_paras)
            #regex searches for pattern of successful...(case)
            successful = re.search(r'successful(.+)?.+?', last_paras)
            #regex searches for dismiss....
            dismiss = re.search(r'(dismiss(.+)?.+)|(adjourned.+?)|(negative(.+)?)', last_paras)
            costs = re.search(r'costs.+?(award(.+)?|cause).+?', last_paras)
            damage = re.search(r'(D|d)amage(.+)?.+|(fault(.+)?)', last_paras)

            if dismiss and "not dismissed" not in last_paras:
                return False
            elif damage:
                return True
            elif awarded:
                return True
            elif entiteled:
                return True
            elif successful:
                return True
            elif costs:
                return True
            else:
                return "OpenCase"


def assign_classification_CN(predictions, min_score = 0, min_predict_proba = 0.5):
    '''Given a set of predictions for CN for a single case, function will
    pull out the most probably answer and return it

    Arguments:
    predictions (list) - List of predictions made for the case
    [Optional] min_score (float) - Minimum location ratio of the match. 0 being beginning of case, 1 being end of case. Default 0
    [Optional] min_predict_proba (float) - Minimum confidence in prediction to be assigned. Default 0.5

    Returns:
    best_result (float) - Returns the most probably CN percent reduction. Or None if it can't find one.
    '''
    percent = 0
    temporary_damages = defaultdict(list)
    for value, prediction_type, ratio, predict_proba in predictions:
        if ratio < min_score:
            continue
        
        # not currently handling sub-cnd
        if prediction_type == 'cnp':
            if max(predict_proba) > min_predict_proba: 
                # print(value, max(predict_proba))
                temporary_damages['cnp'].append((value, max(predict_proba)))
        elif prediction_type == 'cnd':
            if max(predict_proba) > min_predict_proba: 
                # print(value, max(predict_proba))
                temporary_damages['cnp'].append((1-value, max(predict_proba)))
    
    # choose most probable value
    best_value = 0
    best_prob = 0
    if len(temporary_damages['cnp']) > 0:
        for pair in temporary_damages['cnp']:
            prob = pair[-1]
            value = pair[0]
            if prob > best_prob:
                best_prob = prob
                best_value = value

#     return percent or None
    if best_value == 0:
        return 
    else:
        return best_value
            
def train_CN_classifier(path, clf = MultinomialNB(), min_para_score = 0, min_predict_proba = 0.5, context_length = 2):
    '''Trains a classifier based on the given training data path
    Arguments:
    path (String) - Path to .txt containing training data
    clf - untrained sklearn classifier, ie MultinomialNB()
    [Optional] min_score (float) - Minimum location ratio of the match. 0 being beginning of case, 1 being end of case. Default 0
    [Optional] min_predict_proba (float) - Minimum confidence in prediction to be assigned. Default 0.5
    [Optional] context_length (int) - Number of words around a percent to include in the features
    
    Returns:
    model (sklearn model) - Trained model
    vectorizer (sklearn DictVectorizer) - fit-transformed vectorizer
    case_percents (dict) - Dictionary mapping annotated percentages to their cross-validated predictions
    '''

    tag_extractor = re.compile('''<damage type ?= ?['"](.*?)['"]> ?(\$?.*?) ?<\/damage>''')
    CN_tag_extractor = re.compile('''<percentage type ?= ?['"](.*?)['"]> ?(\$?.*?) ?<\/percentage>''')
    stop_words = set(stopwords.words('english'))

    with open(path, encoding='utf-8') as document:
        document_data = document.read()
    document_data = document_data.split('End of Document\n')
    examples_per_case = [] # Each element contains all examples in a case
    answers_per_case = [] # Each element contains all answers in a case 
    case_titles = []
    num_cases = len(document_data)
    for i in range(len(document_data)):
        print('Reading training data and extracting features...', i / num_cases * 100, '%', end='\r')
        
        case = document_data[i]
        case = case.strip() # Make sure to strip!
        if len(case) == 0: # Skip empty lines
            continue
        lines = case.split('\n')
        case_title = lines[0]
        try:
            case_type = lines[1]
        except:
            print(case)

        case_examples = []
        case_answers = []
        if filter_unwanted_cases(case, case_title, case_type):

            # plaintiff/defendant entities
            plaintiff_defendant_pattern = r'(^[a-z].*)+ v\. ([a-z|-]+)+.*\n?(?:British Columbia Judgments)' # group 1 is plaintiff group 2 is defendant
            entity_match = re.search(plaintiff_defendant_pattern, "\n".join(case.split('\n')[:3]), re.IGNORECASE)
            if entity_match:
                plaitiff_defendant = entity_match.groups() # tuple (plaintiff, defendant)
            else:
                plaitiff_defendant = ('Plaintiff', 'Defendant')
            plaintiff_split = [word.lower() for word in plaitiff_defendant[0].split()]
            defendant_split = [word.lower() for word in plaitiff_defendant[1].split()]

            # lower case and remove stopwords
            case = ' '.join([word for word in case.lower().split() if word not in stop_words])
            summary, summary_start_idx, summary_end_idx = summary_tokenize(case)

            matches = CN_tag_extractor.finditer(case) # Extract all <damage ...>$x</damage> tags used for training
            for match in matches:
                features, answer = extract_CN_features(match, case, tag_extractor, CN_tag_extractor, plaintiff_split, defendant_split, context_length = context_length)
                if summary:
                    if match.start() >= summary_start_idx and match.end() <= summary_end_idx and answer != 'other':
                        features['start_idx_ratio'] = 1
                
                case_examples.append(features)
                case_answers.append(answer)
        if len(case_examples) > 0 and len(case_answers) > 0:
            examples_per_case.append(case_examples)
            answers_per_case.append(case_answers)
            case_titles.append(case_title)
        else:
            print('Didnt find any tags in', case_title)
    assert len(examples_per_case) == len(answers_per_case)

    feats = list(chain.from_iterable(examples_per_case)) # Puts it into one big list

    values = [feat['float'] for feat in feats]
    value_locations = [feat['start_idx_ratio'] for feat in feats]

    # Delete start_idx_ratio feature before training classifier
    for feat in feats:
        del feat['start_idx_ratio']
    assert 'start_idx_ratio' not in feats[0]
    
    print('\nVectorizing...')
    vectorizer= DictVectorizer()
    X = vectorizer.fit_transform(feats)
    y = list(chain.from_iterable(answers_per_case))

    print('Tag Distribution')
    dist = Counter(y)
    print(dist)

    
    y_pred = cross_val_predict(clf, X, y, cv = 10) 
    y_prob = cross_val_predict(clf, X, y, cv = 10, method='predict_proba')
    
    values_per_case = [len(vals) for vals in examples_per_case] #number of values in each case
    
    # for all cases in our annotations, get separated value, prediction, location, prob 
    prediction_features = list(zip(values, y_pred, value_locations, y_prob))
    prediction_feats_per_case = []
    number_visited = 0
    for i in range(len(values_per_case)):
        if i == 0:
            prediction_feats_per_case.append(prediction_features[:values_per_case[i]]) 
        elif i < len(values_per_case)-1:
            prediction_feats_per_case.append(prediction_features[number_visited:number_visited + values_per_case[i]])
        else:
            prediction_feats_per_case.append(prediction_features[-values_per_case[i]:])
        number_visited += values_per_case[i]
        assert len(prediction_feats_per_case[i]) == values_per_case[i]
    assert sum([len(feats) for feats in prediction_feats_per_case]) == sum(values_per_case)
    assert prediction_feats_per_case[-1][-1] == prediction_features[-1]
    assert len(case_titles) == len(prediction_feats_per_case)
    
    # assign percents to case
    case_percents = defaultdict(float)
    for i in range(len(prediction_feats_per_case)):
        case_preds = prediction_feats_per_case[i]
        percent = assign_classification_CN(case_preds, min_score = min_para_score, min_predict_proba = min_predict_proba)
        case_percents[case_titles[i]] = percent

    print('Cross validation evaluation...')
    print(classification_report(y, y_pred))

    print('Training final model...')
    clf.fit(X, y)
    return clf, vectorizer, case_percents #feats_train, feats_test, y_train, y_test #clf, vectorizer


def extract_CN_features(match, case, dmg_pattern, cn_pattern = None, plaintiff_split = ['plaintiff'], defendant_split = ['defendant'], context_length = 2, purpose = 'train'):
    '''Given a match will return the features associated with the specific example
    Extracts the examples by finding the percent annotation tags
    in the form <percentage type = "TYPE">50 per cent</percentage>
    Arguments:
    match (Match Object) - Match object with the type as group 1 and value as group 2 if purpose = train, otherwise match group 0 is the value
    case (str) - The case data in string format
    dmg_pattern (str, regex pattern) - The regex pattern being used to find damages.
                                      Used to remove the tags in features using context around value.
    cn_pattern (str, regex pattern) - The regex pattern being used to find percentages.
    [Optional] context_length (int) - The number of words to use around the value for context
    [Optional] purpose (str) - Default is 'train', used to determine pattern type
    Returns:
    features (dict) - Dictionary containing each feature for the current match
    damage_type (str) or None - The type of damage associated with the value if purpose = 'train'
    '''
    features = dict()
    if purpose == 'train':
        percent_type = match.group(1).strip()
        percent_value = match.group(2).strip()
    else:
        percent_type = None
        percent_value = match.group(0).strip()
    start_idx = match.start()
    end_idx = match.end()
    # Get 10 + Context Length on each side 
    # Used to get rid of damage tags within context around our match
    # We want to avoid getting half a damage/percentage tag else it wont be removed
    # So we get more than we need.
    start_tokenized = ' '.join(case[:start_idx].split()[-context_length-10:])
    end_tokenized = ' '.join(case[end_idx:].split()[:context_length+10])

    #lexicons
    reduce_words = ['reduce', 'liable', 'liability', 'fault', 'responsible', 'against', 'less', 'failure']
    other_words = ['apportion', 'recover', 'contributor']

    if purpose == 'train':
        if cn_pattern is None:
            print('Error: Didnt include percentage regex')
            return None
        # Remove damage tags AND percentage in context around match
        start_matches_dmg = dmg_pattern.finditer(start_tokenized)
        for s_dmg in start_matches_dmg:
            start_tokenized = start_tokenized.replace(s_dmg.group(0), s_dmg.group(2))
        start_matches_cn = cn_pattern.finditer(start_tokenized)
        for s_cn in start_matches_cn:
            start_tokenized = start_tokenized.replace(s_cn.group(0), s_cn.group(2))

        end_matches_dmg = dmg_pattern.finditer(end_tokenized)
        for e_dmg in end_matches_dmg:
            end_tokenized = end_tokenized.replace(e_dmg.group(0), e_dmg.group(2))
        end_matches_cn = cn_pattern.finditer(end_tokenized)
        for e_cn in end_matches_cn:
            end_tokenized = end_tokenized.replace(e_cn.group(0), e_cn.group(2))
    # Reconstruct sentence
    start_tokenized = start_tokenized.split()[-context_length:]
    end_tokenized = end_tokenized.split()[:context_length]
    tokens = ' '.join(start_tokenized) + " " + percent_value + " " + ' '.join(end_tokenized)
    value_start_idx = len(start_tokenized) # Location of value in relation to sentence (token level)
    if len(percent_value.split()) > 1: # Deals with problems like '2 million' (where value is multiple tokens)
        value_end_idx = value_start_idx + len(percent_value.split()) - 1
    else:
        value_end_idx = value_start_idx

    tokens = tokens.replace('per cent', 'percent').split()

    # Features: BOW before, BOW after, BOW, contributory negligence in text, plaintiff/defendant name matches, value, location
    start_boundary = value_start_idx - context_length if value_start_idx - context_length >= 0 else 0
    end_boundary = value_end_idx + context_length + 1 if value_end_idx + context_length + 1 < len(tokens) else len(tokens)

    # remove alpha-numerics for BOW and substitute plaintiff/defendant names
    tokens = re.sub(r'[\W0-9]', ' ', " ".join(tokens))
    tokens = re.sub('%s|%s'%(re.escape(plaintiff_split[0]), re.escape(plaintiff_split[-1])), ' plaintiff ', tokens)
    tokens = re.sub('%s|%s'%(re.escape(defendant_split[0]), re.escape(defendant_split[-1])), ' defendant ', tokens).split()

    before = tokens[start_boundary : value_start_idx]
    after = tokens[value_end_idx + 1 : end_boundary]

    #BOW features - befor and after value
    features_bow_b = dict(Counter(before))
    features_bow_b = {k+'@Before': v for k, v in features_bow_b.items()}
    features_bow_a = dict(Counter(after))
    features_bow_a = {k+'@After': v for k, v in features_bow_a.items()}
    features.update(Counter(tokens))

    # features['contributory_negligence'] = True if 'contributory negligence' in case.lower() else False
    
    # plaintiff/defendant entities
    plaintiff_entities = ['plaintiff'] + plaintiff_split
    defendant_entities = ['defendant'] + defendant_split

    features['plaintiff_mentioned'] = True if any('plaintiff' in item for item in tokens) else False
    features['defendant_mentioned'] = True if any('defendant' in item for item in tokens) else False
    # features['value'] = percent_value
    features['start_idx_ratio'] = match.start()/len(case)
    features['reduce_lexicon'] = any(item in reduce_word for item in tokens for reduce_word in reduce_words)
    features['defendant and reduction'] = features['defendant_mentioned'] and features['reduce_lexicon']
    features['plaintiff and reduction'] = features['plaintiff_mentioned'] and features['reduce_lexicon']
    features['CN_lexicon'] =  any(item in other_word for item in tokens for other_word in other_words)
    features['plaintiff_name'] = plaintiff_split[0] in " ".join(tokens)
    features['defendant_name'] = defendant_split[0] in " ".join(tokens)

    # add float feature
    if '/' in percent_value: #to convert factions to float such as 2/3 percent
        features['float'] = float(percent_value.split()[0][0])/float(percent_value.split()[0].strip('%')[-1])
    elif not any(char.isdigit() for char in percent_value): # handles rare case of 'ten percent' - cant reasonable handle these
        features['float'] = False
    else:
        features['float'] = float(percent_value.split()[0].strip('%'))*0.01

    # Percent "bins"
    if features['float']:
        if features['float'] < 0.1:
            features['range'] = '< 10%'
        elif features['float'] < 0.5:
            features['range'] = '10% - 50%'
        elif features['float'] < 0.75:
            features['range'] = '50%-75%'
        else:
            features['range'] = '75%+' 

    if purpose == 'train':
        return features, percent_type
    else:
        return features, percent_value

    