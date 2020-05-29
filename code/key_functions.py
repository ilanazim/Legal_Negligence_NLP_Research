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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report
import math

import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords
from itertools import chain

def rule_based_parse_BCJ(path, damage_model = None, damage_vectorizer = None):
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

        if filter_unwanted_cases(case, case_title, case_type):
            # Fields that can be found via pattern matching
            if re.search('contributory negligence', case, re.IGNORECASE):
                contributory_negligence_raised = True
            else:
                contributory_negligence_raised = False
            case_number = re.search(r'\/P([0-9]+)\.txt', path).group(1)
            decision_len = re.search(r'\(([0-9]+) paras\.?\)', case) # e.g.) (100 paras.)
            registry = re.search(r'(Registry|Registries): ?([A-Za-z0-9 ]+)', case) # e.g.) Registry: Vancouver
            written_decision = True if int(decision_len.group(1)) > 1 else False
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
            # Fields that are always in the same place
            judge_name = lines[4].strip()
            case_title = lines[0].strip()
            # Extract year from case_title (in case we want to make visualizations, etc.)
            year = re.search(r'20[0-2][0-9]', case_title) # Limit regex to be from 2000 to 2029
            if year:
                year = year.group(0)
            else:
                # Rare case: Sometimes the title is too long. Rely on Heard date.
                year = re.search(r'Heard:.* ([2][0][0-2][0-9])', case)
                if year:
                    year = year.group(1)
                else:
                    print('WARNING: Year not found')
            case_dict['case_number'] = '%s of %s'%(i+1+((int(case_number)-1)*50), case_number)
            case_dict['case_title'] = case_title
            case_dict['year'] = year
            case_dict['registry'] = registry
            case_dict['judge'] = judge_name
            case_dict['decision_length'] = decision_len.group(1)
            case_dict['multiple_defendants'] = rule_based_multiple_defendants_parse(case)
            case_dict['contributory_negligence_raised'] = contributory_negligence_raised
            case_dict['written_decision'] = written_decision
            
            case_dict['plaintiff_wins'] = plaintiff_wins(case)
                        
            if damage_model and damage_vectorizer:
                predictions = predict(case, damage_model, damage_vectorizer)
                case_dict['damages'] = assign_classification_damages(predictions)
            else:
                case_dict['damages'] = rule_based_damage_extraction(case)
            
            
            percent_reduction, contributory_negligence_successful = get_percent_reduction_and_contributory_negligence_success(case_dict, case)
            case_dict['percent_reduction'] = percent_reduction
            case_dict['contributory_negligence_successful'] = contributory_negligence_successful
             
        
        # don't add empty dictionaries (non BCJ cases) to list
        if case_dict != dict(): 
            case_parsed_data.append(case_dict)
    return case_parsed_data

def rule_based_multiple_defendants_parse(doc):
    ''' Helper function for rule_based_parse_BCJ
    
    Given a case. Uses regex/pattern-matching to determine whether we have multiple defendants.
    For the most part the logic relies on whether the langauge used implies plurality or not.
    
    Arguments: doc (String): The case in text format following the form used in the DOCX to TXT notebook
    Returns: response (String, 'Y', 'N', or 'UNK')
    '''

    # Case 1)
    # Traditional/most common. Of form "Between A, B, C, Plaintiff(s), X, Y, Z Defendant(s)"
    # Will also allow "IN THE MATTER OF ... Plaintiff .... Defendant..."
    # Can successfully cover ~98% of data
    regex_between_plaintiff_claimant = re.search(r'([Between|IN THE MATTER OF].*([P|p]laintiff[s]?|[C|c]laimant[s]?|[A|a]ppellant[s]?|[P|p]etitioner[s]?|[R|r]espondent[s]?).*([D|d]efendant[s]?|[R|r]espondent[s]?|[A|a]pplicant[s]?).*\n)', doc)
    
    # Match found
    if regex_between_plaintiff_claimant:
        text = regex_between_plaintiff_claimant.group(0).lower()
        if 'defendants' in text or 'respondents' in text or 'applicants' in text: # Defendant/respondent same thing.
            return 'Y'
        elif 'defendant' in text or 'respondent' in text or 'applicant' in text:
            return 'N'
    
    # If not found, try other less common cases
    else:
        # Case 2)
        # Sometimes it does not mention the name of the second item. (Defendent/Respondent)
        # We can estimate if there are multiple based on the number of "," in the line (Covers all cases in initial data)
        regex_missing_defendent = re.search(r'(Between.*([P|p]laintiff[s]?|[C|c]laimant[s]?|[A|a]ppellant[s]?|[P|p]etitioner[s]?).*\n)', doc)
        if regex_missing_defendent:
            text = regex_missing_defendent.group(0).lower()
            if len(text.split(',')) > 5:
                return 'Y'
            else:
                return 'N'
            
        else:
            print('Multiple defendants: Unknown! Unable to regex match')
            return 'UNK'
        
def rule_based_damage_extraction(doc, min_score = 0.9, max_match_len_split = 10):
    '''Helper function for rule_based_parse_BCJ
    
    Given a case, attempts to extract damages using regex patterns
    
    Arguments: doc (String): The case in text format following the form used in the DOCX to TXT notebook
    min_score (float): The minimum paragraph score to consider having a valid $ number
                       Paragraph has score 1 if its the last paragraph
                       Paragraph has score 0 if its the first paragraph
    max_match_len_split (int): The max amount of items that can appear in a regex match after splitting (no. words)
    
    Returns: damages (Dict): Contains any found damages
    
    '''
    damages = defaultdict(float)
    repetition_detection = defaultdict(set) # try to stem the repeated values
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
    #regex_damages = r'(?:[\w|-]* ?){0,3}(?:damage|loss|capacity|cost).+?\$? ?[0-9][0-9|,|.]+[0-9]'
    #regex_in_trust = r'(?:in-?trust|award).*?\$? ?[0-9][0-9|,|.]+[0-9]'
    #regex_damages = r'(?![and])(?:[\w|-]* ?){0,2} ?(?:damage|loss|capacity|cost).+?\$? ?[0-9][0-9|,|.]+[0-9]'
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
    
    # Get money mounts from the text
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

    # Extract $ value. Determine correct column
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
            value_mapped = assign_damage_to_category(extracted_value, non_pecuniary_damage_keywords, match, score, matches, 'Non-pecuniary', damages, repetition_detection, repetition_key = ('non','pecuniary'))
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
                            damages['Total'] = damages['Pecuniary Total'] + damages['Non-pecuniary']
                            if damages['Total'] == 0:
                                total = extracted_value
                                repetition_detection[('total',)].add(extracted_value)
                        
    damages['Pecuniary Total'] = damages['Special'] + damages['General'] + damages['Punitive'] + damages['Aggravated'] + damages['Future Care']
    damages['Total'] = damages['Pecuniary Total'] + damages['Non-pecuniary']
    
    if damages['Total'] == 0 and total is not None: # Only use the "total" if we couldnt find anything else!
        damages['Total'] = total
        damages['General'] = total
        
    columns = ['Total', 'Pecuniary Total', 'Non-pecuniary', 'Special', 'General', 'Punitive', 'Aggravated', 'Future Care']
    for c in columns:
        damages[c] = None if damages[c] == 0 else damages[c]
    
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
    '''Helper function for rule based damage extraction.
    
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
    
    Removes crown cases 'R.v.'
    Removes '(Re)' cases
    Removes client-solicitor cases
    Removes IN THE MATTER OF cases where plaintiff/defendant is not mentioned
    Removes non 'British Columbia Judgments' cases
    
    Arguments:
    case (string) - Case data in string form
    case_title (string) - Case title (line 1 of case)
    case_type (string) - Case type (line 2 of case)
    
    Returns:
    boolean - True if case should be analyzed. False if it should be skipped.
    '''
    
    if 'R. v.' in case_title or '(Re)' in case_title: # Skip crown cases, Skip (Re) cases
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
    ''' Takes string input the of wntire document (case) and returns list of lists of paragraphs in the document.
    ---------
    Input: case (str) - string of single legal case
    Return: case_data(list) - list of of numbrered paragraphs in the document where the first item is the case_title'''
    
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
    ---------
    Input: case (str) - string of single legal case
    Return: summary - summary and HELD section of case (str)'''
    
    # split paragraphs on newline, paragraph number, two spaces
    summary = re.search(r'\([0-9]{1,3} paras\.\)\ncase summary\n((.*\n+?)+)(?=HELD|(Statutes, Regulations and Rules Cited:)|(Counsel\n))', case, re.IGNORECASE)
    if summary:
        summary = summary.group(1)
    else:
        return None

    return summary

def get_context_and_float(value, text, context_length = 8, plaintiff_name = 'Plaintiff', defendant_name = 'Defendant'):
    '''Given a string value found in a body of text, 
    return its context, and its float equivalent.
    -----------------
    Arguments:
    value - percent match found in text
    text - string value where matches were extracted from, eg paragraph or summary (str)
    context_length - the length of context around each quantity to return
    Rerturn:
    value_context - string of context around value (str)
    extracted_value - string quantity value extracted to its float equivalent'''
    
    
    # get context for monetary/percent values 
    context = ''
    amount = re.findall(r'[0-9]+[0-9|,]*(?:\.[0-9]+)?', value)
    extracted_value = clean_money_amount(amount) #use helper function to get float of dollar/percent value
    if not extracted_value:
        print('ERROR: cant convert string, %s'%value)
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
    Arugments:
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

def get_percent_reduction_and_contributory_negligence_success(case_dict, case, min_score = 0.9):
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
        
        if best_score == 0 or not best_percent or not contributory_negligence_successful:
            # no percents found in paragraphs - time to check summary - same process
            summary = summary_tokenize(case)
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

def train_classifier(path, clf = MultinomialNB()):
    '''Trains a classifier based on the given training data path
    
    Arguments:
    path (String) - Path to .txt containing training data
    clf - untrained sklearn classifier, ie MultinomialNB()
    
    Returns:
    model (sklearn model) - Trained model
    vectorizer (sklearn DictVectorizer) - fit-transformed vectorizer
    '''
    tag_extractor = re.compile('''<damage type ?= ?['"](.*?)['"]> ?(\$?.*?) ?<\/damage>''')
    stop_words = set(stopwords.words('english'))
    
    with open(path, encoding='utf-8') as document:
        document_data = document.read()
        
    document_data = document_data.split('End of Document\n') # Always split on 'End of Document\n'
    
    examples_per_case = [] # Each element contains all examples in a case
    answers_per_case = [] # Each element contains all answers in a case 
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
            
            matches = tag_extractor.finditer(case) # Extract all <damage ...>$x</damage> tags used for training
            for match in matches:
                features, answer = extract_features(match, case, tag_extractor)
                # if value is found in case summary, replace start_idx_ratio with 1
                if summary:
                    if match.group(0) in summary.lower():
                        features['start_idx_ratio'] = 1
                    
                case_examples.append(features)
                case_answers.append(answer)
                
        if len(case_examples) > 0 and len(case_answers) > 0:
            examples_per_case.append(case_examples)
            answers_per_case.append(case_answers)
        else:
            print('Didnt find any tags in', case_title)
                    
    print('\nVectorizing...')    
    vectorizer = DictVectorizer()
    feats = list(chain.from_iterable(examples_per_case)) # Puts it into one big list
    X = vectorizer.fit_transform(feats)
    y = list(chain.from_iterable(answers_per_case))
    
    print('Tag Distribution')
    dist = Counter(y)
    print(dist)
    
    print('Cross validation evaluation...')
    print('Scores (F1-MACRO):', np.mean(cross_val_score(clf, X, y, cv = 5, scoring = 'f1_macro')))
    print('Scores (F1-MICRO):', np.mean(cross_val_score(clf, X, y, cv = 5, scoring = 'f1_micro')))
    print('Scores (F1-WEIGHTED):', np.mean(cross_val_score(clf, X, y, cv = 5, scoring = 'f1_weighted')))
    
    print('Training final model...')
    clf.fit(X, y)
    return clf, vectorizer

def extract_features(match, case, pattern, context_length = 5, purpose = 'train'):
    '''Given a match will return the features associated with the specific example
    Extracts the examples by finding the damage annotation tags
    in the form <damage type = "TYPE">$5000</damage>
    
    Arguments:
    match (Match Object) - Match object with the type as group 1 and value as group 2 if purpose = train, otherwise match group 0 is the value
    case (str) - The case data in string format
    pattern (str, regex pattern) - The regex pattern being used to find damages.
                                      Used to remove the tags in features using context around value.
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
    
    # Get 3 * Context Length on each side 
    # Used to get rid of damage tags within context around our match
    # We want to avoid getting half a damage tag else it wont be removed
    # So we get more than we need.
    start_tokenized = ' '.join(case[:start_idx].split()[context_length*3:])
    end_tokenized = ' '.join(case[end_idx:].split()[:context_length*3])

    if purpose == 'train':
        # Remove damage tags in context around match
        start_matches = pattern.finditer(start_tokenized)
        for s in start_matches:
            start_tokenized = start_tokenized.replace(s.group(0), s.group(2))
        end_matches = pattern.finditer(end_tokenized)
        for e in end_matches:
            end_tokenized = end_tokenized.replace(e.group(0), e.group(2))

    # Reconstruct sentence
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
    
    # next and previous word
    if len(before) > 0:
        features['prev word'] = before[-1]
    else:
        features['prev word'] = ''
    if len(after) > 0:
        features['next word'] = after[0]
    else:
        features['next word'] = ''
    
    # BOW features
    features_bow_b = dict(Counter(before))
    features_bow_b = {k+'@Before': v for k, v in features_bow_b.items()}
    features_bow_a = dict(Counter(after))
    features_bow_a = {k+'@After': v for k, v in features_bow_a.items()}
    features.update(features_bow_b)
    features.update(features_bow_a)
    features.update(Counter(context.split()))
    
    return features, damage_type

def predict(case, clf, vectorizer):
    '''Given a legal negligence case (str), a trained classifier, and a fit_transformed DictVectorizer(), 
    Return a list of tuples of (value, prediction, value_location), where value_location is the ratio of the 
    character start index 
    ----------------------
    Arguments:
    case: legal negligence case (str)
    clf: trained classifier with .fit method
    vecotrizer: fit_transformed vectorizer (sklean DictVectorizer())
    ----------------------
    Return: list of tuples or an empty list if no matches in the case
    Example: 
    case = 'I award $5,000 in punitive damages.'
    predict(case, clf, vectorizer)
    > [($5,000, 'punitive', 0.023)]'''
    
    stop_words = set(stopwords.words('english'))
    value_extractor = re.compile('''\$ ?[1-9]+[0-9|,|\.]+''')
    case_examples = []
    value_locations = []
    values = []

    # lower case and remove stopwords
    case = ' '.join([word for word in case.lower().split() if word not in stop_words])
    matches = value_extractor.finditer(case) # Extract all <damage ...>$x</damage> tags used for training
    for match in matches:
        # extract features per match found
        features, _ = extract_features(match, case, value_extractor, purpose = 'predict')
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
    
def assign_classification_damages(predictions, min_score = 0.7):
    '''Helper function for rule based BCJ
    Handles assigning predictions into final damage amounts
    
    Arguments:
    predictions (tuple returned from predict function)
    min_score (float) - If a prediction appears before this point in the case it is discarded
    
    Returns:
    damages (defaultdict(float)) - Damages with values filled in based on predictions
    '''
    
    damages = defaultdict(float)
    temporary_damages = defaultdict(list)
    for value, prediction_type, ratio, predict_proba in predictions:
        if ratio < min_score:
            continue
        
        if prediction_type == 'total':
            if max(predict_proba) > 0.8: # Max will be 'total' since it is the prediction_type
                temporary_damages[prediction_type].append(value)
        else:
            temporary_damages[prediction_type].append(value)

    # Currently not dealing with "reduction" or "total after" (or total - manually adding)
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
                         else None
    
    damages['General'] += damages['Future Wage Loss']
    damages['Special'] += damages['Past Wage Loss'] + damages['In Trust']

    damages['Pecuniary Total'] = damages['Special'] + damages['General'] + damages['Future Care']
    
    # Only sum up damages if we weren't able to properly extract it from the text.
#     if damages['Total'] is None:
#         damages['Total'] = damages['Pecuniary Total'] + damages['Non Pecuniary'] + damages['Aggravated']
    
    columns = ['Total', 'Pecuniary Total', 'Non Pecuniary', 'Special', 'General', 'Punitive', 'Aggravated', 'Future Care']
    for c in columns:
        damages[c] = None if damages[c] == 0 else damages[c]
    
    return damages     

def rule_based_convert_cases_to_DF(cases):
    '''Given a list of parsed cases returns a dataframe'''

    lists = defaultdict(list)    
    for case in cases:
        lists['Case Number'].append(case['case_number'])
        lists['Case Name'].append(case['case_title'])
        lists['Year'].append(case['year'])
        lists['Total Damage'].append(case['damages']['Total'] if case['damages'] != None else None)
        lists['Total Pecuniary'].append(case['damages']['Pecuniary Total'] if case['damages'] != None else None)
        lists['Non Pecuniary'].append(case['damages']['Non-pecuniary'] if case['damages'] != None else None)
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
    for key in lists.keys():
        df[key] = lists[key]
    
    return df

def evaluate(dev_data, gold_data, subset=None):
    '''Evaluates the results against a gold standard set
    
    Arguments:
    dev_data (dataframe) - Dataframe containing results from rule based parse BCJ
    gold_data (dataframe) - Dataframe containing manually annotated data
    (Optional) subset (list/string) - Specific columns to evaluate

    '''
    
    print('#### Evaluation ####')
    
    # Use case name as 'primary key'
    dev_case_names = list(dev_data['Case Name'])
    gold_case_names = list(gold_data['Case Name'])
    
    # Filter data to only use overlapping items
    gold_data = gold_data[gold_data['Case Name'].isin(dev_case_names)]
    dev_data = dev_data[dev_data['Case Name'].isin(gold_case_names)]
    
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
                                errors.append(abs(dev_value - gold_value))
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

def plaintiff_wins(line):
    '''This function will search the cases and returns a dictionary
    with case names as keys and Y or N for value, Y if the plaintiff
    wins the case and N if plaintiff looses'''
    
    plaintiff_dict = {}
    lines = line.strip().split("\n")
    name = lines[0]        
    #check if it's a British columbia case    
    if "B.C.J" in name:
#         #check if it's not a crown case    
#         if 'R. v.' in name or '(Re)' in name:
#             continue
            # regex search for keyword HELD in cases, which determines if case was allowed or dismissed
        HELD = re.search(r'HELD(.+)?', line)
        if HELD:
            matched = HELD.group(0)  
            # regex searching for words such as liablity, liable, negligance, negligant, convicted, convict in matched
            liable = re.search(r'(l|L)iab(.+)?.+|(neglige(.+)?)|(convict(.+)?)', matched)
            # regex searching fot dissmiss/dissmissed/adjourned, negative in matched
            dismiss = re.search(r'(dismiss(.+)?.+)|(adjourned.+?)|(negative(.+)?)', matched)
            # regex searching for damage/Damage/fault/faulty
            damage = re.search(r'(D|d)amage(.+)?.+|(fault(.+)?)', matched)
            if "allowed" in matched or "favour" in matched or "awarded" in matched or "granted" in matched or "accepted" in matched or "entitled" in matched or "guilty" in matched or liable or damage:
                return "Y"

            elif dismiss:
                return "N"

        else:
            if line and name not in plaintiff_dict :

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
                    return "N"
                elif damage:
                    return "Y"
                elif awarded:
                    return "Y"
                elif entiteled:
                    return "Y"
                elif successful:
                    return "Y"
                elif costs:
                    return "Y"
                else:
                    return "OpenCase"


