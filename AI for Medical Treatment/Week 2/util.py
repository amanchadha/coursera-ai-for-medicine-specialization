### MOVE TO UTIL
import urllib
import os
import re
import sklearn.metrics as metrics
import numpy as np
import stanfordnlp
import pandas as pd
from bllipparser import RerankingParser
from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget
import svgling
import pickle
from negbio.pipeline import text2bioc
import bioc
import itertools
from textblob import TextBlob
from tqdm import tqdm_notebook
import os
import bioc
import tqdm
from pathlib2 import Path
from negbio.chexpert.stages.aggregate import NegBioAggregator
from negbio.chexpert.stages.classify import ModifiedDetector, CATEGORIES
from negbio.chexpert.stages.extract import NegBioExtractor
from negbio.chexpert.stages.load import NegBioLoader
from negbio.pipeline import text2bioc, negdetect
from negbio.pipeline.parse import NegBioParser
from negbio.pipeline.ptb2ud import NegBioPtb2DepConverter, Lemmatizer
from negbio.pipeline.ssplit import NegBioSSplitter
from negbio.main_chexpert import pipeline

PARSING_MODEL_DIR = "~/.local/share/bllipparser/GENIA+PubMed"
CHEXPERT_PATH = "NegBio/negbio/chexpert/"
MENTION_PATH =f"{CHEXPERT_PATH}phrases/mention"
UNMENTION_PATH = f"{CHEXPERT_PATH}phrases/"
NEG_PATH = f'{CHEXPERT_PATH}patterns/negation.txt'
PRE_NEG_PATH = f'{CHEXPERT_PATH}patterns/pre_negation_uncertainty.txt'
POST_NEG_PATH = f'{CHEXPERT_PATH}patterns/post_negation_uncertainty.txt'

PHRASES_PATH = f"{CHEXPERT_PATH}phrases/"
TEST_PATH = "stanford_report_test.csv"
test_df = pd.read_csv(TEST_PATH)
CATEGORIES = ["Cardiomegaly",
              "Lung Lesion", "Airspace Opacity", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture"]

test_df = test_df[['Report Impression'] + CATEGORIES]
test_df = test_df.replace(1, True).fillna(False).replace(0, False).replace(-1, False)

def get_dict(path):
    label_to_mention = {}
    mention_files = os.listdir(path)

    for f in mention_files:
        with open(os.path.join(path, f)) as mention_file:
            condition = os.path.basename(f)[:-4]
            condition = condition.replace("_", " ").title()
            if condition not in label_to_mention:
                label_to_mention[condition] = []

            for line in mention_file:
                label_to_mention[condition].append(line.split("\n")[0])
    return label_to_mention

mentions = get_dict(PHRASES_PATH + "mention")
unmentions = get_dict(PHRASES_PATH + "unmention")


mentions_pk = "mentions.pkl"
unmentions_pk = "unmentions.pkl"
pickle.dump(mentions, open(mentions_pk, "wb"))
pickle.dump(unmentions, open(unmentions_pk, "wb"))

mentions = pickle.load(open(mentions_pk, "rb"))
unmentions = pickle.load(open(unmentions_pk, "rb"))


## MOVE TO UTIL
def get_mention_keywords(observation):
    if observation in mentions:
        return mentions[observation]
    else:
        return []
    

chexpert_results_mention = {
    'No Finding': 0.769,
    'Lung Lesion': 0.896,
    'Fracture': 0.975,
    'Pleural Other': 0.850,
    'Pleural Effusion': 0.985,
    'Pneumonia': 0.660,
    'Pneumothorax': 1.000,
    'Lung Opacity': 0.966,
    'Edema': 0.996,
    'Support Devices': 0.933,
    'Atelectasis': 0.998,
    'Enlarged Cardiomediastinum': 0.935,
    'Cardiomegaly': 0.973,
    'Consolidation': 0.999
}

chexpert_results_unmention = {
    'No Finding': float("nan"),
    'Lung Lesion': 0.900,
    'Fracture': 0.807,
    'Pleural Other': 1.00,
    'Pleural Effusion': 0.971,
    'Pneumonia': 0.750,
    'Pneumothorax': 0.977,
    'Lung Opacity': 0.914,
    'Edema': 0.962,
    'Support Devices': 0.720,
    'Atelectasis': 0.833,
    'Enlarged Cardiomediastinum': 0.959,
    'Cardiomegaly': 0.909,
    'Consolidation': 0.981
}


## MOVE TO UTIL
def get_bioc_collection(df):
    collection = bioc.BioCCollection()
    splitter = NegBioSSplitter()
    for i, report in enumerate(df["Report Impression"]):
        document = text2bioc.text2document(str(i), report)
        document = splitter.split_doc(document)
        collection.add_document(document)
    return collection


def clean(sentence):
    """Clean the text."""
    punctuation_spacer = str.maketrans({key: f"{key} " for key in ".,"})
    
    lower_sentence = sentence.lower()
    # Change `and/or` to `or`.
    corrected_sentence = re.sub('and/or',
                              'or',
                              lower_sentence)
    # Change any `XXX/YYY` to `XXX or YYY`.
    corrected_sentence = re.sub('(?<=[a-zA-Z])/(?=[a-zA-Z])',
                              ' or ',
                              corrected_sentence)
    # Clean double periods
    clean_sentence = corrected_sentence.replace("..", ".")
    # Insert space after commas and periods.
    clean_sentence = clean_sentence.translate(punctuation_spacer)
    # Convert any multi white spaces to single white spaces.
    clean_sentence = ' '.join(clean_sentence.split())

    return clean_sentence

def calculate_f1(df, pred_frame):
    # calculate F1
    results = pd.DataFrame()
    for cat in CATEGORIES:
        gt = df[cat]
        pred = pred_frame[cat]
        f1 = metrics.f1_score(gt, pred)
        results = results.append({ "Label": cat, "F1": round(f1, 3)}, ignore_index=True)
    results = results.append({ "Label": "Average", "F1": round(results["F1"].mean(), 3)}, ignore_index=True)
    return results[["Label", "F1"]]

def get_preds(classfication_function, df, cleanup=False):
    # generate labels
    collection = get_bioc_collection(df)
    docs = collection.documents
    pred_frame = pd.DataFrame()
    for i, doc in enumerate(docs):
        sentences_array = [s.text for s in doc.passages[0].sentences]
        if cleanup:
            sentences_array = [clean(s) for s in sentences_array]
        preds = classfication_function(sentences_array)
        pred_frame = pred_frame.append(preds, ignore_index=True)
    
    return pred_frame.astype("bool")

def get_f1_table(classfication_function, df, cleanup=False):
    pred_frame = get_preds(classfication_function, df, cleanup=cleanup)
    
    # calculate F1
    return calculate_f1(df, pred_frame)

def get_negbio_preds(df):
    collection = get_bioc_collection(df)
    lemmatizer = Lemmatizer()
    ptb2dep = NegBioPtb2DepConverter(lemmatizer, universal=True)
    ssplitter = NegBioSSplitter(newline=True)
    parser = NegBioParser(model_dir=PARSING_MODEL_DIR)
    loader = NegBioLoader()
    extractor = NegBioExtractor(Path(MENTION_PATH), Path(UNMENTION_PATH))
    neg_detector = ModifiedDetector(PRE_NEG_PATH,
                                    NEG_PATH,
                                    POST_NEG_PATH)
    aggregator = NegBioAggregator(CATEGORIES)
    collection = pipeline(collection, loader, ssplitter, extractor, 
                          parser, ptb2dep, neg_detector, aggregator, verbose=True)
    
    # convert BioC collection to dataframe for reporting
    negbio_pred = pd.DataFrame()
    for doc in collection.documents:
        dictionary = {}
        for key, val in doc.infons.items():
            dictionary[key[9:]] = val
        negbio_pred = negbio_pred.append(dictionary, ignore_index=True)
    negbio_pred = negbio_pred.replace(
        "Positive", True).replace(
        "Negative", False).replace("Uncertain", False).fillna(False)
    return negbio_pred