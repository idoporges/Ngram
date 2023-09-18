import math
import os
import nltk
import random
import matplotlib.pyplot as plt
import pickle
# nltk.download('punkt')
# nltk.download('gutenberg')
from nltk.corpus import gutenberg
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.corpus import gutenberg
from itertools import islice


# Preprocess the text data
def preprocess_text(text):
    sentences_list = nltk.sent_tokenize(text)
    pattern = r'Mr\.|Mrs\.|Ms\.|Dr\.|St\.|<S>|<E>|\w+|\$[\d\.]+|\S+'  # Make special recognizable tokens.
    modified_sentences = ['<S>' + s + '<E>' for s in sentences_list]
    tokens = []
    for s in modified_sentences:
        tokens += nltk.regexp_tokenize(s, pattern)

    return tokens


# Extract n-grams from the tokenized text
def extract_tuples(tokens, ngram_order):
    return list(ngrams(tokens, ngram_order))


def default_probability(default_prob):
    return default_prob


# Train the n-gram language model
def train_ngram_lm(tokens, ngram_order):
    model = defaultdict(Counter)
    tuples = extract_tuples(tokens, ngram_order)

    for t in tuples:
        context, word = tuple(t[:-1]), t[-1]
        model[context][word] += 1

    k = 1
    for context, word_counts in model.items():
        for word, count in word_counts.items():
            model[context][word] += k
        model[context]['<OOV>'] = 1

    # Initialize Probs as a regular dictionary of dictionaries
    Probs = {}

    for context, word_counts in model.items():
        total_word_count = sum(word_counts.values())

        # Initialize an empty dictionary for the given context
        Probs[context] = {}

        for word, count in word_counts.items():
            Probs[context][word] = float(count) / total_word_count

    # TODO find a good calculation for OOV context
    Probs[('<OOV>',)] = {}
    Probs[('<OOV>',)]['<OOV>'] = 0.1  # 0.999999
    # too low and perplexity will explode and be incalculable
    # too high and the best perplexity will be awarded to the worst models
    return Probs


# Load the Gutenberg dataset and preprocess it
def load_gutenberg_data(text):
    return preprocess_text(text)


def load_file_data(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        text = file.read()
    return preprocess_text(text)


def load_text_data(folder):
    tokens = []
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                tokens.extend(preprocess_text(text))
    return tokens


def print_first_n_items(model, n):
    for context, word_probs in islice(model.items(), n):
        print(f"{context}: {word_probs}")


def get_num_of_all_words_from_ngram(model):
    words = set()

    for context, word_probs in model.items():
        words.update(word_probs.keys())

    return len(words)


if __name__ == "__main__":

    # folder_path = "Austen"
    # Change to folder name. want to train on all Bronte and austen texts.
    train_set_austen = load_text_data("Austen")
    train_set_bronte = load_text_data("Bronte")

    # Test sets.
    test_set_austen = gutenberg.raw("austen-sense.txt")
    # test_set_austen = load_file_data("Sense.txt")
    test_set = test_set_austen

    n_values = list(range(2, 10))

    """
    # generate
    for n in n_values:
        Ngram_austen = train_ngram_lm(train_set_austen, n)
        text = generate_text(Ngram_austen, n, 30)
        print("Austen ", n, "-gram: ", text)
    """
    # """
    # graph:
    Ngrams_austen = []
    for n in n_values:
        Ngram_austen = train_ngram_lm(train_set_austen, n)
        Ngram_bronte = train_ngram_lm(train_set_bronte, n)

        # Creating a dictionary to save model, n-value, and a placeholder for perplexity
        austen_model_info = {
            'model': Ngram_austen,
            'author': "Austen",
            'n_value': n,
            'perplexity': None  # Placeholder
        }

        bronte_model_info = {
            'model': Ngram_bronte,
            'author': "Bronte",
            'n_value': n,
            'perplexity': None  # Placeholder
        }

        # Saving the models
        with open(f'models/Ngram_austen_{n}.pkl', 'wb') as f:
            pickle.dump(austen_model_info, f)

        with open(f'models/Ngram_bronte_{n}.pkl', 'wb') as f:
            pickle.dump(bronte_model_info, f)
    # """
