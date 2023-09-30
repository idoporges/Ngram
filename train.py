import os
import nltk
import pickle
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.corpus import gutenberg


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


def load_file_data(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        text = file.read()
    return preprocess_text(text)


# Load the Gutenberg dataset and preprocess it
def load_gutenberg_data(text):
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


# Save tokens to a pickle file
def save_tokens(tokens, filename):
    with open(filename, 'wb') as f:
        pickle.dump(set(tokens), f)


if __name__ == "__main__":
    author = "Austen"
    # Change to folder name. want to train on all Bronte and austen texts.
    train_set = load_text_data("Austen")
    vocab = extract_tuples(train_set, 1)
    save_tokens(vocab, 'models/Vocab/vocab.pkl')
    # Test sets.
    test_set = gutenberg.raw("austen-sense.txt")
    n_values = list(range(2, 10))

    Ngrams_austen = []
    for n in n_values:
        Ngram_austen = train_ngram_lm(train_set, n)

        # Creating a dictionary to save model, n-value, and a placeholder for perplexity
        austen_model_info = {
            'model': Ngram_austen,
            'author': author,
            'n_value': n,
            'perplexity': None  # Placeholder
        }

        # Saving the models
        with open(f'models/Ngram_austen_{n}.pkl', 'wb') as f:
            pickle.dump(austen_model_info, f)
