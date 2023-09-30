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
        model[context][word] += 1  # counter of how many times the tuple exists in the n-gram model.

    k = 1
    for context, word_counts in model.items():
        for word, count in word_counts.items():
            model[context][word] += k
        model[context]['<OOV>'] = 1


    Probs = defaultdict(lambda: defaultdict(lambda: default_probability(0)))
    # Calculate probabilities
    for context, word_counts in model.items():
        total_word_count = sum(word_counts.values())
        for word, count in word_counts.items():
            Probs[context][word] = float(count) / total_word_count  # Ensure float division

    """
    # Print the probabilities dictionary
    for context, word_probs in Probs.items():
        print(f"Context: {context}")
        prob_sum = 0
        for word, prob in word_probs.items():
            prob_sum += prob
            print(f"  Word: {word}, Probability: {prob}")

        # Define a tolerance threshold
        tolerance = 1e-6  # Adjust the threshold as needed

        # Check if prob_sum is close to 1.0 within the defined tolerance
        if abs(1.0 - prob_sum) <= tolerance:
            prob_sum = 1.0
        if prob_sum != 1.0:
            print("prob sum = ", prob_sum)
    sleep(10000)
    """
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


def calc_perplexity_batch(model, n, test_set, k=1, batch_size=50):
    tokens = preprocess_text(test_set)
    perplexities = []
    context_OOV_cnt = 0
    word_OOV_cnt = 0
    OOV_cap = 1000
    for i in range(0, len(tokens) - n, batch_size):
        batch_tokens = tokens[i:i + batch_size]
        batch_perplexity = 1

        for j in range(len(batch_tokens) - n):
            context = tuple(batch_tokens[j: j + n - 1])
            word = batch_tokens[j + n - 1]

            if context not in model:
                probability = model[context][word]
                if context_OOV_cnt >= OOV_cap:
                    probability = 1
                context_OOV_cnt += 1
            else:
                if word not in model[context]:
                    probability = model[context]['<OOV>']
                    if word_OOV_cnt >= OOV_cap:
                        probability = 1
                    word_OOV_cnt += 1
                else:
                    probability = model[context][word]

            batch_perplexity = batch_perplexity / probability

        batch_perplexity = batch_perplexity ** (1.0 / (len(batch_tokens) - n))
        perplexities.append(batch_perplexity)

    if word_OOV_cnt + context_OOV_cnt != 0:
        print("Num of OOV: ", context_OOV_cnt + word_OOV_cnt)
        print("Num of context OOV: ", context_OOV_cnt)
        print("Num of word OOV: ", word_OOV_cnt)
    average_perplexity = sum(perplexities) / len(perplexities)
    return average_perplexity


def generate_text(model, n, max_length=100):
    # Choose a random starting context that starts at a beginning of a sentence.
    filtered_keys = [key for key in model.keys() if key[0] == '<S>']
    starting_context = random.choice(filtered_keys)
    generated_text = list(starting_context)

    for i in range(max_length):
        # Extract the last n-1 tokens from the generated text as the context
        context = tuple(generated_text[-(n - 1):])

        # Get the probabilities for the next word based on the context
        word_probs = model[context]

        if not word_probs:
            break  # If there are no more words to predict, stop generating

        # Choose the next word based on the probabilities
        next_word = random.choices(list(word_probs.keys()), weights=list(word_probs.values()))[0]
        while next_word == '<OOV>':
            next_word = random.choices(list(word_probs.keys()), weights=list(word_probs.values()))[0]

        # Append the chosen word to the generated text
        generated_text.append(next_word)

    return ' '.join(generated_text)


if __name__ == "__main__":
    print("test test push git")
    model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
    perplexity_values = []
    n_values = []

    for model_file in model_files:
        with open(f'models/{model_file}', 'rb') as f:
            model_info = pickle.load(f)

        n_value = model_info['n_value']
        perplexity = model_info['perplexity']

        n_values.append(n_value)
        perplexity_values.append(perplexity)

    # Sort by n_values for plotting
    sorted_indices = sorted(range(len(n_values)), key=lambda k: n_values[k])
    perplexity_values = [perplexity_values[i] for i in sorted_indices]
    n_values = [n_values[i] for i in sorted_indices]

    # Debugging
    print("n_values: ", n_values)
    print("perplexity_values: ", perplexity_values)

    # Plotting code
    plt.plot(n_values, perplexity_values)
    plt.xlabel('n (Order of the Language Model)')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. n')
    plt.grid(True)
    plt.show()
