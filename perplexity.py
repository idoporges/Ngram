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


def calc_perplexity_batch(model, n, test_set, k, batch_size=50):
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

    # Test sets.
    test_set_austen = gutenberg.raw("austen-sense.txt")
    test_set = test_set_austen
    K = 1
    # Loop over all files in the 'models' directory
    for filename in os.listdir('models'):
        if filename.endswith('.pkl'):
            filepath = os.path.join('models', filename)

            # Load the model
            with open(filepath, 'rb') as f:
                loaded_info = pickle.load(f)
                Ngram_model = loaded_info['model']
                n_value = loaded_info['n_value']

            # Calculate the perplexity
            perplexity = calc_perplexity_batch(Ngram_model, n_value, test_set, K)

            # Update the perplexity placeholder
            loaded_info['perplexity'] = perplexity

            # Save the updated info back to the file
            with open(filepath, 'wb') as f:
                pickle.dump(loaded_info, f)

            print(f"Updated perplexity for {filename} to {perplexity}")
