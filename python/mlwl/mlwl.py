import sys
import os
import gensim
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import nltk
import string
import math
import argparse
import json

from mlwl import dedup_wordlist # Rust library

# Gensim Word2Vec models to use. MAX_SIMILAR words will be retrieved from all models listed
# here. Keep this in mind when defining MAX_SIMILAR and when creating your input wordlist,
# as this may generate a very large wordlist.
GENSIM_W2V_MODELS = ["conceptnet-numberbatch-17-06-300", "word2vec-google-news-300", "glove-twitter-200", "fasttext-wiki-news-subwords-300", "glove-wiki-gigaword-300"]

MIN_SIMILARITY_SPACY = 0.2
MAX_SIMILAR = 1000 # Maximum amount of similar words to find for each keyword.
WORD_VARIATIONS = True # Should variate words on input and output. Faster if False.
WORD_SPLIT = True # Gensim outputs multiple words separated by underscore. If this is true, will copy and split them on output.
DEDUP = True # Deduplicate input and output words.
NUM_THREADS = 6

# preprocess_keywords returns an expanded keywords list by preprocessing the input keywords list.
# This ensures that we have good tokens for finding similar words without getting rid
# of any keywords from the user's input.
def preprocess_keywords(keywords):
    # Replace space with _
    tokens = []
    for keyword in keywords:
        tokens.append(keyword.replace(" ", "_"))

    # Tokenize and remove stop words.
    tokens_new = []
    for keyword in tokens:
        for token in nltk.tokenize.word_tokenize(keyword.lower()):
            for t in token:
                tokens_new.append(t)
    tokens = tokens_new
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove punctuation.
    tokens = [token.translate(str.maketrans('','', string.punctuation)) for token in tokens]

    # Lemmatize.
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return keywords + tokens

# get_similar_from_gensim takes an input list of strings and returns the most similar strings
# by getting similar from each Word2Vec model that is passed.
def get_similar_from_gensim(keywords, progress, progress_i):
    models = []
    for model_name in GENSIM_W2V_MODELS:
        models.append((model_name, gensim.models.keyedvectors.KeyedVectors.load("{}.normed".format(model_name), mmap="r")))
        models[-1][1].vectors_norm = models[-1][1].vectors

    similar_words = []

    for keyword in keywords:
        similar_words.append("{}\n".format(keyword))

        # Get the most similar words to the given word.
        for model_name, model in models:
            most_similar = None
            try:
                # Handle conceptnet URI
                if model_name == "conceptnet-numberbatch-17-06-300":
                    most_similar = model.most_similar("/c/en/{}".format(keyword), topn=MAX_SIMILAR)
                else:
                    most_similar = model.most_similar(keyword, topn=MAX_SIMILAR)
            except Exception as e: # Model does not contain this keyword so do nothing.
                #print(e)
                continue
            for word, _ in most_similar:
                w = word

                # Handle conceptnet URI
                if model_name == "conceptnet-numberbatch-17-06-300":
                    w = w.split("/")[-1]
                for variation in get_word_variations(w):
                    similar_words.append("{}\n".format(variation))
        
        progress[progress_i] += 1
    
    return similar_words

# most_similar_gensim takes input keywords string list, preprocess them concurrently with nltk, 
# and concurrently gets similar words using all the chosen Word2Vec models. Returns a list
# of the keywords + similar words. The returned list is not deduplicated, so there will
# likely be repeated words. 
def most_similar_gensim(keywords):
    import gensim.downloader as api
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # See https://stackoverflow.com/questions/51616074/sharing-memory-for-gensims-keyedvectors-objects-between-docker-containers
    for model_name in GENSIM_W2V_MODELS:
        if not os.path.exists("{}.normed".format(model_name)):
            print("Saving normed gensim Word2Vec model {}...".format(model_name))
            model = api.load(model_name)
            model.init_sims(replace=True)
            model.save("{}.normed".format(model_name))

    similar_words = None

    print("Splitting keywords among threads...")
    keywords_threads = [[]] # Input to concurrent func.

    for keyword in keywords:
        # If we already added the max amount we are submitting to each thread, go to next input list.
        if len(keywords_threads[-1]) == round(len(keywords) / NUM_THREADS) and len(keywords_threads) < NUM_THREADS:
            keywords_threads.append([])
        
        keywords_threads[-1].append(keyword)

    print("Getting similar words concurrently with {} threads".format(NUM_THREADS))
    with multiprocessing.Manager() as manager:
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
            # First preprocess all keywords concurrently.
            print("Preprocessing keywords concurrently...")
            preprocess_queue = manager.list()
            futures_preprocess = []
            for keywords_thread in keywords_threads:
                futures_preprocess.append(executor.submit(preprocess_keywords, keywords_thread))

            # Create progress bar.
            pbar = tqdm(total=len(futures_preprocess))

            # Add all keywords to set and wait til done.
            is_done = manager.Value(bool, False)
            last_num_done = 0
            while not is_done.value:
                num_done = 0
                for i in range(len(futures_preprocess)):
                    if futures_preprocess[i].done():
                        num_done += 1
                
                if num_done != last_num_done:
                    pbar.update(num_done - last_num_done)
                    last_num_done = num_done

                if num_done == len(futures_preprocess):
                    is_done.value = True
            pbar.clear()
            pbar.close()
            print("Done preprocessing.")

            # Create new keywords for each thread from newly created set.
            keywords_threads = [[]] 
            keyword_set = []
            for future in futures_preprocess:
                keyword_set += future.result()

            if DEDUP:
                print("Deduping preprocessed keywords...")
                keyword_set = dedup_wordlist(keyword_set)

            print("Concurrently getting words based on {} preprocessed keywords".format(len(keyword_set)))
            for keyword in keyword_set:
                # If we already added the max amount we are submitting to each thread, go to next input list.
                if len(keywords_threads[-1]) == round(len(keyword_set) // NUM_THREADS) and len(keywords_threads) < NUM_THREADS:
                    keywords_threads.append([])
                
                keywords_threads[-1].append(keyword)

            futures = []
            progress = manager.list([0 for i in range(len(keywords_threads))])
            totals = []
            wq = manager.list()

            progress_index = 0
            for keywords_thread in keywords_threads:
                future = executor.submit(get_similar_from_gensim, keywords_thread, progress, progress_index)
                futures.append(future)

                totals.append(len(keywords_thread))
                progress_index += 1
        
            pbars = []
            last_vals = [0 for i in range(len(keywords_threads))]
            for i in range(len(progress)):
                pbars.append(tqdm(total=totals[i], position=i))


            is_done = manager.Value(bool, False)
            while not is_done.value:
                for i in range(len(progress)):
                    if progress[i] != last_vals[i]:
                        pbars[i].update(progress[i] - last_vals[i])
                        last_vals[i] = progress[i]

                num_done = 0
                for i in range(len(futures)):
                    if futures[i].done():
                        num_done += 1

                        pbars[i].n = pbars[i].total
                        pbars[i].refresh()

                if num_done == len(futures):
                    is_done.value = True

            # Clear all progress bars
            for pbar in pbars:
                pbar.clear()
                pbar.close()

            similar_words = []
            for future in futures:
                similar_words += future.result()
            

    return similar_words

# get_word_variations takes an input string and returns a list of variations of the word. 
# Looks for certain seperator characters and replaces them.
def get_word_variations(word):
    variations = [word]

    replace_chars = [".", " ", "-", "_", "@"]

    if WORD_VARIATIONS:
        for c in replace_chars:
            if c in word:
                # Replace all replace chars with all other replace chars.
                for c_other in replace_chars:
                    if c != c_other:
                        variations.append(word.replace(c, c_other))
                        variations.append(word.replace(c, ""))

                # Also add each 'token' by splitting on each replace char.
                if len(word.split(c)) > 1:
                    for token in word.split(c):
                        variations.append(token)

    # Split into multiple words on underscore. Also remove the underscore and combine them too.
    if WORD_SPLIT:
        full_word = ""
        if len(word.split("_")) > 1:
            for token in word.split("_"):
                variations.append(token)
                full_word += token
        variations.append(full_word)
    

    return variations

def main():
    # Get user input arguments.
    parser = argparse.ArgumentParser(
        prog='mlwl',
        description='Use machine learning to generate wordlists for hash cracking!')
    parser.add_argument('in_keywords_path')
    parser.add_argument('out_wordlist_path')
    
    args = parser.parse_args()
    file_path_in = args.in_keywords_path
    file_path_out = args.out_wordlist_path

    # Get keywords from file. These will be used to find similar words.
    print("Retrieving keywords from input file...")
    keywords = []
    try:
        with open(file_path_in, 'r') as f:
            for line in f:
                k = line.strip().lower()

                if WORD_VARIATIONS:
                    for k_variation in get_word_variations(k):
                        if k_variation not in keywords:
                            keywords.append(k_variation)
                else:
                    if k not in keywords:
                        keywords.append(k)

    except Exception as e:
        print("Error: Could not read input file. {}".format(e))

    if len(keywords) == 0:
        print("Error: No keywords were found in the file.")
        return

    print("Read {} keywords from {}".format(len(keywords), file_path_in))

    # Validate that we can open the output file.
    try:
        with open(file_path_out, 'w'):
            pass
    except Exception as e:
        print("Error: Could not open output file. {}".format(e))

    final_list = []

    similar_words = most_similar_gensim(keywords)

    if DEDUP:
        print("Deduping wordlist...")
        similar_words = dedup_wordlist(similar_words)

    print("Writing results to file...")
    try:
        with open(file_path_out, 'w') as f:
            f.writelines(similar_words)
    except Exception as e:
        print("Error: Could not write results to {}. {}".format(file_path_out, e))


if __name__ == "__main__":
    main()