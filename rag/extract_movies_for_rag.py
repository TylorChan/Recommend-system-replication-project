# modified from
# https://github.com/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys/blob/master/src/utils.py
# https://github.com/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys/blob/master/src/vicuna/general/extract.py#L29

import re
import numpy as np
from editdistance import eval as distance

def del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)


def del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()


def del_numbering(text):
    pattern = r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?"
    return re.sub(pattern, "", text)


def nearest(text, items):
    """ given the raw text name and all candidates, 
        return movie_name
    """
    # calculate the edit distance
    dists = [distance(text.lower(), i.lower()) for i in items]
    # find the nearest movie
    nearest_idx = np.argmin(dists)
    nearest_movie = items[nearest_idx]

    return nearest_movie


def extract_movies(response, predicted_candidates, movie2id):
    text = response

    llm_rec_list = [del_numbering(del_space(del_parentheses(i.strip()))) for i in text.split('\n')]
    processed_rec_list = []

    if predicted_candidates is not None:
        for i in llm_rec_list:
            nearest_movie = nearest(i, predicted_candidates)
            processed_rec_list.append(movie2id[nearest_movie])

    return processed_rec_list
