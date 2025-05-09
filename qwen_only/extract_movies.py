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
        return {movie_name:, min_edit_distance: , nearest_movie: }
    """
    # calculate the edit distance
    dists = [distance(text.lower(), i.lower()) for i in items]
    # find the nearest movie
    nearest_idx = np.argmin(dists)
    # nearest_movie = items[nearest_idx]

    return nearest_idx


def extract_movies(response, candidates):
    text = response

    rec_list = [del_numbering(del_space(del_parentheses(i.strip()))) for i in text.split('\n')]

    if candidates is not None:
        rec_list = [nearest(i, candidates) for i in rec_list]
    return rec_list
