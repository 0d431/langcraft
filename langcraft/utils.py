import re
import numpy as np
from typing import List


def extract_tag(tag: str, data: str) -> List[str]:
    """
    Extracts all string contents enclosed in any set of pairs of XML tags in the given string.

    Parameters:
        tag (str): The XML tag to search for.
        data (str): The string containing the XML data.

    Returns:
        List[str]: The extracted string contents enclosed in the XML tags.
    """
    return [
        m.strip(" \n\t") for m in re.findall(rf"<{tag}>(.*?)</{tag}>", data, re.DOTALL)
    ]


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
        vec1 (list or numpy.ndarray): The first vector.
        vec2 (list or numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return similarity
