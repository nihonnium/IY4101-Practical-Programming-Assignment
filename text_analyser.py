"""
IY499 Introduction to Programming – Practical Programming Assignment

A Natural Language Processing (NLP) Text Analyser that evaluates sentiment
and readability of a given text file. The programme uses the Python
standard library, string and matplotlib.

Student Name : Nihan Dilay Boz
Student Number: 303065789
Course  : IY499 Introduction to Programming
"""

import string
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  FILE I/O FUNCTIONS
# ---------------------------------------------------------------------------

def load_text_file(file_path: str) -> str:
    """
    Read and return the contents of a text file.

    Parameters
    ----------
    file_path : str
        Path to the .txt file to be read.

    Returns
    -------
    str
        The raw text content of the file, or an empty string if the file
        could not be opened.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            content = text_file.read()

        # Result for completely empty files
        if not content.strip():
            print(f"  Warning: '{file_path}' is empty.")
            return ""

        print(f"  Successfully loaded '{file_path}'.")
        return content

    except FileNotFoundError:
        print(f"  Error: The file '{file_path}' was not found.")
        return ""
    except PermissionError:
        print(f"  Error: Permission denied when reading '{file_path}'.")
        return ""
    except OSError as error:
        print(f"  Error reading '{file_path}': {error}")
        return ""


def load_lexicon_file(file_path: str) -> list:
    """
    Read a sentiment lexicon file and return a list of words.

    Each line in the lexicon should contain a single word.  Lines that begin
    with '#' are treated as comments and are ignored.

    Parameters
    ----------
    file_path : str
        Path to the lexicon .txt file.

    Returns
    -------
    list of str
        A list of lower-case words loaded from the lexicon, or an empty list
        if the file could not be opened.
    """
    word_list = []

    try:
        with open(file_path, "r", encoding="utf-8") as lexicon_file:
            for line in lexicon_file:
                stripped_line = line.strip().lower()
                # Skip comment lines and blank lines
                if stripped_line and not stripped_line.startswith("#"):
                    word_list.append(stripped_line)

        print(f"  Loaded {len(word_list)} words from '{file_path}'.")
        return word_list

    except FileNotFoundError:
        print(f"  Warning: Lexicon file '{file_path}' not found. "
              "Sentiment analysis will be limited.")
        return []
    except OSError as error:
        print(f"  Error reading lexicon '{file_path}': {error}")
        return []


# ---------------------------------------------------------------------------
# 2.  TEXT PRE-PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------

def tokenise_words(raw_text: str) -> list:
    """
    Convert a raw text string into a list of lower-case words, removing
    all punctuation.

    Parameters
    ----------
    raw_text : str
        The original text to tokenise.

    Returns
    -------
    list of str
        A list of individual words with punctuation stripped and converted
        to lower case.
    """
    # Build a translation table that removes every punctuation character
    removal_table = str.maketrans("", "", string.punctuation)
    cleaned_text = raw_text.translate(removal_table).lower()

    # Split on whitespace to obtain individual tokens
    all_tokens = cleaned_text.split()

    # Discard any tokens that contain no alphabetic characters (e.g. numbers)
    word_tokens = [token for token in all_tokens if token.isalpha()]
    return word_tokens


def split_into_sentences(raw_text: str) -> list:
    """
    Split a block of raw text into individual sentences.

    Sentences are assumed to end with a full stop, exclamation mark, or
    question mark.

    Parameters
    ----------
    raw_text : str
        The original text to split.

    Returns
    -------
    list of str
        A list of non-empty sentence strings.
    """
    sentence_list = []
    # Replace the other terminal punctuation marks with full stops
    normalised_text = raw_text.replace("!", ".").replace("?", ".")

    for fragment in normalised_text.split("."):
        stripped_fragment = fragment.strip()
        if stripped_fragment:  # Ignore empty fragments
            sentence_list.append(stripped_fragment)

    return sentence_list

# ---------------------------------------------------------------------------
# 3.  WORD FREQUENCY FUNCTIONS
# ---------------------------------------------------------------------------

def build_word_frequency_dict(word_list: list) -> dict:
    """
    Count how many times each word appears in a list of words.

    Parameters
    ----------
    word_list : list of str
        Tokenised words from the source text.

    Returns
    -------
    dict
        A dictionary mapping each unique word (str) to its frequency (int).
    """
    frequency_dict = {}

    for word in word_list:
        if word in frequency_dict:
            frequency_dict[word] += 1
        else:
            frequency_dict[word] = 1

    return frequency_dict


# ---------------------------------------------------------------------------
# 4.  SORTING ALGORITHM
# ---------------------------------------------------------------------------

def insertion_sort_by_frequency(word_freq_pairs: list) -> list:
    """
    Sort a list of (word, frequency) tuples in descending order of frequency
    using the Insertion Sort algorithm.

    Insertion Sort works by building a sorted sub-list one element at a time.
    Each new element is compared against the already-sorted portion and
    inserted into its correct position.

    Time complexity: O(n^2) in the worst case.

    Parameters
    ----------
    word_freq_pairs : list of tuple
        Each tuple is (word: str, frequency: int).

    Returns
    -------
    list of tuple
        The same list sorted from highest to lowest frequency.
    """
    # Work on a copy so that the original data structure is not modified
    sorted_pairs = list(word_freq_pairs)
    number_of_items = len(sorted_pairs)

    # Outer loop: iterate over every element starting from the second one
    for current_index in range(1, number_of_items):
        current_pair = sorted_pairs[current_index]

        # Inner loop: shift elements that are smaller than current_pair
        comparison_index = current_index - 1
        while (comparison_index >= 0 and
               sorted_pairs[comparison_index][1] < current_pair[1]):
            # Move the smaller element one step to the right
            sorted_pairs[comparison_index + 1] = sorted_pairs[comparison_index]
            comparison_index -= 1

        # Place the current pair in its correct sorted position
        sorted_pairs[comparison_index + 1] = current_pair

    return sorted_pairs


def get_top_n_words(frequency_dict: dict, top_n: int = 10) -> list:
    """
    Return the top-n most frequent words from a frequency dictionary,
    sorted using the manual insertion sort function.

    Parameters
    ----------
    frequency_dict : dict
        Mapping of word -> frequency count.
    top_n : int
        How many top words to return (default: 10).

    Returns
    -------
    list of tuple
        Up to top_n tuples of (word, frequency) in descending order.
    """
    # Convert the dictionary into a list of (word, frequency) tuples
    all_pairs = list(frequency_dict.items())

    # Sort using insertion sort
    sorted_pairs = insertion_sort_by_frequency(all_pairs)

    # Return only the requested number of top results
    return sorted_pairs[:top_n]


