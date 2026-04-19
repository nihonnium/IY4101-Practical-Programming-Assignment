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
