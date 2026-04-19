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

