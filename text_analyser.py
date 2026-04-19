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

# ---------------------------------------------------------------------------
# 5.  SENTIMENT ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------

def analyse_sentiment(word_list: list,
                      positive_words: list,
                      negative_words: list) -> dict:
    """
    Perform a simple lexicon-based sentiment analysis on a list of words.

    Each word is checked against the positive and negative lexicons.  The
    result is a dictionary containing counts and a percentage breakdown.

    Parameters
    ----------
    word_list : list of str
        Tokenised words from the source text.
    positive_words : list of str
        Words considered to carry positive sentiment.
    negative_words : list of str
        Words considered to carry negative sentiment.

    Returns
    -------
    dict with keys:
        'positive_count'  – number of positive words found
        'negative_count'  – number of negative words found
        'neutral_count'   – remaining words
        'total_words'     – total number of words analysed
        'positive_pct'    – percentage of positive words (float)
        'negative_pct'    – percentage of negative words (float)
        'neutral_pct'     – percentage of neutral words (float)
        'overall_label'   – 'Positive', 'Negative', or 'Neutral'
        'matched_positive'– list of positive words actually found in text
        'matched_negative'– list of negative words actually found in text
    """

    positive_set = set(positive_words)
    negative_set = set(negative_words)

    positive_count = 0
    negative_count = 0
    matched_positive_words = []
    matched_negative_words = []

    for word in word_list:
        if word in positive_set:
            positive_count += 1
            matched_positive_words.append(word)
        elif word in negative_set:
            negative_count += 1
            matched_negative_words.append(word)

    total_word_count = len(word_list)
    neutral_count = total_word_count - positive_count - negative_count

    # Avoid division by zero if the text is empty
    if total_word_count > 0:
        positive_percentage = (positive_count / total_word_count) * 100
        negative_percentage = (negative_count / total_word_count) * 100
        neutral_percentage = (neutral_count / total_word_count) * 100
    else:
        positive_percentage = negative_percentage = neutral_percentage = 0.0

    # Determine an overall sentiment label
    if positive_count > negative_count:
        overall_sentiment_label = "Positive"
    elif negative_count > positive_count:
        overall_sentiment_label = "Negative"
    else:
        overall_sentiment_label = "Neutral"

    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_words": total_word_count,
        "positive_pct": positive_percentage,
        "negative_pct": negative_percentage,
        "neutral_pct": neutral_percentage,
        "overall_label": overall_sentiment_label,
        "matched_positive": matched_positive_words,
        "matched_negative": matched_negative_words,
    }



# ---------------------------------------------------------------------------
# 6.  READABILITY FUNCTIONS
# ---------------------------------------------------------------------------

def count_syllables_in_word(word: str) -> int:
    """
    Estimate the number of syllables in an English word.

    The algorithm:
      1. Count groups of vowels (a, e, i, o, u) as one syllable.
      2. Subtract one syllable if the word ends in a silent 'e'.
      3. Ensure every word has at least one syllable.

    This is an approximation, it will not be perfectly accurate for all words.

    Parameters
    ----------
    word : str
        A single lower-case word.

    Returns
    -------
    int
        Estimated syllable count (minimum 1).
    """
    vowels = "aeiou"
    word_lower = word.lower()
    syllable_count = 0
    previous_char_was_vowel = False

    for character in word_lower:
        is_vowel = character in vowels
        # Only count the start of a vowel cluster
        if is_vowel and not previous_char_was_vowel:
            syllable_count += 1
        previous_char_was_vowel = is_vowel

    # Subtract a syllable for words ending in a silent 'e'
    if word_lower.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    # Every word must have at least one syllable
    return max(1, syllable_count)


def calculate_readability_metrics(raw_text: str, word_list: list) -> dict:
    """
    Calculate several standard readability metrics for the text.

    Metrics computed:
      - Total word count
      - Total sentence count
      - Average words per sentence
      - Average syllables per word
      - Flesch Reading Ease score (higher = easier to read)
      - Flesch-Kincaid Grade Level (approximate US school grade)
      - Sentence length data (list of int) for further analysis

    Formulae:
      Flesch Reading Ease = 206.835
                            - 1.015  × (words / sentences)
                            - 84.6   × (syllables / words)

      Flesch-Kincaid Grade = 0.39 × (words / sentences)
                           + 11.8 × (syllables / words)
                           - 15.59

    Parameters
    ----------
    raw_text : str
        The original unprocessed text.
    word_list : list of str
        Pre-tokenised word list from the same text.

    Returns
    -------
    dict
        A dictionary of readability metrics.
    """
    sentence_list = split_into_sentences(raw_text)
    total_sentence_count = len(sentence_list)
    total_word_count = len(word_list)

    # Calculate syllable count for every word in the text
    total_syllable_count = sum(count_syllables_in_word(w) for w in word_list)

    # Build a list recording the number of words in each sentence
    sentence_word_lengths = []
    for sentence in sentence_list:
        sentence_words = tokenise_words(sentence)
        sentence_word_lengths.append(len(sentence_words))

    # Avoid division by zero for very short or empty texts
    if total_sentence_count > 0 and total_word_count > 0:
        avg_words_per_sentence = total_word_count / total_sentence_count
        avg_syllables_per_word = total_syllable_count / total_word_count

        # Flesch Reading Ease: 0-100 scale; higher means more readable
        flesch_reading_ease = (206.835
                               - 1.015 * avg_words_per_sentence
                               - 84.6 * avg_syllables_per_word)

        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * avg_words_per_sentence
                                + 11.8 * avg_syllables_per_word
                                - 15.59)
    else:
        avg_words_per_sentence = 0.0
        avg_syllables_per_word = 0.0
        flesch_reading_ease = 0.0
        flesch_kincaid_grade = 0.0

    return {
        "total_words": total_word_count,
        "total_sentences": total_sentence_count,
        "total_syllables": total_syllable_count,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_syllables_per_word": avg_syllables_per_word,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "sentence_word_lengths": sentence_word_lengths,
    }


def interpret_flesch_score(flesch_score: float) -> str:
    """
    Return a human-readable description of a Flesch Reading Ease score.

    Parameters
    ----------
    flesch_score : float
        A Flesch Reading Ease score.

    Returns
    -------
    str
        Description of the difficulty level.
    """
    if flesch_score >= 90:
        return "Very Easy (suitable for young children)"
    elif flesch_score >= 80:
        return "Easy (conversational English)"
    elif flesch_score >= 70:
        return "Fairly Easy"
    elif flesch_score >= 60:
        return "Standard (plain English)"
    elif flesch_score >= 50:
        return "Fairly Difficult"
    elif flesch_score >= 30:
        return "Difficult (academic / professional)"
    else:
        return "Very Difficult (specialist or technical)"



# ---------------------------------------------------------------------------
# 7.  DATA VISUALISATION FUNCTIONS
# ---------------------------------------------------------------------------

def plot_word_frequency_bar_chart(top_words: list, output_file: str = "word_frequency.png") -> None:
    """
    Create and save a horizontal bar chart showing the top word frequencies.

    Parameters
    ----------
    top_words : list of tuple
        Each tuple is (word: str, frequency: int), sorted descending.
    output_file : str
        File name to save the chart image (default: 'word_frequency.png').
    """
    if not top_words:
        print("  No word data available to plot.")
        return

    # Separate words and their counts for plotting
    words = [pair[0] for pair in top_words]
    counts = [pair[1] for pair in top_words]

    # Reverse the lists so the most frequent word appears at the top
    words.reverse()
    counts.reverse()

    fig, axes = plt.subplots(figsize=(9, 6))
    bar_colours = plt.cm.Blues([0.4 + 0.5 * (i / len(words))
                                for i in range(len(words))])
    axes.barh(words, counts, color=bar_colours)

    axes.set_xlabel("Frequency", fontsize=12)
    axes.set_title("Top Word Frequencies", fontsize=14, fontweight="bold")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=150)
        print(f"  Bar chart saved as '{output_file}'.")
    except OSError as error:
        print(f"  Could not save chart: {error}")

    plt.show()
    plt.close()


def plot_sentiment_pie_chart(sentiment_results: dict,
                             output_file: str = "sentiment_pie.png") -> None:
    """
    Create and save a pie chart illustrating the sentiment distribution.

    Parameters
    ----------
    sentiment_results : dict
        The dictionary returned by analyse_sentiment().
    output_file : str
        File name to save the chart image (default: 'sentiment_pie.png').
    """
    positive_count = sentiment_results["positive_count"]
    negative_count = sentiment_results["negative_count"]
    neutral_count = sentiment_results["neutral_count"]

    # Only include non-zero segments to avoid a confusing empty slice
    labels = []
    sizes = []
    colours = []

    if positive_count > 0:
        labels.append(f"Positive\n({positive_count} words)")
        sizes.append(positive_count)
        colours.append("#4CAF50")   # Green

    if negative_count > 0:
        labels.append(f"Negative\n({negative_count} words)")
        sizes.append(negative_count)
        colours.append("#F44336")   # Red

    if neutral_count > 0:
        labels.append(f"Neutral\n({neutral_count} words)")
        sizes.append(neutral_count)
        colours.append("#9E9E9E")   # Grey

    if not sizes:
        print("  No sentiment data available to plot.")
        return

    fig, axes = plt.subplots(figsize=(7, 6))
    axes.pie(sizes,
             labels=labels,
             colors=colours,
             autopct="%1.1f%%",
             startangle=140,
             pctdistance=0.8)

    axes.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=150)
        print(f"  Pie chart saved as '{output_file}'.")
    except OSError as error:
        print(f"  Could not save chart: {error}")

    plt.show()
    plt.close()


def plot_sentence_length_histogram(sentence_lengths: list,
                                   output_file: str = "sentence_lengths.png") -> None:
    """
    Create and save a histogram of sentence lengths (words per sentence).

    Parameters
    ----------
    sentence_lengths : list of int
        Number of words in each sentence.
    output_file : str
        File name to save the chart image.
    """
    if not sentence_lengths:
        print("  No sentence data available to plot.")
        return

    fig, axes = plt.subplots(figsize=(8, 5))
    axes.hist(sentence_lengths, bins=10, color="#5C85D6", edgecolor="white")
    axes.set_xlabel("Words per Sentence", fontsize=12)
    axes.set_ylabel("Number of Sentences", fontsize=12)
    axes.set_title("Sentence Length Distribution", fontsize=14, fontweight="bold")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=150)
        print(f"  Histogram saved as '{output_file}'.")
    except OSError as error:
        print(f"  Could not save chart: {error}")

    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 8.  DISPLAY / PRINT FUNCTIONS
# ---------------------------------------------------------------------------

def display_word_frequency_results(top_words: list) -> None:
    """
    Print the top word frequencies to the console in a formatted table.

    Parameters
    ----------
    top_words : list of tuple
        Each tuple is (word: str, frequency: int).
    """
    print("\n" + "=" * 45)
    print(f"  {'RANK':<6} {'WORD':<20} {'FREQUENCY'}")
    print("=" * 45)

    for rank, (word, frequency) in enumerate(top_words, start=1):
        print(f"  {rank:<6} {word:<20} {frequency}")

    print("=" * 45)


def display_sentiment_results(sentiment_results: dict) -> None:
    """
    Print a formatted summary of the sentiment analysis results.

    Parameters
    ----------
    sentiment_results : dict
        The dictionary returned by analyse_sentiment().
    """
    print("\n" + "=" * 45)
    print("  SENTIMENT ANALYSIS RESULTS")
    print("=" * 45)
    print(f"  Total words analysed : {sentiment_results['total_words']}")
    print(f"  Positive words found : {sentiment_results['positive_count']} "
          f"({sentiment_results['positive_pct']:.1f}%)")
    print(f"  Negative words found : {sentiment_results['negative_count']} "
          f"({sentiment_results['negative_pct']:.1f}%)")
    print(f"  Neutral / Other      : {sentiment_results['neutral_count']} "
          f"({sentiment_results['neutral_pct']:.1f}%)")
    print(f"\n  Overall Sentiment    : {sentiment_results['overall_label']}")
    print("=" * 45)

    if sentiment_results["matched_positive"]:
        unique_pos = sorted(set(sentiment_results["matched_positive"]))
        print(f"\n  Positive words detected: {', '.join(unique_pos)}")

    if sentiment_results["matched_negative"]:
        unique_neg = sorted(set(sentiment_results["matched_negative"]))
        print(f"  Negative words detected: {', '.join(unique_neg)}")


def display_readability_results(readability_metrics: dict) -> None:
    """
    Print a formatted summary of the readability metrics.

    Parameters
    ----------
    readability_metrics : dict
        The dictionary returned by calculate_readability_metrics().
    """
    flesch_score = readability_metrics["flesch_reading_ease"]
    flesch_description = interpret_flesch_score(flesch_score)

    print("\n" + "=" * 55)
    print("  READABILITY METRICS")
    print("=" * 55)
    print(f"  Total words              : {readability_metrics['total_words']}")
    print(f"  Total sentences          : {readability_metrics['total_sentences']}")
    print(f"  Total syllables          : {readability_metrics['total_syllables']}")
    print(f"  Avg words per sentence   : {readability_metrics['avg_words_per_sentence']:.2f}")
    print(f"  Avg syllables per word   : {readability_metrics['avg_syllables_per_word']:.2f}")
    print(f"\n  Flesch Reading Ease      : {flesch_score:.2f}")
    print(f"  Difficulty level         : {flesch_description}")
    print(f"\n  Flesch-Kincaid Grade     : {readability_metrics['flesch_kincaid_grade']:.1f}")
    print("=" * 55)


