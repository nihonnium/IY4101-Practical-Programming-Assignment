# NLP Text Analyser
### IY499 Introduction to Programming – Practical Programming Assignment

---

## Identifying Information

| Field         | Detail                        |
|---------------|-------------------------------|
| **Name**      | Nihan Dilay Boz               |
| **S-Number**  | 303065789                     |
| **Course**    | IY499 Introduction to Programming |
| **Marker**    | Dr Dena S Y Nuuman            |

---

## Declaration of Own Work

> *I confirm that this assignment is my own work.*
> *Where I have referred to online sources, I have provided comments detailing the reference and included a link to the source.*

---

## Programme Description

The NLP Text Analyser is a command-line Python application that reads a plain-text
file and performs two forms of linguistic analysis: sentiment analysis and readability
assessment. The programme uses only the Python standard library alongside `matplotlib`.

When a text file is loaded, the programme tokenises the text by removing punctuation
and converting all words to lower case. It then builds a word-frequency dictionary and
sorts the entries using an insertion sort algorithm to identify the top
ten most common words. 

Sentiment is determined by matching every word in the text against two separate lexicon
files, one containing positive words and one containing negative words. The programme
counts how many words from each list appear in the text and calculates percentage
breakdowns, producing an overall sentiment label of Positive, Negative, or Neutral.

Readability is evaluated using the standard Flesch Reading Ease and
Flesch–Kincaid Grade Level formulae. Syllable counts are estimated by counting the 
vowels.

Three `matplotlib` charts are available from the menu: a bar chart of the top ten word
frequencies, a pie chart of the sentiment distribution, and a histogram of sentence
lengths.

Robust error handling covers missing files, empty documents, and invalid menu choices
throughout the programme.

---

## Required Packages

| Package      | Purpose                                      | Version |
|--------------|----------------------------------------------|---------|
| `matplotlib` | Generating bar charts, pie charts and histograms | ≥ 3.5   |

> All other imports (`string`, `os`) are part of the Python standard library and do
> not require separate installation.

---

## Installation Instructions

### Prerequisites

- Python 3.9 or later

### 1. Clone or unzip the project

If you received a `.zip` file, extract it to a folder of your choice.

```
unzip IY499_NLP_Analyser.zip -d nlp_analyser
cd nlp_analyser
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
matplotlib>=3.5
```

If you prefer to install manually:

```bash
pip install matplotlib
```

---

## How to Run the Programme

1. Make sure you are in the project directory and (optionally) your virtual environment
   is active.

2. Run the main script:

   ```bash
   python text_analyser.py
   ```

3. The programme will display a numbered menu. Select **option 1** first to load your
   text file. You will be prompted for:
   - The path to your `.txt` file (e.g. `sample_text.txt`)
   - The path to your positive-words lexicon (e.g. `positive_words.txt`)
   - The path to your negative-words lexicon (e.g. `negative_words.txt`)

   > Press **Enter** to skip either lexicon file; sentiment analysis will still run
   > but will report zero matches.

4. Once a file is loaded, all other menu options become available.

### Sample files included

-These test data (negative words, positive words and sample text) are created using 
Gemini to test the application.


| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `sample_text.txt`     | Example paragraph text for testing the analyser  |
| `positive_words.txt`  | Default positive-sentiment lexicon (40 words)    |
| `negative_words.txt`  | Default negative-sentiment lexicon (40 words)    |

### Menu overview

| Option | Action                                        |
|--------|-----------------------------------------------|
| 1      | Load a text file (required first step)        |
| 2      | Display top 10 most frequent words (console)  |
| 3      | Run sentiment analysis (console output)       |
| 4      | Show readability metrics (console output)     |
| 5      | Plot and save word frequency bar chart        |
| 6      | Plot and save sentiment distribution pie chart|
| 7      | Plot and save sentence length histogram       |
| 8      | Exit the programme                            |

Charts are saved as PNG files (`word_frequency.png`, `sentiment_pie.png`,
`sentence_lengths.png`) in the same directory as the script.

---

## File Structure

```
nlp_analyser/
│
├── text_analyser.py       # Main Python programme
├── sample_text.txt        # Example input text
├── positive_words.txt     # Positive sentiment lexicon
├── negative_words.txt     # Negative sentiment lexicon
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

---

## Troubleshooting

| Problem                          | Solution                                                  |
|----------------------------------|-----------------------------------------------------------|
| `FileNotFoundError`              | Check the file path and ensure the file is in the correct directory |
| `ModuleNotFoundError: matplotlib`| Run `pip install -r requirements.txt`                    |
| Charts open but do not display   | Ensure a GUI backend is available; on headless servers use `matplotlib.use('Agg')` |
| Empty sentiment results          | Verify lexicon files are loaded and contain uncommented words |
