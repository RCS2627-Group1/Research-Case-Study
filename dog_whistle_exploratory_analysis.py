"""Exploratory analysis of a Reddit dog-whistle sample.

Methods implemented
-------------------
1. Frequency analysis of assigned expressions, canonical roots, subreddits,
   and in-group labels.
2. Dictionary matching against a reference lexicon, both at the label level
   and in the raw Reddit comment text.
3. Unigram, bigram, and trigram extraction, with occurrence and document
   frequencies, followed by exact comparison with the reference lexicon.

The script performs lexical exploration only. A dictionary match or a frequent
n-gram does not establish that an expression is being used with a coded meaning.
"""

from __future__ import annotations

import re
from collections import Counter
from html import unescape
from pathlib import Path
from typing import Iterable

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REDDIT_FILE = BASE_DIR / "dog_whistle_reddit_sample_5000.csv"
DICTIONARY_FILE = BASE_DIR / "silent_signals_reference_dictionary.csv"
OUTPUT_DIR = BASE_DIR / "analysis_results"
TOP_N = 20

REDDIT_REQUIRED_COLUMNS = {
    "content",
    "subreddit",
    "dog_whistle",
    "date",
    "ingroup",
    "source",
}
DICTIONARY_REQUIRED_COLUMNS = {
    "dog_whistle",
    "dog_whistle_root",
    "definition",
    "ingroup",
}

# Words (including Unicode letters), contractions, hyphenated forms, and
# numeric tokens such as 13% are retained. Emojis are not tokenized here.
TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|\d+(?:[.,]\d+)?%?",
    flags=re.UNICODE,
)

# A lightweight sentence-boundary heuristic. It prevents n-grams from being
# created across full stops, question marks, exclamation marks, or newlines.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


# -----------------------------------------------------------------------------
# General utilities and input validation
# -----------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    """Decode HTML entities, case-fold text, and normalize whitespace."""
    if pd.isna(value):
        return ""
    text = unescape(str(value)).casefold()
    return " ".join(text.split())


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Raise a clear error if required columns are absent."""
    missing = required_columns.difference(dataframe.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"{dataset_name} is missing required column(s): {missing_list}"
        )


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Reddit sample and reference dictionary from the script folder."""
    for path in (REDDIT_FILE, DICTIONARY_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"Could not find: {path}\n"
                "Place both CSV files in the same folder as this Python script."
            )

    reddit_df = pd.read_csv(REDDIT_FILE, encoding="utf-8-sig")
    dictionary_df = pd.read_csv(DICTIONARY_FILE, encoding="utf-8-sig")

    validate_columns(reddit_df, REDDIT_REQUIRED_COLUMNS, "Reddit dataset")
    validate_columns(dictionary_df, DICTIONARY_REQUIRED_COLUMNS, "Reference dictionary")

    return reddit_df, dictionary_df


def print_section(title: str) -> None:
    """Print a consistent section heading."""
    print("\n" + "=" * 72)
    print(title.upper())
    print("=" * 72)


def save_series(
    series: pd.Series,
    output_path: Path,
    index_name: str,
    value_name: str,
) -> None:
    """Save a frequency series as a consistently named CSV table."""
    output = series.rename(value_name).rename_axis(index_name).reset_index()
    output.to_csv(output_path, index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# Dataset inspection
# -----------------------------------------------------------------------------

def inspect_datasets(
    reddit_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
) -> None:
    """Report dataset dimensions, columns, missing values, and duplicates."""
    print_section("Dataset overview")
    print(f"Reddit dataset: {reddit_df.shape[0]:,} rows × {reddit_df.shape[1]} columns")
    print(f"Reference dictionary: {dictionary_df.shape[0]:,} rows × {dictionary_df.shape[1]} columns")

    print("\nReddit columns:", reddit_df.columns.tolist())
    print("Dictionary columns:", dictionary_df.columns.tolist())

    print("\nMissing values in Reddit dataset:")
    print(reddit_df.isna().sum().to_string())

    print(f"\nExact duplicate Reddit rows: {reddit_df.duplicated().sum():,}")
    print(f"Unique comment texts: {reddit_df['content'].nunique(dropna=True):,}")
    print(f"Unique non-missing expression labels: {reddit_df['dog_whistle'].nunique(dropna=True):,}")
    print(f"Unique non-missing subreddits: {reddit_df['subreddit'].nunique(dropna=True):,}")


# -----------------------------------------------------------------------------
# Method 1: Frequency analysis
# -----------------------------------------------------------------------------

def run_frequency_analysis(
    reddit_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Calculate frequencies for expression labels, roots, subreddits, and groups."""
    print_section("Method 1: Frequency analysis")

    result_df = reddit_df.copy()
    dictionary = dictionary_df.copy()

    result_df["expression_key"] = result_df["dog_whistle"].map(normalize_text)
    dictionary["expression_key"] = dictionary["dog_whistle"].map(normalize_text)

    # Build a normalized expression-to-root lookup. If a label is not in the
    # reference dictionary, retain the normalized label rather than dropping it.
    root_mapping = (
        dictionary.loc[dictionary["expression_key"].ne(""),
                       ["expression_key", "dog_whistle_root"]]
        .drop_duplicates(subset="expression_key")
        .set_index("expression_key")["dog_whistle_root"]
        .to_dict()
    )

    result_df["canonical_root"] = result_df["expression_key"].map(root_mapping)
    result_df["canonical_root"] = result_df["canonical_root"].map(normalize_text)
    result_df.loc[result_df["canonical_root"].eq(""), "canonical_root"] = (
        result_df.loc[result_df["canonical_root"].eq(""), "expression_key"]
    )

    expression_frequency = (
        result_df.loc[result_df["expression_key"].ne(""), "expression_key"]
        .value_counts()
    )

    root_frequency = (
        result_df.loc[result_df["canonical_root"].ne(""), "canonical_root"]
        .value_counts()
    )

    subreddit_frequency = result_df["subreddit"].value_counts()
    ingroup_frequency = result_df["ingroup"].value_counts()

    # Number of distinct comment texts associated with each assigned expression.
    unique_comment_frequency = (
        result_df.loc[result_df["expression_key"].ne("")]
        .groupby("expression_key")["content"]
        .nunique()
        .sort_values(ascending=False)
    )

    tables = {
        "expression_frequency": expression_frequency,
        "root_frequency": root_frequency,
        "subreddit_frequency": subreddit_frequency,
        "ingroup_frequency": ingroup_frequency,
        "unique_comment_frequency": unique_comment_frequency,
    }

    for title, series in tables.items():
        print(f"\nTop {TOP_N}: {title.replace('_', ' ')}")
        print(series.head(TOP_N).to_string())

    save_series(expression_frequency, OUTPUT_DIR / "expression_frequency.csv",
                "expression", "record_frequency")
    save_series(root_frequency, OUTPUT_DIR / "root_frequency.csv",
                "canonical_root", "record_frequency")
    save_series(subreddit_frequency, OUTPUT_DIR / "subreddit_frequency.csv",
                "subreddit", "record_frequency")
    save_series(ingroup_frequency, OUTPUT_DIR / "ingroup_frequency.csv",
                "ingroup", "record_frequency")
    save_series(unique_comment_frequency, OUTPUT_DIR / "unique_comment_frequency.csv",
                "expression", "unique_comment_frequency")

    return tables


# -----------------------------------------------------------------------------
# Method 2: Dictionary matching
# -----------------------------------------------------------------------------

def build_dictionary_pattern(dictionary_terms: Iterable[str]) -> re.Pattern[str] | None:
    """Compile a case-insensitive lexicon matcher with longest phrases first.

    Matching is whole-token based and permits flexible whitespace between words.
    When dictionary entries overlap (e.g. a single-word entry and a longer
    multiword entry), the longer entry is prioritized.
    """
    terms = sorted(
        {term for term in dictionary_terms if term},
        key=lambda term: (-len(term.split()), -len(term), term),
    )

    if not terms:
        return None

    pattern_parts = []
    for term in terms:
        escaped_words = [re.escape(word) for word in term.split()]
        phrase_pattern = r"\s+".join(escaped_words)
        pattern_parts.append(r"(?<!\w)" + phrase_pattern + r"(?!\w)")

    return re.compile("|".join(pattern_parts), flags=re.UNICODE)


def find_dictionary_matches(
    comment: object,
    dictionary_pattern: re.Pattern[str] | None,
) -> list[str]:
    """Return non-overlapping lexicon matches in one comment."""
    if pd.isna(comment) or dictionary_pattern is None:
        return []

    normalized_comment = normalize_text(comment)
    return [
        normalize_text(match.group(0))
        for match in dictionary_pattern.finditer(normalized_comment)
    ]


def run_dictionary_matching(
    reddit_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compare assigned labels with the lexicon and search the comment text."""
    print_section("Method 2: Dictionary matching")

    result_df = reddit_df.copy()
    dictionary = dictionary_df.copy()

    result_df["expression_key"] = result_df["dog_whistle"].map(normalize_text)
    dictionary["expression_key"] = dictionary["dog_whistle"].map(normalize_text)

    reddit_expressions = {
        term for term in result_df["expression_key"].dropna().tolist() if term
    }
    dictionary_expressions = {
        term for term in dictionary["expression_key"].dropna().tolist() if term
    }

    matching_expressions = sorted(reddit_expressions & dictionary_expressions)
    reddit_only = sorted(reddit_expressions - dictionary_expressions)
    dictionary_only = sorted(dictionary_expressions - reddit_expressions)

    print(f"Unique Reddit expression labels: {len(reddit_expressions):,}")
    print(f"Unique dictionary expressions: {len(dictionary_expressions):,}")
    print(f"Overlapping expressions: {len(matching_expressions):,}")
    print(f"Reddit-only expression labels: {len(reddit_only):,}")
    print(f"Dictionary-only expressions: {len(dictionary_only):,}")

    overlap_df = pd.DataFrame({
        "expression": matching_expressions,
        "present_in_reddit_labels": True,
        "present_in_dictionary": True,
    })
    overlap_df.to_csv(OUTPUT_DIR / "label_dictionary_overlap.csv",
                      index=False, encoding="utf-8-sig")
    pd.DataFrame({"expression": reddit_only}).to_csv(
        OUTPUT_DIR / "reddit_only_expression_labels.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame({"expression": dictionary_only}).to_csv(
        OUTPUT_DIR / "dictionary_only_expressions.csv",
        index=False, encoding="utf-8-sig")

    dictionary_pattern = build_dictionary_pattern(dictionary_expressions)
    result_df["matched_expressions"] = result_df["content"].apply(
        lambda comment: find_dictionary_matches(comment, dictionary_pattern)
    )
    result_df["number_of_dictionary_matches"] = result_df["matched_expressions"].str.len()

    # One row per matched occurrence. Repeated appearances in one comment are
    # retained in this table and therefore contribute to occurrence frequency.
    occurrence_records: list[dict[str, object]] = []
    for record_id, row in result_df.reset_index(drop=True).iterrows():
        for expression in row["matched_expressions"]:
            occurrence_records.append({
                "record_id": record_id + 1,
                "matched_expression": expression,
                "subreddit": row["subreddit"],
                "ingroup": row["ingroup"],
                "assigned_expression": row["dog_whistle"],
            })

    occurrences_df = pd.DataFrame(
        occurrence_records,
        columns=[
            "record_id", "matched_expression", "subreddit", "ingroup",
            "assigned_expression",
        ],
    )

    if not occurrences_df.empty:
        occurrence_frequency = (
            occurrences_df.groupby("matched_expression").size()
            .sort_values(ascending=False)
        )
        record_frequency = (
            occurrences_df.groupby("matched_expression")["record_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        match_summary = pd.concat(
            [occurrence_frequency.rename("occurrence_frequency"),
             record_frequency.rename("record_frequency")],
            axis=1,
        ).reset_index()
    else:
        match_summary = pd.DataFrame(columns=[
            "matched_expression", "occurrence_frequency", "record_frequency"
        ])

    print(f"\nRecords containing at least one dictionary match: "
          f"{result_df['number_of_dictionary_matches'].gt(0).sum():,}")
    print(f"Total non-overlapping dictionary matches: {len(occurrences_df):,}")
    print("\nTop dictionary matches in comment text:")
    if not match_summary.empty:
        print(match_summary.head(TOP_N).to_string(index=False))
    else:
        print("No dictionary matches found in comment text.")

    occurrences_df.to_csv(
        OUTPUT_DIR / "dictionary_match_occurrences.csv",
        index=False, encoding="utf-8-sig")
    match_summary.to_csv(
        OUTPUT_DIR / "dictionary_match_frequency.csv",
        index=False, encoding="utf-8-sig")

    comments_output = result_df.copy()
    comments_output["matched_expressions"] = comments_output["matched_expressions"].apply(
        lambda matches: " | ".join(matches)
    )
    comments_output.to_csv(
        OUTPUT_DIR / "comments_with_dictionary_matches.csv",
        index=False, encoding="utf-8-sig")

    return {
        "label_overlap": overlap_df,
        "occurrences": occurrences_df,
        "match_summary": match_summary,
        "comments": comments_output,
    }


# -----------------------------------------------------------------------------
# Method 3: N-gram extraction
# -----------------------------------------------------------------------------

def tokenize_sentence(sentence: str) -> list[str]:
    """Tokenize one sentence, preserving words, contractions, and numeric tokens."""
    return TOKEN_PATTERN.findall(sentence.casefold())


def extract_ngrams_from_document(
    document: object,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Extract unigram, bigram, and trigram occurrences from one comment.

    Sentence segmentation is applied before n-gram generation so that a
    sequence does not cross a detected sentence boundary.
    """
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    trigram_counts: Counter[str] = Counter()

    if pd.isna(document):
        return unigram_counts, bigram_counts, trigram_counts

    text = unescape(str(document)).casefold()
    sentences = SENTENCE_SPLIT_PATTERN.split(text)

    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        unigram_counts.update(tokens)

        bigram_counts.update(
            " ".join(tokens[index:index + 2])
            for index in range(len(tokens) - 1)
        )
        trigram_counts.update(
            " ".join(tokens[index:index + 3])
            for index in range(len(tokens) - 2)
        )

    return unigram_counts, bigram_counts, trigram_counts


def build_ngram_table(
    occurrence_counts: Counter[str],
    document_counts: Counter[str],
) -> pd.DataFrame:
    """Create a ranked table with occurrence and document frequencies."""
    terms = occurrence_counts.most_common()
    return pd.DataFrame(
        [
            {
                "ngram": term,
                "occurrence_frequency": count,
                "document_frequency": document_counts.get(term, 0),
            }
            for term, count in terms
        ],
        columns=["ngram", "occurrence_frequency", "document_frequency"],
    )


def run_ngram_extraction(reddit_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Extract and save unigram, bigram, and trigram frequency tables."""
    print_section("Method 3: N-gram extraction")

    # Each unique, non-missing comment text is treated as one document.
    documents = (
        reddit_df["content"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    print(f"Reddit records: {len(reddit_df):,}")
    print(f"Unique comment texts treated as documents: {len(documents):,}")

    occurrence_counts = {
        1: Counter(),
        2: Counter(),
        3: Counter(),
    }
    document_counts = {
        1: Counter(),
        2: Counter(),
        3: Counter(),
    }

    for document in documents:
        unigram_doc, bigram_doc, trigram_doc = extract_ngrams_from_document(document)
        per_document = {
            1: unigram_doc,
            2: bigram_doc,
            3: trigram_doc,
        }

        for n_value, counts in per_document.items():
            occurrence_counts[n_value].update(counts)
            # Count each n-gram no more than once per document for document frequency.
            document_counts[n_value].update(counts.keys())

    labels = {1: "unigram", 2: "bigram", 3: "trigram"}
    tables: dict[int, pd.DataFrame] = {}

    for n_value in (1, 2, 3):
        table = build_ngram_table(
            occurrence_counts[n_value], document_counts[n_value]
        )
        tables[n_value] = table
        table.to_csv(
            OUTPUT_DIR / f"{labels[n_value]}_frequency.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"\nTop {TOP_N} {labels[n_value]}s:")
        print(table.head(TOP_N).to_string(index=False))

    return tables


def compare_ngrams_with_dictionary(
    ngram_tables: dict[int, pd.DataFrame],
    dictionary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find exact bigram/trigram matches in the reference dictionary."""
    print_section("N-gram and dictionary comparison")

    dictionary_terms = {
        normalize_text(term)
        for term in dictionary_df["dog_whistle"].dropna()
        if normalize_text(term)
    }

    matching_tables = []
    for n_value in (2, 3):
        table = ngram_tables[n_value]
        matches = table[table["ngram"].isin(dictionary_terms)].copy()
        matches.insert(0, "n", n_value)
        matching_tables.append(matches)

        label = "bigrams" if n_value == 2 else "trigrams"
        print(f"\nDictionary-matching {label}: {len(matches):,} distinct expressions")
        if not matches.empty:
            print(matches.head(TOP_N).to_string(index=False))

    comparison_df = pd.concat(matching_tables, ignore_index=True)
    comparison_df.to_csv(
        OUTPUT_DIR / "ngram_dictionary_matches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return comparison_df


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    """Run all three exploratory analyses and save their outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reddit_df, dictionary_df = load_datasets()
    inspect_datasets(reddit_df, dictionary_df)

    run_frequency_analysis(reddit_df, dictionary_df)
    run_dictionary_matching(reddit_df, dictionary_df)
    ngram_tables = run_ngram_extraction(reddit_df)
    compare_ngrams_with_dictionary(ngram_tables, dictionary_df)

    print_section("Analysis completed")
    print(f"All result files have been saved to: {OUTPUT_DIR}")
    print(
        "Interpretation note: frequency, dictionary, and n-gram matches are "
        "lexical observations. They do not by themselves establish coded usage."
    )


if __name__ == "__main__":
    main()
