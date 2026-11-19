"""N-gram frequency analysis of a Reddit dog-whistle comment sample.

Scope
-----
This script performs only:
1. Bigram and trigram extraction from Reddit comment text.
2. Minimum occurrence/document-frequency filtering.
3. Ranking by occurrence and document frequency.
4. Extraction of example comments for high-ranking phrases not listed as
   exact entries in the Silent Signals reference dictionary.
5. Exact phrase-level comparison of the filtered n-grams with that dictionary.

It does not perform the separate assigned-label frequency analysis or
comment-level dictionary matching. An n-gram absent from the reference
lexicon is only an unlisted candidate for human review, not a confirmed
new dog whistle.
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
OUTPUT_DIR = BASE_DIR / "ngram_analysis_results"

# Keep an n-gram only if it occurs at least this many times in total AND
# appears in at least this many distinct comment texts.
MIN_OCCURRENCE_FREQUENCY = 5
MIN_DOCUMENT_FREQUENCY = 3

# Export the top 100 qualifying bigrams and top 100 qualifying trigrams.
TOP_N = 100

# Show examples for the highest-ranked unlisted phrases of each n-gram size.
EXAMPLE_CANDIDATES_PER_N = 10
EXAMPLES_PER_CANDIDATE = 2
MAX_EXAMPLE_CHARS_IN_TERMINAL = 350

REDDIT_REQUIRED_COLUMNS = {"content", "subreddit"}
DICTIONARY_REQUIRED_COLUMNS = {"dog_whistle"}

# Words, contractions, hyphenated forms and numeric tokens (e.g. 13%) are kept.
# Emojis and punctuation are not treated as tokens.
TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|\d+(?:[.,]\d+)?%?",
    flags=re.UNICODE,
)

# A lightweight sentence-boundary heuristic. N-grams are not generated across
# full stops, question marks, exclamation marks or line breaks.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


# -----------------------------------------------------------------------------
# Input validation and text processing
# -----------------------------------------------------------------------------


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Raise a readable error if an input CSV lacks required columns."""
    missing = required_columns.difference(dataframe.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing column(s): {missing_text}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Reddit comments and lexicon used for phrase comparison."""
    for path in (REDDIT_FILE, DICTIONARY_FILE):
        if not path.is_file():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Place both CSV files in the same folder as this script."
            )

    reddit_df = pd.read_csv(REDDIT_FILE, encoding="utf-8-sig")
    dictionary_df = pd.read_csv(DICTIONARY_FILE, encoding="utf-8-sig")

    validate_columns(reddit_df, REDDIT_REQUIRED_COLUMNS, "Reddit dataset")
    validate_columns(dictionary_df, DICTIONARY_REQUIRED_COLUMNS, "Reference dictionary")
    return reddit_df, dictionary_df


def normalize_phrase(value: object) -> str:
    """Decode HTML entities, case-fold, and normalize whitespace."""
    if pd.isna(value):
        return ""
    return " ".join(unescape(str(value)).casefold().split())


def tokenize_document(document: object) -> list[list[str]]:
    """Return token lists sentence by sentence for one comment."""
    if pd.isna(document):
        return []

    text = unescape(str(document)).casefold()
    sentences = SENTENCE_SPLIT_PATTERN.split(text)
    tokenized_sentences: list[list[str]] = []
    for sentence in sentences:
        tokens = TOKEN_PATTERN.findall(sentence)
        if tokens:
            tokenized_sentences.append(tokens)
    return tokenized_sentences


def count_ngrams_in_sentences(
    sentence_tokens: Iterable[list[str]],
    n_value: int,
) -> Counter[str]:
    """Count n-gram occurrences within one document, without crossing sentences."""
    counts: Counter[str] = Counter()
    for tokens in sentence_tokens:
        if len(tokens) < n_value:
            continue
        counts.update(
            " ".join(tokens[index:index + n_value])
            for index in range(len(tokens) - n_value + 1)
        )
    return counts


def count_phrase_in_sentences(
    sentence_tokens: list[list[str]],
    phrase_tokens: list[str],
) -> int:
    """Count exact consecutive token-sequence matches in one document."""
    n_value = len(phrase_tokens)
    total = 0
    for tokens in sentence_tokens:
        for index in range(len(tokens) - n_value + 1):
            if tokens[index:index + n_value] == phrase_tokens:
                total += 1
    return total


# -----------------------------------------------------------------------------
# N-gram extraction and frequency filtering
# -----------------------------------------------------------------------------


def prepare_documents(reddit_df: pd.DataFrame) -> pd.DataFrame:
    """Treat each distinct, non-empty comment text as one document.

    If identical comment text occurs in more than one row, the first row is
    retained for the example metadata. Document frequency therefore means
    number of distinct comment texts, not number of source rows.
    """
    documents = reddit_df[["content", "subreddit"]].copy()
    documents = documents.dropna(subset=["content"])
    documents["content"] = documents["content"].astype(str)
    # Use exact comment-text deduplication to match the earlier analysis.
    # Normalization is used only to remove comments containing whitespace alone.
    documents = documents[documents["content"].map(normalize_phrase).ne("")]
    documents = documents.drop_duplicates(subset="content").reset_index(drop=True)
    documents.insert(0, "document_id", range(1, len(documents) + 1))
    return documents


def build_ngram_table(
    occurrence_counts: Counter[str],
    document_counts: Counter[str],
) -> pd.DataFrame:
    """Create a table containing total occurrences and distinct-document counts."""
    records = [
        {
            "ngram": phrase,
            "occurrence_frequency": occurrence_frequency,
            "document_frequency": document_counts[phrase],
        }
        for phrase, occurrence_frequency in occurrence_counts.items()
    ]
    if not records:
        return pd.DataFrame(
            columns=["ngram", "occurrence_frequency", "document_frequency"]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["occurrence_frequency", "document_frequency", "ngram"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def filter_ngram_table(table: pd.DataFrame) -> pd.DataFrame:
    """Apply both minimum occurrence-frequency and document-frequency cutoffs."""
    if table.empty:
        return table.copy()

    filtered = table[
        (table["occurrence_frequency"] >= MIN_OCCURRENCE_FREQUENCY)
        & (table["document_frequency"] >= MIN_DOCUMENT_FREQUENCY)
    ].copy()

    return filtered.sort_values(
        ["occurrence_frequency", "document_frequency", "ngram"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def extract_and_filter_ngrams(
    documents: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], list[dict[str, object]]]:
    """Extract bigrams/trigrams, count occurrences and documents, then filter."""
    occurrence_counts = {2: Counter(), 3: Counter()}
    document_counts = {2: Counter(), 3: Counter()}
    tokenized_documents: list[dict[str, object]] = []

    for row in documents.itertuples(index=False):
        sentence_tokens = tokenize_document(row.content)
        tokenized_documents.append(
            {
                "document_id": row.document_id,
                "subreddit": row.subreddit,
                "content": row.content,
                "sentence_tokens": sentence_tokens,
            }
        )

        for n_value in (2, 3):
            document_ngram_counts = count_ngrams_in_sentences(
                sentence_tokens, n_value
            )
            occurrence_counts[n_value].update(document_ngram_counts)
            # Each phrase contributes only once per document to document frequency.
            document_counts[n_value].update(document_ngram_counts.keys())

    filtered_tables: dict[int, pd.DataFrame] = {}
    for n_value in (2, 3):
        unfiltered_table = build_ngram_table(
            occurrence_counts[n_value], document_counts[n_value]
        )
        filtered_table = filter_ngram_table(unfiltered_table)
        filtered_tables[n_value] = filtered_table

        label = "bigrams" if n_value == 2 else "trigrams"
        print(f"\n{label.capitalize()} before filtering: {len(unfiltered_table):,}")
        print(f"{label.capitalize()} after filtering:  {len(filtered_table):,}")

    return filtered_tables, tokenized_documents


# -----------------------------------------------------------------------------
# Phrase-level comparison with the Silent Signals reference lexicon
# -----------------------------------------------------------------------------


def add_reference_status(
    ngram_table: pd.DataFrame,
    dictionary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mark whether each extracted phrase is an exact lexicon entry.

    This is an exact normalized phrase lookup only. It is not a second
    dictionary scan of the raw comments.
    """
    result = ngram_table.copy()
    lexicon_terms = {
        normalize_phrase(term)
        for term in dictionary_df["dog_whistle"].dropna()
        if normalize_phrase(term)
    }

    result["reference_status"] = result["ngram"].map(
        lambda phrase: (
            "listed_in_silent_signals"
            if phrase in lexicon_terms
            else "not_listed_exactly_in_silent_signals"
        )
    )
    return result


# -----------------------------------------------------------------------------
# Example comments for high-ranking unlisted candidates
# -----------------------------------------------------------------------------


def collect_candidate_examples(
    filtered_tables: dict[int, pd.DataFrame],
    tokenized_documents: list[dict[str, object]],
    dictionary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Collect a small number of source comments for high-ranking unlisted n-grams.

    Candidates are selected separately for bigrams and trigrams. Ranking uses
    document frequency first, then occurrence frequency, so phrases occurring
    across more distinct comments are prioritized over phrases repeated many
    times in only a few comments.
    """
    lexicon_terms = {
        normalize_phrase(term)
        for term in dictionary_df["dog_whistle"].dropna()
        if normalize_phrase(term)
    }
    example_records: list[dict[str, object]] = []

    for n_value in (2, 3):
        table = filtered_tables[n_value]
        if table.empty:
            continue

        unlisted = table[~table["ngram"].isin(lexicon_terms)].copy()
        unlisted = unlisted.sort_values(
            ["document_frequency", "occurrence_frequency", "ngram"],
            ascending=[False, False, True],
        ).head(EXAMPLE_CANDIDATES_PER_N)

        for candidate in unlisted.itertuples(index=False):
            phrase_tokens = candidate.ngram.split()
            examples_found = 0

            for document in tokenized_documents:
                sentence_tokens = document["sentence_tokens"]
                count_in_document = count_phrase_in_sentences(
                    sentence_tokens, phrase_tokens
                )
                if count_in_document == 0:
                    continue

                example_records.append(
                    {
                        "n": n_value,
                        "ngram": candidate.ngram,
                        "occurrence_frequency": candidate.occurrence_frequency,
                        "document_frequency": candidate.document_frequency,
                        "reference_status": "not_listed_exactly_in_silent_signals",
                        "document_id": document["document_id"],
                        "subreddit": document["subreddit"],
                        "occurrences_in_this_comment": count_in_document,
                        "example_comment": document["content"],
                    }
                )
                examples_found += 1
                if examples_found >= EXAMPLES_PER_CANDIDATE:
                    break

    columns = [
        "n", "ngram", "occurrence_frequency", "document_frequency",
        "reference_status", "document_id", "subreddit",
        "occurrences_in_this_comment", "example_comment",
    ]
    return pd.DataFrame(example_records, columns=columns)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the n-gram-only analysis and write its outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reddit_df, dictionary_df = load_inputs()
    documents = prepare_documents(reddit_df)

    print("=" * 72)
    print("N-GRAM FREQUENCY ANALYSIS")
    print(f"Reddit rows: {len(reddit_df):,}")
    print(f"Distinct comment texts (documents): {len(documents):,}")
    print(f"Reference lexicon entries: {dictionary_df['dog_whistle'].nunique():,}")
    print(f"Minimum occurrence frequency: {MIN_OCCURRENCE_FREQUENCY}")
    print(f"Minimum document frequency: {MIN_DOCUMENT_FREQUENCY}")
    print(f"Top n-grams exported per size: {TOP_N}")

    filtered_tables, tokenized_documents = extract_and_filter_ngrams(documents)

    # Add exact lexicon status to the filtered bigram and trigram tables.
    tables_with_status: dict[int, pd.DataFrame] = {}
    for n_value, table in filtered_tables.items():
        tables_with_status[n_value] = add_reference_status(table, dictionary_df)
        ngram_name = "bigrams" if n_value == 2 else "trigrams"

        all_path = OUTPUT_DIR / f"all_filtered_{ngram_name}.csv"
        top_path = OUTPUT_DIR / f"top_{TOP_N}_{ngram_name}.csv"

        tables_with_status[n_value].to_csv(
            all_path, index=False, encoding="utf-8-sig"
        )
        tables_with_status[n_value].head(TOP_N).to_csv(
            top_path, index=False, encoding="utf-8-sig"
        )

        print(f"\nTop {min(TOP_N, len(tables_with_status[n_value]))} {ngram_name}:")
        print(tables_with_status[n_value].head(TOP_N).to_string(index=False))

        print("\nReference-lexicon coverage among filtered phrases:")
        print(
            tables_with_status[n_value]["reference_status"]
            .value_counts()
            .to_string()
        )

    # Combined filtered candidate table (bigram/trigram only).
    combined_table = pd.concat(
        [tables_with_status[2].assign(n=2), tables_with_status[3].assign(n=3)],
        ignore_index=True,
    )
    combined_table = combined_table[
        ["n", "ngram", "occurrence_frequency", "document_frequency", "reference_status"]
    ]
    combined_table.to_csv(
        OUTPUT_DIR / "all_filtered_bigrams_trigrams_with_reference_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Example comments are collected only for high-ranking n-grams not listed
    # as exact entries in the Silent Signals lexicon.
    examples_df = collect_candidate_examples(
        filtered_tables, tokenized_documents, dictionary_df
    )
    examples_path = OUTPUT_DIR / "examples_for_unlisted_candidates.csv"
    examples_df.to_csv(examples_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("EXAMPLE COMMENTS FOR HIGH-RANKING UNLISTED CANDIDATES")
    if examples_df.empty:
        print("No unlisted candidates met the configured frequency thresholds.")
    else:
        for row in examples_df.head(12).itertuples(index=False):
            comment = str(row.example_comment).replace("\n", " ")
            if len(comment) > MAX_EXAMPLE_CHARS_IN_TERMINAL:
                comment = comment[:MAX_EXAMPLE_CHARS_IN_TERMINAL] + "..."
            print(
                f"\nN={row.n} | Phrase: {row.ngram} | "
                f"Occurrences: {row.occurrence_frequency} | "
                f"Documents: {row.document_frequency} | "
                f"Subreddit: {row.subreddit}"
            )
            print(f"Comment: {comment}")

    print("\n"f"Results directory: {OUTPUT_DIR}")
    print(
        "Interpretation note: an n-gram absent from the reference lexicon is "
        "not necessarily a new dog whistle. It is an unlisted phrase for "
        "further examination. A lexicon match also does not establish coded "
        "usage in a particular comment."
    )


if __name__ == "__main__":
    main()
