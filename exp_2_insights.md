# Experiment 2: PMI and LLR-Based Dog Whistle Candidate Detection

## Objective
To investigate whether statistical association measures can identify potentially interesting word combinations in Reddit discussions beyond simple frequency-based extraction.

## Methodology
- *Dataset:* Shared sample of 5,000 Reddit comments from the SALT-NLP dataset.
- *Preprocessing:* Text cleaning, normalisation and tokenisation.
- *Candidate extraction:* Bigrams and trigrams.
- *Statistical measures:* Pointwise Mutual Information (PMI) and Log-Likelihood Ratio (LLR).
- *Frequency thresholds:* 5, 10 and 20.
- *Reference comparison:* Silent Signals glossary (706 entries).
- *Contextual analysis:* Examined up to 10 Reddit comments per candidate.

## Key Findings

1. **PMI:** Identified less frequent but strongly associated expressions, such as *honk honk* and *virtue signaling*. However, its rankings were sensitive to frequency thresholds.

2. **LLR:** Produced more stable candidate rankings across thresholds, including expressions such as *blue pill*, *clown world* and *gender critical*. It also identified ordinary word combinations.

3. **Dictionary comparison:** Of the 98 selected candidate expressions, 22 matched the Silent Signals reference dictionary, while 76 did not. A missing dictionary match does not establish that an expression is new or has a coded meaning.

4. **Contextual analysis:** Expressions such as *red pill*, *blue pill* and *blue pilled* showed ideological usage in MGTOW discussions. In contrast, *trans people* and *child support* were generally descriptive. *Single mothers* also appeared in derogatory contexts, although the expression itself is not inherently a dog whistle.

## Main Insights
- PMI and LLR identify different types of statistically associated expressions.
- Frequency thresholds are important, particularly because PMI can assign high scores to rare combinations.
- Statistical association and ranking stability do not establish dog-whistle meaning.
- Reference dictionaries are useful but may not capture all potentially relevant expressions.
- Examining usage within specific communities is essential for interpreting candidate expressions.
- Dictionary matching may be affected by differences in phrase length and word forms.
- Contextual examples provided evidence for interpretation but do not constitute definitive human-validated classifications.
