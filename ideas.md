Put down your ideas for our project


idea 1:
  - test pre-trained models e.g. on hate speech, toxicity, euphemism detection on silent signals dataset. compare results with fetch! baseline
  - fine-tune pre-trained models and then test, compare results
  - research in literature which should perform best
  - strengths of proposal: three tasks that can be worked on simultaneously in sub-groups.
  - Q: How good are other models at detecting dog whistles? Could they be used as a cheap alternative to EarShot?

idea 2 (sidequest):
  - analyse dog whistle datasets. Are there terms that appear together often that can help detect dog whistles in a cheap way?

--------------------------------------------------------------------------------------------------------------------------------------------------------------

**A Hybrid Method for Detecting and Explaining Emerging Dog Whistles:**
We could use a hybrid approach that combines statistical methods, contextual analysis and LLMs to discover emerging dog whistles. Instead of relying only on a fixed dictionary of known expressions, we would try to identify new candidate expressions from their usage patterns and then investigate whether they have a coded meaning.
I propose dividing the process into three main stages.
Stage 1: Generate Candidate Expressions Using Statistical Methods
What we do: First, we establish a dictionary-based baseline and then use statistical methods to identify potentially interesting expressions that are not already in the dictionary.
Datasets: We could start with the Silent Signals dataset from SALT-NLP, which contains known dog-whistle examples and their contexts. We could also explore the Informal Potential Dog Whistles from Reddit dataset to investigate candidate expressions in a larger collection of informal online discussions.
Methods:
•	Dictionary matching: Identify known dog-whistle expressions in the text. This gives us a baseline for comparison.
•	N-gram extraction: Extract bigrams (two-word combinations) and trigrams (three-word combinations) from the Reddit text.
•	Frequency filtering: Remove extremely rare combinations that might otherwise receive misleadingly high statistical scores.
•	Pointwise Mutual Information (PMI): Measure how strongly two words are associated compared with what we would expect if they occurred independently.
•	Log-likelihood ratio (LLR): Optionally use this additional association measure to help rank candidate combinations.
We would then examine phrases that are statistically interesting but absent from our existing dictionary.
For example, we might investigate a phrase such as “soy face” alongside ordinary expressions such as “soy sauce” and “soy milk.” The goal would not be to assume that the first phrase is a dog whistle, but to investigate whether its usage provides evidence of a coded meaning.
Purpose: This stage generates a shortlist of candidate expressions that dictionary matching alone might miss. Statistical scores help us prioritise candidates, but they cannot establish whether an expression is actually a dog whistle.
Stage 2: Expand and Investigate Candidates Using Context
What we do: Statistical methods may identify interesting word combinations, but they might also miss expressions whose individual words are not strongly associated or whose coded meaning depends heavily on context. To address this, we could use the contexts of known and newly identified expressions to find additional candidates.
Datasets: We would use the candidate expressions and their surrounding posts from Stage 1, together with known examples and contextual information from Silent Signals.
Methods:
•	Context collection: Gather multiple posts containing each candidate expression and examine the surrounding text.
•	Context-based candidate expansion: Identify other phrases that appear in similar contexts to known dog whistles or promising candidates, even if those phrases have not yet appeared in our dictionary.
•	Contextual embeddings (optional): Represent expressions according to their surrounding text and compare their usage patterns. We could use a pretrained sentence-embedding model to identify potentially similar contexts.
•	Candidate filtering: Remove duplicates and prioritise expressions that have recurring contextual patterns for further investigation.
For example, if a known dog-whistle expression repeatedly appears in discussions about a particular social or political topic, we could investigate other phrases that occur in similar contexts. These phrases would become additional candidates, not confirmed dog whistles.
Purpose: This stage expands candidate discovery beyond simple word-frequency statistics. It allows us to investigate phrases based on how and where they are used, rather than relying exclusively on their statistical association scores.
We would need to be careful because expressions can appear in similar discussions without sharing the same coded meaning. Contextual similarity would therefore be a signal for further investigation, not proof of a dog whistle.
Stage 3: Use an LLM to Interpret and Verify Candidate Uses
What we do: Finally, we use an LLM to help interpret the candidate expressions in context. Rather than asking the model to discover and confirm dog whistles on its own, we provide it with candidate expressions and their surrounding posts and ask it to evaluate their possible meanings.
Model: We could begin with Llama 3.1 8B Instruct, an open-weight model, and consider a second model only if time and resources permit.
Method: For each candidate occurrence, we ask the model to classify its use into categories such as:
•	CODED: The context supports a coded meaning.
•	LITERAL: The expression is being used in its ordinary meaning.
•	MENTION OR QUOTATION: The author is discussing or quoting the expression rather than using it with that meaning.
•	OTHER: The expression has another use, such as an ordinary insult, that does not qualify as a dog whistle.
•	UNCERTAIN: The available evidence is insufficient to make a reliable judgement.
We would also ask the model to explain its interpretation using evidence from the post and consider alternative explanations. Human annotators would then review a sample of the predictions to determine whether the interpretations are justified.
Purpose: This stage helps us distinguish genuine coded usage from literal, quoted or otherwise non-coded usage. It also allows us to investigate whether the candidates identified through statistical and contextual methods actually have the meanings we suspect.
Evaluation: Does the Combined Approach Work Better?
We would compare our approach with the dictionary-based baseline and evaluate the contribution of the additional stages.
For candidate discovery, we could measure Precision@20 or Precision@50, which tells us how many of the top-ranked candidates human annotators judge to be relevant. For contextual classification, we could measure precision, recall and macro-F1 across the different usage categories.
We would also examine false positives, such as ordinary phrases mistakenly identified as dog whistles, and false negatives, where genuine coded uses are missed.
To test whether the method can generalise to new expressions, we should hold out entire expressions from our project’s development data rather than randomly splitting posts containing the same expressions across training and test sets.
We could begin with a small pilot, evaluate which stages contribute useful results, and only add more complex components, such as contextual clustering, if the initial experiments justify them.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Hybrid Discovery and Fine-Tuned Classification for Emerging Dog Whistles:**
I propose a hybrid approach that combines dictionary-based detection, statistical candidate discovery and machine-learning classification. The goal is to investigate whether we can identify emerging dog whistles that are missing from existing dictionaries and distinguish their coded uses from ordinary uses in context.
We could divide the process into four stages.
Stage 1: Establish a Dictionary-Based Baseline and Discover Candidate Expressions
What we do: First, we use a dictionary of known dog-whistle expressions to identify matching phrases in the text. This gives us our baseline. We then use statistical methods to discover potentially interesting expressions that are not already in the dictionary.
Datasets: We could start with the Silent Signals dataset from SALT-NLP and investigate the Informal Potential Dog Whistles from Reddit dataset for additional candidate expressions and their contexts.
Methods:
•	Dictionary matching: Search the text for known dog-whistle expressions and record their occurrences.
•	N-gram extraction: Extract bigrams and trigrams from the Reddit text.
•	Frequency filtering: Remove extremely rare combinations.
•	PMI and log-likelihood ratio (LLR): Rank word combinations according to their statistical associations.
•	Context-based candidate expansion: Look for additional phrases that appear in contexts similar to known dog whistles or promising candidates.
We would prioritise expressions that are absent from the dictionary and collect their surrounding posts for further analysis.
Purpose: This stage establishes a simple baseline while generating candidates that dictionary matching alone might miss. Statistical and contextual signals identify phrases worth investigating; they do not prove that those phrases are dog whistles.
Stage 2: Prepare a Labelled Dataset
What we do: We prepare labelled examples to train and evaluate the classification models.
Dataset: We would use known coded examples from Silent Signals and supplement them with appropriate literal, quoted and other non-coded examples. We would inspect the dataset's existing labels and manually annotate additional examples where necessary.
Each example would contain a candidate expression and its surrounding text, labelled as one of the following:
•	CODED: The context supports a coded meaning.
•	LITERAL: The expression is used in its ordinary meaning.
•	MENTION OR QUOTATION: The expression is being discussed or quoted.
•	OTHER: Another use that does not qualify as a dog whistle, such as an ordinary insult.
We could also allow an UNCERTAIN outcome when the evidence is insufficient to make a reliable judgement.
Purpose: This stage creates the labelled data needed to train the classifiers and evaluate whether they can distinguish coded usage from non-coded usage.
Stage 3: Train and Compare Two Classification Models
What we do: We train two models to classify candidate expressions based on their surrounding text.
Model A: TF-IDF + Logistic Regression
This is our simpler machine-learning model. TF-IDF converts the text into numerical features, and Logistic Regression learns patterns associated with the different usage categories.
It is relatively fast to train and gives us a useful machine-learning baseline.
Model B: Fine-tuned DistilBERT
We fine-tune the pretrained DistilBERT model for sequence classification. We provide the candidate expression and its surrounding text, and the model learns to predict the appropriate category.
Because DistilBERT uses contextual representations, it may be better able to distinguish different uses of the same expression. We would test this rather than assume it will outperform Logistic Regression.
Purpose: This stage compares a conventional machine-learning classifier with a fine-tuned contextual language model.
Stage 4: Evaluate the Complete Pipeline
What we do: We evaluate candidate discovery and contextual classification separately.
First, we compare dictionary matching with the statistical and context-based discovery methods to determine whether they identify useful candidates missing from the dictionary.
Next, we compare TF-IDF + Logistic Regression with fine-tuned DistilBERT to determine whether contextual fine-tuning improves classification.
For candidate discovery, we could measure Precision@20 or Precision@50 using human judgements. For classification, we could measure precision, recall and macro-F1 across the usage categories. We would also examine false positives, including ordinary phrases mistakenly classified as dog whistles.
We would hold out entire expressions from training and development to test whether the classifiers generalise to expressions excluded from our project data.
Purpose: This stage tells us whether the discovery methods add value beyond dictionary matching and whether the fine-tuned model improves on the simpler classifier.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Glossary-Grounded Retrieval for Emerging Dog Whistles:**
1. Research Questions
This project investigates whether retrieving information about known dog whistles can help an LLM interpret expressions it has not encountered before.
Our main research question could be:
Does providing an LLM with relevant examples of known dog whistles improve its ability to identify and interpret previously unseen dog whistles?
We will also investigate whether differences in how expressions are used across online communities can help discover potential new dog whistles, and whether the retrieval-based approach generalises to expressions excluded from the retrieval collection during testing.
2. Data
We will use the Silent Signals dataset as our primary resource because it contains known dog whistles and examples of their usage. Its glossary will provide expressions, coded meanings, and contextual examples for retrieval.
The Informal Potential Dog Whistles from Reddit dataset will support candidate discovery. We will also use a general-text reference corpus, selected after inspecting available data, to compare language usage across communities.
We will prepare a held-out evaluation set containing examples of coded, literal, quoted or mentioned, and other uses. The final size will depend on data availability and annotation effort. Two annotators will label examples independently, with disagreements reviewed and resolved. We will establish clear annotation guidelines and allow an uncertain category when evidence is insufficient.
3. Methodology
Stage 1: Candidate discovery through semantic shift.
We will use a contextual language model to represent words and phrases based on their surrounding text. We will compare how expressions are used in a particular online community with how they are used in general text. Expressions showing substantial differences will be ranked as potential candidates. We will also consider how closely their contexts resemble those of known dog whistles. Minimum frequency thresholds will reduce unreliable results. We will compare this approach with statistical methods such as pointwise mutual information (PMI) and log-likelihood ratio (LLR). These signals will identify candidates for investigation, not establish that they are genuine dog whistles.
Stage 2: Retrieval of similar known expressions.
We will build a searchable glossary containing known dog whistles, their coded meanings, and contextual examples. For each candidate, the system will retrieve the three or five most relevant entries by comparing their meanings and contexts with the original post. We will compare semantic search, keyword-based retrieval using BM25, and a hybrid of both methods.
Stage 3: LLM-based interpretation.
We will initially use Llama 3.1 8B Instruct to interpret each candidate using the original post and retrieved glossary entries. The model will assign one of five labels: CODED, LITERAL, MENTION/QUOTATION, OTHER, or UNCERTAIN. It will also provide a short explanation of the possible coded meaning, an explicit-meaning rewrite where appropriate, the glossary entry supporting its decision, and a confidence estimate. We will compare its predictions with those of the same LLM without retrieval to determine whether relevant examples improve interpretation. If resources permit, we will also test a larger model.
Stage 4: Verification.
Human annotators will review a sample of the model's classifications and explanations. They will assess whether the predicted label is correct and whether the explanation is supported by the retrieved evidence. Unsupported explanations and incorrect interpretations will be recorded for error analysis.
4. Evaluation
We will compare our approach against dictionary matching, statistical candidate discovery, an LLM without retrieval, and a conventional classifier such as TF-IDF with Logistic Regression. We will also test the LLM with unrelated retrieved entries to determine whether improvements result from relevant evidence rather than simply adding more text to the prompt.
Candidate discovery will be evaluated using Precision@20 and Precision@50, based on human assessment of the highest-ranked candidates. Classification will be evaluated using per-class precision, recall, macro-F1, and a confusion matrix. Human annotators will also assess the correctness of the generated explanations.
To test generalisation, we will hold out entire expressions during development and testing rather than randomly splitting posts containing the same expression. At test time, the target expression and its known variants will be excluded from the retrieval glossary. This will help us assess whether the system can interpret expressions without relying on their presence in the glossary.
5. Expected Outcomes and Limitations
We expect retrieval to be most useful when a new expression resembles a known dog whistle in meaning or usage. It may be less effective for entirely new expressions with no close examples in the glossary.
Semantic shift may also reflect changes in topic or community style rather than coded meaning. Furthermore, an LLM may produce plausible explanations that are not supported by the evidence. Statistical comparisons, human annotation, matched controls, and explicit evidence verification will help address these limitations.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Cluster-Based Detection of Emerging Dog Whistles Using LLMs:**
1. Research Questions
This project investigates whether changes in how words and phrases are used across online communities can help identify emerging dog whistles.
Our main research question is:
Can clustering different uses of an expression help us discover previously unseen dog whistles, and can an LLM interpret the meaning of these usage patterns accurately?
We will also investigate whether interpreting groups of similar examples is as effective as analysing each post separately, and whether providing information about known dog whistles improves the LLM's interpretations.
2. Data
We will primarily use the Informal Potential Dog Whistles from Reddit dataset to identify expressions and collect their occurrences across online communities. Where the available metadata permits, we will separate posts from the community being investigated and a general-text reference corpus.
The Silent Signals dataset will provide labelled examples and a glossary of known dog whistles and their coded meanings.
We aim to investigate approximately 60 expressions, including potential dog whistles and ordinary expressions used as controls. Each expression should have enough occurrences in both corpora to support meaningful clustering. Expressions with insufficient data will be analysed separately. The final sample size will depend on dataset availability and annotation effort.
Two annotators will label selected examples according to whether the expression is used in a coded, literal, quoted or mentioned, or other sense. Disagreements will be reviewed, and an uncertain category will be available when the context is insufficient.
3. Methodology
Our approach consists of five stages.
Stage 1: Represent expressions in context.
We will use a contextual language model to represent each occurrence of a word or phrase based on its surrounding text. The same expression can have different representations depending on how it is used. For example, an expression may have an ordinary meaning in one post but a coded meaning in another. We will prepare these representations for the clustering stage.
Stage 2: Group similar uses through clustering.
We will apply K-means clustering to group occurrences with similar contextual representations. We will initially test between two and six clusters per expression and select suitable settings using a clustering quality measure. We will also check whether the clusters remain similar across repeated runs. Where feasible, we will compare K-means with HDBSCAN, an alternative clustering method.
Stage 3: Measure differences between communities.
We will compare how frequently an expression's occurrences fall into each cluster in the investigated community and the reference corpus. Jensen-Shannon divergence will measure how different these distributions are. Expressions with larger differences will be ranked as potential dog whistles. We will also examine whether clusters found mainly in one community have meanings that differ from the expression's other uses. Statistical methods such as PMI and log-likelihood ratio (LLR) will serve as discovery baselines. Differences between communities will be treated as signals for further investigation, not proof of coded meaning.
Stage 4: Use an LLM to interpret each cluster.
We will provide an LLM, initially Llama 3.1 8B Instruct, with five to eight representative examples from each cluster. The model will assign a label—CODED, LITERAL, MENTION/QUOTATION, OTHER, or UNCERTAIN—and describe the shared meaning of the examples. Where appropriate, it will also rewrite the expression using its explicit meaning and identify the examples supporting its interpretation. We will compare cluster-level interpretation with analysing individual posts to determine whether grouping examples reduces the number of LLM calls without substantially reducing accuracy.
Stage 5: Improve and verify interpretations using known examples.
We will retrieve up to three similar entries from the Silent Signals glossary and provide them to the LLM alongside the cluster examples. This will allow us to test whether information about known dog whistles improves the interpretation of unseen expressions. Human annotators will review selected outputs and flag explanations that are incorrect or unsupported by the examples.
4. Evaluation
We will compare our approach against several baselines:
•	Dictionary matching: Checks for expressions already listed in a known dog-whistle glossary.
•	PMI/LLR: Identifies potentially interesting expressions using word co-occurrence statistics.
•	Average contextual distance: Measures differences between community usage without clustering, to test whether clustering provides additional value.
•	LLM per post: Interprets individual occurrences rather than groups of examples.
•	Cluster + LLM without glossary: Tests whether glossary information improves interpretation.
•	Our proposed approach: Combines clustering, LLM interpretation, and glossary-based evidence.
For discovery, we will measure how well the methods rank known dog whistles above matched ordinary expressions using AUROC and Precision@20 and Precision@50. For clustering, we will assess consistency across repeated runs and, where reliable sense labels are available, compare the clusters with human annotations. For interpretation, we will measure accuracy, macro-F1, and per-class performance. Human annotators will also assess whether the explanations correctly describe the coded meanings.
We will hold out entire expressions during development and testing rather than randomly splitting posts containing the same expressions. The glossary will exclude the target expression and its known variants during testing. This will help us assess generalisation to expressions not used during development.
5. Expected Outcomes and Limitations
We expect clustering to help identify expressions whose uses differ across communities, particularly when an expression has distinct literal and coded meanings. Analysing representative examples at the cluster level may also reduce the number of LLM calls required.
However, clusters may reflect differences in topics or writing styles rather than differences in meaning. Clustering can also be unreliable when expressions have few occurrences or when their meanings overlap. Finally, LLMs may produce plausible but unsupported interpretations.
We will address these risks through matched control expressions, comparisons with a general-text corpus, repeated clustering runs, human annotation, and evidence-based verification.
The overall goal is to determine whether clustering contextual uses can help discover emerging dog whistles and whether LLMs can interpret these usage patterns accurately and efficiently.
