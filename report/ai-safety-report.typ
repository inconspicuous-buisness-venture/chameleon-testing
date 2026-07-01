#set page(
  paper: "us-letter",
  margin: (x: 0.75in, top: 0.9in, bottom: 0.9in),
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 9.7pt)
#set par(justify: true, leading: 0.5em, first-line-indent: 1em)
#let lc = rgb("#173a5e")
#show link: it => text(fill: lc, it)

// IEEE-style section headings: centered small-caps roman numerals; italic subsections.
#set heading(numbering: "I.A.")
#show heading.where(level: 1): it => {
  set align(center)
  set text(size: 9.7pt, weight: "regular")
  v(0.7em)
  smallcaps(it)
  v(0.35em)
}
#show heading.where(level: 2): it => {
  set text(size: 9.7pt, style: "italic", weight: "regular")
  v(0.45em)
  block(it)
  v(0.1em)
}

// ── Title & author block (spans both columns) ──
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Coherence and Detection Approaches for\ Identifying AI-Generated Text
  ]
  #v(0.7em)
  #text(size: 11pt)[
    Alexander Castronovo, #h(0.3em) Jossaya Camille, #h(0.3em) Makai Pindell, #h(0.3em)
    Amarnath S. Patel, #h(0.3em) Thandi Menelas, #h(0.3em) Zach Lopez
  ]
  #v(0.3em)
  #text(size: 9.5pt, style: "italic")[
    Grant-Funded AI Safety Research Project, Florida Atlantic University\
    Faculty supervisor: Tucker Hindle
  ]
]
#v(1em)

#columns(2, gutter: 0.32in)[

#set par(first-line-indent: 0em)
#text(weight: "bold")[#h(1em)*Abstract*---]
Reliably identifying machine-generated text is a moving target: as detectors
improve, so do the rewriting strategies used to evade them. This report documents
an empirical study of that arms race. We implement and evaluate ten coherence- and
detection-based methods for distinguishing human from machine-generated text, then
build a suite of "humanization" pipelines---iterative rewriting, tree-search
decoding, reinforcement learning, and heuristic content filtering---that attempt to
defeat those detectors. Two results anchor the work. First, among lightweight
coherence signals, GPT-2 perplexity is the single most reliable discriminator,
separating representative coherent and incoherent passages by a factor of
$approx 3.35 times$ (17.48 vs. 58.50), while BERT next-sentence prediction fails
outright (both classes score $approx 0.99999$); of the dedicated detectors tested,
only a fine-tuned RoBERTa classifier is effective. Second, every evasion method we
build exposes a consistent quality--evasion tradeoff: detectability can be driven
down, but only at measurable cost to text quality.

#v(0.4em)
#text(weight: "bold")[#h(1em)*Index Terms*---]
AI-generated text detection, text coherence, adversarial evasion, language models,
perplexity, reinforcement learning.

#set par(first-line-indent: 1em)

= Introduction

Large language models now produce prose that is difficult to distinguish from human
writing, which has made automated detection of AI-generated text a practical concern
for academic integrity, content provenance, and safety. Detection is inherently
adversarial: any published detector invites countermeasures---paraphrasers,
"humanizers," and decoding tricks designed to strip the statistical fingerprints
detectors rely on. Understanding detection therefore requires studying evasion in the
same breath.

This project pursues two linked questions. #emph[(1) What signals actually separate
human from machine-generated text?] We survey coherence-based measures---which ask
whether a passage reads as internally consistent---and dedicated detectors trained to
spot generation artifacts. #emph[(2) How robust are those signals to deliberate
evasion?] We construct several rewriting pipelines and measure how far detectability
can be reduced, and at what cost. The remainder of the report covers the dataset
(Section II), coherence methods (Section III), detectors (Section IV), evasion
pipelines (Section V), the resulting quality--evasion tradeoff (Section VI), and the
tooling built to study it (Section VII).

= Dataset

Experiments draw on a public human-vs-AI text corpus from the Kaggle "LLM---Detect
AI-Generated Text" family [3], stored as a two-column table of `text` and a binary
`generated` label ($0 =$ human, $1 =$ AI), comprising roughly $2.55 times 10^5$
labeled passages. The corpus pairs student-style essays written by humans with
model-generated counterparts on the same prompts, which makes it well suited to
controlled coherent-vs-incoherent and human-vs-AI comparisons. For the decoding
experiments in Section V, additional text is generated on demand with open-weight and
API models rather than drawn from the corpus.

= Coherence Measurement

We group coherence signals into lightweight #emph[basic] methods and reimplementations
of published #emph[paper-based] metrics. Each was probed on matched coherent and
deliberately-incoherent passages; the headline numbers are illustrative single-passage
probes, not full-corpus benchmarks, and are reported to show the direction and
magnitude of each signal.

== Basic methods

GPT-2 perplexity is the strongest lightweight signal, cleanly separating the two
classes (17.48 coherent vs. 58.50 incoherent, an $approx 3.35 times$ gap): lower
perplexity---text the model finds unsurprising---tracks human-like coherence. BERT
next-sentence prediction fails, saturating at $approx 0.9999997$ for both classes; a
sentence-transformer (SBERT) cosine variant discriminates only marginally (0.50 vs.
0.59). Latent semantic analysis (TF-IDF $arrow$ truncated SVD $arrow$ inter-sentence
cosine) inverts, scoring incoherent text #emph[higher] (0.61 vs. 0.40), and an NLI
entailment score (BART-MNLI) did not yield a usable value in our setup.

== Paper-based metrics

We reimplement the LongWanjuan long-text quality framework [1], combining a
#emph[coherence] term (long- vs. short-context next-token accuracy under GPT-2), a
#emph[cohesion] term (connective and pronoun density), and a #emph[complexity] term
(type--token ratio and paragraph length) into a structured, multi-axis quality profile
rather than a single score. A word2vec (google-news-300) n-gram model with
inter-sentence cosine similarity was also tested; like LSA, its ordering was
unreliable on our probes.

#figure(
  table(
    columns: (1.35fr, 0.7fr, 0.7fr),
    align: (left, center, center),
    stroke: (x: none, y: 0.4pt),
    inset: (x: 3pt, y: 4pt),
    table.header([*Method*], [*Coh.*], [*Incoh.*]),
    [GPT-2 perplexity #box[($arrow.b$)]], [*17.48*], [*58.50*],
    [BERT NSP], [.9999967], [.9999969],
    [SBERT cosine], [0.500], [0.589],
    [Latent Semantic Analysis], [0.401], [0.607],
    [NLI (BART-MNLI)], [---], [---],
    [Word2Vec n-grams], [0.609], [0.682],
  ),
  caption: [Coherence signals on matched coherent/incoherent passages. Only GPT-2
  perplexity shows a large, correctly-ordered separation ($3.35 times$).],
)

= Detection Methods

Three dedicated detectors were tested against the corpus. #emph[Burstiness]
(sentence-length variance, word-frequency entropy, type--token ratio, coefficient of
variation) and a #emph[perplexity-threshold] classifier both proved unreliable in
practice. Only the fine-tuned RoBERTa OpenAI detector
(`roberta-base-openai-detector`) [2] discriminated effectively, assigning confident,
correctly-ordered human/AI probabilities. RoBERTa is consequently adopted as the
reference detector for the evasion experiments below.

= Evasion / Humanization Pipelines

Having fixed RoBERTa as the target detector, we built four families of
rewriting/decoding pipelines that attempt to lower a passage's detection score.

#emph[Iterative rewriting.] Starting from AI text, score it with the detector; if it
reads as machine-generated, rewrite via an API model (Gemini / GPT-4o / Claude) and
repeat, up to 15 iterations or until the score falls below threshold---a closed
generate $arrow$ detect $arrow$ rewrite loop.

#emph[Tree-search decoding.] With an open-weight model (Llama-3.1-8B), expand the
top-$k$ next-token continuations into a search tree, annotate each branch with
perplexity and coherence, and keep paths that stay low-detectability while remaining
fluent. Generation trees are persisted as RDF (Turtle) for analysis.

#emph[List-branching.] A systematic sweep of the branching idea: with $k = 5$ over 5
steps, the pipeline enumerates all #box[$5^5$] = 3,125 candidate sequences, scoring
each on four axes---the text, GPT-2 coherence, perplexity, and a log-likelihood
ratio---to map the fluency/detectability frontier.

#emph[Reinforcement learning.] Wrap generation as a Gym environment whose reward is
the negative detector score, and train a PPO agent [4] (10,000 timesteps) /
fine-tune Llama-3.1-8B to minimize detectability directly.

A fifth component, #emph[heuristic content filtering] (14 line-level quality
filters---capitalization, repetition ratio, punctuation, stopword and grammar checks,
length bounds---weighted by each filter's perplexity improvement over the unfiltered
corpus, in the LongWanjuan style [1]), scores and prunes generated text for quality
rather than evasion, and serves as the quality yardstick for the tradeoff analysis.

= The Quality--Evasion Tradeoff

The central empirical finding is consistent across every evasion pipeline:
detectability can be reduced, but not for free. Iterative rewriting lowers detector
scores while BLEU against the original text drops and drift accumulates; RL
fine-tuning optimizes the evasion reward directly and degrades fluency as it learns to
avoid detectable patterns; tree- and list-branching make the frontier
explicit---the lowest-detectability paths are rarely the most coherent ones. In other
words, the same statistical regularities that make text read as high-quality are what
detectors key on, so suppressing the detector signal tends to suppress quality with
it. This tradeoff, rather than any single "winning" evader, is the practical takeaway
for reasoning about the durability of AI-text detection.

= Tooling

Two tools support the study. An interactive Flask application closes the evasion loop
end to end: it takes a prompt and parameters (threshold, temperature, iteration
count), generates with a selected model, and---via Playwright browser
automation---queries a live detector (ZeroGPT) for each iteration, logging the full
timestamped history to JSON. A separate analysis notebook suite visualizes
detection-score evolution across iterations, tracks BLEU across original $arrow$
paraphrased $arrow$ iterative rewrites for several models, and applies Kendall's-$tau$
significance testing to the resulting rank correlations.

= Discussion and Conclusions

Across ten coherence and detection methods, robustness was the exception rather than
the rule: most lightweight coherence signals failed or inverted, and only GPT-2
perplexity (as a coherence proxy) and a fine-tuned RoBERTa classifier (as a detector)
proved dependable. Against that reference, four distinct evasion strategies could all
reduce detectability---but each surfaced the same quality--evasion tradeoff,
suggesting the tension is structural rather than an artifact of any one method. Two
directions follow: training a purpose-built detector on the available large-scale
corpora rather than relying on off-the-shelf models, and quantifying the tradeoff
frontier more rigorously with paired quality/detectability metrics across many
generations. The full implementation---coherence and detection notebooks, the four
evasion pipelines, and the interactive tool---is available in the project repository.

#v(0.5em)
#text(size: 8.6pt)[
  #set par(first-line-indent: 0em, leading: 0.42em, hanging-indent: 1em)
  #smallcaps[*Acknowledgments*]\
  This work was conducted as a grant-funded AI-safety research project at Florida
  Atlantic University under the supervision of Tucker Hindle, with high-performance
  computing resources provided for model inference and training. Findings were
  presented at the Wilkes Honors College Undergraduate Research Symposium, 2025.

  #v(0.5em)
  #smallcaps[*References*]\
  [1] Y. Liu et al., "LongWanjuan: Towards Systematic Measurement for Long Text
  Quality," arXiv preprint, 2024.\
  [2] OpenAI, "GPT-2 Output Detector (`roberta-base-openai-detector`)," 2019.\
  [3] The Learning Agency Lab, "LLM---Detect AI Generated Text," Kaggle, 2023.\
  [4] J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347,
  2017.
]

]
