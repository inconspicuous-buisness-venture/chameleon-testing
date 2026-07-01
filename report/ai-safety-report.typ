#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#show heading: set block(above: 1.1em, below: 0.6em)
#set heading(numbering: "1.1")
#let lc = rgb("#173a5e")
#show link: it => text(fill: lc, it)

// ── Title block ──
#align(center)[
  #text(size: 17pt, weight: "bold")[
    Coherence and Detection Approaches for\ Identifying AI-Generated Text
  ]
  #v(0.4em)
  #text(size: 11pt)[A technical report on measuring text quality and the limits of detection under adversarial rewriting]
  #v(0.8em)
  #text(size: 11pt, weight: "bold")[Amarnath S. Patel]
  #v(0.2em)
  #text(size: 10pt)[
    Grant-Funded AI Safety Research Project — Florida Atlantic University\
    Supervisor: Tucker Hindle · 2025
  ]
]
#v(0.6em)
#line(length: 100%, stroke: 0.5pt)
#v(0.6em)

// ── Abstract ──
#block(inset: (x: 1.2em))[
  #text(weight: "bold")[Abstract.]
  Reliably identifying machine-generated text is a moving target: as detectors improve, so do the
  rewriting strategies used to evade them. This report documents an empirical study of that arms race.
  We implement and evaluate ten coherence- and detection-based methods for distinguishing human from
  AI-generated text, then build a set of "humanization" pipelines — iterative rewriting, tree-search
  decoding, reinforcement learning, and heuristic content filtering — that attempt to defeat those
  detectors. Two results anchor the work. First, among lightweight coherence signals, *GPT-2 perplexity*
  is the single most reliable discriminator, separating representative coherent and incoherent passages
  by a factor of #box[$approx 3.35 times$] (17.48 vs. 58.50), while *BERT Next-Sentence-Prediction*
  fails outright (both classes score #box[$approx 0.99999$]); of the dedicated detectors tested, only a
  fine-tuned RoBERTa classifier is effective. Second, every evasion method we build exposes a consistent
  *quality–evasion tradeoff*: detectability can be driven down, but only at measurable cost to text
  quality. We release the full notebook suite and an interactive tool that closes the
  generate → detect → rewrite loop against a live detector.
]
#v(0.4em)

= Introduction

Large language models now produce prose that is difficult to distinguish from human writing, which has
made automated detection of AI-generated text a practical concern for academic integrity, content
provenance, and safety. Detection is inherently adversarial: any published detector invites
countermeasures — paraphrasers, "humanizers," and decoding tricks designed to strip the statistical
fingerprints detectors rely on. Understanding detection therefore requires studying evasion in the same
breath.

This project pursues two linked questions. *(1) What signals actually separate human from
machine-generated text?* We survey coherence-based measures — which ask whether a passage reads as
internally consistent — and dedicated detectors trained to spot generation artifacts. *(2) How robust are
those signals to deliberate evasion?* We construct several rewriting pipelines and measure how far
detectability can be reduced, and what it costs. The remainder of the report covers the dataset
(#ref(<sec:data>)), coherence methods (#ref(<sec:coh>)), detectors (#ref(<sec:det>)), evasion pipelines
(#ref(<sec:evasion>)), the resulting quality–evasion tradeoff (#ref(<sec:tradeoff>)), and the tooling
built to study it (#ref(<sec:tools>)).

= Dataset <sec:data>

Experiments draw on a public human-vs-AI text corpus (Kaggle "LLM — Detect AI-Generated Text" family),
stored as a two-column table of `text` and a binary `generated` label (0 = human, 1 = AI), comprising
roughly #box[$2.55 times 10^5$] labeled passages. The corpus mixes student-style essays written by
humans with model-generated counterparts on the same prompts, which makes it well suited to controlled
coherent-vs-incoherent and human-vs-AI comparisons. For the decoding experiments in #ref(<sec:evasion>),
additional text is generated on demand with open-weight and API models rather than drawn from the corpus.

= Coherence measurement <sec:coh>

We group coherence signals into lightweight *basic* methods and reimplementations of published
*paper-based* metrics. Each was probed on matched coherent and deliberately-incoherent passages; the
headline numbers are illustrative single-passage probes, not full-corpus benchmarks, and are reported to
show the *direction and magnitude* of each signal.

== Basic methods

- *GPT-2 perplexity (best).* Scoring passages by the perplexity of a pretrained GPT-2 language model
  cleanly separates the two classes: 17.48 (coherent) vs. 58.50 (incoherent), a
  #box[$approx 3.35 times$] gap. Lower perplexity — text the model finds unsurprising — tracks human-like
  coherence.
- *BERT Next-Sentence Prediction (failed).* Pretrained NSP saturates: coherent and incoherent passages
  both score #box[$approx 0.9999997$], giving no usable separation. A sentence-transformer (SBERT) cosine
  variant discriminates only marginally (0.50 vs. 0.59).
- *Latent Semantic Analysis (failed).* TF-IDF → truncated SVD → inter-sentence cosine similarity inverts:
  incoherent text scores *higher* (0.61 vs. 0.40).
- *Natural Language Inference (failed).* Entailment scoring with a BART-MNLI model did not yield a working
  score in our setup.

== Paper-based metrics

- *LongWanjuan (DMR + TTR).* A reimplementation of the LongWanjuan long-text quality framework, combining
  a *coherence* term (long- vs. short-context next-token accuracy under GPT-2), a *cohesion* term
  (connective and pronoun density), and a *complexity* term (type–token ratio and paragraph length). It
  produces a structured, multi-axis quality profile rather than a single score.
- *Word2Vec n-grams.* Averaged word2vec (google-news-300) n-gram vectors with inter-sentence cosine
  similarity. Like LSA, its ordering was unreliable on our probes.

#figure(
  table(
    columns: (2.2fr, 1fr, 1fr, 1.4fr),
    align: (left, center, center, left),
    stroke: 0.4pt,
    inset: 6pt,
    table.header(
      [*Method*], [*Coherent*], [*Incoherent*], [*Verdict*],
    ),
    [GPT-2 perplexity #box[($arrow.b$ better)]], [17.48], [58.50], [*Best — 3.35× gap*],
    [BERT NSP], [0.9999967], [0.9999969], [Failed (saturated)],
    [SBERT cosine], [0.500], [0.589], [Marginal],
    [Latent Semantic Analysis], [0.401], [0.607], [Failed (inverted)],
    [NLI (BART-MNLI)], [—], [—], [No working score],
    [Word2Vec n-grams], [0.609], [0.682], [Unreliable],
  ),
  caption: [Coherence signals on matched coherent/incoherent passages. GPT-2 perplexity is the only
  lightweight measure with a large, correctly-ordered separation.],
)

= Detection methods <sec:det>

Three dedicated detectors were tested against the corpus. *Burstiness* (sentence-length variance, word-
frequency entropy, type–token ratio, coefficient of variation) and a *perplexity-threshold* classifier
both proved unreliable in practice. Only the fine-tuned *RoBERTa OpenAI detector*
(`roberta-base-openai-detector`) discriminated effectively, assigning confident, correctly-ordered
human/AI probabilities. RoBERTa is consequently used as the reference detector for the evasion
experiments below.

= Evasion / humanization pipelines <sec:evasion>

Having fixed RoBERTa as the target detector, we built four families of rewriting/decoding pipelines that
attempt to lower a passage's detection score.

+ *Iterative rewriting.* Starting from AI text, score it with the detector; if it reads as machine-
  generated, rewrite via an API model (Gemini / GPT-4o / Claude) and repeat, up to 15 iterations or until
  the score falls below threshold. A closed generate → detect → rewrite loop.
+ *Tree-search decoding (branching).* With an open-weight model (Llama-3.1-8B), expand the top-#box[$k$]
  next-token continuations into a search tree, annotate each branch with perplexity and coherence, and
  keep paths that stay low-detectability while remaining fluent. Generation trees are persisted as RDF
  (Turtle) for analysis.
+ *List-branching.* A systematic sweep of the branching idea: with top_k = 5 over 5 steps, the pipeline
  enumerates all #box[$5^5$] = 3,125 candidate sequences, scoring each on four axes — the text itself,
  GPT-2 coherence, perplexity, and a log-likelihood ratio — to map the fluency/detectability frontier.
+ *Reinforcement learning (RLHF-style).* Wrap generation as a Gym environment whose reward is the
  negative detector score, and train a PPO agent (10,000 timesteps) / fine-tune Llama-3.1-8B to minimize
  detectability directly.

A fifth component, *heuristic content filtering* (14 line-level quality filters — capitalization,
repetition ratio, punctuation, stopword and grammar checks, length bounds — weighted by each filter's
perplexity improvement over the unfiltered corpus, in the LongWanjuan style), scores and prunes generated
text for quality rather than evasion, and serves as the quality yardstick for the tradeoff analysis.

= The quality–evasion tradeoff <sec:tradeoff>

The central empirical finding is consistent across every evasion pipeline: *detectability can be reduced,
but not for free.* Iterative rewriting lowers detector scores while BLEU against the original text drops
and drift accumulates; RL fine-tuning optimizes the detector-evasion reward directly and degrades fluency
as it learns to avoid detectable patterns; tree- and list-branching make the frontier explicit — the
lowest-detectability paths are rarely the most coherent ones. In other words, the same statistical
regularities that make text read as high-quality are what detectors key on, so suppressing the detector
signal tends to suppress quality with it. This tradeoff, rather than any single "winning" evader, is the
practical takeaway for anyone reasoning about the durability of AI-text detection.

= Tooling <sec:tools>

Two tools support the study. An interactive *Flask application* closes the evasion loop end to end:
it takes a prompt and parameters (threshold, temperature, iteration count), generates with a selected
model (Gemini / GPT-4o / Claude), and — via Playwright browser automation — queries a live detector
(ZeroGPT) for each iteration, logging the full timestamped chat/score history to JSON. A separate
*analysis notebook suite* visualizes detection-score evolution across iterations, tracks BLEU across
original → paraphrased → iterative rewrites for several models, and applies Kendall's-#box[$tau$]
significance testing to the resulting rank correlations.

= Discussion and conclusions

Across ten coherence and detection methods, robustness was the exception rather than the rule: most
lightweight coherence signals failed or inverted, and only GPT-2 perplexity (as a coherence proxy) and a
fine-tuned RoBERTa classifier (as a detector) proved dependable. Against that RoBERTa reference, four
distinct evasion strategies could all reduce detectability — but each surfaced the same quality–evasion
tradeoff, suggesting the tension is structural rather than an artifact of any one method. Two directions
follow naturally: training a purpose-built detector on the available large-scale corpora rather than
relying on off-the-shelf models, and quantifying the tradeoff frontier more rigorously with paired
quality/detectability metrics across many generations. The full implementation — coherence and detection
notebooks, the four evasion pipelines, and the interactive tool — is available in the project repository.

#v(0.4em)
#line(length: 100%, stroke: 0.4pt)
#text(size: 9.5pt)[
  *Acknowledgments.* This work was conducted as a grant-funded AI-safety research project at Florida
  Atlantic University under the supervision of Tucker Hindle, with high-performance computing resources
  provided for model inference and training. Findings were presented at the Wilkes Honors College
  Undergraduate Research Symposium (2025).

  #v(0.3em)
  *Selected references.* Y. Liu et al., "LongWanjuan: Towards Systematic Measurement for Long Text
  Quality," 2024. · OpenAI, `roberta-base-openai-detector` (GPT-2 output detector). · Kaggle,
  "LLM — Detect AI-Generated Text" dataset family.
]
