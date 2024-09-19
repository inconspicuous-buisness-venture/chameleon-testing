# Chameleon Testing

Ok so basically we've been moving pretty fast so I made a bunch of notebooks to show the summaries of our information. We can document **existing product research** [here](research.md) for humanizers, and all the elements that are important when measuring reinforcement are contained within a series of notebooks. 

🌉 **Coherence**: I implemented two different sections of coherence, first testing a series of [basic coherence measurements](coherence/basic/) (none of which worked). 

- 🔮 **Implementations of BERT for Next Sentence Prediction**: [<kbd>BERT_NSP.ipynb</kbd>](coherence/basic/BERT_NSP.ipynb) [<kbd>BERT_NSP2.ipynb</kbd>](coherence/basic/BERT_NSP2.ipynb)
- ❓ **Implementations of Perplexity Estimations**: [<kbd>GPT2_PEP.ipynb</kbd>](coherence/basic/GPT2_PEP.ipynb)
- 🌌 **Latent Semantic Analysis**: [<kbd>LSA.ipynb</kbd>](coherence/basic/LSA.ipynb)
- 🎲 **Natural Language Inference**: [<kbd>NLI.ipynb</kbd>](coherence/basic/NLI.ipynb)

I then moved towards the [implementation of research papers](coherence/paper/). You can find all of them here:

- 📖 **LongWanjuan**: [<kbd>dmr_ttr.ipynb</kbd>](coherence/paper/dmr_ttr.ipynb) [<kbd>Official Implementation</kbd>](https://github.com/OpenLMLab/LongWanjuan)
- ↗️ **Word2Vec N-Grams**: [<kbd>ngrams.ipynb</kbd>](coherence/paper/ngrams.ipynb) 

I also implemented a few [detection algorithms](detection/), but none of them seemed to work effectively, except for the ROBERTA model from Hugging Face. 

- 🎆 **Burstiness**: [<kbd>burstiness.ipynb</kbd>](detection/burstiness.ipynb)
- 🤯 **Perplexity**: [<kbd>perplexity.ipynb</kbd>](detection/perplexity.ipynb)
- 🤗 **ROBERTA**: [<kbd>roberta.ipynb</kbd>](detection/roberta.ipynb)


However, I then discovered that there are many publically available datasets that are all viable options for fine-tuning (or training our own model), which I can use here: 

- https://github.com/iamjr15/Ensemble-AI-Text-Detection
- https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset
- https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data
- https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
- https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro