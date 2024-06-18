# RAGs for Open Domain Complex QA
As question answering (QA) systems increasingly rely on large language models (LLMs), integrating retrieval mechanisms to provide context is essential for handling complex queries, which may require information missing from its training data. This work explores the impact of different contexts on Retrieval-Augmented Generation (RAG) systems for complex QA, through comparing the performance of several LLMs with relevant, negative and random contexts. Our results indicate that injecting negative contexts and using different prompting techniques can impact QA performance differently depending on the LLM.
## Repository structure
 - `llm-responses` contains QA responses from the experiments in json format.
 - `llm-inference` contains notebooks used for running inference with tested LLMs.
 - `llm-evaluation-metrics.ipynb` can be used to compute evaluation metrics from the llm's responses
 - `adore-notebooks` contains code used for preprocessing, training and infernce of the ADORE Dense 
 - `qualitative-analysis` contains qualitative results for 100  queries and code to produce them (TBD: cleaning)
 Retrieval model (see [GitHub](https://github.com/jingtaozhan/DRhard) and [Paper](https://dl.acm.org/doi/abs/10.1145/3404835.3462880)).
 - `rag` contains miscellaneous code used for data processing and evaluation of the experiments.
