## 📒 `notebooks` &ndash; Jupyter Notebooks for parsing, evaluating, and generating

* `read_constructions.ipynb` → Parsing of data from the German constructicon.
* `read_constructions_bert.ipynb` → Parsing of data from the German constructicon, but specialized for training with BERT.
* `read_constructions_deps.ipynb` → Parsing of data from the German constructicon, but specialized for dependency/pos tag analysis.


* `get_common_dependencies.ipynb` → Find common dependencies in the constructicon data and match possible candidates of the HDT-UD corpus.
* `eval_matches.ipynb` → Evaluate the matches' quality.


* `compare_embeddings.ipynb` → Analyze the distances between pseudoword embeddings and sentence embeddings. 
* `eval_compare.ipynb` → Evaluate the comparisons of pseudowords to sentence embeddings.


* `llama_generate_examples.ipynb` → Generation of construction examples with Llama 2 using zero-shot and few-shot prompting.
* `mbart_generate_examples.ipynb` → Generation of construction examples with mBART-50 using <mask> completion.
* `bert_comapp_generate_examples.ipynb` → Generation of construction examples with BERT and pseudoword embeddings using [MASK] completion.
* `mbart_comapp_generate_examples.ipynb` → Generation of construction examples with mBART-50 and pseudoword embeddings using <mask> completion.
* `evaluate_llama.ipynb` → Evaluation of generated examples by Llama 2.
* `eval_generate.ipynb` → More evaluation of Llama 2 examples.
* `eval_generate_human.ipynb` → Prepare generated data from human annotation.


* `llama_find_construction.ipynb` → Detection of construction examples with Llama 2.
* `bert_find_construction.ipynb` → Detection of construction examples with BERT NSP.
* `eval_llama.ipynb` → Analysis of detection results for Llama 2.
* `eval_bert.ipynb` → Analysis of detection results for BERT (with and without added pseudoword embeddings); significance tests.


* `llama_x_bert_comapp_generate_examples.ipynb` → Draft of a joint approach for combining the generation of construction examples with Llama 2 while double-checking the quality with BERT NSP.
