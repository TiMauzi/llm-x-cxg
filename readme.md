# 🦙 `llm-x-cxg`: Language Models and Construction Grammar 📖
## 🏭 Text Generation and Construction Detection 👀

![banner](misc/training.gif)

---

[![Static Badge](https://zenodo.org/badge/DOI/10.5281/zenodo.10957260.svg)](https://doi.org/10.5281/zenodo.10957260/)
[![Static Badge](https://img.shields.io/badge/license-CC--4.0--BY-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/legalcode)

This repository contains code and output data from probing and experimenting with construction grammar (CxG) and large
language models (LLMs). This project is based on a master's thesis at Ludwig-Maximilians-Universität München 
(University of Munich; LMU) with the title _Sprachmodelle und Konstruktionsgrammatiken zur Textgenerierung und 
Konstruktionserkennung_ (English: _Language Models and Construction Grammar: Text Generation and Construction
Detection_). Thanks to [Leonie Weissweiler](https://github.com/LeonieWeissweiler/) for 
supervising this work!

Each sub-directory contains a `readme.md` file outlining each file's content. 
For citation of this work, please see the BibLaTeX snippet below.

---

### 🤔 Usage

In case you want to run any of the code provided, feel free to install the necessary dependencies using `conda`:
```shell
conda create --name llm-cxg --file llm-cxg.txt
conda activate llm-cxg
```
---

### 🙏 Acknowledgements

Both `src/pseudowords/get_bsb_bert_kee_pseudowords_avg.py` and `src/pseudowords/get_kee_pseudowords_avg.py` are based 
on the BERT-based 
[pseudoword tool by Karidi et al. (2021)](https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth). All
other files are built by the creator of `llm-x-cxg` (this repository).

### 📑 Citation

If you want to use the content of this repository, feel free to use the following template:

```bibtex
@thesis{sockel_llm_x_cxg_2024,
 author = {Sockel, Tim},
 year = {2024},
 title = {{Sprachmodelle und Konstruktionsgrammatiken zur Textgenerierung und Konstruktionserkennung}},
 keywords = {Computer;FOS: Computer;FOS: Languages;General language studies;information sciences;Linguistics;literature;Natural language processing},
 type = {{Master's Thesis}},
 institution = {{Ludwig-Maximilians-Universität München}},
 location = {Munich},
 language = {German},
 titleaddon = {Centrum für Informations- und Sprachverarbeitung},
 doi = {10.5281/zenodo.10957259}
}
```