## 🕵️‍♀️ `pseudowords` &ndash; generate and train pseudoword embeddings

* `get_bsb_bert_kee_pseudowords_avg.py` → Generate pseudowords for German BERT (using the version by the Bavarian State Library)
* `get_kee_pseudowords_avg.py` → Generate pseudowords for mBART-50


* `script_bsb.sh` → Script for starting 15 SLURM jobs for BERT pseudoword generation; needs to be adjusted if used.


* `CoMaPP_test_bert.json` → Template for JSON files that are needed for pseudoword training on BERT.
* `CoMaPP_test.json` → Template for JSON files that are needed for pseudoword training on mBART-50.

### Acknowledgements

Both `get_bsb_bert_kee_pseudowords_avg.py` and `get_kee_pseudowords_avg.py` are based on the BERT-based 
[pseudoword tool by Karidi et al. (2021)](https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth). 