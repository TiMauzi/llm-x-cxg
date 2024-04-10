## 🏭 `generation` &ndash; results for sequence generation for constructions

* `{a}_shot_data.tsv` → Results for generating sentences with Llama 2, with `a` being the number of added examples to the prompt.


* `data_mbart_complete.tsv` → Results for <mask> filling with mBART-50 and pseudowords.
* `data_mbart_vanilla_complete.tsv` → Results for <mask> filling with mBART-50 without pseudowords.
* `loss_min_max.tsv` → Min and max losses during the training of mBART-50 pseudoword tokens.