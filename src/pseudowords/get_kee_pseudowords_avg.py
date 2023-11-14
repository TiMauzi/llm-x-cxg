"""
Bert Coercion
"""
import csv
import itertools
from typing import List, Tuple, TextIO

from transformers import AutoTokenizer, MBart50Tokenizer, MBartForConditionalGeneration, Text2TextGenerationPipeline, \
    MBartTokenizer, MBart50TokenizerFast
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import torch
import torch.nn as nn
import torch.optim
from tqdm import trange, tqdm
import jsonlines
import json
import numpy as np
import os
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.metrics import mean_squared_error
import pickle

NEW_TOKEN = '#TOKEN#'

Item = Tuple[str, int]
Example = Tuple[Item, Item]

# ARGS
QUERIES_PATH = "../../data/pseudowords/CoMaPP_all.json"  # path to queries
DATASET_PATH = "../../data/pseudowords/CoMapp_Dataset.csv"
DIR_OUT = "../../out/"  # path to dir to save the pseudowords
CACHE = "../../out/cache/"  # path to cach directory


################################################

class DataBuilder:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, max_length=None):
        tokens = text.split()
        # Build token indices
        _, gather_indexes = self._manual_tokenize(tokens)
        # Tokenization
        if max_length:
            # TODO Note by huggingface:
            #  "We strongly recommend passing in an `attention_mask` since your input_ids may be padded."
            encode_dict = self.tokenizer(
                text, return_attention_mask=True,
                return_token_type_ids=False, return_tensors='pt',
                padding='max_length', max_length=max_length)
        else:
            encode_dict = self.tokenizer(
                text, return_attention_mask=True,
                return_token_type_ids=False, return_tensors='pt')
        input_ids = encode_dict['input_ids']
        return input_ids, gather_indexes

    def _manual_tokenize(self, tokens: List[str]):
        split_tokens = []
        gather_indexes = []
        for token in tokens:
            indexs = []
            for sub_token in self.tokenizer.tokenize(token):
                indexs.append(len(split_tokens))
                split_tokens.append(sub_token)
            gather_indexes.append(indexs)

        gather_indexes = [(min(t), max(t) + 1) for t in gather_indexes]

        # Adjust for CLS and SEP
        indices = [(a + 1, b + 1) for a, b in gather_indexes]
        # Add of CLS and SEP
        indices = [(0, 1)] + indices + [(indices[-1][1], indices[-1][1] + 1)]
        return split_tokens, indices


class Coercion:
    def __init__(self, builder: DataBuilder):
        self.builder = builder

    def coercion(self,
                 group,
                 k: int = 5):
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50",
                                                              return_dict=True)  #, decoder_start_token_id=250003)  # 250003 == de_DE
        model.to('cuda')

        self.builder.tokenizer.add_tokens(NEW_TOKEN)
        model.resize_token_embeddings(len(self.builder.tokenizer))

        new_queries = []
        queries = []
        targets1 = []
        vec_targets = []

        # Print targets (and their id's) and the query (and its id)
        for entry in group:
            i = 0
            while True:
                i = i + 1
                if ('target' + str(i)) not in entry.keys():
                    break
                print(f'target {i}: {entry["target" + str(i)]}, {entry["target" + str(i) + "_idx"]}')
            print(f'query: {entry["query"]}, {entry["query_idx"]}')

            nlp = Text2TextGenerationPipeline(model=model, tokenizer=self.builder.tokenizer, device=0)
            output = nlp(entry["query"], max_length=30, num_return_sequences=5, num_beams=100)
            output = self._format(output)
            print(f"output: {output}")
            # print('<mask> = ' + str(outputs))  # TODO Just show the replaced <mask> token

            for j in range(1, i):
                vec_targets.append(
                    self._get_target_embed((entry["target" + str(j)], entry["target" + str(j) + "_idx"]), model)
                )

            new_query = entry["query"].split()
            new_query[entry["query_idx"]] = NEW_TOKEN
            new_query = ' '.join(new_query)
            query = (new_query, entry["query_idx"])
            print(query)
            new_queries.append(new_query)
            queries.append(query)
            targets1.append((entry["target1"], entry["target1_idx"]))

        model = self._freeze(model)

        model.eval()

        for i in range(k):
            print('-' * 40)
            print('Random {a}'.format(a=i))

            # Random initialization, same initialization as huggingface
            weight = model.model.shared.weight.data[-1]
            nn.init.normal_(weight, mean=0.0, std=model.config.init_std)

            # Before training
            # print('Before training:')
            # We need a Text2TextGeneration here, because mBart is created for translation, originally.
            # Only this way, there can be multiple predicted words for one <mask>.
            # nlp = Text2TextGenerationPipeline(model=model, tokenizer=self.builder.tokenizer, device=0)

            model = self._train(model, vec_targets, queries, targets1)

            print("*************************************************************************")
            # After training
            print('After training:')
            nlp = Text2TextGenerationPipeline(model=model, tokenizer=self.builder.tokenizer, device=0)

            # For determining the original token's length, you take a random (the first) target, whitespace-tokenize it
            # and extract the token string. Then you tokenize it using the tokenizer. You can then count the input_ids,
            # ignoring the first id and the final id.
            token_length = len(tokenizer(targets1[0][0].split()[targets1[0][1]])["input_ids"][1:-1])
            for new_query in set(new_queries):  # only view different queries
                print(f"query: {new_query}")

                target_length = len(new_query) - 1 + token_length  # length of new query - #TOKEN# + target token
                output = nlp(new_query, max_length=target_length, num_return_sequences=5, num_beams=100)  # TODO output is fishy...
                output = self._format(output)
                print(f'output: {output}')

                outputs_list.append(output)

                #predictions = tokenizer(output, return_tensors="pt").input_ids
                output = self._predict_z(model, query, output)  # todo predictions testen
                output = self._format(output)
                print(f'{NEW_TOKEN} {output}')
            print("*************************************************************************")

    def _train(self, model, vec_targets, queries, targets1):
        loss_fct = nn.MSELoss(reduction='mean')  # mean will be computed later
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, eps=1e-8)
        epoch = 10  # 1000 was the default for BERT; but 400 seems to be enough to practically minimize the loss
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch)

        # This snippet, retrieving the possible padding, does the following:
        #  (a) encode each query's text (first [0]),
        #  (b) get the input_ids (second [0]),
        #  (c) count the input_ids (.shape[-1], because the number of input_ids is stored in the second/last dimension).
        # Then, you can take the max to know how much you should pad the rest.
        max_length = max([(self.builder.encode(query[0])[0]).shape[-1] for query in queries])
        max_labels_length = max([(self.builder.encode(target1[0])[0]).shape[-1] for target1 in targets1])

        input_ids_and_gather_indexes = [self.builder.encode(query[0], max_length=max_length) for query in queries]
        input_ids = torch.cat([input_id for input_id in [i for i, _ in input_ids_and_gather_indexes]], dim=0).to("cuda")
        # We actually don't need the index tuples here, because the real indices are stored within the labels:
        # gather_indexes = [gather_index for gather_index in [g for _, g in input_ids_and_gather_indexes]]

        # This is needed for computing the loss. This is because mBart is a generative model unlike Bert, so
        # the decoder needs the solution during training time. It also needs to be shifted right
        # (happens automatically here).
        # [0] because I don't need "gather_indexes"
        labels_and_gather_indexes = [self.builder.encode(target1[0], max_length=max_labels_length) for target1 in targets1]
        labels = torch.cat([label for label in [lab for lab, _ in labels_and_gather_indexes]], dim=0).to("cuda")
        gather_indexes = [gather_index for gather_index in [g for _, g in labels_and_gather_indexes]]

        # target_idx is the index of target word in the token list.
        # TODO Statt g[q[1] + 1][0]: Suche alle Indizes, die zu #TOKEN# werden sollen. -> Statt (8, 1) sollte es z. B. (8, 3) sein
        target_idxs = [g[q[1] + 1] for g, q in zip(gather_indexes, queries)]
        # target_idxs = torch.tensor(target_idxs, device="cuda").unsqueeze(-1)
        target_idxs = torch.tensor([range(*i) for i in target_idxs], device="cuda")

        # token_idx is the index of target word in the vocabulary of BERT
        # token_idxs = input_ids.gather(dim=-1, index=target_idxs)
        token_idxs = input_ids.gather(dim=-1, index=target_idxs[:, 0].unsqueeze(-1))
        vocab_size = len(tokenizer)  # can be checked with tokenizer.get_added_vocab()
        min_token_idx = min(token_idxs)
        # Get all indices smaller than the new token_idx:
        indices = torch.tensor([i for i in range(vocab_size) if i < min_token_idx], device="cuda", dtype=torch.long)

        for _ in trange(epoch):
            optimizer.zero_grad()
            outputs = model(input_ids, output_hidden_states=True, labels=labels)
            # loss for generation needs to be calculated manually because of smaller vocab size:
            # masked_lm_loss = loss_fct(outputs.logits.view(-1, model.config.vocab_size-1), labels.view(-1))
            z = torch.gather(outputs.decoder_hidden_states[-1], dim=1,
                             index=target_idxs.unsqueeze(-1).expand(-1, -1, model.config.d_model)) # d_model == 1024
            # todo outputs.decoder_hidden_states[-1][:, slice(*target_idx)]

            loss = loss_fct(z, torch.stack(vec_targets).squeeze())
            print(f"\ttrain loss = {float(loss)}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.model.encoder.embed_tokens.weight.grad[indices] = 0
            optimizer.step()
            scheduler.step()

            # try to fix the feed-forward bug
            # outputs = model(input_ids)
            # bert_z = torch.index_select(outputs.encoder_last_hidden_state[0], dim=0, index=target_idxs.squeeze(-1))

        # get the z* for classification
        vec = model.get_input_embeddings()(token_idxs).squeeze(1)[0]  # this is z*; [0] because all the same
        vec_array = vec.cpu().detach().numpy()
        z_list.append(vec_array)
        loss_list.append(str(loss.cpu().detach().numpy()))

        # save checkpoints
        np.save(CACHE + "temp_z_arrays_mbart.npy", np.array(z_list))
        np.save(CACHE + "temp_loss_arrays_mbart.npy", np.array(loss_list))

        s = 'Final loss = {a}'.format(a=str(loss.cpu().detach().numpy()))
        print(s)

        return model

    def _get_target_embed(self, target, model):
        input_ids, gather_indexes = self.builder.encode(target[0])
        target_idx = gather_indexes[target[1] + 1]
        model.eval()
        with torch.no_grad():
            # Find the learning target x
            input_ids = input_ids.to('cuda')
            outputs = model(input_ids=input_ids, output_hidden_states=True)  # labels are shifted right automatically
            # get all indices that are part of the KEE; slice is needed for converting the tuple to a slice
            x_target = outputs.decoder_hidden_states[-1][:, slice(*target_idx)]
        return x_target

    def _freeze(self, model):
        # Freeze all the parameters except the word embeddings
        for name, param in model.named_parameters():
            if 'model.shared' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # The "decoder" in BERT maps from hidden to output. This is analogous to "lm_head" in mBART.
        original_weight = model.lm_head.weight
        original_bias = model.final_logits_bias
        # The vocabulary of the decoder doesn't need #TOKEN# and would be too big:
        original_decoder_embed_tokens_weight = model.model.decoder.embed_tokens.weight
        # original_decoder_embed_tokens_bias = model.model.decoder.embed_tokens.bias

        # The argument len(tokenizer)-1 should prevent the model from outputting #TOKEN#:
        lm_head = nn.Linear(in_features=1024, out_features=len(tokenizer)-1, bias=False)
        lm_head.weight.requires_grad = False
        model.register_buffer("final_logits_bias", torch.zeros((1, model.model.shared.num_embeddings-1)))
        lm_head.weight.data.copy_(original_weight.data[:-1])
        model.final_logits_bias.copy_(original_bias[:, :-1])
        model.lm_head = lm_head
        decoder_embed_tokens = nn.Embedding(len(tokenizer)-1, model.config.d_model, model.config.pad_token_id)
        # For decoder, see above:
        decoder_embed_tokens.weight.data.copy_(original_decoder_embed_tokens_weight.data[:-1])
        # decoder_embed_tokens.bias.data.copy_(original_decoder_embed_tokens_bias.data[:, :-1])
        decoder_embed_tokens.requires_grad_(False)
        model.model.decoder.embed_tokens = decoder_embed_tokens
        model.config.vocab_size -= 1

        return model.to(model.device)

    def _format(self, results):  # new format

        reval = []
        for item in results:
            if "generated_text" in item.keys():
                generated_text = item["generated_text"]
                reval.append(generated_text)
            else:
                token_str = item['token_str']
                score = item['score']
                s = ':'.join([token_str, str(score)])
                reval.append(s)
        return reval

    def _predict_z(self, model, query, predictions):
        max_length = max([self.builder.encode(pred)[0].size(-1) for pred in predictions])

        input_ids, _ = self.builder.encode(query[0])

        pred_and_gather_indexes = [self.builder.encode(pred, max_length=max_length) for pred in predictions]
        prediction_ids = torch.stack([i for i, _ in pred_and_gather_indexes])

        input_ids = input_ids.to('cuda')
        prediction_ids = prediction_ids.to('cuda')
        with torch.no_grad():
            logits = [model(input_ids, labels=prediction.to("cuda")).logits[0, :, :] for prediction in prediction_ids]  # todo predicitions müssen mit rein!
            logits = torch.stack(logits, dim=0)
            # Use torch.gather to select values from probabilities based on prediction_ids
            selected_logits = torch.gather(logits, 2, prediction_ids)
            logits_sum = selected_logits.sum(dim=1, keepdim=True)

        probs = logits_sum.softmax(dim=-1)
        # values, output = probs.topk(5, dim=-1)
        reval = []
        for prob, pred in zip(probs.tolist(), predictions):
            s = {
                'score': prob,
                'prediction': pred  # self.builder.tokenizer.convert_ids_to_tokens(p)
            }
            reval.append(s)
        return reval


def load_data(path: TextIO) -> List[Example]:
    reval = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            target = (obj['target'], obj['target_idx'])
            query = (obj['query'], obj['query_idx'])
            reval.append((target, query))
    return reval


def get_lowest_loss_arrays(z_list, loss_list):
    z_array = np.array(z_list)
    loss_array = np.array(loss_list)

    loss_list = loss_array.tolist()
    z_list = []  # list of arrays

    loss_list = list(map(float, loss_list))

    # print(z_array)
    for vec in z_array:
        # print("vec.shape", vec.shape) #(768,)
        z_list.append(vec)

    # empty lists
    z_temp = []
    loss_temp = []

    # 5 initializations
    r = int(len(loss_list) / 5)

    for i in range(r):
        k = 0
        for j in range(5):
            if k == 0:

                k = loss_list[5 * i + j]
                z = z_list[5 * i + j]
            else:
                if loss_list[5 * i + j] < k:
                    k = loss_list[5 * i + j]
                    z = z_list[5 * i + j]
                else:
                    continue

        z_temp.append(z)
        loss_temp.append(k)

    z_temp_array = np.array(z_temp)

    return z_temp_array


if __name__ == '__main__':

    z_list = []
    z_eps_list = []
    loss_list = []
    outputs_list = []

    with open(QUERIES_PATH) as json_file:
        data = json.load(json_file)

    # Group the dataset into a list of lists where the label of the dictionaries is identical:
    data.sort(key=lambda x: x["label"])  # Grouping doesn't work without sorting first!
    data = [list(group) for _, group in itertools.groupby(data, key=lambda x: x["label"])]

    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="de_DE", tgt_lang="de_DE")
    builder = DataBuilder(tokenizer)
    co = Coercion(builder)
    for group in data:
        co.coercion(group)
        print('==' * 40)
        break  # TODO test

    result = get_lowest_loss_arrays(z_list, loss_list)

    # save the pseudowords
    np.save(DIR_OUT + 'pseudowords_comapp.npy', result)
