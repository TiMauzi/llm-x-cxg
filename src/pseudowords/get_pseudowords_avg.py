"""
Bert Coercion
"""
import csv
import itertools
from typing import List, Tuple, TextIO

from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForMaskedLM
from transformers import FillMaskPipeline
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import torch
import torch.nn as nn
import torch.optim
from tqdm import trange
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
QUERIES_PATH = "../../data/pseudowords/MaPP_all.txt"  # path to queries
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
        model = BertForMaskedLM.from_pretrained(
            'bert-base-cased', return_dict=True)
        model.to('cuda')

        self.builder.tokenizer.add_tokens(NEW_TOKEN)
        model.resize_token_embeddings(len(self.builder.tokenizer))

        # TODO Muss für mehrere Sätze umgeschrieben werden!

        new_queries = []
        queries = []
        vec_targets = []

        # Print targets (and their id's) and the query (and its id)
        for entry in group:
            i = 0
            while True:
                i = i + 1
                if ('target' + str(i)) not in entry.keys():
                    break
                print('target ' + str(i) + ': ' + entry["target" + str(i)] + " , " + str(entry["target" + str(i) + "_idx"]))
            print('query:' + entry["query"] + " , " + str(entry["query_idx"]))

            # Model output
            nlp = FillMaskPipeline(model, self.builder.tokenizer, device=0)
            output = nlp(entry["query"])
            output = self._format(output)
            print('[MASK]=' + str(output))

            for j in range(1, i):
                vec_targets.append(
                    self._get_target_embed((entry["target" + str(j)], entry["target" + str(j) + "_idx"]), model))

            new_query = entry["query"].split()
            new_query[entry["query_idx"]] = NEW_TOKEN
            new_query = ' '.join(new_query)
            query = (new_query, entry["query_idx"])
            print(query)
            new_queries.append(new_query)
            queries.append(query)

        model = self._freeze(model)

        model.eval()

        for i in range(k):
            print('-' * 40)
            print('Random {a}'.format(a=i))

            # Random initialization, same initialization as huggingface
            weight = model.bert.embeddings.word_embeddings.weight.data[-1]
            nn.init.normal_(weight, mean=0.0,
                            std=model.config.initializer_range)

            # Before training
            print('Before training:')
            nlp = FillMaskPipeline(model, self.builder.tokenizer, device=0)

            model = self._train(model, vec_targets, queries)

            print("*************************************************************************")
            # After training
            print('After training:')
            nlp = FillMaskPipeline(model, self.builder.tokenizer, device=0)
            for new_query in set(new_queries):  # only view different queries
                print("query: " + new_query)
                output = nlp(new_query)
                output = self._format(output)
                print('[MASK]=' + str(output))

                outputs_list.append(output)

                output = self._predict_z(model, query)
                output = self._format(output)
                print(NEW_TOKEN + '=' + str(output))
            print("*************************************************************************")

    def _train(self, model, vec_targets, queries):
        loss_fct = nn.MSELoss(reduction='mean')  # mean will be computed later
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.3, eps=1e-8)
        epoch = 1000
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch)

        # TODO Why is it different from get_kee_pseudowords_avg?
        max_length = 1 + max([len(self.builder.encode(query[0])[1]) for query in queries])  # possible padding
        input_ids_and_gather_indexes = [self.builder.encode(query[0], max_length=max_length) for query in queries]
        input_ids = torch.cat([input_id for input_id in [i for i, _ in input_ids_and_gather_indexes]], dim=0).to("cuda")
        gather_indexes = [gather_index for gather_index in [g for _, g in input_ids_and_gather_indexes]]

        # target_idx is the index of target word in the token list.
        target_idxs = [g[q[1] + 1][0] for g, q in zip(gather_indexes, queries)]
        target_idxs = torch.tensor(target_idxs, device="cuda").unsqueeze(-1)
        # token_idx is the index of target word in the vocabulary of BERT
        token_idxs = input_ids.gather(dim=-1, index=target_idxs)
        vocab_size = len(tokenizer.get_vocab())  # can be checked with tokenizer.get_added_vocab()
        min_token_idx = min(token_idxs)
        indices = torch.tensor([i for i in range(vocab_size) if i < min_token_idx], device="cuda", dtype=torch.long)

        vec_arrays = []

        for _ in trange(epoch):
            model.zero_grad()
            outputs = model(input_ids, output_hidden_states=True)
            z = torch.index_select(outputs.hidden_states[12][0], dim=0, index=target_idxs.squeeze(-1))

            loss = loss_fct(z, torch.stack(vec_targets))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.bert.embeddings.word_embeddings.weight.grad[indices] = 0
            optimizer.step()
            scheduler.step()

            # try to fix the feed-forward bug
            outputs = model(input_ids, output_hidden_states=True)
            bert_z = torch.index_select(outputs.hidden_states[12][0], dim=0, index=target_idxs.squeeze(-1))

        # get the z* for classification
        vec = model.bert.embeddings.word_embeddings(token_idxs).squeeze(1)[0]  # this is z*; [0] because all the same
        vec_array = vec.cpu().detach().numpy()
        z_list.append(vec_array)
        loss_list.append(str(loss.cpu().detach().numpy()))

        # save checkpoints
        try:
            np.save(CACHE + "temp_z_arrays.npy", np.array(z_list))
            np.save(CACHE + "temp_loss_arrays.npy", np.array(loss_list))
        except:
            # TODO Why?
            print("Skip saving this time...")

        s = 'Final loss={a}'.format(a=str(loss.cpu().detach().numpy()))
        print(s)

        return model

    def _get_target_embed(self, target, model):
        input_ids, gather_indexes = self.builder.encode(target[0])
        target_idx = gather_indexes[target[1] + 1][0]
        model.eval()
        with torch.no_grad():
            # Find the learning target x
            input_ids = input_ids.to('cuda')
            outputs = model(input_ids, output_hidden_states=True)
            x_target = outputs.hidden_states[12][0][target_idx]
        return x_target

    def _freeze(self, model):
        # Freeze all the parameters except the word embeddings
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name == 'bert.embeddings.word_embeddings.weight':
                param.requires_grad = True

        # Manually break the connection of decoder and embeddings.
        original_weight = model.cls.predictions.decoder.weight
        original_bias = model.cls.predictions.decoder.bias
        decoder = nn.Linear(768, len(tokenizer) - 1, bias=True)
        decoder.weight.requires_grad = False
        decoder.bias.requires_grad = False
        decoder.weight.data.copy_(original_weight.data[:-1])
        decoder.bias.data.copy_(original_bias.data[:-1])
        model.cls.predictions.decoder = decoder

        return model

    def _format(self, results):  # new format

        reval = []
        for item in results:
            token_str = item['token_str']
            score = item['score']
            s = ':'.join([token_str, str(score)])
            reval.append(s)
        return reval

    def _predict_z(self, model, query):
        input_ids, gather_indexes = self.builder.encode(query[0])
        # target_idx is the index of target word in the token list.
        target_idx = gather_indexes[query[1] + 1][0]
        input_ids = input_ids.to('cuda')
        outputs = model(input_ids)
        with torch.no_grad():
            logits = outputs.logits[0, target_idx, :]
        probs = logits.softmax(dim=0)
        values, predictions = probs.topk(5)
        reval = []
        for v, p in zip(values.tolist(), predictions.tolist()):
            s = {
                'score': v,
                'token_str': self.builder.tokenizer.convert_ids_to_tokens(p)
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
    # Read the columns "query" and "label" from "./data/MaPP_Dataset.csv" and save them as a dictionary:
    with open("../../data/pseudowords/MaPP_Dataset.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        labels = {row[0]: row[2] + row[5] for row in csv_reader}
    # Add labels to data:
    for d in data:
        try:
            d["label"] = labels[d["target1"]]
        except KeyError:
            d["label"] = labels[d["target1"].strip()]
    # Group the dataset into a list of lists where the label of the dictionaries is identical:
    data = [list(g) for _, g in itertools.groupby(data, key=lambda x: x["label"])]

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    builder = DataBuilder(tokenizer)
    co = Coercion(builder)
    for group in data:
        co.coercion(group)
        print('==' * 40)

result = get_lowest_loss_arrays(z_list, loss_list)

# save the pseudowords
np.save(DIR_OUT + 'pseudowords.npy', result)
