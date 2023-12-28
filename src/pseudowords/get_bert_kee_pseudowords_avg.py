"""
Bert Coercion
"""
import argparse
import csv
import itertools
import random
from typing import List, Tuple, TextIO

from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForMaskedLM
from transformers import FillMaskPipeline
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
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

NEW_TOKEN = '#TOKEN#'

Item = Tuple[str, int]
Example = Tuple[Item, Item]

# ARGS
QUERIES_PATH = "../../data/pseudowords/CoMaPP_all_bert.json"  # path to queries
DIR_OUT = "../../out/"  # path to dir to save the pseudowords
CACHE = "../../out/cache/"  # path to cach directory
DOC_PATH = "../../out/cache/documents/"

device = "cuda"


################################################

# Helper functions:

def setup_seed(seed: int, strict: bool = True):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(min(seed, 2 ** 32 - 1))
    if strict:
        setup_reproducible()


def setup_reproducible():
    torch.backends.cudnn.benchmark = False  # accelerate code speed

    # "A handful of CUDA operations are nondeterministic if the CUDA version is 10.2 or greater, unless the environment
    # variable CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8 is set."
    # (from: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

################################################

class DataBuilder:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, max_length=None):
        tokens = text.split()
        #tokens = word_tokenize(text)
        #if '[MASK]' in text:
        #    tokens = rejoin_mask(tokens)
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


def rejoin_mask(new_query):
    start_index = new_query.index('[')
    mask_index = new_query.index('MASK')
    end_index = new_query.index(']')
    assert start_index + 1 == mask_index and mask_index + 1 == end_index
    # Join '[', 'MASK', and ']' to '[MASK]'
    new_query[start_index:end_index + 1] = [''.join(new_query[start_index:end_index + 1])]
    return new_query


class Coercion:
    def __init__(self, builder: DataBuilder, batch_size: int = 8):
        self.builder = builder
        self.batch_size = batch_size

    def coercion(self, group_no, group, k: int = 5):
        model = BertForMaskedLM.from_pretrained('bert-base-german-cased', return_dict=True)
        model.to(device)

        new_queries = []
        queries = []
        targets1 = []
        vec_targets = []

        # Print targets (and their id's) and the query (and its id)
        for entry in group:
            i = 1

            # Make sure there are no tokenization mismatches between target and query:
            #entry["target1"] = " ".join(word_tokenize(entry["target1"])).replace("``", '"')
            #entry["query"] = " ".join(rejoin_mask(word_tokenize(entry["query"]))).replace("``", '"')

            print(f'target1: {entry["target1"]}, {entry["target1_idx"]}')
            print(f'query: {entry["query"]}, {entry["query_idx"]}')

            # Model output
            nlp = FillMaskPipeline(model=model, tokenizer=self.builder.tokenizer, device=device)
            output = nlp(entry["query"])
            output = self._format(output)
            print('[MASK]=' + str(output))

        self.builder.tokenizer.add_tokens(NEW_TOKEN)  # add the temporary token #TOKEN#
        model.resize_token_embeddings(len(self.builder.tokenizer))  # resize the model to fit the new token

        try:
            with open(DOC_PATH + "new_queries_bert_" + str(group_no), "rb") as file:
                new_queries = pickle.load(file)
            with open(DOC_PATH + "queries_bert_" + str(group_no), "rb") as file:
                queries = pickle.load(file)
            with open(DOC_PATH + "vec_targets_bert_" + str(group_no), "rb") as file:
                vec_targets = pickle.load(file)

        except FileNotFoundError:
            for entry in group:

                vec_targets.append(
                    self._get_target_embed((entry["target1"], entry["target1_idx"]), model)
                )

                new_query = entry["query"].split()
                if new_query[entry["query_idx"]] == "[MASK]":
                    continue  # don't let #TOKEN# and [MASK] overlap
                else:
                    new_query[entry["query_idx"]] = NEW_TOKEN
                    new_query = ' '.join(new_query)
                    query = (new_query, entry["query_idx"])
                    print(query)
                    new_queries.append(new_query)
                    queries.append(query)

            with open(DOC_PATH + "new_queries_bert_" + str(group_no), "wb") as file:
                pickle.dump(new_queries, file)
            with open(DOC_PATH + "queries_bert_" + str(group_no), "wb") as file:
                pickle.dump(queries, file)
            with open(DOC_PATH + "vec_targets_bert_" + str(group_no), "wb") as file:
                pickle.dump(vec_targets, file)

        model = self._freeze(model)

        model.eval()

        for i in range(k):
            print('-' * 40)
            print('Random {a}'.format(a=i))

            # Random initialization, same initialization as huggingface
            weight = model.bert.embeddings.word_embeddings.weight.data[-1]
            nn.init.normal_(weight, mean=0.0, std=model.config.initializer_range)

            model = self._train(model, vec_targets, queries)

            print("*************************************************************************")
            print('After training:')
            nlp = FillMaskPipeline(model, self.builder.tokenizer, device=0)
            for new_query, query in set(zip(new_queries, queries)):  # only view different queries
                print("query: " + new_query)
                assert "[MASK]" in new_query
                output = nlp(new_query)
                output = self._format(output)
                print('[MASK]=' + str(output))

                output = self._predict_z(model, query)
                output = self._format(output)
                print(NEW_TOKEN + '=' + str(output))
            print("*************************************************************************")

    def _train(self, model, vec_targets, queries):
        loss_fct = nn.MSELoss(reduction='mean')  # mean will be computed later
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.3, eps=1e-8)
        epoch = 100 // len(queries)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch)

        # TODO Why is it different from get_kee_pseudowords_avg?
        max_length = 1 + max([self.builder.encode(query[0])[0].shape[1] for query in queries])  # 1 + max([len(self.builder.encode(query[0])[1]) for query in queries])  # possible padding
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

        vec_targets = torch.stack(vec_targets)

        dataloader = torch.utils.data.DataLoader(list(zip(input_ids, target_idxs, vec_targets)),
                                                 batch_size=self.batch_size)

        with tqdm(total=6, desc="Train Loss", position=2, disable=True) as loss_bar:
            for _ in trange(epoch, position=1, desc="Epoch", leave=True, disable=False):
                for batched_input_ids, batched_target_idxs, batched_vec_targets in dataloader:
                    model.to(device)
                    optimizer.zero_grad()
                    outputs = model(batched_input_ids, output_hidden_states=True)
                    z = torch.index_select(outputs.hidden_states[12][0], dim=0, index=batched_target_idxs.squeeze(-1))

                    loss = loss_fct(z, batched_vec_targets)

                    loss_bar.n = float(loss)
                    loss_bar.refresh()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    model.bert.embeddings.word_embeddings.weight.grad[indices] = 0
                    optimizer.step()
                    scheduler.step()

        # get the z* for classification
        vec = model.bert.embeddings.word_embeddings(token_idxs).squeeze(1)[0]  # this is z*; [0] because all the same
        vec_array = vec.cpu().detach().numpy()
        z_list.append(vec_array)
        loss_list.append(str(loss.cpu().detach().numpy()))

        # save checkpoints
        np.save(CACHE + f"temp_z_arrays_bert_{temp}.npy", np.array(z_list))
        np.save(CACHE + f"temp_loss_arrays_bert_{temp}.npy", np.array(loss_list))

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
    setup_seed(15)
    batch_size = 4
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda", type=str, required=False, help='Task ID for the current task')
    parser.add_argument('--start', default=0, type=int, required=False, help='Which construction to start with')
    parser.add_argument('--end', default=562, type=int, required=False, help='Which construction to stop at')
    parser.add_argument('--temp', default=False, type=int, required=False, help='Which temp files to use.')
    args = parser.parse_args()

    device = args.device

    z_list = []
    z_eps_list = []
    loss_list = []

    temp = args.temp

    # load checkpoints if available
    if os.path.isfile(CACHE + f"temp_z_arrays_bert_{temp}.npy"):
        z_list = np.load(CACHE + f"temp_z_arrays_bert_{temp}.npy").tolist()
    if os.path.isfile(CACHE + f"temp_loss_arrays_bert_{temp}.npy"):
        loss_list = np.load(CACHE + f"temp_loss_arrays_bert_{temp}.npy").tolist()

    with open(QUERIES_PATH) as json_file:
        data = json.load(json_file)

    # Group the dataset into a list of lists where the label of the dictionaries is identical:
    data.sort(key=lambda x: x["label"])  # Grouping doesn't work without sorting first!
    data = [list(group) for _, group in itertools.groupby(data, key=lambda x: x["label"])]

    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    builder = DataBuilder(tokenizer)
    co = Coercion(builder, batch_size)

    start = args.start
    end = args.end

    print(f"Started at construction number {start}.")

    i = start
    for group in tqdm(data[start:end], initial=start, total=len(data),
                      desc="Construction", position=0, leave=True):
        print(i, group[0]["label"])
        co.coercion(i, group)
        print('==' * 40)
        result = get_lowest_loss_arrays(z_list, loss_list)

        # save the pseudowords
        np.save(DIR_OUT + f'pseudowords_comapp_bert_{start}_{end}.npy', result)

        with open(DIR_OUT + f"order_bert_{temp}.csv", "a+") as order_file:
            order_file.write(f"{i};" + group[0]["label"] + "\n")

        i += 1

    result = get_lowest_loss_arrays(z_list, loss_list)
    np.save(DIR_OUT + f'pseudowords_comapp_bert_{start}_{end}.npy', result)
