"""
Bert Coercion
"""
import argparse
import csv
import itertools
import random
from typing import List, Tuple, TextIO

from transformers import AutoTokenizer, MBart50Tokenizer, MBartForConditionalGeneration, Text2TextGenerationPipeline
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from tqdm import trange, tqdm
import jsonlines
import json
import numpy as np
import os

NEW_TOKEN = '#TOKEN#'

Item = Tuple[str, int]
Example = Tuple[Item, Item]

# ARGS
QUERIES_PATH = "../../data/pseudowords/CoMaPP_all.json"  # path to queries
DATASET_PATH = "../../data/pseudowords/CoMapp_Dataset.csv"
DIR_OUT = "../../out/"  # path to dir to save the pseudowords
CACHE = "../../out/cache/"  # path to cach directory

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
        # Build token indices
        _, gather_indexes = self._manual_tokenize(tokens)
        # Tokenization
        if max_length:
            # Note by huggingface:
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
    def __init__(self, builder: DataBuilder, batch_size: int = 8):
        self.builder = builder
        self.batch_size = batch_size

    def coercion(self, group, k: int = 5):
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50",
                                                              return_dict=True)  # load model and save to cuda
        model.to(device)

        self.builder.tokenizer.add_tokens(NEW_TOKEN)  # add the temporary token #TOKEN#
        model.resize_token_embeddings(len(self.builder.tokenizer))  # resize the model to fit the new token

        new_queries = []
        queries = []
        targets1 = []
        vec_targets = []

        # Print targets (and their ids) and the query (and its id)
        for entry in group:
            i = 1
            while ('target' + str(i)) in entry.keys():
                print(f'target {i}: {entry["target" + str(i)]}, {entry["target" + str(i) + "_idx"]}')
                i += 1
            print(f'query: {entry["query"]}, {entry["query_idx"]}')

            # We need a Text2TextGeneration here, because mBart is created for translation, originally.
            # Only this way, there can be multiple predicted words for one <mask>.
            nlp = Text2TextGenerationPipeline(model=model, tokenizer=self.builder.tokenizer, device=device)
            output = nlp(entry["query"], max_length=30, num_return_sequences=5, num_beams=20)
            output = self._format(output)
            print(f"output: {output}")

            #
            for j in range(1, i):
                target_j = entry["target" + str(j)]
                target_begin = entry["target" + str(j) + "_idx"]
                # the end of the target sequence is the begin plus the difference of target and query lengths:
                #target_end = target_begin + (len(target_j.split()) - len(entry["query"].split())) + 1
                vec_targets.append(
                    # self._get_target_embed((target_j, (target_begin, target_end)), model)
                    self._get_target_embed((target_j, target_begin), model)
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

            model = self._train(model, vec_targets, queries, targets1)

            print("*************************************************************************")
            print('After training:')

            # For determining the original token's length, you take a random (the first) target, whitespace-tokenize it
            # and extract the token string. Then you tokenize it using the tokenizer. You can then count the input_ids,
            # ignoring the first id and the final id.
            token_length = len(tokenizer(targets1[0][0].split()[targets1[0][1]])["input_ids"][1:-1])
            for new_query in set(new_queries):  # only view different queries
                print(f"query: {new_query}")

                target_length = len(new_query) - 1 + token_length  # length of new query - #TOKEN# + target token

                outputs = tokenizer(new_query, return_tensors="pt").to(model.device)
                outputs = model.generate(outputs["input_ids"], max_length=target_length, num_return_sequences=5,
                                         num_beams=20, output_scores=True, return_dict_in_generate=True)
                output_strings = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                output_probs = torch.exp(outputs.sequences_scores)

                print([f'output: {output}, score: {score}'
                       for output, score in zip(output_strings, output_probs)])

                outputs_list.append(output_strings)

            print("*************************************************************************")

    def _train(self, model, vec_targets, queries, targets1):
        loss_fct = nn.MSELoss(reduction='mean')  # mean will be computed later
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, eps=1e-8)
        epoch = 5000 // len(queries)  # 1000 was the default for BERT; but 400 seems to be enough to practically minimize the loss
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
        input_ids = torch.cat([input_id for input_id in [i for i, _ in input_ids_and_gather_indexes]], dim=0).to(device)

        # This is needed for computing the loss. This is because mBart is a generative model unlike Bert, so
        # the decoder needs the solution during training time. It also needs to be shifted right
        # (happens automatically here).
        # [0] because I don't need "gather_indexes"
        labels_and_gather_indexes = [self.builder.encode(target1[0], max_length=max_labels_length)
                                     for target1 in targets1]
        labels = torch.cat([label for label in [lab for lab, _ in labels_and_gather_indexes]], dim=0).to(device)

        gather_indexes = [gather_index for gather_index in [g for _, g in labels_and_gather_indexes]]

        # target_idx is the index of target word in the token list.
        target_idxs = [g[q[1] + 1] for g, q in zip(gather_indexes, targets1)]

        target_ranges = [range(*i) for i in target_idxs]
        target_lengths = {len(r) for r in target_ranges}

        removed = 0
        # check if all tokens have the same length (should usually be the case, but not always)
        if len(target_lengths) > 1:
            # TODO The new token has different lengths in different examples. For now, we remove the sentences with "different lengths"...
            # in case there are a few new tokens with different lengths, remove the corresponding sentences
            most_common_target_length = max({len(r) for r in target_ranges}, key=target_ranges.count)

            for i in range(len(target_ranges)):
                if len(target_ranges[i]) != most_common_target_length:
                    gather_indexes.pop(i-removed)
                    input_ids = torch.cat((input_ids[:i-removed], input_ids[i-removed+1:]))  # equivalent to "pop"
                    input_ids_and_gather_indexes.pop(i - removed)
                    labels = torch.cat((labels[:i-removed], labels[i-removed+1:]))  # equivalent to "pop"
                    labels_and_gather_indexes.pop(i-removed)
                    queries.pop(i-removed)
                    target_idxs.pop(i-removed)
                    targets1.pop(i-removed)
                    vec_targets.pop(i-removed)
                    removed += 1
        target_idxs = torch.tensor([range(*i) for i in target_idxs], device=device)

        # token_idx is the index of target word in the vocabulary of BERT
        token_idxs = input_ids.gather(dim=-1, index=target_idxs[:, 0].unsqueeze(-1))
        vocab_size = len(tokenizer)  # can be checked with tokenizer.get_added_vocab()
        min_token_idx = min(token_idxs)
        # Get all indices smaller than the new token_idx:
        indices = torch.tensor([i for i in range(vocab_size) if i < min_token_idx], device=device, dtype=torch.long)

        vec_targets = torch.stack(vec_targets).squeeze(1)

        dataloader = torch.utils.data.DataLoader(list(zip(input_ids, labels, target_idxs, vec_targets)),
                                                 batch_size=self.batch_size)

        # with tqdm(total=6, desc="Train Loss", position=2) as loss_bar:
        for _ in range(epoch):  # trange(epoch, position=1, desc="Epoch", leave=True):
            for batched_input_ids, batched_labels, batched_target_idxs, batched_vec_targets in dataloader:
                optimizer.zero_grad()

                # "Automatic mixed-precision" (AMP) is faster and helps reducing the workload of the GPU:
                # with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=False):  # maybe float16 if bfloat16 doesn't work
                # ... this does seem to be buggy, though...

                outputs = model(batched_input_ids, output_hidden_states=True, labels=batched_labels)

                z = torch.gather(
                    outputs.decoder_hidden_states[-1], dim=1,
                    index=batched_target_idxs.unsqueeze(-1).expand(-1, -1, model.config.d_model)  # d_model == 1024
                )

                loss = loss_fct(z, batched_vec_targets)
                # loss_bar.n = float(loss)
                # loss_bar.refresh()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model.model.encoder.embed_tokens.weight.grad[indices] = 0
                optimizer.step()
                scheduler.step()

        # get the z* for classification
        vec = model.get_input_embeddings()(token_idxs).squeeze(1)[0]  # this is z*; [0] because all the same
        vec_array = vec.cpu().detach().numpy()
        z_list.append(vec_array)
        loss_list.append(str(loss.cpu().detach().numpy()))

        # save checkpoints
        np.save(CACHE + f"temp_z_arrays_mbart_{temp}.npy", np.array(z_list))
        np.save(CACHE + f"temp_loss_arrays_mbart_{temp}.npy", np.array(loss_list))

        s = '\n\nFinal loss = {a}'.format(a=str(loss.cpu().detach().numpy())) + f"\n(Number of removed sentences: {removed})"
        print(s)

        return model

    def _get_target_embed(self, target, model):
        input_ids, gather_indexes = self.builder.encode(target[0])
        target_idx = gather_indexes[target[1] + 1]
        # Variant for multi-word targets (if both start and end are given and not only start):
        # target_idxs = gather_indexes[target[1][0] + 1 : target[1][1] + 1]  # [1][0] start; [1][1] end; +1 <s>
        model.eval()
        with torch.no_grad():
            # Find the learning target x
            input_ids = input_ids.to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)  # labels are shifted right automatically
            # get all indices that are part of the KEE; slice is needed for converting the tuple to a slice
            x_target = outputs.decoder_hidden_states[-1][:, slice(*target_idx)]
            # x_target = torch.cat([outputs.decoder_hidden_states[-1][:, slice(*target_idx)] for target_idx in target_idxs], dim=1)
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

        # The argument len(tokenizer)-1 should prevent the model from outputting #TOKEN#:
        lm_head = nn.Linear(in_features=1024, out_features=len(tokenizer) - 1, bias=False)
        lm_head.weight.requires_grad = False
        model.register_buffer("final_logits_bias", torch.zeros((1, model.model.shared.num_embeddings - 1)))
        lm_head.weight.data.copy_(original_weight.data[:-1])
        model.final_logits_bias.copy_(original_bias[:, :-1])
        model.lm_head = lm_head
        decoder_embed_tokens = nn.Embedding(len(tokenizer) - 1, model.config.d_model, model.config.pad_token_id)
        # For decoder, see above:
        decoder_embed_tokens.weight.data.copy_(original_decoder_embed_tokens_weight.data[:-1])
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
                token_str = item['prediction']  # item['token_str']
                score = item['score']
                s = ':'.join([token_str, str(score)])
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

    for vec in z_array:
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
    outputs_list = []

    temp = args.temp

    # load checkpoints if available
    if os.path.isfile(CACHE + f"temp_z_arrays_mbart_{temp}.npy"):
        z_list = np.load(CACHE + f"temp_z_arrays_mbart_{temp}.npy").tolist()
    if os.path.isfile(CACHE + f"temp_loss_arrays_mbart_{temp}.npy"):
        loss_list = np.load(CACHE + f"temp_loss_arrays_mbart_{temp}.npy").tolist()

    with open(QUERIES_PATH) as json_file:
        data = json.load(json_file)

    # Group the dataset into a list of lists where the label of the dictionaries is identical:
    data.sort(key=lambda x: x["label"])  # Grouping doesn't work without sorting first!
    data = [list(group) for _, group in itertools.groupby(data, key=lambda x: x["label"])]

    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="de_DE", tgt_lang="de_DE")
    builder = DataBuilder(tokenizer)
    co = Coercion(builder, batch_size)

    start = args.start
    end = args.end
    #if args.use_checkpoint:
    #    start = len(z_list) // 5
    #    assert len(z_list) % 5 == 0
    #    assert start < end

    print(f"Started at construction number {start}.")
    i = start
    for group in tqdm(data[start:end], initial=start, total=len(data),
                      desc="Construction", position=0, leave=True):
        try:
            co.coercion(group)  # , devices)
            print('==' * 40)
            result = get_lowest_loss_arrays(z_list, loss_list)

            # save the pseudowords
            np.save(DIR_OUT + f'pseudowords_comapp_{start}_{end}.npy', result)
        except Exception as e:
            if type(e) != KeyboardInterrupt:
                print(f"Construction with index {i} threw an error!\n" + e)
        i += 1

    result = get_lowest_loss_arrays(z_list, loss_list)
    np.save(DIR_OUT + f'pseudowords_comapp_{start}_{end}.npy', result)
