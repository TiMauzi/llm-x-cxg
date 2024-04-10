'''
This code is based on the codes from this repository:
https://github.com/tai314159/PWIBM-Putting-Words-in-Bert-s-Mouth
It has been extended to fit the need of construction detection.
'''
import argparse
import itertools
import random
from typing import List, Tuple, TextIO
import sys
import logging
from statistics import mean

from tokenizers import AddedToken
from transformers import (AutoTokenizer, MBartForConditionalGeneration,
                          Text2TextGenerationPipeline, MBart50TokenizerFast, get_linear_schedule_with_warmup)
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from tqdm import trange, tqdm
import jsonlines
import json
import numpy as np
import os

logging_handlers = [logging.StreamHandler(sys.stdout)]
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=logging_handlers
)

NEW_TOKEN = AddedToken('#TOKEN#', single_word=False, lstrip=True, rstrip=True, normalized=False)

Item = Tuple[str, int]
Example = Tuple[Item, Item]

# ARGS
QUERIES_PATH = ("../../data/pseudowords/CoMaPP_all.json")  # path to queries
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
        if max_length:
            encode_dict = self.tokenizer(
                tokens, return_attention_mask=True,
                return_token_type_ids=False, return_tensors='pt',
                padding='max_length', max_length=max_length, is_split_into_words=True, add_special_tokens=False)
        else:
            encode_dict = self.tokenizer(
                tokens, return_attention_mask=True,
                return_token_type_ids=False, return_tensors='pt', is_split_into_words=True, add_special_tokens=False)
        input_ids = encode_dict['input_ids']

        gather_indexes = encode_dict.word_ids()
        gather_indexes = [-1 if w is None else w for w in gather_indexes]
        gather_indexes = torch.tensor(gather_indexes)
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

    def coercion(self, group_no, group, k: int = 5):
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50",
                                                              return_dict=True)  # load model and save to cuda
        model.to(device)

        new_queries = []
        queries = []
        targets1 = []
        vec_targets = []

        # Print targets (and their ids) and the query (and its id)
        for entry in group:
            # The inputs need to be tweaked slightly in order to work with BART's index shift:
            entry["target1"] = "<s> " + entry["target1"] + " </s> de_DE"
            entry["query"] = "<s> " + entry["query"] + " </s> de_DE"
            entry["target1_idx"] += 1
            entry["query_idx"] += 1

            print(f'target1: {entry["target1"]}, {entry["target1_idx"]}')
            print(f'query: {entry["query"]}, {entry["query_idx"]}')

            # We need a Text2TextGeneration here, because mBart is created for translation, originally.
            # Only this way, there can be multiple predicted words for one <mask>.
            nlp = Text2TextGenerationPipeline(model=model, tokenizer=self.builder.tokenizer, device=device)
            output = nlp(entry["query"], max_length=int(len(entry["target1"]) * 1.5), num_return_sequences=5,
                         num_beams=20)
            output = self._format(output)
            print(f"output: {output}")

        self.builder.tokenizer.add_tokens(NEW_TOKEN)  # add the temporary token #TOKEN#
        model.resize_token_embeddings(len(self.builder.tokenizer))  # resize the model to fit the new token

        document_path = "../../out/cache/documents/"
        for entry in group:
            vec_targets.append(
                self._get_target_embed((entry["target1"], entry["target1_idx"]), model)
            )

            new_query = entry["query"].split()
            if new_query[entry["query_idx"]] == "[MASK]":
                continue  # don't let #TOKEN# and [MASK] overlap
            else:
                new_query[entry["query_idx"]] = NEW_TOKEN.content
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
            weight = model.model.encoder.embed_tokens.weight.data[-1]
            nn.init.normal_(weight, mean=0.0, std=model.config.init_std)

            model, losses = self._train(model, vec_targets, queries, targets1)

            print("*************************************************************************")
            print('After training:')

            # For determining the original token's length, you take a random (the first) target, whitespace-tokenize it
            # and extract the token string. Then you tokenize it using the tokenizer. You can then count the input_ids,
            # ignoring the first id and the final id.
            token_length = len(tokenizer(targets1[0][0].split()[targets1[0][1]])["input_ids"][1:-1])
            for new_query in set(new_queries):  # only view different queries
                print(f"query: {new_query}")
                target_length = len(new_query) - 1 + token_length  # length of new query - #TOKEN# + target token

                try:
                    outputs = tokenizer(new_query, return_tensors="pt").to(device)
                    outputs = model.generate(outputs["input_ids"], max_length=target_length,
                                                       num_return_sequences=5,
                                                       num_beams=20, output_scores=True, return_dict_in_generate=True)
                    output_strings = tokenizer.batch_decode(outputs.sequences,
                                                            clean_up_tokenization_spaces=True)
                    output_probs = torch.exp(outputs.sequences_scores)
                    print([f'output: {output}, score: {score}'
                           for output, score in zip(output_strings, output_probs)])
                except torch.cuda.OutOfMemoryError:
                    print("Output too large for CUDA.")

            print("*************************************************************************")

    def _train(self, model, vec_targets, queries, targets1):
        loss_fct = nn.MSELoss(reduction='mean')  # mean will be computed later
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035)
        epoch = 5000 // len(queries)  # 1000 == 5000//5 was the default for BERT
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=epoch)

        # This snippet, retrieving the possible padding, does the following:
        #  (a) encode each query's text (first [0]),
        #  (b) get the input_ids (second [0]),
        #  (c) count the input_ids (.shape[-1], because the number of input_ids is stored in the second/last dimension).
        # Then, you can take the max to know how much you should pad the rest.
        max_length = max([(self.builder.encode(query[0])[0]).shape[1] for query in queries])
        max_labels_length = max([(self.builder.encode(target1[0])[0]).shape[1] for target1 in targets1])

        input_ids_and_gather_indexes = [self.builder.encode(query[0], max_length=max_length) for query in queries]
        input_ids = torch.cat([input_id for input_id in [i for i, _ in input_ids_and_gather_indexes]], dim=0).to(device)

        # This is needed for computing the loss. mBart is a generative model unlike Bert, so
        # the decoder needs the solution during training time. It also needs to be shifted right
        # (happens automatically here).
        labels_and_gather_indexes = [self.builder.encode(target1[0], max_length=max_labels_length)
                                     for target1 in targets1]
        labels = torch.cat([label for label in [lab for lab, _ in labels_and_gather_indexes]], dim=0).to(device)
        gather_indexes = [g for _, g in labels_and_gather_indexes]

        vec_target_lengths = [t.shape[1] for t in vec_targets]
        vec_targets = torch.nn.utils.rnn.pad_sequence([vec_target.squeeze() for vec_target in vec_targets],
                                                      batch_first=True, padding_value=1)

        z_lengths = torch.tensor([[g.eq(t[1]).sum().item()] for g, t in zip(gather_indexes, targets1)], device=device)

        dataloader = torch.utils.data.DataLoader(
            list(zip(input_ids, labels, vec_targets, vec_target_lengths, z_lengths)),
            batch_size=self.batch_size
        )

        indices = torch.tensor(
            [i for i in range(len(tokenizer)) if i < tokenizer.convert_tokens_to_ids(NEW_TOKEN.content)],
            device=device, dtype=torch.long
        )
        # model.train()

        mean_loss = 0.0
        mean_losses = []
        with tqdm(total=100, desc="Train Loss", position=2, disable=True) as loss_bar:
            with trange(epoch, position=1, desc="Epoch", leave=True, disable=False) as epoch_bar:
                for _ in epoch_bar:
                    losses = []
                    for batched_input_ids, batched_labels, batched_vec_targets, \
                            batched_vec_target_lengths, batched_z_lengths in dataloader:
                        optimizer.zero_grad()

                        outputs = model(batched_input_ids, output_hidden_states=True, labels=batched_labels)
                        output_ids = outputs.logits.argmax(dim=-1)

                        offsets = torch.tensor([
                            batched_labels[d].shape[-1] - output_ids[d].shape[-1] + 1
                            if decoded.index("<mask>") < decoded.index("#TOKEN#") else 0
                            for d, decoded in enumerate(tokenizer.batch_decode(batched_input_ids))
                        ], device=device).unsqueeze(-1)

                        # Idea taken from here:
                        # https://huggingface.co/docs/transformers/model_doc/mbart#transformers.MBartForConditionalGeneration.forward.example-2
                        z_idxs = torch.stack([
                            torch.arange(
                                (i == torch.tensor(
                                    tokenizer.convert_tokens_to_ids(NEW_TOKEN.content))).nonzero().item(),
                                (i == torch.tensor(tokenizer.convert_tokens_to_ids(
                                    NEW_TOKEN.content))).nonzero().item() + z_length.item()
                            )
                            for i, z_length in zip(batched_input_ids, batched_z_lengths)
                        ]).to(device)

                        z_pred_idxs = z_idxs + offsets
                        z_pred_tokens = torch.gather(output_ids, -1, z_pred_idxs)
                        z_targets = torch.gather(batched_vec_targets, 1, z_idxs.unsqueeze(-1)
                                                 .repeat(1, 1, model.config.d_model))
                        z_preds = torch.gather(outputs.decoder_hidden_states[-1], 1, z_pred_idxs.unsqueeze(-1)
                                               .repeat(1, 1, model.config.d_model))

                        sum_loss = 0.0
                        for p, t, z_l in zip(z_preds, z_targets, batched_z_lengths):
                                sum_loss += loss_fct(p, t) * z_l.item()
                        loss = sum_loss / len(z_preds)  # get the mean of all losses
                        losses.append(float(loss))
                        predictions = [(tokenizer.decode(z), tokenizer.decode(o)) for z, o in zip(z_pred_tokens, output_ids)]
                        # print("\n" + str(predictions))
                        epoch_bar.set_postfix({"loss": mean_loss, "pred": predictions[0][0]})

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        model.model.encoder.embed_tokens.weight.grad[indices] = 0
                        optimizer.step()
                        scheduler.step()
                    mean_loss = mean(losses)
                    mean_losses.append(mean_losses)
                    loss_bar.n = mean_loss
                    loss_bar.refresh()

        # get the z* for classification
        vec = model.model.shared.weight.data[-1]  # == model.model.encoder.embed_tokens.weight.data[-1]  # this is z*
        vec_array = vec.cpu().detach().numpy()
        z_list.append(vec_array)
        loss_list.append(str(mean_loss))  # add the last mean loss

        # save checkpoints
        np.save(CACHE + f"temp_z_arrays_mbart_{temp}.npy", np.array(z_list))
        np.save(CACHE + f"temp_loss_arrays_mbart_{temp}.npy", np.array(loss_list))

        s = f'\n\nFinal loss = {mean_loss}'
        print(s)

        return model, mean_losses

    def _get_target_embed(self, target, model):
        input_ids, gather_indexes = self.builder.encode(target[0])
        model.eval()
        with torch.no_grad():
            # Find the learning target x
            input_ids = input_ids.to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True,
                            return_dict=True)  # labels are shifted right automatically
            output_ids = outputs.logits.argmax(dim=-1)
        return outputs.decoder_hidden_states[-1]

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

    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50",
                                                     src_lang="de_DE",
                                                     tgt_lang="de_DE")  # MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="de_DE", tgt_lang="de_DE")
    # bug fix for <mask> token
    mask_addedtoken = AddedToken('<mask>', single_word=False, lstrip=True, rstrip=True, normalized=True)
    tokenizer.add_tokens(mask_addedtoken)
    # tokenizer.padding_side = "left"

    builder = DataBuilder(tokenizer)
    co = Coercion(builder, batch_size)

    start = args.start
    end = args.end

    print(f"Started at construction number {start}.")
    i = start
    for group in tqdm(data[start:end], initial=start, total=len(data),
                      desc="Construction", position=0, leave=True):
        try:
            print(i, group[0]["label"])

            co.coercion(i, group)  # , devices)
            print('==' * 40)
            result = get_lowest_loss_arrays(z_list, loss_list)

            # save the pseudowords
            np.save(DIR_OUT + f'pseudowords_comapp_{start}_{end}.npy', result)

            with open(DIR_OUT + f"order_{temp}.csv", "a+") as order_file:
                order_file.write(f"{i};" + group[0]["label"] + "\n")

        except Exception as e:
            if type(e) != KeyboardInterrupt:
                print(f"Construction with index {i} threw an error!\n", e, "\n")
        i += 1

    result = get_lowest_loss_arrays(z_list, loss_list)
    np.save(DIR_OUT + f'pseudowords_comapp_{start}_{end}.npy', result)
