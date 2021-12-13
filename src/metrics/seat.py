from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import math
from ..data import professions

data_file = professions.load_dataset()

male_list = data_file.male_list
female_list = data_file.female_list
# female_list
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
XX = [
    "female",
    "woman",
    "girl",
    "sister",
    "daughter",
    "mother",
    "aunt",
    "grandmother",
    "daughter",
]
YY = ["male", "man", "boy", "brother", "son", "father", "uncle", "grandfather", "son"]


def get_index(sentence, word):
    toks = tokenizer(sentence).input_ids
    wordpieces = tokenizer(word).input_ids
    #     print(toks)
    word = wordpieces[1]  # use first wordpiece
    for i, t in enumerate(toks):
        if t == word:
            return i


def sentence_embedding(template, word, ind):
    # CLS embedding
    if ind == 0:
        sentence = template.replace("_", word)
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        token_embeddings = last_hidden_states
        return token_embeddings[0][0].cpu().detach().numpy()

    # no context
    if ind == 1:
        template = "_"
        sentence = template.replace("_", word)
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        token_embeddings = last_hidden_states
        return token_embeddings[0][0].cpu().detach().numpy()

    # no context pooled
    if ind == 2:
        template = "_"
        start = 1
        end = -1
        sentence = template.replace("_", word)
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        token_embeddings = last_hidden_states
        input_mask_expanded = (
            inputs.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        start = get_index(sentence, word)
        sum_embeddings = torch.sum(token_embeddings[0][start:end], 0)
        pooled_output = sum_embeddings
        return pooled_output.cpu().detach().numpy()

    # SWP pooled
    if ind == 3:
        start = 4
        end = -2
        sentence = template.replace("_", word)
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        token_embeddings = last_hidden_states
        input_mask_expanded = (
            inputs.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        start = get_index(sentence, word)
        sum_embeddings = torch.sum(token_embeddings[0][start:end], 0)
        pooled_output = sum_embeddings
        return pooled_output.cpu().detach().numpy()

    # SWP first embedding
    if ind == 4:
        start = 4
        sentence = template.replace("_", word)
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        token_embeddings = last_hidden_states
        input_mask_expanded = (
            inputs.attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        start = get_index(sentence, word)
        embeddings = token_embeddings[0][start]
        #         pooled_output = (sum_embeddings)
        #         print(embeddings.shape)
        return embeddings.cpu().detach().numpy()

    # Vulic
    if ind == 5:
        template = "_"
        sentence = template.replace("_", word)
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states_1_4 = outputs["hidden_states"][1:5]
        hidden_states = torch.stack(
            [torch.FloatTensor(i[0]) for i in hidden_states_1_4]
        )
        mean_embeddings = torch.mean(hidden_states[:, 1:-1, :], 0)
        mean_embeddings = torch.mean(mean_embeddings, 0)
        #         print((mean_embeddings.shape))
        return mean_embeddings.cpu().detach().numpy()


def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    Returns an array of size (len(XY), len(AB)) containing cosine similarities
    between items in XY and items in AB.
    """

    cossims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
    return cossims


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )


def s_XAB(A, s_wAB_memo):
    return s_wAB_memo[A].sum()


def s_wAB(X, Y, cossims):
    """
    Return vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cossims[X, :].mean(axis=0) - cossims[Y, :].mean(axis=0)


def s_XAB_df(A, B, s_wAB_memo):
    df1 = pd.DataFrame(s_wAB_memo[A])
    df2 = pd.DataFrame(s_wAB_memo[B])
    return df1, df2


def s_XYAB(A, B, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    the WEAT test statistic for p-value computation.
    """
    return s_XAB(A, s_wAB_memo) - s_XAB(B, s_wAB_memo)


def WEAT_test(X, Y, A, B, n_samples, cossims):
    """Compute the p-val for the permutation test, which is defined as
    the probability that a random even partition X_i, Y_i of X u Y
    satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    """
    X = np.array(list(X), dtype=np.int)
    Y = np.array(list(Y), dtype=np.int)
    A = np.array(list(A), dtype=np.int)
    B = np.array(list(B), dtype=np.int)

    assert len(X) == len(Y)
    size = len(X)
    s_wAB_memo = s_wAB(X, Y, cossims=cossims)
    #     print(s_wAB_memo)
    XY = np.concatenate((X, Y))

    #     if parametric:
    #     log.info('Using parametric test')
    s = s_XYAB(A, B, s_wAB_memo)
    return s


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )


def get_effect_size(df1, df2, k=0):
    diff = df1[k].mean() - df2[k].mean()
    std_ = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
    return diff / std_
