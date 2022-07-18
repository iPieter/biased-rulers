
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from pytest import approx

from biased_rulers.metrics import disco

def test_disco_bert_base_names():
    model_type = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    print(f"Loaded {model_type}")

    attribute_template = "This is the _."
    target_template = "This is the _."

    score = disco.disco_test(tokenizer, model)

    # Score for the bert-base names test (taken from https://arxiv.org/pdf/2010.06032.pdf)
    assert score == approx(3.7, abs=1e-1)