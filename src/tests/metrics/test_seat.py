
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from pytest import approx

from src.metrics import seat

def test_seat_bert_base():
    model_type = "bert-base-multilingual-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModel.from_pretrained(model_type)
    print(f"Loaded {model_type}")

    attribute_template = "This is the _."
    target_template = "This is the _."

    results = seat.seat_test(attribute_template, target_template, tokenizer, model)
    score = np.fromiter(results.values(), dtype=float).mean()
    print(score)

    assert score == approx(0.4365, abs=1e-3)