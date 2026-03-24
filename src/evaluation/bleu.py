# src/evaluation/bleu.py

from nltk.translate.bleu_score import corpus_bleu


def compute_bleu(predictions, references):
    """
    predictions: list of strings
    references: list of strings
    """

    # Tokenize
    preds_tokens = [pred.split() for pred in predictions]
    refs_tokens = [[ref.split()] for ref in references]

    score = corpus_bleu(refs_tokens, preds_tokens)

    return score