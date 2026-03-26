from src.inference.translator import Translator
from src.evaluation.bleu import compute_bleu


def evaluate(model, dataset, tokenizer_src, tokenizer_tgt, device, num_samples=100):

    translator = Translator(model, tokenizer_src, tokenizer_tgt, device)

    predictions = []
    references = []

    for i in range(num_samples):
        src = dataset['test'][i]['en']
        tgt = dataset['test'][i]['de']

        pred, _, _, _ = translator.translate(src)

        predictions.append(pred)
        references.append(tgt)

        if i < 5:
            print(f"\nSRC: {src}")
            print(f"REF: {tgt}")
            print(f"PRED: {pred}")

    bleu = compute_bleu(predictions, references)

    print(f"\nBLEU Score: {bleu:.4f}")

    return bleu