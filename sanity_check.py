import pickle
import sacrebleu
import evaluate

# -------------------------
# Load predictions & references
# -------------------------
with open('results_greedy.pkl', 'rb') as f:
    data = pickle.load(f)

predictions = data['predictions']
references = data['references']

refs = [[r] for r in references]

# -------------------------
# Method 1: Official sacrebleu (corpus_bleu)
# -------------------------
result_corpus = sacrebleu.corpus_bleu(predictions, refs)
print(f"Corpus BLEU (default = BLEU-4): {result_corpus.score:.4f}")

# BLEU-1..4 using precisions
for n in range(1, 5):
    # compute full BLEU, then just take precision for that n
    bleu = sacrebleu.corpus_bleu(predictions, refs, smooth_method="exp")
    precision = bleu.precisions[n-1]
    print(f"BLEU-{n}: {precision:.4f}")

# -------------------------
# Method 2: HuggingFace evaluate wrapper
# -------------------------
sacrebleu_hf = evaluate.load("sacrebleu")
result_hf = sacrebleu_hf.compute(predictions=predictions, references=[[r] for r in references])
print(f"HuggingFace BLEU: {result_hf['score']:.4f}")
