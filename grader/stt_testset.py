"""Build the Phase 0 STT test set: the lexicon of signal words and the
sentence set the human reader (and the TTS voices) will record.

Run:  .venv\\Scripts\\python grader\\stt_testset.py [--print]

Writes grader/stt_lexicon.json and grader/stt_sentences.jsonl. Both are
deterministic (seeded) so they can be regenerated from the public
question banks; a private overlay data/stt/lexicon_extra.json (resume and
project vocabulary, gitignored) is merged if present.

Lexicon entry: {"term", "category", "aliases", "short", "common", "source"}
  term      the written form a good transcript should contain
  aliases   spoken/written variants that still count as recovered under
            lenient term matching ("quadratic weighted kappa" for QWK)
  short     a <= 20-character form for Scribe Realtime keyterms
  common    an ordinary English word STT already knows (recall, precision):
            counted, but ranked last by the keyterm policy
  category  metric | model | library | concept | mixed | tool | product

Sentence item: {"id", "kind", "text", "chunk_id", "terms", "words"}
  kind = "sentence" (curated hard-term sentences and bank sentences, for
  WER and term error rate) or "answer" (a whole bank model answer, <= 70
  words, for downstream grade damage: it is graded against its chunk's
  rubric before and after transcription).
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from grader.stt_text import Lexicon, normalize  # noqa: E402

LEXICON_PATH = BASE_DIR / "grader" / "stt_lexicon.json"
SENTENCES_PATH = BASE_DIR / "grader" / "stt_sentences.jsonl"
EXTRA_PATH = BASE_DIR / "data" / "stt" / "lexicon_extra.json"
CORPORA = {"ml": BASE_DIR / "rag_ml" / "all_chunks.jsonl",
           "ai": BASE_DIR / "rag_ai" / "all_chunks.jsonl"}
SEED = 20260830
N_BANK_SENTENCES = 36
N_ANSWERS = 20

# --------------------------------------------------------------- curated
# (term, category, aliases, short, common). The order is the "naive" keyterm
# baseline order, so it is deliberately not sorted by importance.

CURATED = [
    # metrics
    ("QWK", "metric", ["quadratic weighted kappa"], None, False),
    ("MAE", "metric", ["mean absolute error"], None, False),
    ("MAPE", "metric", ["mean absolute percentage error"], None, False),
    ("RMSE", "metric", ["root mean squared error", "root mean square error"], None, False),
    ("MSE", "metric", ["mean squared error"], None, False),
    ("F1", "metric", ["f1 score", "f one"], None, False),
    ("macro-F1", "metric", ["macro f1"], None, False),
    ("AUC", "metric", ["area under the curve"], None, False),
    ("ROC", "metric", [], None, False),
    ("AUC-ROC", "metric", ["roc auc", "auc roc"], None, False),
    ("PR-AUC", "metric", ["precision recall auc", "pr auc"], None, False),
    ("log-loss", "metric", ["logloss", "logarithmic loss"], None, False),
    ("cross-entropy", "metric", [], None, False),
    ("Recall@5", "metric", ["recall at five", "recall at 5"], None, False),
    ("Recall@k", "metric", ["recall at k"], None, False),
    ("MRR", "metric", ["mean reciprocal rank"], None, False),
    ("NDCG", "metric", [], None, False),
    ("R2", "metric", ["r squared", "r square"], None, False),
    ("p-value", "metric", ["p value"], None, False),
    ("p95", "metric", ["p 95", "ninety fifth percentile"], None, False),
    ("p50", "metric", ["p 50", "median latency"], None, False),
    ("TTFT", "metric", ["time to first token"], None, False),
    ("WER", "metric", ["word error rate"], None, False),
    ("Brier score", "metric", [], None, False),
    ("KL divergence", "metric", ["kullback leibler divergence"], None, False),
    ("precision", "metric", [], None, True),
    ("recall", "metric", [], None, True),
    ("accuracy", "metric", [], None, True),
    ("calibration", "metric", [], None, True),
    # models and algorithms
    ("LightGBM", "model", ["light gbm"], None, False),
    ("XGBoost", "model", ["xg boost", "extreme gradient boosting"], None, False),
    ("CatBoost", "model", ["cat boost"], None, False),
    ("HistGradientBoosting", "model", ["hist gradient boosting", "histogram gradient boosting"], "HistGradientBoost", False),
    ("random forest", "model", [], None, False),
    ("gradient boosting", "model", [], None, False),
    ("logistic regression", "model", [], None, False),
    ("linear regression", "model", [], None, False),
    ("Ridge", "model", ["ridge regression"], None, False),
    ("Lasso", "model", ["lasso regression"], None, False),
    ("ElasticNet", "model", ["elastic net"], None, False),
    ("SVM", "model", ["support vector machine"], None, False),
    ("SVC", "model", ["support vector classifier"], None, False),
    ("KNN", "model", ["k nearest neighbors", "k nearest neighbours", "k nn"], None, False),
    ("k-means", "model", ["kmeans", "k means"], None, False),
    ("DBSCAN", "model", ["db scan"], None, False),
    ("PCA", "model", ["principal component analysis"], None, False),
    ("SVD", "model", ["singular value decomposition"], None, False),
    ("t-SNE", "model", ["tsne", "t sne"], None, False),
    ("UMAP", "model", ["u map"], None, False),
    ("ARIMA", "model", [], None, False),
    ("SARIMAX", "model", [], None, False),
    ("Prophet", "model", [], None, False),
    ("LSTM", "model", [], None, False),
    ("transformer", "model", [], None, False),
    ("BERT", "model", [], None, False),
    ("Sentence-BERT", "model", ["sentence bert", "sbert"], None, False),
    ("GPT-4o", "model", ["gpt 4 o", "gpt four o"], None, False),
    ("Claude Opus", "model", [], None, False),
    ("Claude", "model", [], None, False),
    ("DeepSeek", "model", ["deep seek"], None, False),
    ("DeepSeek V4 Flash", "model", ["deepseek v4 flash", "deepseek v 4 flash"], "DeepSeek V4 Flash", False),
    ("Qwen3", "model", ["qwen 3", "qwen three"], None, False),
    ("Llama 3", "model", ["llama three"], None, False),
    ("Mistral", "model", [], None, False),
    ("Whisper", "model", [], None, False),
    ("faster-whisper", "model", ["faster whisper"], None, False),
    ("large-v3-turbo", "model", ["large v3 turbo", "large v 3 turbo"], None, False),
    ("Kokoro", "model", [], None, False),
    ("Silero VAD", "model", ["silero"], None, False),
    ("Scribe v2", "model", ["scribe v 2", "scribe version 2"], None, False),
    ("MiniLM", "model", ["mini lm"], None, False),
    ("bge-m3", "model", ["bge m3", "b g e m 3"], None, False),
    # libraries, tools, products
    ("scikit-learn", "library", ["sklearn", "sci kit learn"], None, False),
    ("PyTorch", "library", ["py torch"], None, False),
    ("TensorFlow", "library", ["tensor flow"], None, False),
    ("NumPy", "library", ["numpy"], None, False),
    ("pandas", "library", [], None, False),
    ("DataFrame", "library", ["data frame"], None, False),
    ("GridSearchCV", "library", ["grid search cv", "grid search c v"], None, False),
    ("RandomizedSearchCV", "library", ["randomized search cv"], "RandomizedSearch", False),
    ("TimeSeriesSplit", "library", ["time series split"], None, False),
    ("GroupKFold", "library", ["group k fold"], None, False),
    ("StandardScaler", "library", ["standard scaler"], None, False),
    ("ColumnTransformer", "library", ["column transformer"], "ColumnTransformer", False),
    ("Pipeline", "library", [], None, True),
    ("Optuna", "library", [], None, False),
    ("MLflow", "library", ["ml flow"], None, False),
    ("Airflow", "library", ["air flow"], None, False),
    ("Spark", "library", [], None, False),
    ("Kafka", "library", [], None, False),
    ("Docker", "tool", [], None, False),
    ("Kubernetes", "tool", ["k8s"], None, False),
    ("EC2", "tool", ["ec 2", "e c 2"], None, False),
    ("t3.micro", "tool", ["t3 micro", "t 3 micro"], None, False),
    ("S3", "tool", ["s 3"], None, False),
    ("SageMaker", "tool", ["sage maker"], None, False),
    ("nginx", "tool", ["engine x"], None, False),
    ("WebSocket", "tool", ["web socket", "websockets"], None, False),
    ("FastAPI", "tool", ["fast api"], None, False),
    ("SQL", "tool", ["sequel"], None, False),
    ("JSON", "tool", [], None, False),
    ("JSON schema", "tool", [], None, False),
    ("CUDA", "tool", [], None, False),
    ("cuDNN", "tool", ["cu dnn"], None, False),
    ("GPU", "tool", [], None, False),
    ("VRAM", "tool", ["v ram"], None, False),
    ("RTX 5080", "tool", ["rtx fifty eighty", "rtx 50 80"], None, False),
    ("vLLM", "tool", ["v llm"], None, False),
    ("ONNX", "tool", ["onyx"], None, False),
    ("TensorRT", "tool", ["tensor rt"], None, False),
    ("CTranslate2", "tool", ["ctranslate 2", "c translate 2"], None, False),
    ("Ollama", "tool", ["o llama"], None, False),
    ("Hugging Face", "tool", ["huggingface"], None, False),
    ("LangChain", "tool", ["lang chain"], None, False),
    ("LangGraph", "tool", ["lang graph"], None, False),
    ("LlamaIndex", "tool", ["llama index"], None, False),
    ("FAISS", "tool", ["face", "fais"], None, False),
    ("HNSW", "tool", [], None, False),
    ("pgvector", "tool", ["pg vector"], None, False),
    ("Pinecone", "tool", ["pine cone"], None, False),
    ("BM25", "tool", ["bm 25", "b m 25"], None, False),
    ("ElevenLabs", "product", ["eleven labs"], None, False),
    ("Anthropic", "product", [], None, False),
    ("OpenAI", "product", ["open ai"], None, False),
    ("MCP", "tool", ["model context protocol"], None, False),
    ("ReAct", "concept", ["re act"], None, False),
    ("GGUF", "tool", [], None, False),
    ("LoRA", "concept", ["low rank adaptation", "lora"], None, False),
    ("QLoRA", "concept", ["q lora"], None, False),
    ("RLHF", "concept", [], None, False),
    ("DPO", "concept", ["direct preference optimization"], None, False),
    ("PPO", "concept", [], None, False),
    ("SFT", "concept", ["supervised fine tuning"], None, False),
    ("RAG", "concept", ["retrieval augmented generation"], None, False),
    # concepts
    ("quantization", "concept", ["quantisation"], None, False),
    ("regularization", "concept", ["regularisation"], None, False),
    ("collinearity", "concept", ["colinearity"], None, False),
    ("multicollinearity", "concept", ["multi collinearity"], None, False),
    ("heteroscedasticity", "concept", ["heteroskedasticity"], "heteroscedasticity", False),
    ("leakage", "concept", ["data leakage"], None, False),
    ("diarization", "concept", ["diarisation", "speaker diarization"], None, False),
    ("tokenization", "concept", ["tokenisation"], None, False),
    ("tokenizer", "concept", ["tokeniser"], None, False),
    ("embeddings", "concept", ["embedding"], None, False),
    ("logits", "concept", ["logit"], None, False),
    ("softmax", "concept", ["soft max"], None, False),
    ("sigmoid", "concept", [], None, False),
    ("k-fold", "concept", ["kfold", "k fold"], None, False),
    ("cross-validation", "concept", ["cross validation", "cv"], None, False),
    ("stratified", "concept", [], None, False),
    ("L1", "concept", ["l one", "l 1"], None, False),
    ("L2", "concept", ["l two", "l 2"], None, False),
    ("one-hot", "concept", ["one hot"], None, False),
    ("target encoding", "concept", [], None, False),
    ("bias-variance", "concept", ["bias variance"], None, False),
    ("overfitting", "concept", ["over fitting"], None, False),
    ("underfitting", "concept", ["under fitting"], None, False),
    ("hyperparameter", "concept", ["hyper parameter"], None, False),
    ("Bayesian", "concept", [], None, False),
    ("bootstrap", "concept", [], None, False),
    ("SMOTE", "concept", [], None, False),
    ("class imbalance", "concept", [], None, False),
    ("SHAP", "concept", ["shapley"], None, False),
    ("Box-Cox", "concept", ["box cox"], None, False),
    ("Yeo-Johnson", "concept", ["yeo johnson"], None, False),
    ("VIF", "concept", ["variance inflation factor"], None, False),
    ("stationarity", "concept", [], None, False),
    ("autocorrelation", "concept", ["auto correlation"], None, False),
    ("ACF", "concept", [], None, False),
    ("PACF", "concept", [], None, False),
    ("A/B test", "concept", ["ab test", "a b test", "a/b testing"], None, False),
    ("CUPED", "concept", [], None, False),
    ("confidence interval", "concept", [], None, False),
    ("drift", "concept", ["data drift", "concept drift"], None, True),
    ("attention", "concept", ["self attention"], None, True),
    ("KV cache", "concept", ["kv cache", "key value cache"], None, False),
    ("RoPE", "concept", ["rope"], None, False),
    ("context window", "concept", [], None, False),
    ("chunking", "concept", [], None, False),
    ("reranker", "concept", ["re ranker", "reranking"], None, False),
    ("cross-encoder", "concept", ["cross encoder"], None, False),
    ("bi-encoder", "concept", ["bi encoder"], None, False),
    ("cosine similarity", "concept", [], None, False),
    ("top-p", "concept", ["top p", "nucleus sampling"], None, False),
    ("temperature", "concept", [], None, True),
    ("hallucination", "concept", ["hallucinations"], None, False),
    ("function calling", "concept", ["tool calling", "tool use"], None, False),
    ("chain-of-thought", "concept", ["chain of thought"], None, False),
    ("few-shot", "concept", ["few shot"], None, False),
    ("zero-shot", "concept", ["zero shot"], None, False),
    ("system prompt", "concept", [], None, False),
    ("prompt caching", "concept", [], None, False),
    ("speculative decoding", "concept", [], None, False),
    ("mixed precision", "concept", [], None, False),
    ("distillation", "concept", ["knowledge distillation"], None, False),
    ("endpointing", "concept", ["end pointing"], None, False),
    ("barge-in", "concept", ["barge in"], None, False),
    ("latency", "concept", [], None, True),
    ("throughput", "concept", [], None, True),
    # letter-number mixes and magnitudes
    ("int8", "mixed", ["int 8", "8 bit integer"], None, False),
    ("int4", "mixed", ["int 4"], None, False),
    ("4-bit", "mixed", ["4 bit", "four bit"], None, False),
    ("fp16", "mixed", ["fp 16", "float 16", "half precision"], None, False),
    ("bf16", "mixed", ["bf 16", "bfloat 16"], None, False),
    ("float32", "mixed", ["float 32"], None, False),
    ("v2.5", "mixed", ["v 2.5", "version 2.5"], None, False),
    ("v3", "mixed", ["v 3", "version 3"], None, False),
    ("14B", "mixed", ["14 b", "14 billion"], None, False),
    ("7B", "mixed", ["7 b", "7 billion"], None, False),
    ("82M", "mixed", ["82 m", "82 million"], None, False),
    ("128k", "mixed", ["128 k", "128 thousand"], None, False),
    ("1e-4", "mixed", ["1 e 4", "1e 4", "10 to the minus 4"], None, False),
    ("seed 42", "mixed", [], None, False),
    ("5-fold", "mixed", ["5 fold", "five fold"], None, False),
    ("80/20 split", "mixed", ["80 20 split", "eighty twenty split"], None, False),
]

# Corpus-derived terms (acronyms, class names, letter-number mixes) that the
# scan below accepts. The stoplist removes look-alikes that are ordinary
# words or would never be spoken as written.
_CORPUS_STOP = {"NOT", "AND", "OR", "NA", "ID", "IDS", "DB", "PR", "SM", "PC", "AR", "MA",
                "QA", "UI", "GB", "TP", "FP", "FN", "TN", "NN", "LR", "RF", "SKILL", "EOS",
                "CSV", "NAT", "CLI", "SDK", "ABI", "RAM", "CPU", "CSR", "GLM", "PLS", "AI",
                "ML", "LLM", "LLMS", "APIS", "API", "GPT", "QK", "KV", "SVMS", "RNNS", "NANS",
                "B0", "B1", "N", "K", "T",
                # all-caps emphasis and ordinary words in the banks
                "BOTH", "CAN", "DO", "ERROR", "FIRST", "IT", "KEY", "LINEAR", "MODEL", "MORE",
                "OK", "OS", "SET", "STATES", "TARGET", "TRAIN", "WHEN", "WITH", "MS", "SD",
                "SS", "UX", "VM", "CI", "MAP", "Q3", "SETTINGWITHCOPY"}
# Single-letter-plus-digit algebra (a0, x1, e6) is notation, not vocabulary.
_NOTATION = re.compile(r"^[a-zA-Z][0-9]+$")
_CORPUS_KEEP_HYPHEN = {"one-hot", "k-means", "k-fold", "t-SNE", "Box-Cox", "Yeo-Johnson",
                       "cross-entropy", "log-loss", "macro-F1", "out-of-bag", "top-k",
                       "top-p", "few-shot", "zero-shot", "chain-of-thought", "log-odds",
                       "bag-of-words", "out-of-fold", "leave-one-out", "min-max", "F-beta",
                       "self-attention", "cross-attention", "decoder-only", "encoder-decoder",
                       "cost-complexity", "max-margin", "soft-margin", "in-context"}
_ACRONYM = re.compile(r"^[A-Z][A-Z0-9]{1,6}$")
_CAMEL = re.compile(r"^[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+)+$|^[a-z]+[A-Z][A-Za-z0-9]+$")
_LETTER_NUMBER = re.compile(r"^[A-Za-z]{1,8}[0-9]{1,3}(?:\.[0-9]+)?$")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9.-]*[A-Za-z0-9]")


def load_chunks():
    chunks = []
    for corpus, path in CORPORA.items():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    chunk["_corpus"] = corpus
                    chunks.append(chunk)
    return chunks


def chunk_text(chunk):
    interview = chunk["interview"]
    return " ".join([interview["question"], interview["model_answer"]]
                    + interview["key_points"] + interview["common_mistakes"]
                    + interview["followups"])


def corpus_terms(chunks, known):
    """Acronyms, CamelCase class names, letter-number mixes and allow-listed
    hyphenated terms that occur in the banks and are not curated already."""
    counts = {}
    for chunk in chunks:
        for token in _TOKEN.findall(chunk_text(chunk)):
            token = token.rstrip(".")
            if token.lower() in known or token.upper() in _CORPUS_STOP or _NOTATION.match(token):
                continue
            if (_ACRONYM.match(token) or _CAMEL.match(token) or _LETTER_NUMBER.match(token)
                    or token in _CORPUS_KEEP_HYPHEN):
                counts[token] = counts.get(token, 0) + 1
    entries = []
    for token, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < 1:
            continue
        if _ACRONYM.match(token):
            category = "concept"
        elif _CAMEL.match(token):
            category = "library"
        elif _LETTER_NUMBER.match(token):
            category = "mixed"
        else:
            category = "concept"
        entries.append({"term": token, "category": category, "aliases": [],
                        "short": None, "common": False, "source": "corpus",
                        "bank_count": count})
    return entries


def build_lexicon(chunks):
    entries = []
    known = set()
    for term, category, aliases, short, common in CURATED:
        entries.append({"term": term, "category": category, "aliases": aliases,
                        "short": short, "common": common, "source": "curated"})
        known.add(term.lower())
        known.update(a.lower() for a in aliases)
    entries.extend(corpus_terms(chunks, known))
    if EXTRA_PATH.exists():
        extra = json.loads(EXTRA_PATH.read_text(encoding="utf-8"))
        for entry in extra.get("terms", extra if isinstance(extra, list) else []):
            if entry["term"].lower() not in known:
                entries.append({"aliases": [], "short": None, "common": False,
                                **entry, "source": "extra"})
                known.add(entry["term"].lower())
    return entries


# ------------------------------------------------------------- sentences

# Curated hard-term sentences: interview-answer prose dense in the words the
# lexicon says carry the signal (acronyms, product names, letter-number
# mixes). The reader reads them exactly as written.
CURATED_SENTENCES = [
    "I trained a LightGBM model and an XGBoost baseline, then compared them with QWK, MAE and MAPE on a held-out fold.",
    "The distilled grader reached a QWK of 0.93 and an MAE of 0.59 against the Claude teacher labels.",
    "For retrieval I used BM25 over the question bank and measured Recall@5 and MRR on 23 labeled cases.",
    "We fine-tuned a 7B Llama 3 model with QLoRA in 4-bit, then quantized the merged weights to int8 for serving on vLLM.",
    "Running faster-whisper large-v3-turbo in fp16 on an RTX 5080 gave real-time transcription with about 16 gigabytes of VRAM.",
    "Kokoro is an 82M parameter text-to-speech model, and Silero VAD handles endpointing and barge-in in the local loop.",
    "ElevenLabs Scribe v2 supports keyterm prompting, and the Realtime version allows up to 50 keyterms per session.",
    "DeepSeek V4 Flash grades the paid tier by default, while Claude Opus stays the teacher for distillation.",
    "The backend runs in Docker behind nginx on an EC2 t3.micro instance, with the model artifacts stored in S3.",
    "I used a HistGradientBoosting classifier from scikit-learn with GroupKFold so that no chunk leaked across folds.",
    "The pipeline had a ColumnTransformer with a StandardScaler for numeric features and one-hot encoding for categoricals.",
    "I tuned the hyperparameters with GridSearchCV over a five-fold TimeSeriesSplit instead of a random 80/20 split.",
    "Ridge and Lasso differ in the penalty: L2 shrinks coefficients smoothly while L1 drives some of them exactly to zero.",
    "Because of multicollinearity the VIF was above ten, so I dropped one of the correlated features before fitting logistic regression.",
    "The residual plot showed heteroscedasticity, so I applied a Box-Cox transform to the target and refit the model.",
    "For imbalanced fraud data I compared SMOTE against class weights and reported PR-AUC rather than plain accuracy.",
    "Our RAG pipeline chunks documents to 512 tokens, embeds them with bge-m3, stores them in pgvector with HNSW, and reranks with a cross-encoder.",
    "The agent uses MCP tools with function calling, a ReAct loop, and a system prompt that is cached with prompt caching.",
    "We reduced TTFT by streaming tokens, disabling extended thinking for conversational turns, and keeping the KV cache warm.",
    "At temperature zero with top-p sampling disabled, the JSON schema output was still malformed about 2.6 percent of the time.",
    "The ARIMA baseline underperformed SARIMAX once we added the weekly seasonality and checked the ACF and PACF plots.",
    "After PCA and t-SNE the k-means clusters were clearly separated, but DBSCAN handled the outliers better.",
    "Cross-entropy is the loss for softmax outputs, while log-loss is the same idea reported as a metric on the sigmoid probabilities.",
    "RLHF and DPO both align the model after SFT, but DPO skips training a separate reward model with PPO.",
    "I set the learning rate to 1e-4 with mixed precision in bf16, gradient clipping at one, and seed 42 for reproducibility.",
    "The Qwen3 14B model ran locally through Ollama in GGUF format, which was good enough to generate paraphrase training rows.",
    "We measured p50 and p95 latency end to end, from the last user audio to the first agent audio, over a WebSocket.",
    "The A/B test used CUPED to reduce variance, and the confidence interval on the lift excluded zero with a p-value below 0.01.",
    "Speaker diarization and tokenization both happen before the embeddings are computed, so errors there propagate downstream.",
    "SHAP values explained the CatBoost predictions, and calibration was checked with a Brier score and a reliability diagram.",
    "The word error rate was low, but the term error rate on words like QWK, MAPE and LightGBM was what actually moved the grade.",
    "Whisper's initial prompt is a weak form of vocabulary biasing compared with Scribe v2 keyterms, which is exactly what condition six measures.",
]

# Anything a reader could not say as written: code, math, dotted names,
# leftover parentheses, URLs, slash commands.
_CODEISH = re.compile(
    r"[`_=^{}\[\]|<>#\\()+]|->|\bdf\b|\s\.[a-z]|[a-z]{2,}\.[a-z]{2,}|\s/|\s-[a-z]")
_PAREN = re.compile(r"\s*\([^)]*\)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_MARKDOWN = re.compile(r"[*`]")


def clean_prose(text):
    text = _PAREN.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = text.replace("—", ", ").replace("–", ", ").replace(" ,", ",")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bank_sentences(chunk):
    interview = chunk["interview"]
    out = []
    for source in (interview["question"], interview["model_answer"]):
        for sentence in _SENTENCE_END.split(clean_prose(source)):
            sentence = sentence.strip()
            words = sentence.split()
            if not 8 <= len(words) <= 32 or _CODEISH.search(sentence):
                continue
            if not sentence[0].isupper() and not sentence[0].isdigit():
                continue
            out.append(sentence)
    return out


def bank_answer(chunk, max_words=70):
    """The chunk's model answer trimmed at a sentence boundary to max_words."""
    kept = []
    total = 0
    for sentence in _SENTENCE_END.split(clean_prose(chunk["interview"]["model_answer"])):
        n = len(sentence.split())
        if _CODEISH.search(sentence):
            return None
        if total + n > max_words:
            break
        kept.append(sentence.strip())
        total += n
    if total < 35:
        return None
    return " ".join(kept)


def item_terms(lexicon, text):
    return sorted(lexicon.occurrences(normalize(text)))


def pick_diverse(candidates, lexicon, n, rng):
    """Greedy pick maximizing new lexicon-term coverage, one item per chunk."""
    scored = []
    for chunk_id, text in candidates:
        terms = set(item_terms(lexicon, text))
        hard = {t for t in terms if not lexicon.by_term[t].get("common")}
        if len(hard) >= 2:
            scored.append((chunk_id, text, terms, hard))
    rng.shuffle(scored)
    chosen, covered, used_chunks = [], set(), set()
    while scored and len(chosen) < n:
        best = max(scored, key=lambda c: (len(c[3] - covered), len(c[3]) / len(c[1].split())))
        scored = [c for c in scored if c[0] != best[0]]
        if best[0] in used_chunks:
            continue
        used_chunks.add(best[0])
        covered |= best[3]
        chosen.append(best)
    return chosen


def build_sentences(chunks, lexicon):
    rng = random.Random(SEED)
    items = []
    for i, text in enumerate(CURATED_SENTENCES):
        items.append({"id": f"cur_{i + 1:02d}", "kind": "sentence", "text": text,
                      "chunk_id": None})
    candidates = [(chunk["id"], s) for chunk in chunks for s in bank_sentences(chunk)]
    for i, (chunk_id, text, _terms, _hard) in enumerate(
            pick_diverse(candidates, lexicon, N_BANK_SENTENCES, rng)):
        items.append({"id": f"bank_{i + 1:02d}", "kind": "sentence", "text": text,
                      "chunk_id": chunk_id})
    used = {item["chunk_id"] for item in items}
    answer_candidates = []
    for chunk in chunks:
        if chunk["id"] in used:
            continue
        text = bank_answer(chunk)
        if text:
            answer_candidates.append((chunk["id"], text))
    for i, (chunk_id, text, _terms, _hard) in enumerate(
            pick_diverse(answer_candidates, lexicon, N_ANSWERS, rng)):
        items.append({"id": f"ans_{i + 1:02d}", "kind": "answer", "text": text,
                      "chunk_id": chunk_id})
    # Fixed reading order: interleave kinds so fatigue is not confounded
    # with item kind, deterministic across rebuilds.
    order = list(range(len(items)))
    rng.shuffle(order)
    items = [items[i] for i in order]
    for item in items:
        item["terms"] = item_terms(lexicon, item["text"])
        item["words"] = len(item["text"].split())
    return items


def main():
    parser = argparse.ArgumentParser(description="Build the STT lexicon and sentence set")
    parser.add_argument("--print", action="store_true", help="print the sentence set")
    args = parser.parse_args()

    chunks = load_chunks()
    entries = build_lexicon(chunks)
    lexicon = Lexicon(entries)
    LEXICON_PATH.write_text(
        json.dumps({"built_from": "curated + rag_ml/rag_ai banks"
                    + (" + data/stt/lexicon_extra.json" if EXTRA_PATH.exists() else ""),
                    "count": len(entries), "terms": entries}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    items = build_sentences(chunks, lexicon)
    with SENTENCES_PATH.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_source = {}
    for entry in entries:
        by_source[entry["source"]] = by_source.get(entry["source"], 0) + 1
    print(f"Lexicon: {len(entries)} terms {by_source} -> {LEXICON_PATH.relative_to(BASE_DIR)}")
    kinds = {}
    words = 0
    covered = set()
    for item in items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
        words += item["words"]
        covered.update(item["terms"])
    print(f"Sentences: {len(items)} items {kinds}, {words} words "
          f"(~{words / 140:.0f} min at 140 wpm), {len(covered)} distinct lexicon terms "
          f"-> {SENTENCES_PATH.relative_to(BASE_DIR)}")
    if args.print:
        for item in items:
            print(f"[{item['id']}] ({item['words']}w, {len(item['terms'])} terms) {item['text']}")


if __name__ == "__main__":
    main()
