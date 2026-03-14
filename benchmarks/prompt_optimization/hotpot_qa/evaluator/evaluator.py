from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
import os
import random
from typing import Callable, Dict, Any, List, Type

import dspy
from dspy import Signature
from dspy.evaluate import EM
from dspy.evaluate.metrics import F1


@dataclass
class EvaluationResult:
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)


from litellm import completion

import bm25s
import Stemmer

from datasets import load_dataset

dataset_size = {"full": None, "lite": 500, "tiny": 200, "test": 50}


class Benchmark(ABC):
    def __init__(self, dataset_mode="lite"):
        # dataset for training and validation
        self.dataset = None
        # dataset for the actual benchmarking
        self.train_set = None
        self.test_set = None
        self.val_set = None

        self.init_dataset()
        assert self.dataset is not None, "Dataset not initialized"
        self.max_testset_size = dataset_size[dataset_mode]

        if dataset_mode == "test":
            self.dataset = self.trim_dataset(self.dataset, 60)
            self.create_splits()

        if not self.train_set or not self.test_set or not self.val_set:
            self.create_splits()

        self.train_set = self.trim_dataset(self.train_set, 300)
        self.test_set = self.trim_dataset(self.test_set, 300)
        self.val_set = self.trim_dataset(self.val_set, 300)

        assert self.train_set is not None, "Train set not initialized"
        assert self.test_set is not None, "Dev set not initialized"
        assert self.val_set is not None, "Val set not initialized"

    @abstractmethod
    def init_dataset(self) -> None:
        """
        Initializes the dataset for the benchmark, and sets it to self.dataset.
        Each element in the dataset should be an instance of dspy.Example.
        """
        return

    def trim_dataset(self, dataset, size: int) -> list:
        if size is None or size >= len(dataset):
            return dataset
        rng = random.Random()
        rng.seed(1)
        return rng.sample(dataset, size)

    def create_splits(self) -> None:
        """
        Creates the splits for the dataset (not including test).
        Upon completion, self.train_set, self.test_set, and self.val_set should be set.
        """

        total_len = len(self.dataset)
        self.test_set = self.dataset[: int(0.4 * total_len)]
        self.val_set = self.dataset[int(0.4 * total_len) : int(0.8 * total_len)]
        self.train_set = self.dataset[int(0.8 * total_len) :]

    def get_dataset(self):
        return self.dataset

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set


@dataclass
class BenchmarkMeta:
    benchmark: Type[Benchmark]
    program: List[dspy.Module]
    metric: Callable
    dataset_mode: str = "lite"
    # BenchmarkMeta.num_threads has higher priority than run time argument of num_threads
    # use this as an upper bound for the number of threads to use
    num_threads: int = None
    name: str = None
    metric_with_feedback: Callable = None
    feedback_fn_maps: list[dict] = None

class HotpotQABench(Benchmark):
    def init_dataset(self):
        raw_datasets = load_dataset("hotpot_qa", "fullwiki")
        self.dataset = [
            dspy.Example(**x).with_inputs("question") for x in raw_datasets["train"]
        ]


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )


stemmer = None
retriever = None
corpus = None
initialized = False

from diskcache import Cache

import threading

init_lock = threading.Lock()

# Cache benchmark at module level — same pattern as the BM25 retriever.
# Avoids reloading 90K examples + re-splitting on every evaluate() call.
_benchmark = None
_benchmark_lock = threading.Lock()


def _get_benchmark():
    global _benchmark
    if _benchmark is not None:
        return _benchmark
    with _benchmark_lock:
        if _benchmark is None:
            _benchmark = HotpotQABench()
        return _benchmark


def initialize_bm25s_retriever_and_corpus(directory):
    from dspy.utils import download

    download(
        "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
    )
    # !tar -xzvf wiki.abstracts.2017.tar.gz
    import tarfile

    with tarfile.open("wiki.abstracts.2017.tar.gz", "r:gz") as tar:
        tar.extractall(path=directory)

    import ujson

    corpus = []

    assert os.path.exists(os.path.join(directory, "wiki.abstracts.2017.jsonl")), (
        "Corpus file not found. Please ensure the corpus is downloaded and extracted correctly."
    )

    with open(os.path.join(directory, "wiki.abstracts.2017.jsonl")) as f:
        for line in f:
            line = ujson.loads(line)
            corpus.append(f"{line['title']} | {' '.join(line['text'])}")

    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)

    retriever.save(os.path.join(directory, "bm25s_retriever"))
    assert os.path.exists(os.path.join(directory, "bm25s_retriever")), (
        "Retriever not saved correctly."
    )


def init_retriever():
    global retriever, stemmer, corpus, initialized
    if initialized:
        return
    with init_lock:
        if not initialized:
            if not os.path.exists(
                os.path.join(os.path.dirname(__file__), "bm25s_retriever")
            ) or not os.path.exists(
                os.path.join(os.path.dirname(__file__), "wiki.abstracts.2017.jsonl")
            ):
                initialize_bm25s_retriever_and_corpus(os.path.dirname(__file__))
            retriever = bm25s.BM25.load(
                os.path.join(os.path.dirname(__file__), "bm25s_retriever")
            )
            stemmer = Stemmer.Stemmer("english")
            import ujson

            corpus_data = []
            with open(
                os.path.join(os.path.dirname(__file__), "wiki.abstracts.2017.jsonl")
            ) as f:
                for line in f:
                    line = ujson.loads(line)
                    corpus_data.append(f"{line['title']} | {' '.join(line['text'])}")
            corpus = corpus_data
            initialized = True


# Initialize cache with a dedicated directory
cache = Cache(os.path.join(os.path.dirname(__file__), "retriever_cache"))


@cache.memoize()
def _search_cached(query: str, k: int) -> dict:
    init_retriever()
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return {"passages": list(run.keys())[:k]}


def search(query: str, k: int):
    return DotDict(_search_cached(query, k))


class LangProBeDSPyMetaProgram(dspy.Module):
    def setup_lm(self, lm, api_key=None, api_base=None):
        dspy.settings.experimental = True
        self.lm = dspy.LM(lm, api_key=api_key, api_base=api_base)
        self.set_lm(self.lm)

    def program_type(self):
        return "dspy"


class Predict(dspy.Predict, LangProBeDSPyMetaProgram):
    pass


class CoT(dspy.ChainOfThought, LangProBeDSPyMetaProgram):
    pass


class HotpotSingleHop(LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self, prompt: str):
        super().__init__()
        self.k = 7
        self.retrieve_k = partial(search, k=self.k)  # dspy.Retrieve(k=self.k)
        self.final_answer = dspy.ChainOfThought(
            Signature("question,passages->answer").with_instructions(prompt)
        )

    def forward(self, question):
        # HOP 1
        hop1_docs = self.retrieve_k(question).passages
        final_answer = self.final_answer(question=question, passages=hop1_docs).answer

        return dspy.Prediction(answer=final_answer, hop1_docs=hop1_docs)


def generate_feedback(questions, outputs):
    feedbacks = []
    count = 0
    for question, output in zip(questions, outputs):
        if count >= 10:
            break
        score_obj = output[2]
        # score_obj is a dspy.Prediction with .score/.feedback on success,
        # or a plain float (0.0) on evaluation failure
        if isinstance(score_obj, (int, float)):
            continue
        if score_obj.score:
            continue
        feedbacks.append((question["question"], score_obj.feedback))
    CONDENSE_PROMPT = """**System behavior overview:**
This system aims to answer hotpot QA questions in an accurate manner.
The following has been provided as feedback about the current function of the system. Extract repeating themes and issues and provide a condensed version of the feedback that can be executed on more efficiently by an prompt-optimizing agent.
"""
    if not feedbacks:
        return "All evaluated examples were answered correctly."
    condense_prompt = CONDENSE_PROMPT
    for question, feedback in feedbacks:
        condense_prompt += f'\nFeedback for "{question}":\n' + feedback
    response = completion(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": condense_prompt}],
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    return response.choices[0].message.content


def create_lm(lm_config: dict):
    config = lm_config.copy()
    config["model"] = config.pop("new_model_name", config["model"])

    provider = None
    fixed_config = {
        "max_tokens": 16384,  # overriding the dspy defaults
        "num_retries": 0,
        "provider": provider,
    }
    config = {k: v for k, v in config.items() if k != "name"}
    return dspy.LM(**config, **fixed_config)


lm_for_optimizer = create_lm({
    "model": "openai/gpt-5-mini",
    "temperature": 1,
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
})
adapter = dspy.settings.adapter  # if "qwen" not in lm_name else XMLAdapter()
dspy.configure(lm=lm_for_optimizer, adapter=adapter)


def get_textual_context(d):
    title_to_sentences = {
        title: sentences
        for title, sentences in zip(d["context"]["title"], d["context"]["sentences"])
    }
    text = ""

    useful_titles = set(d["supporting_facts"]["title"])

    for title in useful_titles:
        text += title + " | " + "".join(title_to_sentences[title])

    return text


def answer_match_fn(prediction, answers, frac=1.0):
    """Returns True if the prediction matches any of the answers."""

    if frac >= 1.0:
        return EM(prediction, answers)

    return F1(prediction, answers) >= frac


def answer_exact_match_with_feedback(example, pred, trace=None, frac=1.0):
    ans_match = None
    if isinstance(example.answer, str):
        ans_match = answer_match_fn(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        ans_match = answer_match_fn(pred.answer, example.answer, frac=frac)

    textual_context = ""
    if hasattr(pred, "feedback_text"):
        textual_context = pred.feedback_text + "\n\n"

    textual_context += get_textual_context(example)

    if ans_match:
        return dspy.Prediction(
            score=ans_match,
            feedback=f"The provided answer, '{pred.answer}' is correct. Here's some additional context behind the answer:\n{textual_context}",
        )
    else:
        return dspy.Prediction(
            score=ans_match,
            feedback=f"The provided answer, '{pred.answer}' is incorrect. The correct answer is: {example.answer}. Here's some context behind the answer, and how you could have reasoned to get the correct answer:\n{textual_context}",
        )


def evaluate(prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()

    benchmark = _get_benchmark()
    # Evolve on train_set; val_set and test_set are held out for final reporting.
    eval_set = benchmark.train_set
    program = HotpotSingleHop(prompt)
    evaluate_prog = dspy.Evaluate(
        devset=eval_set,
        metric=answer_exact_match_with_feedback,
        num_threads=8,
        display_progress=True,
        max_errors=len(eval_set) * 10,
        provide_traceback=True,
    )
    output = evaluate_prog(program)
    score = output.score
    outputs = output.results
    print("Generating feedback")
    feedback = generate_feedback(eval_set, outputs)
    return EvaluationResult(
        metrics={
            "combined_score": score / 100,
            "prompt_length": len(prompt),
        },
        artifacts={"feedback": feedback},
    )


if __name__ == "__main__":
    import json
    import sys
    import traceback

    if len(sys.argv) < 2:
        print("Usage: evaluator.py <program_path>", file=sys.stderr)
        sys.exit(1)

    program_path = sys.argv[1]
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = evaluate(program_path)
    except Exception as e:
        sys.stdout = real_stdout
        print(json.dumps({
            "status": "error",
            "combined_score": 0.0,
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": str(e), "traceback": traceback.format_exc()},
        }))
        sys.exit(0)
    sys.stdout = real_stdout

    # result is an EvaluationResult with .metrics and .artifacts
    metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float, bool))}
    artifacts = {k: str(v) for k, v in result.artifacts.items()}
    if "combined_score" not in metrics:
        metrics["combined_score"] = 0.0
    status = "error" if "error" in artifacts else "success"
    output = {"status": status, "combined_score": metrics.get("combined_score", 0.0), "metrics": metrics}
    if artifacts:
        output["artifacts"] = artifacts
    print(json.dumps(output))
