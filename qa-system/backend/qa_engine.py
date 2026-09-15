"""
BERT extractive-QA inference.

Loads the fine-tuned model you exported from Colab via `save_pretrained()`
(config.json + model.safetensors + tokenizer.json + tokenizer_config.json),
sitting locally in `backend/bert_qa_model/`. `local_files_only=True` means
this never tries to hit the Hugging Face Hub -- it only ever reads that
folder.

Expected return shape from `answer(question, context)`:
    {
        "answer": str,          # the extracted span
        "score": float,         # confidence, 0..1
        "start": int,           # char offset of the answer *within `context`*
        "end": int,             # char offset of the answer end within `context`
    }
"""
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_qa_model")


class BertQAEngine:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._pipeline = None

    def _lazy_load(self):
        if self._pipeline is not None:
            return
        if not os.path.isdir(self.model_dir):
            raise RuntimeError(
                f"Model folder not found at {self.model_dir}. "
                "Place your save_pretrained() export there "
                "(config.json, model.safetensors, tokenizer.json, tokenizer_config.json)."
            )
        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_dir, local_files_only=True)
        self._pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    def answer(self, question: str, context: str) -> dict:
        self._lazy_load()

        result = self._pipeline(question=question, context=context)
        # result looks like: {"answer": ..., "score": ..., "start": ..., "end": ...}

        return {
            "answer": result["answer"],
            "score": float(result["score"]),
            "start": int(result["start"]),
            "end": int(result["end"]),
        }


engine = BertQAEngine()
