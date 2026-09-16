"""
BERT extractive-QA inference.

Supports two model sources:

1. Local development:
   backend/bert_qa_model/

2. Deployment:
   Hugging Face Hub using the HF_MODEL_ID environment variable.

Example:
    HF_MODEL_ID=your-username/your-bert-qa-model

The model must have been exported using save_pretrained().
"""

import os

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bert_qa_model"
)


class BertQAEngine:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model_id = os.getenv("HF_MODEL_ID")
        self._pipeline = None

    def _lazy_load(self):
        if self._pipeline is not None:
            return

        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            pipeline,
        )

        # ============================================================
        # OPTION 1: Hugging Face model
        # ============================================================

        if self.model_id:
            print(f"Loading BERT QA model from Hugging Face: {self.model_id}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id
            )

            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_id
            )

        # ============================================================
        # OPTION 2: Local model
        # ============================================================

        else:
            print(f"Loading BERT QA model from local directory: {self.model_dir}")

            if not os.path.isdir(self.model_dir):
                raise RuntimeError(
                    f"Model folder not found at {self.model_dir}. "
                    "Place your save_pretrained() export there "
                    "(config.json, model.safetensors, tokenizer.json, "
                    "tokenizer_config.json)."
                )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True
            )

            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_dir,
                local_files_only=True
            )

        # ============================================================
        # Create QA pipeline
        # ============================================================

        self._pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer
        )

        print("BERT QA model loaded successfully!")

    def answer(self, question: str, context: str) -> dict:
        self._lazy_load()

        result = self._pipeline(
            question=question,
            context=context
        )

        return {
            "answer": result["answer"],
            "score": float(result["score"]),
            "start": int(result["start"]),
            "end": int(result["end"]),
        }


engine = BertQAEngine()