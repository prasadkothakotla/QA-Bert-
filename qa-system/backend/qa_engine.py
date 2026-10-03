```python
"""
BERT extractive-QA inference.

Production:
    Set HF_MODEL_ID to your Hugging Face repository ID.

Local development:
    If HF_MODEL_ID is not set, load from backend/bert_qa_model/.
"""

import os

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bert_qa_model",
)


class BertQAEngine:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model_id = os.getenv("HF_MODEL_ID")
        self.hf_token = os.getenv("HF_TOKEN")
        self._pipeline = None

    def _lazy_load(self):
        if self._pipeline is not None:
            return

        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            pipeline,
        )

        if self.model_id:
            print(f"Loading BERT QA model from Hugging Face: {self.model_id}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token,
                clean_up_tokenization_spaces=True,
            )

            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_id,
                token=self.hf_token,
            )

        else:
            print(f"Loading BERT QA model from local directory: {self.model_dir}")

            if not os.path.isdir(self.model_dir):
                raise RuntimeError(
                    f"Model folder not found at {self.model_dir}. "
                    "Set HF_MODEL_ID to your Hugging Face repository ID, "
                    "or provide a local save_pretrained() model."
                )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )

            model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )

        self._pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
        )

        print("BERT QA model loaded successfully!")

    def answer(self, question: str, context: str) -> dict:
        self._lazy_load()

        result = self._pipeline(
            question=question,
            context=context,
        )

        return {
            "answer": result["answer"],
            "score": float(result["score"]),
            "start": int(result["start"]),
            "end": int(result["end"]),
        }


engine = BertQAEngine()
```
