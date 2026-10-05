"""The LLM-as-judge role from the Apple paper, run locally instead of via a
paid API: given a query and a document, output a relevance grade on the
same 0/1/2 scale NFCorpus's human annotators used. A small open-weight
instruct model (not a fine-tuned 3B model trained on millions of real
labels, which is what the paper actually does) is the honest stand-in for
what's reproducible without their data or budget.
"""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT_TEMPLATE = """You are a search relevance judge for a medical \
information search engine. Users ask health questions; documents are \
PubMed-style abstracts that may or may not answer them.

Rate relevance on this scale:
0 = the document does not address the query topic at all
1 = the document is on the same general topic but does not directly \
answer the query
2 = the document directly addresses or answers the query

Example:
Query: Is coffee good for your heart?
Document: A randomized trial found daily coffee consumption was associated \
with improved arterial flow-mediated dilation in healthy adults.
Relevance: 2

Now rate this one. Think for one short sentence, then give your answer as \
"Relevance: X".

Query: {query}

Document: {document}"""

# A bare "respond with only a digit" prompt (no worked example, 5-token
# budget) was tried first and was badly miscalibrated: it answered 0 on 6
# of 8 documents humans rated as highly relevant. Letting the model reason
# for one sentence before answering, with one worked example, fixed most
# of that -- the failure was the prompt, not the model's capability.
RELEVANCE_PATTERN = re.compile(r"relevance:\s*([0-2])", re.IGNORECASE)


class LocalLLMJudge:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)
        self.model.eval()

    def judge(self, query: str, document: str, max_doc_chars: int = 2000) -> int:
        prompt = PROMPT_TEMPLATE.format(query=query, document=document[:max_doc_chars])
        messages = [{"role": "user", "content": prompt}]
        # apply_chat_template with return_dict=True returns a BatchEncoding
        # (input_ids + attention_mask), not a bare tensor -- passing it
        # positionally to generate() fails on newer transformers versions.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=60, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        prompt_len = inputs["input_ids"].shape[1]
        reply = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        return self._parse_grade(reply)

    @staticmethod
    def _parse_grade(reply: str) -> int:
        # Prefer the labeled "Relevance: X" line; a bare digit elsewhere in
        # the reasoning sentence (e.g. "type 2 diabetes") would otherwise
        # risk being picked up by a looser digit search.
        match = RELEVANCE_PATTERN.search(reply)
        if match is not None:
            return int(match.group(1))
        fallback = re.search(r"[0-2]", reply)
        if fallback is None:
            return 0  # unparseable response treated as "not relevant", not dropped silently
        return int(fallback.group())
