# gsm8k_collect_alllayers.py
import os, re, math, json, argparse, sys
from collections import deque
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList,
)

# ----------------------------
# 文末判定（短い末尾ウィンドウをdecodeして正規表現で判定）
# ----------------------------
SENT_END_RE = re.compile(r"[.!?]+(\s|$)")

def is_sentence_end(tokenizer, input_ids: torch.LongTensor, tail_tokens: int = 1) -> bool:
    tail = input_ids[0, -tail_tokens:]  # B=1を想定。B>1なら行ループにする
    text = tokenizer.decode(tail.tolist(), skip_special_tokens=True)
    return bool(SENT_END_RE.search(text))

# ----------------------------
# フック管理（全レイヤー：層出力HS & MLP出力）
# ----------------------------
class AllLayerHooks:
    def __init__(self, model):
        self.layers = model.model.layers
        self.L = len(self.layers)
        self._ctx_hs  = [None] * self.L
        self._ctx_mlp = [None] * self.L
        self._handles = []

        def make_layer_hook(i):
            def h(module, inp, out):
                self._ctx_hs[i] = (
                    out[0][:, -1, :].detach().cpu().numpy().squeeze(0)
                )
            return h

        def make_mlp_hook(i):
            def h(module, inp, out):
                self._ctx_mlp[i] = (
                    out[:, -1, :].detach().cpu().numpy().squeeze(0)
                )
            return h

        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(make_layer_hook(i)))
            self._handles.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    def stack_current(self, row: int = 0):
        hs  = np.stack([t for t in self._ctx_hs],  axis=0)  # (L,H)
        mlp = np.stack([t for t in self._ctx_mlp], axis=0)  # (L,H)
        return hs.copy(), mlp.copy()

    def remove(self):
        for h in self._handles:
            h.remove()

# ----------------------------
# 停止しない logger（停止するStoppingCriteriaと併用）.
# 文末に当たった「直近生成トークン」を pending → 次ステップのhookで確定保存
# ----------------------------
class LoggingAllLayers(StoppingCriteria):
    def __init__(self, tokenizer, hook_mgr: AllLayerHooks):
        self.tok = tokenizer
        self.hooks = hook_mgr
        self.step = 0
        self.sentences = []        # list of {"step":int,"hs":(L,H),"mlp":(L,H)}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 新しい末尾をdecodeして文末判定
        if is_sentence_end(self.tok, input_ids):
            hs, mlp = self.hooks.stack_current(0)
            self.sentences.append({"step": self.step, "hs": hs, "mlp": mlp})

        self.step += 1
        return False  # 停止は他Criteriaに任せる (Trueだと停止）.

# ----------------------------
# 実際の停止用
# ----------------------------
class SpecificStringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[self.input_len:]
        
        return any(stop_string in current_text for stop_string in self.stop_strings)

# Define a stopping condition for generation
generation_util = [
    "Q:",
    "</s>",
    "<|im_end|>"
]

# ----------------------------
# GSM8Kのプロンプト整形（test split）
# ----------------------------
USER_TEMPLATE = "Q: {q}\nA: Let's think step by step."

def make_prompt(example: Dict[str, Any]) -> str:
    q = example["question"].strip()
    return (USER_TEMPLATE.format(q=q)).strip()

# ----------------------------
# 1サンプル実行：プロンプト末尾(全層)＋文末ごと末尾(全層)を収集
# ----------------------------
@torch.no_grad()
def run_one(model, tok, prompt: str, max_new_tokens: int, pad_id, eos_id) -> Dict[str, Any]:
    device = model.device
    hooks = AllLayerHooks(model)
    logger = LoggingAllLayers(tok, hooks)

    inputs = tok(prompt, return_tensors="pt").to(device)

    stopper = SpecificStringStoppingCriteria(tokenizer=tok, stop_strings=generation_util, input_len=inputs["input_ids"].shape[1])
    stoppers = StoppingCriteriaList([
        logger,
        stopper,
    ])

    out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stoppers,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        return_dict_in_generate=True
    )

    # テキスト出力
    gen_text = tok.decode(out.sequences[0], skip_special_tokens=True)
    print(gen_text)

    # 片付け
    hooks.remove()

    # テンソルまとめ（文末ごとに (L,H) を積む → (S,L,H)）
    sent_hs  = np.stack([rec["hs"]  for rec in logger.sentences], axis=0) if logger.sentences else np.empty((0,))
    sent_mlp = np.stack([rec["mlp"] for rec in logger.sentences], axis=0) if logger.sentences else np.empty((0,))

    return {
        "sent_hs":    sent_hs,                 # (S,L,H)
        "sent_mlp":   sent_mlp,                # (S,L,H)
        "gen_text":   gen_text,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out_path", type=str, default="gsm8k_all.pkl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<PAD>"})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id
    ).to(device).eval()

    ds = load_dataset("gsm8k", "main", split=args.split)

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    all_records = []

    for i in tqdm(range(len(ds)), ncols=80):
        ex = ds[i]
        prompt = make_prompt(ex)
        rec = run_one(
            model, tok, prompt,
            max_new_tokens=args.max_new_tokens,
            pad_id=pad_id, eos_id=eos_id
        )

        meta = {
            "idx": i,
            "question": ex["question"],
            "answer_gt": ex.get("answer", None),
            "prompt": prompt,
            "gen_text": rec["gen_text"],
        }
        all_records.append({
            "meta": meta,
            "sent_hs": rec["sent_hs"],     # (S,L,H)
            "sent_mlp": rec["sent_mlp"],   # (S,L,H)
        })
        print(all_records[0]['sent_hs'].shape)
        print(all_records[0]['sent_hs'][9, 31, :])
        print(all_records[0]['sent_mlp'][9, 1, :])
        sys.exit()

    return all_records

if __name__ == "__main__":
    main()