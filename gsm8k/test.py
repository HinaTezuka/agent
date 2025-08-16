# gsm8k_collect_alllayers.py
import os, re, math, json, argparse
from collections import deque
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList,
    MaxNewTokensCriteria, EosTokenCriteria,
)

# ----------------------------
# 文末判定（短い末尾ウィンドウをdecodeして正規表現で判定）
# ----------------------------
SENT_END_RE = re.compile(r"[.!?]+(\s|$)")

def is_sentence_end(tokenizer, input_ids: torch.LongTensor, tail_tokens: int = 8) -> bool:
    tail = input_ids[0, -tail_tokens:]  # B=1を想定。B>1なら行ループにする
    text = tokenizer.decode(tail.tolist(), skip_special_tokens=True)
    return bool(SENT_END_RE.search(text))

# ----------------------------
# フック管理（全レイヤー：層出力HS & MLP出力）
# ----------------------------
class AllLayerHooks:
    def __init__(self, model, to_dtype=torch.float16):
        # LLaMA/Mistral/GPT-NeoX など decoder-only 系を想定
        self.layers = model.model.layers
        self.L = len(self.layers)
        self._ctx_hs  = [None] * self.L
        self._ctx_mlp = [None] * self.L
        self._handles = []
        self._to_dtype = to_dtype

        def make_layer_hook(i):
            def h(module, inp, out):
                self._ctx_hs[i] = out[:, -1, :].detach().to("cpu").to(self._to_dtype)
            return h

        def make_mlp_hook(i):
            def h(module, inp, out):
                self._ctx_mlp[i] = out[:, -1, :].detach().to("cpu").to(self._to_dtype)
            return h

        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(make_layer_hook(i)))
            # モデルにより名前が異なる場合あり（例: layer.ffn など）。適宜変更してください。
            self._handles.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    def stack_current(self, row: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """現ステップ forward の“末尾トークン”について、全レイヤー(L,H)で積む"""
        hs  = torch.stack([t[row] for t in self._ctx_hs],  dim=0)  # (L,H)
        mlp = torch.stack([t[row] for t in self._ctx_mlp], dim=0)  # (L,H)
        return hs.clone(), mlp.clone()

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
        self.pending = deque()     # 文末を検知した生成トークン（次ステップで確定）
        self.prompt_hs = None      # (L,H)
        self.prompt_mlp = None     # (L,H)
        self.sentences = []        # list of {"step":int,"hs":(L,H),"mlp":(L,H)}

    def on_prompt_forward_done(self):
        hs, mlp = self.hooks.stack_current(0)
        self.prompt_hs, self.prompt_mlp = hs, mlp

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 1) いま追加された新トークンまで含めて文末かを判定
        if is_sentence_end(self.tok, input_ids):
            self.pending.append(self.step)  # 次ステップのhookで確定

        # 2) 1ステップ遅れで pending を確定（今回のhookは前回生成トークンの表現）
        if self.pending:
            s = self.pending.popleft()
            hs, mlp = self.hooks.stack_current(0)
            self.sentences.append({"step": s, "hs": hs, "mlp": mlp})

        self.step += 1
        return False  # 停止は他Criteriaに任せる（併用のため常にFalse）

    def flush_last(self, full_seq_ids: torch.LongTensor, model, device):
        """EOSで終わり次ステップが来ない pending を回収（1回だけ teacher-forcing）"""
        if not self.pending:
            return
        with torch.no_grad():
            _ = model(full_seq_ids.to(device))  # 全レイヤー末尾をhookで再度埋める
        s = self.pending.popleft()
        hs, mlp = self.hooks.stack_current(0)
        self.sentences.append({"step": s, "hs": hs, "mlp": mlp})

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
    hooks = AllLayerHooks(model, to_dtype=torch.float16)
    logger = LoggingAllLayers(tok, hooks)

    inputs = tok(prompt, return_tensors="pt").to(device)

    # まずプロンプトを一括forward → 末尾1トークン（全層）を確定
    _ = model(**inputs, use_cache=True)
    logger.on_prompt_forward_done()

    stoppers = StoppingCriteriaList([
        logger,
        # MaxNewTokensCriteria(max_new_tokens=max_new_tokens),
        EosTokenCriteria(eos_token_id=eos_id),
    ])

    out = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
        stopping_criteria=stoppers,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        return_dict_in_generate=True
        # output_hidden_states=False（hookで収集するため不要）
    )
    logger.flush_last(out.sequences, model=model, device=device)  # 最後の1文（たいていEOS直前）を回収

    # テキスト出力
    gen_text = tok.decode(out.sequences[0], skip_special_tokens=True)

    # 片付け
    hooks.remove()

    # テンソルまとめ（文末ごとに (L,H) を積む → (S,L,H)）
    sent_hs  = torch.stack([rec["hs"]  for rec in logger.sentences], dim=0) if logger.sentences else torch.empty(0)
    sent_mlp = torch.stack([rec["mlp"] for rec in logger.sentences], dim=0) if logger.sentences else torch.empty(0)

    return {
        "prompt_hs":  logger.prompt_hs,        # (L,H)
        "prompt_mlp": logger.prompt_mlp,       # (L,H)
        "sent_hs":    sent_hs,                 # (S,L,H)
        "sent_mlp":   sent_mlp,                # (S,L,H)
        "gen_text":   gen_text,
    }

# ----------------------------
# メイン：GSM8K test split を回してシャード保存
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3-8B-Instruct")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--shard_size", type=int, default=100)  # 100件ごとに1シャード
    ap.add_argument("--out_dir", type=str, default="gsm8k_captures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    # PAD≠EOSを推奨（なければ追加）
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<PAD>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    # データ読み込み（gsm8k: config="main"）
    ds = load_dataset("gsm8k", "main", split=args.split)

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    shard_meta: List[Dict[str, Any]] = []
    shard_id = 0
    buf: List[Dict[str, Any]] = []

    for i in tqdm(range(len(ds)), ncols=80):
        ex = ds[i]
        prompt = make_prompt(ex)
        rec = run_one(
            model, tok, prompt,
            max_new_tokens=args.max_new_tokens,
            pad_id=pad_id, eos_id=eos_id
        )
        # メタ
        meta = {
            "idx": i,
            "question": ex["question"],
            "answer_gt": ex.get("answer", None),
            "prompt": prompt,
            "gen_text": rec["gen_text"],
        }
        buf.append({
            "meta": meta,
            "prompt_hs": rec["prompt_hs"],   # (L,H)
            "prompt_mlp": rec["prompt_mlp"], # (L,H)
            "sent_hs": rec["sent_hs"],       # (S,L,H)
            "sent_mlp": rec["sent_mlp"],     # (S,L,H)
        })

        # シャード保存
        if len(buf) >= args.shard_size or i == len(ds) - 1:
            shard_path_pt = os.path.join(args.out_dir, f"shard_{shard_id:05d}.pt")
            torch.save(
                {
                    "samples": [
                        {
                            "meta": b["meta"],
                            # ベクトルはそのまま保存（f16）
                            "prompt_hs": b["prompt_hs"],
                            "prompt_mlp": b["prompt_mlp"],
                            "sent_hs": b["sent_hs"],
                            "sent_mlp": b["sent_mlp"],
                        }
                        for b in buf
                    ]
                },
                shard_path_pt
            )
            shard_meta.append({"shard_id": shard_id, "file": shard_path_pt, "count": len(buf)})
            buf = []
            shard_id += 1

    # インデックス
    with open(os.path.join(args.out_dir, "index.jsonl"), "w", encoding="utf-8") as f:
        for m in shard_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Done. {len(shard_meta)} shards written under: {args.out_dir}")

if __name__ == "__main__":
    main()