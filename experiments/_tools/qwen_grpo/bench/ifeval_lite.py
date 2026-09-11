# -*- coding: utf-8 -*-
"""
IFEval-lite：指令遵循评测（自实现轻量版）

- 50 条中文指令，覆盖 IFEval 官方常见的可验证约束类型（关键词/JSON/长度/起止标点/段落数等）
- 每条指令含 1~3 个约束，全部用规则代码自动判分（无需 LLM judge）
- 指标: instruction-level 通过率 = 通过的约束 / 总约束；prompt-level = 全部约束都过的指令占比

用法(实例, base python):
  python ifeval_lite.py --model /path/to/base [--adapter /path/to/lora] \
      --tag grpo_after --out ifeval_results.csv

注意(--no-think): Qwen3 的 chat template 默认开启思考模式，base/GRPO 在 600 token
预算内会一直"想"、答案被截断（实测 GSM8K 首题因此答错）。建议加 --no-think 让三个
模型都在非思考模式下作答，口径一致；不加则保持 Qwen3 官方默认行为。
"""
import argparse
import csv
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- 约束检查器 ----------------

def ck_keywords(text, words, min_count=1):
    low = text.lower()
    return all(low.count(w.lower()) >= min_count for w in words)

def ck_json(text):
    # 尝试提取 JSON(可能被 ``` 包裹)并解析
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.S)
    s = m.group(1) if m else text.strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False

def count_units(text):
    """中英混排的"字数"计数。

    BACKGROUND(必须保留的坑记录): 这里原来用 len(text.split())。中文不用空格分词，
    于是"至少 100 字"的回答（实测 127 个汉字）被 split() 算成 **1 个词**：
      -> min_words 对中文**永远不可能通过**（恒 False）
      -> max_words 对中文**永远通过**（恒 True）
    实测 base/SFT 的 min_words 通过率都是 0%(0/5)、max_words 都是 100%(5/5)，
    纯粹是判分口径造成的，不是模型能力差异。

    现在: 每个 CJK 汉字算 1 个单位，连续的西文字母/数字算 1 个词（与原 split 对英文的行为一致）。
    对纯中文回答，这个数 ≈ 汉字数，与题面里的"N 字"口径对齐。
    """
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*", text))
    return cjk + latin

def ck_max_words(text, n):
    return count_units(text) <= n

def ck_min_words(text, n):
    return count_units(text) >= n

def ck_paragraphs_min(text, n):
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paras) >= n

# 结尾允许的尾部修饰: 空白/换行、中英句末标点、引号、markdown 强调符
_TRAIL = r"[\s。．.!！?？;；:：,，…\"'“”‘’*`_）)】\]】]+$"

def ck_start_with(text, chars):
    return text.strip().startswith(chars)

def ck_end_with(text, chars):
    """以指定短语结尾。

    BACKGROUND: 这里原来是 text.strip().endswith(chars)，过严 —— 实测模型输出
    "……此致敬礼。" 会被判 False，因为结尾多了个句号；"坚持就是胜利。" 同理。
    3 道题（#14/#24/#48）因此"三个模型全挂"，其中 5/6 是误判（不是模型没照做）。
    判分改为: 先严格比对，不通过再剥掉尾部标点/空白/引号/markdown 符后比对 ——
    这样"以该短语结尾"考的是内容，不再因为多一个句号而全盘否定。
    """
    t = text.strip()
    if t.endswith(chars):
        return True
    return re.sub(_TRAIL, "", t).endswith(chars)

def ck_no_commas(text):
    return "," not in text and "，" not in text

def ck_exact_word(text, word):
    # 中文无 \b 词边界, 用子串判断(大小写不敏感, 兼容中英文)
    return word.lower() in text.lower()

def ck_sentences_min(text, n):
    return len(re.findall(r"[。!?！？.…]", text)) >= n

CHECKERS = {
    "keywords": ck_keywords, "json": ck_json, "max_words": ck_max_words,
    "min_words": ck_min_words, "paragraphs_min": ck_paragraphs_min,
    "start_with": ck_start_with, "end_with": ck_end_with,
    "no_commas": ck_no_commas, "exact_word": ck_exact_word,
    "sentences_min": ck_sentences_min,
}

# ---------------- 评测指令集（IFEval 风格，中文） ----------------

PROMPTS = [
    ("请写一段介绍机器学习的文字，并确保包含至少两个以数字编号的要点。", [("paragraphs_min", 1)]),
    ("请用JSON格式回复你的名字与功能，例如 {\"name\": ..., \"abilities\": [...]}。", [("json", 0)]),
    ("写一段 200 字以内的自我介绍。", [("max_words", 200)]),
    ("写一段关于环境保护的建议，至少 100 字。", [("min_words", 100)]),
    ("请列举三种编程语言，用顿号分隔。", [("exact_word", "Python"), ("exact_word", "Java")]),
    ("写一篇不少于三个自然段的短文，主题是人工智能的利弊。", [("paragraphs_min", 3)]),
    ("用一句话介绍量子计算，且必须以句号结尾。", [("end_with", "。")]),
    ("回答：什么是大语言模型？请以'大语言模型'开头。", [("start_with", "大语言模型")]),
    ("请输出一段完全不用逗号的描述性文字，介绍你的家乡。", [("no_commas", 0)]),
    ("请用至少五句话介绍什么是强化学习。", [("sentences_min", 5)]),
    ("请写一段关于健康的文字，其中必须包含'运动'和'睡眠'两个词。", [("keywords", ["运动", "睡眠"])]),
    ("请以JSON格式输出今天需要完成的三件事。", [("json", 0)]),
    ("写一段关于阅读的好处的文字，不少于150字。", [("min_words", 150)]),
    ("请用一个不超过50词的段落解释什么是区块链。", [("max_words", 50)]),
    ("请写一封信的结尾，最后一句必须是'此致敬礼'。", [("end_with", "此致敬礼")]),
    ("请用三个自然段介绍你的学习计划。", [("paragraphs_min", 3)]),
    ("请回答'什么是注意力机制'，开头必须写'注意力机制'四个字。", [("start_with", "注意力机制")]),
    ("写一段关于时间管理的文字，其中提到'优先级'这个词。", [("exact_word", "优先级")]),
    ("请用不少于八句话描述一次旅行。", [("sentences_min", 8)]),
    ("请用JSON格式给出本周书单。", [("json", 0)]),
    ("写一段文字介绍宠物饲养，必须同时包含'责任'与'陪伴'。", [("keywords", ["责任", "陪伴"])]),
    ("请用不超过120字总结这篇文章的要点。（直接输出总结）", [("max_words", 120)]),
    ("请分三段解释什么是碳中和。", [("paragraphs_min", 3)]),
    ("以'今天天气'开头写一段话。", [("start_with", "今天天气")]),
    ("写一段关于坚持的文字，最后以'坚持就是胜利'结尾。", [("end_with", "坚持就是胜利")]),
    ("请列举三种常见的水果，不要用逗号。", [("no_commas", 0)]),
    ("用至少六句话介绍你自己。", [("sentences_min", 6)]),
    ("请写一段环保倡议书，其中必须提到'垃圾分类'。", [("exact_word", "垃圾分类")]),
    ("用JSON格式输出你的一天安排。", [("json", 0)]),
    ("写一段 150 字左右的文字描述秋天的景色。", [("max_words", 250), ("min_words", 100)]),
    ("请以'失败是成功之母'开头，写一段感悟。", [("start_with", "失败是成功之母")]),
    ("写一段关于团队合作的文字，至少包含'沟通'与'信任'两个词。", [("keywords", ["沟通", "信任"])]),
    ("请用三个自然段介绍你最喜欢的季节。", [("paragraphs_min", 3)]),
    ("用一句话给出建议，必须以感叹号结尾。", [("end_with", "！")]),
    ("写一段介绍家乡的文字，不少于200字。", [("min_words", 200)]),
    ("请输出一段不含任何标点的文字描述一个场景。", [("no_commas", 0)]),
    ("请用不少于十句话介绍什么是数据科学。", [("sentences_min", 10)]),
    ("请写一段关于友谊的文字，最后一句以句号结尾。", [("end_with", "。")]),
    ("用JSON格式给出三个学习建议。", [("json", 0)]),
    ("写一段关于音乐的介绍，必须提到'旋律'这个词。", [("exact_word", "旋律")]),
    ("请以'在人生的旅途中'开头写一段话。", [("start_with", "在人生的旅途中")]),
    ("写一篇至少四个自然段的文章，主题是科技改变生活。", [("paragraphs_min", 4)]),
    ("请用不超过30个词回答：你今天开心吗？", [("max_words", 30)]),
    ("写一段关于安全的文字，同时包含'预防'和'检查'。", [("keywords", ["预防", "检查"])]),
    ("请用JSON格式介绍你自己。", [("json", 0)]),
    ("写一段 300 字左右的短文介绍一座你喜欢的城市。", [("min_words", 200)]),
    ("请用至少七句话说明为什么要早起。", [("sentences_min", 7)]),
    ("以'你好'开头，写一段欢迎词。", [("start_with", "你好")]),
    ("写一段关于读书的文字，以'书中自有黄金屋'结尾。", [("end_with", "书中自有黄金屋")]),
    ("请列举至少三种编程语言，不要使用逗号。", [("no_commas", 0), ("exact_word", "Python")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default="ifeval_results.csv")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--no-think", action="store_true",
                    help="给 chat template 传 enable_thinking=False（Qwen3 专用；其它模型自动忽略）")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("gpu:", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev, trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("adapter loaded:", args.adapter)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    total_ck, pass_ck = 0, 0
    total_prompt_ok = 0
    t0 = time.time()
    detail = []
    for i, (prompt, checks) in enumerate(PROMPTS):
        inp = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      tokenize=False, add_generation_prompt=True,
                                      **({"enable_thinking": False} if args.no_think else {}))
        ids = tok(inp, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(input_ids=ids.input_ids, attention_mask=ids.attention_mask,
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        out_text = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        res = []
        for kind, param in checks:
            fn = CHECKERS[kind]
            if kind in ("json", "no_commas"):
                ok = fn(out_text)
            elif kind in ("keywords", "exact_word", "start_with", "end_with"):
                ok = fn(out_text, param) if kind == "keywords" else fn(out_text, param)
            else:
                ok = fn(out_text, param)
            total_ck += 1
            pass_ck += int(ok)
            res.append(ok)
        prompt_ok = all(res)
        total_prompt_ok += int(prompt_ok)
        detail.append({"i": i, "prompt": prompt[:40], "checks_ok": res, "answer": out_text[:120]})
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(PROMPTS)}] instr_acc={pass_ck/total_ck:.3f} ({time.time()-t0:.0f}s)")

    n_instr = total_ck
    instr_acc = pass_ck / n_instr
    prompt_acc = total_prompt_ok / len(PROMPTS)
    print(f"\n===== RESULT {args.tag} =====")
    print(f"instruction-level: {instr_acc*100:.1f}% ({pass_ck}/{n_instr})")
    print(f"prompt-level      : {prompt_acc*100:.1f}% ({total_prompt_ok}/{len(PROMPTS)})")
    new = not os.path.exists(args.out)
    with open(args.out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "instruction_acc", "prompt_acc", "prompts", "constraints"])
        w.writerow([args.tag, f"{instr_acc:.4f}", f"{prompt_acc:.4f}", len(PROMPTS), n_instr])
    with open(args.out.replace(".csv", f"_{args.tag}.jsonl"), "w", encoding="utf-8") as f:
        for d in detail:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
