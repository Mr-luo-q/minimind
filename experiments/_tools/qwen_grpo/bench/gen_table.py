#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 bench/results/ 下的原始结果（CSV + 每题 jsonl 明细）汇总成一个结果表。

在【实例】上执行（因为它要读 gsm8k_results_<tag>.jsonl 等明细文件）：
  python gen_table.py --dir /root/autodl-tmp/qwen_grpo/bench/results

产出（都在同一个 --dir 里）：
  RESULTS.md     Markdown 对比表（贴报告用）
  index.html     单文件网页版（浏览器直接打开，便于查找/查阅明细）
  table.csv      纯表格数据

设计: 幂等、可反复执行——跑批中途执行会如实反映"部分完成"，
      跑完再执行一次即可得到完整表。
"""
import argparse
import csv
import html
import json
import os
import sys
import time

MODELS = [("qwen3-4b_base", "base（Qwen3-4B 底座）"),
          ("qwen3-4b_sft", "SFT（e1 LoRA, 1 epoch）"),
          ("qwen3-4b_grpo", "GRPO（通用 RM, 250 步）")]


def read_csv_map(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["tag"]: r for r in csv.DictReader(f)}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_manifest(d):
    p = os.path.join(d, "manifest.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def collect(d):
    ifile = read_csv_map(os.path.join(d, "ifeval_results.csv"))
    gfile = read_csv_map(os.path.join(d, "gsm8k_results.csv"))
    out = {}
    for tag, label in MODELS:
        i, g = ifile.get(tag), gfile.get(tag)
        detail_i = read_jsonl(os.path.join(d, f"ifeval_results_{tag}.jsonl"))
        detail_g = read_jsonl(os.path.join(d, f"gsm8k_results_{tag}.jsonl"))
        out[tag] = {
            "label": label, "tag": tag,
            "instr": float(i["instruction_acc"]) if i else None,
            "prompt": float(i["prompt_acc"]) if i else None,
            "n_ck": int(i["constraints"]) if i else None,
            "n_p": int(i["prompts"]) if i else None,
            "gsm": float(g["accuracy"]) if g else None,
            "gsm_ok": int(g["correct"]) if g else None,
            "gsm_n": int(g["total"]) if g else None,
            "detail_i": detail_i, "detail_g": detail_g,
            "ifeval_done": i is not None,
            "gsm_done": g is not None,
        }
    return out


def fmt_pct(v, n=None, d=None):
    if v is None:
        return "—"
    s = f"{v*100:.1f}%"
    if n is not None and d is not None:
        s += f" ({n}/{d})"
    return s


def write_md(d, res, stamp, manifest=None):
    L = ["# base vs SFT vs GRPO 评测结果", "",
         f"> 生成时间：{stamp} ｜ 评测口径：IFEval-lite 50 条中文指令（规则自动判分）+ GSM8K test 前 100 题，"
         f"贪心解码，**统一关闭思考模式**（`--no-think`）。", ""]
    if manifest:
        L += [f"> 本次口径（来自 `manifest.json`）：长度判分 `{manifest.get('ifeval_checker','?')}`；"
              f"chat_template={manifest.get('chat_template','?')}；no_think={manifest.get('no_think','?')}；"
              f"创建于 {manifest.get('created','?')}"
              + (f"；备注：{manifest['note']}" if manifest.get("note") else ""), ""]
    else:
        L += ["> ⚠️ 未找到 `manifest.json`：无法确认这批分数用的哪版判分口径。"
              "**v1（split 计长，对中文失效）与修正版的结果不可直接比较**。", ""]
    L += ["> **说明**：IFEval-lite 为自实现轻量版，分数用于**模型间横向对比**，不代表官方 IFEval 榜单成绩。", "",
          "| 模型 | IFEval instruction-level | IFEval prompt-level | GSM8K | 状态 |",
          "|---|---|---|---|---|"]
    for tag, label in MODELS:
        r = res[tag]
        done = r["ifeval_done"] and r["gsm_done"]
        state = "✅ 完成" if done else ("⏳ 部分完成" if (r["ifeval_done"] or r["gsm_done"]) else "⬜ 未跑")
        ic = f"{r['instr']*100:.1f}% ({r['n_ck']} 约束)" if r["instr"] is not None else "—"
        pc = f"{r['prompt']*100:.1f}% ({r['n_p']} 题)" if r["prompt"] is not None else "—"
        gc = fmt_pct(r["gsm"], r["gsm_ok"], r["gsm_n"])
        L.append(f"| {label} | {ic} | {pc} | {gc} | {state} |")
    L += ["", "## 原始数据", "",
          "| 文件 | 说明 |", "|---|---|",
          "| `ifeval_results.csv` / `gsm8k_results.csv` | 汇总分数（追加写） |",
          "| `ifeval_results_<tag>.jsonl` / `gsm8k_results_<tag>.jsonl` | 每题明细（含模型回答片段 / ref vs pred） |",
          "| `ANALYSIS.md` | 约束类型分析（哪类约束最容易被违反） |",
          "| `run_main.log` | 完整运行日志 |", "| `index.html` | 网页版结果表（含每题明细） |", ""]
    md = "\n".join(L) + "\n"
    with open(os.path.join(d, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write(md)
    return md


def write_html(d, res, stamp, manifest=None):
    def esc(x):
        return html.escape(str(x))

    css = """<style>
body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:28px;background:#f6f7f9;color:#1a1d23}
h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:30px 0 10px;padding-bottom:6px;border-bottom:2px solid #e3e6ea}
.meta{color:#6b7280;font-size:13px;margin-bottom:20px}
.note{background:#fff8e1;border-left:4px solid #f0b429;padding:10px 14px;font-size:13px;border-radius:4px;margin:14px 0}
table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden;font-size:14px}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #eef0f3}
th{background:#2d3748;color:#fff;font-weight:600;font-size:13px}
tr:last-child td{border-bottom:none}tr:hover td{background:#f9fafb}
.num{font-variant-numeric:tabular-nums;font-weight:600;font-size:15px}
.ok{color:#0f7b3f}.bad{color:#c0392b}.pend{color:#9aa0a6}
.tag{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;color:#4b5563}
details{background:#fff;border-radius:6px;margin:8px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
summary{cursor:pointer;padding:11px 14px;font-weight:600;font-size:14px}
details[open] summary{border-bottom:1px solid #eef0f3}
.det{padding:10px 14px;font-size:13px}
.q{padding:7px 0;border-bottom:1px dashed #eee}.q:last-child{border-bottom:none}
.qid{color:#9aa0a6;font-size:12px;font-family:ui-monospace,monospace}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#4b5563;word-break:break-all}
</style>"""
    H = [f"<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>base vs SFT vs GRPO 评测结果</title>", css, "</head><body>",
         "<h1>base vs SFT vs GRPO 评测结果</h1>",
         f"<div class='meta'>生成时间 {esc(stamp)} ｜ IFEval-lite 50 条中文指令 + GSM8K test 前 100 题 ｜ 贪心解码 ｜ 统一关闭思考模式(<span class='mono'>--no-think</span>)</div>",
         "<div class='note'><b>口径诚实说明</b>：IFEval-lite 是自实现的轻量版（规则自动判分），"
         "分数只用于 <b>base / SFT / GRPO 之间横向对比</b>，不代表官方 IFEval 榜单成绩。</div>"]
    # 口径来源（manifest.json）——把"这批分数是什么口径"写在结果旁边，避免日后混用 v1/v2
    if manifest:
        H.append("<div class='note' style='background:#e8f4fd;border-left-color:#2b7fd4'>"
                 f"<b>本次口径</b>（来自 <span class='mono'>manifest.json</span>）：长度判分 "
                 f"<span class='mono'>{esc(manifest.get('ifeval_checker','?'))}</span> ｜ "
                 f"chat_template={esc(manifest.get('chat_template','?'))} ｜ "
                 f"no_think={esc(manifest.get('no_think','?'))} ｜ 创建于 {esc(manifest.get('created','?'))}"
                 + (f"<br><b>备注</b>：{esc(manifest['note'])}" if manifest.get("note") else "")
                 + "</div>")
    else:
        H.append("<div class='note'><b>⚠️ 未找到 manifest.json</b>：无法确认这批分数用的哪版判分口径。"
                 "v1（<span class='mono'>split()</span> 计长，对中文恒真/恒假）与修正版的结果"
                 "<b>不可直接比较</b>。</div>")
    H += ["<h2>总分对比</h2>",
          "<table><thead><tr><th>模型</th><th>IFEval instruction-level</th><th>IFEval prompt-level</th>"
          "<th>GSM8K</th><th>状态</th></tr></thead><tbody>"]

    def cell(v, extra=""):
        if v is None:
            return "<td class='num pend'>—</td>"
        return f"<td class='num'>{v*100:.1f}%{extra}</td>"

    for tag, label in MODELS:
        r = res[tag]
        done = r["ifeval_done"] and r["gsm_done"]
        state = "<span class='ok'>✅ 完成</span>" if done else (
            "<span class='pend'>⏳ 部分完成</span>" if (r["ifeval_done"] or r["gsm_done"]) else "<span class='pend'>⬜ 未跑</span>")
        H.append(f"<tr><td><b>{esc(label)}</b><br><span class='tag'>{esc(tag)}</span></td>"
                 + cell(r["instr"], f" <span class='tag'>({r['n_ck']}约束)</span>" if r["n_ck"] else "")
                 + cell(r["prompt"], f" <span class='tag'>({r['n_p']}题)</span>" if r["n_p"] else "")
                 + cell(r["gsm"], f" <span class='tag'>({r['gsm_ok']}/{r['gsm_n']})</span>" if r["gsm_n"] else "")
                 + f"<td>{state}</td></tr>")
    H.append("</tbody></table>")

    # ---- IFEval 明细 ----
    H.append("<h2>IFEval-lite 每题明细</h2>")
    for tag, label in MODELS:
        r = res[tag]
        det = r["detail_i"]
        if not det:
            H.append(f"<details><summary>{esc(label)} — 尚无数据</summary></details>")
            continue
        nfail = sum(1 for x in det if not all(x.get("checks_ok", [])))
        H.append(f"<details><summary>{esc(label)} — {len(det)} 题，{len(det)-nfail} 题全约束通过，{nfail} 题有约束未过</summary><div class='det'>")
        for x in det:
            oks = x.get("checks_ok", [])
            badge = "<span class='ok'>全过</span>" if all(oks) else f"<span class='bad'>未过 {oks.count(False)}/{len(oks)}</span>"
            H.append(f"<div class='q'><span class='qid'>#{x.get('i')}</span> {badge} "
                     f"<div>{esc(x.get('prompt',''))}</div>"
                     f"<div class='mono'>{esc(x.get('answer','')[:110])}</div></div>")
        H.append("</div></details>")

    # ---- GSM8K 明细 ----
    H.append("<h2>GSM8K 错题明细</h2>")
    H.append("<div class='meta'>只列错题——对题数量太大（每模型 100 条），需要看对题请查 jsonl 明细文件。</div>")
    for tag, label in MODELS:
        r = res[tag]
        det = r["detail_g"]
        if not det:
            H.append(f"<details><summary>{esc(label)} — 尚无数据</summary></details>")
            continue
        bad = [x for x in det if not x.get("ok")]
        H.append(f"<details><summary>{esc(label)} — {len(det)} 题中 {len(bad)} 题答错</summary><div class='det'>")
        for x in bad:
            H.append(f"<div class='q'><span class='qid'>#{x.get('i')}</span> "
                     f"<span class='bad'>ref={esc(x.get('ref'))} pred={esc(x.get('pred'))}</span>"
                     f"<div>{esc(x.get('question',''))}</div></div>")
        H.append("</div></details>")

    H.append("</body></html>")
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(H))


def write_csv(d, res):
    with open(os.path.join(d, "table.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "模型", "ifeval_instruction_acc", "ifeval_prompt_acc",
                    "ifeval_constraints", "gsm8k_acc", "gsm8k_correct", "gsm8k_total", "状态"])
        for tag, label in MODELS:
            r = res[tag]
            done = r["ifeval_done"] and r["gsm_done"]
            w.writerow([tag, label,
                        "" if r["instr"] is None else f"{r['instr']:.4f}",
                        "" if r["prompt"] is None else f"{r['prompt']:.4f}",
                        r["n_ck"] or "", "" if r["gsm"] is None else f"{r['gsm']:.4f}",
                        r["gsm_ok"] or "", r["gsm_n"] or "",
                        "完成" if done else ("部分完成" if (r["ifeval_done"] or r["gsm_done"]) else "未跑")])


def safe_print(s):
    """Windows 控制台默认 GBK，直接 print 表里的 ✅/⏳ 会 UnicodeEncodeError 把脚本打挂。
    文件已经按 UTF-8 写好了，这里只做"打印失败也不要崩"的兜底。"""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = (sys.stdout.encoding or "utf-8")
        print(s.encode(enc, errors="replace").decode(enc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="结果目录（含 *_results.csv 与 jsonl 明细）")
    args = ap.parse_args()
    d = args.dir
    res = collect(d)
    manifest = read_manifest(d)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    md = write_md(d, res, stamp, manifest)
    write_html(d, res, stamp, manifest)
    write_csv(d, res)
    safe_print(md)
    n_done = sum(1 for t, _ in MODELS if res[t]["ifeval_done"] and res[t]["gsm_done"])
    safe_print(f"[gen_table] {n_done}/{len(MODELS)} 个模型完整  ->  {d}/RESULTS.md, index.html, table.csv")


if __name__ == "__main__":
    main()
