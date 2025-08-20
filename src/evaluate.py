import json, re, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from collections import Counter

# --------------------------
# 1. Metric: Named Entity Recognition (NER)
# --------------------------
def _to_items(y):
    """Normalize prediction/ground truth JSON into list of (key,value) tuples."""
    try:
        obj = y if isinstance(y, (list,dict)) else json.loads(y)
    except Exception:
        return []
    items = []
    if isinstance(obj, dict):
        for k,v in obj.items():
            if isinstance(v, list):
                for vi in v: items.append((k, str(vi).lower()))
            else:
                items.append((k, str(v).lower()))
    elif isinstance(obj, list):
        for d in obj:
            if isinstance(d, dict):
                for k,v in d.items():
                    if isinstance(v, list):
                        for vi in v: items.append((k, str(vi).lower()))
                    else:
                        items.append((k, str(v).lower()))
    return items

def ner_prf1(preds, gts):
    tp=fp=fn=0
    for p,g in zip(preds,gts):
        pset = Counter(_to_items(p))
        gset = Counter(_to_items(g))
        for t in set(pset)|set(gset):
            ctp = min(pset[t], gset[t])
            tp += ctp
            fp += max(pset[t]-ctp, 0)
            fn += max(gset[t]-ctp, 0)
    prec = tp/(tp+fp+1e-12)
    rec  = tp/(tp+fn+1e-12)
    f1   = 2*prec*rec/(prec+rec+1e-12)
    return {"precision":prec, "recall":rec, "f1":f1}

# --------------------------
# 2. Metric: Summarization (ROUGE-1/2/L)
# --------------------------
def rouge_scores(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    agg = {"rouge1":0,"rouge2":0,"rougeL":0}
    for p,r in zip(preds,refs):
        s = scorer.score(r, p)
        for k in agg: agg[k]+=s[k].fmeasure
    n = max(1,len(preds))
    return {k: v/n for k,v in agg.items()}

# --------------------------
# 3. Metric: Medical QA (Exact accuracy + token F1)
# --------------------------
def _tok(s): return re.findall(r"\\w+", str(s).lower())

def qa_accuracy_f1(preds, refs):
    acc = sum(p.strip()==r.strip() for p,r in zip(preds,refs))/max(1,len(preds))
    tp=fp=fn=0
    for p,r in zip(preds,refs):
        P,R = _tok(p), _tok(r)
        Pset, Rset = {}, {}
        for w in P: Pset[w]=Pset.get(w,0)+1
        for w in R: Rset[w]=Rset.get(w,0)+1
        for w in set(Pset)|set(Rset):
            ctp = min(Pset.get(w,0), Rset.get(w,0))
            tp += ctp
            fp += max(Pset.get(w,0)-ctp,0)
            fn += max(Rset.get(w,0)-ctp,0)
    prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    return {"accuracy":acc, "f1":f1}

# --------------------------
# 4. Evaluation runner
# --------------------------
def evaluate(model_dir, test_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.float16
    )
    ds = load_dataset("json", data_files={"test": test_file})["test"]

    preds, refs, tasks = [], [], []
    for ex in ds:
        messages = [
            {"role":"system","content":"You are a clinical NLP assistant."},
            {"role":"user","content":f"{ex['instruction']}\\n\\nINPUT:\\n{ex.get('input','')}"}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.0)
        out_text = tok.decode(out_ids[0], skip_special_tokens=True)

        # cut after assistant
        pred = out_text.split("assistant")[-1].strip()
        preds.append(pred)
        refs.append(str(ex["output"]))
        tasks.append(ex.get("task",""))

    # --- Compute metrics by task ---
    results = {}
    if any(t=="ner" for t in tasks):
        idx = [i for i,t in enumerate(tasks) if t=="ner"]
        results["NER"] = ner_prf1([preds[i] for i in idx], [refs[i] for i in idx])
    if any(t=="summarization" for t in tasks):
        idx = [i for i,t in enumerate(tasks) if t=="summarization"]
        results["Summarization"] = rouge_scores([preds[i] for i in idx], [refs[i] for i in idx])
    if any(t=="med_qa" for t in tasks):
        idx = [i for i,t in enumerate(tasks) if t=="med_qa"]
        results["MedicalQA"] = qa_accuracy_f1([preds[i] for i in idx], [refs[i] for i in idx])

    return results

if __name__=="__main__":
    model_dir = "outputs/run1/adapter"   # trained adapter path
    test_file = "data/test.jsonl"
    scores = evaluate(model_dir, test_file)
    print("Evaluation Results:")
    print(json.dumps(scores, indent=2))
