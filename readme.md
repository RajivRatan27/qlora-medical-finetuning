# 🩺 LLaMA-3 QLoRA Fine-Tuning for Multi-Task Medical NLP

This repository provides code and examples for fine-tuning the meta-llama/Meta-Llama-3-8B-Instruct model with QLoRA on a multi-task medical dataset. The covered tasks include:

- Named Entity Recognition (NER) for drugs, dosages, and conditions
- Medical Chatbot / Dialogue
- Clinical Text Summarization
- Salt Composition Extraction
- Medical Question Answering (QA)



## 📂 Project Structure

```
llama3-medical-qlora/
├─ README.md                # This file
├─ requirements.txt         # Python dependencies
├─ .gitignore
├─ data/                    # JSONL datasets (not committed to git)
│   ├─ train.jsonl
│   ├─ val.jsonl
│   └─ test.jsonl
├─ src/
│   ├─ collator.py          # Formats instruction prompts into the LLaMA-3 chat template
│   ├─ train_qlora.py       # Main training script
│   └─ evaluate.py          # Evaluation script (NER, ROUGE, QA)
└─ notebooks/
    └─ finetune_llama3_qlora.ipynb   # Colab/Jupyter notebook for a full fine-tuning pipeline
```

## 📊 Dataset Format

Each dataset split (train.jsonl, val.jsonl, test.jsonl) must be a JSON Lines file, where each line is a valid JSON object.

Example Line:
```json
{
  "task": "ner",
  "instruction": "Identify drugs, dosages, and conditions from the clinical note.",
  "input": "Patient was prescribed 500mg Paracetamol for fever.",
  "output": "[{\"drug\":\"Paracetamol\",\"dosage\":\"500mg\",\"condition\":[\"fever\"]}]"
}
```

- Required Keys: instruction, input, output
- Optional Key: task (This is highly recommended for evaluation, as it routes the example to the correct metric calculation).
-The data folder contains only the sample data.
We used a standard 80/10/10 split for the train, validation, and test sets.

## 🚀 Training

You can train the model using either the standalone Python script or the Jupyter notebook.

### Training (Script)

To run a training session, use src/train_qlora.py with your desired arguments.

Example Training Command:
```bash
python src/train_qlora.py \
  --model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --train_file data/train.jsonl \
  --eval_file data/val.jsonl \
  --out_dir outputs/run1 \
  --lora_r 16 \
  --num_train_epochs 10
```

Key Hyperparameters (from paper):
- Batch Size: 4
- Gradient Accumulation: 8 (for an effective batch size of 32)
- Epochs: 10
- Learning Rate: 2e-5 with AdamW optimizer, or an optional two-stage LR scheme (5e-6 → 1e-4)
- LoRA Rank (r): 16 (used in main experiments), or 64 for the Algorithm 1 variant
- Target Modules: q_proj, v_proj

### Training (Notebook)

For an interactive, step-by-step guide, open and run the cells in:
`notebooks/finetune_llama3_qlora.ipynb`

This notebook provides a complete 12-step pipeline suitable for Google Colab or any Jupyter environment.

## 📈 Evaluation

After training is complete and you have saved the adapters, run the evaluation script:

```bash
python src/evaluate.py \
  --model_path meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter_path outputs/run1/final_adapter \
  --test_file data/test.jsonl
```

Metrics Computed:
- NER: Precision, Recall, and F1-score
- Summarization: ROUGE-1, ROUGE-2, and ROUGE-L
- Medical QA: Exact match accuracy and token-level F1-score
-Evaluated the Speed , Memory usage, Peak Vram , Training Throughput ,Training time 
