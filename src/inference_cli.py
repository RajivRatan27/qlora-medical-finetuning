import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

save_dir = "./outputs/adapter"   # adjust to your trained adapter path

tok = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(
    save_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

gen = pipeline("text-generation", model=model, tokenizer=tok)

while True:
    user_input = input("Enter clinical note (or 'exit'): ")
    if user_input.lower()=="exit": break
    prompt = f"Instruction: Analyze the following clinical note.\nInput: {user_input}\nAnswer:"
    out = gen(prompt, max_length=512, temperature=0.7, top_p=0.9)[0]["generated_text"]
    print("\nModel Output:", out.split("Answer:")[-1].strip())
    print("="*60)
