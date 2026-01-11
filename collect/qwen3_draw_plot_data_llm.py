# stage1_generate_transformers_think_budget_resume.py
from __init__ import *
import os, json, argparse, torch
from tqdm import tqdm
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_category_data(category: str, split: str = "test", max_samples: int = -1):
    results = []
    if category == "HuggingFaceH4/MATH-500":
        ds = load_dataset(category, split=split)
        for ex in ds:
            results.append({"problem": ex["problem"], "answer": ex["answer"], "subject": "math500"})
    elif category == "AI-MO/aimo-validation-aime":
        ds = load_dataset(category, split='train')
        for ex in ds:
            results.append({"problem": ex["problem"], "answer": ex["answer"], "subject": "aime22-24"})
    elif category == 'Idavidrein/gpqa':
        ds = load_dataset(category,'gpqa_diamond', split='train')
        instruction_following = 'Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.'
        for ex in ds:
            multiple_choice_string, correct_answer_letter = get_multiple_choice_answers(ex)
            question = instruction_following + " " +  ex["Question"] + " " + multiple_choice_string
            # question = ex["Question"] + " " + multiple_choice_string
            results.append({"problem": question, "answer": correct_answer_letter, "subject": "gpqa"})
    if max_samples > 0:
        results = results[:max_samples]
    return results

# def get_category_data(category: str, split: str = "test", max_samples: int = -1):
#     results = []
#     if category == "HuggingFaceH4/MATH-500":
#         ds = load_dataset(category, split=split)
#         for ex in ds:
#             results.append({
#                 "problem": ex["problem"],
#                 "answer": ex["answer"],
#                 "subject": "math500"
#             })
#     if max_samples > 0:
#         results = results[:max_samples]
#     return results


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_tokens, temperature, top_p, n=1):
    """使用 transformers 生成"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_tokens,
        num_return_sequences=n,
    )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--think_budget", type=int, default=512)
    parser.add_argument("--answer_budget", type=int, default=512)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--save_path", type=str, default="/expert/think_budget_collect")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(
        args.save_path,
        f"{args.model.split('/')[-1]}_{args.dataset.split('/')[-1]}_budget{args.think_budget}.jsonl"
    )

    # ========================
    # 检查已有文件
    # ========================
    existing = []
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            for line in f:
                try:
                    existing.append(json.loads(line))
                except:
                    continue
    done_count = len(existing)
    finished_samples = done_count // args.n
    print(f"🧭 Found {done_count} existing records ({finished_samples} samples done).")

    rows = get_category_data(args.dataset, max_samples=args.max_samples)
    if finished_samples >= len(rows):
        print("🎉 All samples already processed.")
        return
    print(f"Will resume from sample index {finished_samples}/{len(rows)}")

    print(f"🚀 Loading {args.model} with transformers ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager", 
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    model = torch.compile(model, mode="max-autotune")

    with open(save_path, "a") as fout:
        acc = sum(int(x["correct"]) for x in existing if "correct" in x)
        for ex in tqdm(rows[finished_samples:], desc="Generating (resumed)"):
            messages = [{"role": "user", "content": ex["problem"]}]
            base_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            think_texts = generate_text(
                model, tokenizer, base_prompt,
                max_tokens=args.think_budget,
                temperature=args.temperature,
                top_p=args.top_p,
                n=args.n,
            )

            for j, think_text in enumerate(think_texts):
                think_text = think_text.strip()
                if not think_text.endswith("</think>"):
                    think_text += '... Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n'
                    # "\n</think>"

                prompt_after_think = think_text

                answer_text = generate_text(
                    model, tokenizer, prompt_after_think,
                    max_tokens=args.answer_budget,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    n=1,
                )[0]

                full_output = answer_text

                answer_len = len(tokenizer(answer_text, add_special_tokens=False).input_ids)
                correct = check_correctness(full_output, ex["answer"])
                acc += int(correct)

                fout.write(json.dumps({
                    "len": answer_len,
                    "correct": correct,
                    "problem": ex["problem"],
                    "answer": ex["answer"],
                    "generated_text": full_output,
                    # "think_text": think_text,
                    "path_id": j
                }, ensure_ascii=False) + "\n")
                fout.flush()

    total = len(rows) * args.n
    print(f"✅ Done! Acc={acc/total:.4f} ({acc}/{total})")
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()