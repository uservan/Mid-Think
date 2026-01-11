# stage1_generate_vllm_budget_batch.py
from __init__ import *
import os, json, argparse, re
from tqdm import tqdm
from typing import List
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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


def extract_think_block(text: str) -> str:
    m = text.split('</think>')[0]
    # m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    # return m.group(1).strip() if m else text
    return m


def truncate_by_budget(tokenizer, text: str, budget: float) -> str:
    if budget >= 1.0 or len(text.strip()) == 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        return ""
    keep = max(1, int(len(ids) * budget))
    return tokenizer.decode(ids[:keep]), len(ids[:keep])

prompt = 'Solve the problem with minimal reasoning. Think carefully but concisely. Only include steps that are strictly necessary. Avoid redundant explanations or restatements. Provide a brief reasoning, then give the final answer. '
def get_prompt(problem, tokenizer, enable_thinking, suffix, add_okay, add_prompt):
    if add_prompt:
        problem = prompt + problem
        print(problem)
    messages = [{"role": "user", "content": problem}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not enable_thinking:
        if "<think>" in text[-15:]:
            text += "\n</think>\n\n"
        else:
            text += "<think>\n\n</think>\n\n"
    if suffix and len(suffix)>0:
        text = text + suffix+'\nOkay'
    if add_okay:
        text = text + '\nOkay'
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=1024 * 32)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--add_okay", type=bool, default=False)
    parser.add_argument("--add_prompt", type=bool, default=False)
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--save_path", type=str,
                        default="/expert/no_think_prompt")
    args = parser.parse_args()

    filename = f"{args.model.split('/')[-1]}_{args.enable_thinking}_{args.suffix}_budget{args.budget}.json"
    save_path = os.path.join(args.save_path, args.dataset.split('/')[-1], f'{args.n}', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Exiting.")
        return

    rows = get_category_data(args.dataset, max_samples=args.max_samples)
    print(f"Total problems: {len(rows)}")

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=args.max_tokens)

    # ============================================================
    # First-pass prompts
    # ============================================================
    first_prompts = []
    for ex in rows:
        first_prompts.append(get_prompt(ex["problem"], tokenizer, enable_thinking=args.enable_thinking, 
                                        suffix=args.suffix, add_okay=args.add_okay, add_prompt=args.add_prompt))

    # ============================================================
    # First-pass generation (batched, n completions)
    # ============================================================
    print(f"Running first-pass generation with n={args.n}...")
    first_outputs = llm.generate(
        first_prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.n,
            stop=[tokenizer.eos_token]
        )
    )

    # ============================================================
    # Budget condition
    # ============================================================
    need_second_pass = (0 < args.budget < 1)

    # Storage for second pass prompts
    second_pass_prompts = []
    # To keep mapping: each final output must match its problem
    second_pass_meta = []  # (problem, answer)

    # ============================================================
    # Build second-pass prompts (batched)
    # ============================================================
    if need_second_pass:
        print(f"Budget={args.budget}; building second-pass prompts...")

        for ex, out in zip(rows, first_outputs):
            for sample in out.outputs:

                first_text = sample.text
                think_text = extract_think_block(first_text)
                truncated, keep_len = truncate_by_budget(tokenizer, think_text, args.budget)
                base_prompt = get_prompt(ex["problem"], tokenizer, args.enable_thinking, args.suffix, args.add_okay, args.add_prompt)
                # # Build truncated prompt
                # messages = [{"role": "user", "content": ex["problem"]}]
                # base_prompt = tokenizer.apply_chat_template(
                #     messages, tokenize=False, add_generation_prompt=True
                # )
                new_prompt = base_prompt + f"{truncated}\n</think>\n\n"

                second_pass_prompts.append(new_prompt)
                second_pass_meta.append((ex["problem"], ex["answer"], f"{truncated}\n</think>\n\n", keep_len))

        # ============================================================
        # Second-pass generation (FULL BATCH)
        # ============================================================
        print("Running second-pass generation (batched)...")
        second_outputs = llm.generate(
            second_pass_prompts,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                n=1,   # always 1 in second-pass
                stop=[tokenizer.eos_token]
            )
        )

    else:
        print("No budget mode active, using first-pass outputs only.")
        second_outputs = None

    # ============================================================
    # Save minimal results
    # ============================================================
    print("Saving results...")

    result_lines = []
    idx = 0

    for ex, out in zip(rows, first_outputs):
        for sample in out.outputs:
            length, problem, answer = 0, ex["problem"], ex["answer"]
            # -----------------------------------------
            # Get final generated text
            # -----------------------------------------
            if need_second_pass:
                # second pass output
                o = second_outputs[idx]
                problem, answer = second_pass_meta[idx][0], second_pass_meta[idx][1]
                raw_final = second_pass_meta[idx][2]+ o.outputs[0].text 
                length = len(o.outputs[0].token_ids) + second_pass_meta[idx][3]
                idx += 1
            else:
                # first pass output only
                raw_final = sample.text
                length = len(sample.token_ids)

            # -----------------------------------------
            # Build final_text = original_prompt + final_output
            # -----------------------------------------
            # reconstruct original prompt (same as first pass)
            first_prompt = get_prompt(problem, tokenizer, args.enable_thinking, args.suffix, args.add_okay, args.add_prompt)
            final_text = first_prompt + raw_final

            # -----------------------------------------
            # Check correctness
            # -----------------------------------------
            correct = check_correctness(final_text, answer)

            # -----------------------------------------
            # Save json line
            # -----------------------------------------
            result_lines.append(json.dumps({
                # evaluation
                "correct": correct,
                "len": length,
                # data
                "problem": problem,
                "answer": answer,
                "generated_text": final_text,
                # experiment metadata
                "model": args.model,
                "dataset": args.dataset,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "enable_thinking": args.enable_thinking,
                "suffix": args.suffix,
                "budget": args.budget,
            }, ensure_ascii=False))

    with open(save_path, "w") as f:
        f.write("\n".join(result_lines))

    print(f"Saved to {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()