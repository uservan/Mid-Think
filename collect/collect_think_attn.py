from __init__ import *

import os, json, argparse, re
from typing import List, Optional, Tuple
from typing import Dict, Tuple, Optional, Union, List
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_prompt(problem, tokenizer, enable_thinking, suffix):
    messages = [{"role": "user", "content": problem}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if not enable_thinking:
        if "<think>" in text[-15:]:
            text += "\n</think>\n\n"
        else:
            text += "<think>\n\n</think>\n\n"

    # if suffix and len(suffix) > 0:
    text = text + suffix + "\nOkay"
    return text

def get_category_data(category: str, split: str = "test", max_samples: int = -1):
    results = []
    if category == "HuggingFaceH4/MATH-500":
        ds = load_dataset(category, split=split)
        for ex in ds:
            results.append({"problem": ex["problem"], "answer": ex["answer"], "subject": "math500"})

    elif category == "AI-MO/aimo-validation-aime":
        ds = load_dataset(category, split="train")
        for ex in ds:
            results.append({"problem": ex["problem"], "answer": ex["answer"], "subject": "aime22-24"})

    elif category == "Idavidrein/gpqa":
        ds = load_dataset(category, "gpqa_diamond", split="train")
        instruction_following = "Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{{}}."
        for ex in ds:
            multiple_choice_string, correct_answer_letter = get_multiple_choice_answers(ex)
            question = instruction_following + " " + ex["Question"] + " " + multiple_choice_string
            results.append({"problem": question, "answer": correct_answer_letter, "subject": "gpqa"})

    if max_samples > 0:
        results = results[:max_samples]
    return results


@torch.no_grad()
def gen_and_collect_prefix_attn(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    prefix_k: int = 100,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Returns:
      full_ids_1d: Tensor [seq_total]
      out[i] = (token_ids_after, attn_mean_after)
        - token_ids_after: [seq_total - (i+1)]
        - attn_mean_after: [seq_total - (i+1)] where attn_mean_after[t] = mean_{layers,heads} Attn[j, i]
          (j is the corresponding position of token_ids_after)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # 1) tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # 2) generate full sequence
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_attentions=False,
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))

    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        **gen_kwargs
    )
    full_ids = gen_out.sequences  # [1, seq_total]
    seq_total = full_ids.size(1)

    # 3) full forward to get full attentions
    fw_out = model(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids, device=device),
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    attentions = fw_out.attentions
    if attentions is None:
        raise RuntimeError(
            "No attentions returned. If you saw sdpa error before, reload model with attn_implementation='eager'."
        )

    # stack: [layers, 1, heads, seq, seq]
    attn = torch.stack(list(attentions), dim=0)
    # mean over heads + layers -> [seq, seq]
    attn_mean = attn.mean(dim=2).mean(dim=0).squeeze(0)  # [seq, seq]

    full_ids_1d = full_ids[0]  # [seq]
    K = min(prefix_k, seq_total)

    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # 4) for each prefix token position i, collect (token_ids_after, attn_mean_after)
    for i in range(K):
        if i >= seq_total - 1:
            out[i] = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float))
            continue
        tok_id = full_ids_1d[i].item()
        token_ids_after = full_ids_1d[i + 1:]                          # [L_after]
        idx_j = torch.arange(i + 1, seq_total, device=device)          # positions j
        attn_vals = attn_mean[idx_j, i]                                # [L_after], each is attention from j -> i

        out[i] = {
            "token_id": tok_id,
            "token_ids_after": token_ids_after.detach().cpu(),
            "attn_vals": attn_vals.detach().cpu()
        }

    return full_ids_1d.detach().cpu(), out

if __name__ == "__main__":
    # 1) load data
    dataset_name = "HuggingFaceH4/MATH-500"
    rows = get_category_data(dataset_name, max_samples=10)
    print(f"Total problems: {len(rows)}")
    enable_thinking, suffix = False, ''
    # 2) load model
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    # mode = 'think' 
    # if not enable_thinking:
    #     if len(suffix) > 0:
    #         mode = suffix
    #     else:
    #         mode = 'no_think'

    mode = 'think'
    if not enable_thinking: mode = 'no_think'
    if len(suffix) > 0: mode = suffix+mode
    else: mode = 'notag'+mode
    

    save_dir = os.path.join("/expert/no_think_attn", model_name.split("/")[-1], mode, dataset_name.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)

    # 3) build prompts
    prompts = []
    for ex in rows:
        prompts.append(get_prompt(ex["problem"], tokenizer, enable_thinking=enable_thinking, suffix=suffix))

    # 4) run
    for idx, prompt in enumerate(prompts):
        full_ids, d = gen_and_collect_prefix_attn(
            model,
            tokenizer,
            prompt,
            prefix_k=300,
            max_new_tokens=3000
        )

        # 确保都在 CPU，避免之后 load 要 GPU
        full_ids = full_ids.cpu()

        data = {
            "prom`pt": prompt,
            "full_ids": full_ids,  # [seq]
            "d": d              # i -> (after_token_ids, attn_mean)
        }

        save_path = os.path.join(save_dir, f"sample_{idx:05d}.pt")
        torch.save(data, save_path)

        print(f"[saved] {save_path}")