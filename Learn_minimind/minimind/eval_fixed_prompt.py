import gc
import math
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# ============================================================
# 1. Fixed Evaluation Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 8
MAX_SEQ_LEN = 768
EVAL_BATCH_SIZE = 8

TRAIN_DATA = "dataset/pretrain_fixed_train_8k.jsonl"
VAL_DATA = "dataset/pretrain_fixed_val_2k.jsonl"

OUTPUT_PATH = Path("../week4_minimind/generation_results.md")

CHECKPOINTS = {
    "Early": {
        "step": 100,
        "optimizer_step": 25,
        "path": "checkpoints/fixed_prompt_eval/"
                "pretrain_fixed_eval_step0100_768.pth",
    },
    "Middle": {
        "step": 500,
        "optimizer_step": 125,
        "path": "checkpoints/fixed_prompt_eval/"
                "pretrain_fixed_eval_step0500_768.pth",
    },
    "Late": {
        "step": 1000,
        "optimizer_step": 250,
        "path": "checkpoints/fixed_prompt_eval/"
                "pretrain_fixed_eval_step1000_768.pth",
    },
}

PROMPTS = [
    "中国的首都是",
    "机器学习是",
    "1 + 1 =",
]

MAX_NEW_TOKENS = 64


# ============================================================
# 2. Utility
# ============================================================

def get_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()


def load_model(checkpoint_path):
    config = MiniMindConfig(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        use_moe=False,
    )

    model = MiniMindForCausalLM(config)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)

    model = model.half().to(DEVICE)
    model.eval()

    return model


# ============================================================
# 3. Fixed Loss Evaluation
# ============================================================

@torch.inference_mode()
def evaluate_loss(model, dataloader):
    total_nll = 0.0
    total_tokens = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(
            input_ids,
            labels=labels,
        )

        # MiniMind 内部：
        #
        # logits[..., :-1, :]
        # labels[..., 1:]
        #
        # 所以这里统计真正参与 CE 的 target token 数。
        valid_tokens = (
            labels[:, 1:] != -100
        ).sum().item()

        total_nll += (
            outputs.loss.float().item()
            * valid_tokens
        )

        total_tokens += valid_tokens

    return total_nll / total_tokens


# ============================================================
# 4. Fixed Greedy Generation
# ============================================================

@torch.inference_mode()
def generate_text(model, tokenizer, prompt):
    text = tokenizer.bos_token + prompt

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(DEVICE)

    prompt_length = inputs["input_ids"].shape[1]

    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,

        # Pure greedy decoding
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,

        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = generated_ids[0][prompt_length:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )


# ============================================================
# 5. Markdown Output
# ============================================================

def write_results(results, git_commit):
    lines = []

    lines.append("# Week 4 - Fixed Prompt Evaluation")
    lines.append("")

    lines.append("## Experiment Setup")
    lines.append("")
    lines.append(f"- Git commit: `{git_commit}`")
    lines.append("- Model: MiniMind Pretrain")
    lines.append("- hidden_size: `768`")
    lines.append("- num_hidden_layers: `8`")
    lines.append("- max_seq_len: `768`")
    lines.append("- Train samples: `8000`")
    lines.append("- Validation samples: `2000`")
    lines.append("- Split seed: `42`")
    lines.append("")

    lines.append("### Generation Config")
    lines.append("")
    lines.append("- decoding: `greedy`")
    lines.append("- do_sample: `False`")
    lines.append("- max_new_tokens: `64`")
    lines.append("- temperature: `1.0`")
    lines.append("- top_p: `1.0`")
    lines.append("- top_k: `0`")
    lines.append("- repetition_penalty: `1.0`")
    lines.append("")

    lines.append("## Loss Comparison")
    lines.append("")
    lines.append(
        "| Checkpoint | Micro Step | Optimizer Step | "
        "Fixed Train Eval Loss | Validation Loss |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: |"
    )

    for stage, result in results.items():
        lines.append(
            f"| {stage} "
            f"| {result['step']} "
            f"| {result['optimizer_step']} "
            f"| {result['train_loss']:.4f} "
            f"| {result['val_loss']:.4f} |"
        )

    lines.append("")
    lines.append("## Generation Results")
    lines.append("")

    for prompt in PROMPTS:
        lines.append(f"### Prompt: `{prompt}`")
        lines.append("")

        for stage, result in results.items():
            lines.append(f"#### {stage} Checkpoint")
            lines.append("")
            lines.append("```text")
            lines.append(result["generations"][prompt])
            lines.append("```")
            lines.append("")

    lines.append("## Analysis")
    lines.append("")
    lines.append(
        "- Compare whether lower Train Loss also gives lower "
        "Validation Loss."
    )
    lines.append(
        "- Compare whether lower token-level loss improves "
        "fluency or correctness."
    )
    lines.append(
        "- Check whether factual prompts improve monotonically "
        "across checkpoints."
    )
    lines.append(
        "- Remember that this is a pretraining model: generation "
        "should be interpreted primarily as text continuation, "
        "not instruction-following ability."
    )
    lines.append("")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 6. Main
# ============================================================

def main():
    print("Device:", DEVICE)
    print("Git commit:", get_git_commit())

    tokenizer = AutoTokenizer.from_pretrained("model")

    train_dataset = PretrainDataset(
        TRAIN_DATA,
        tokenizer,
        max_length=MAX_SEQ_LEN,
    )

    val_dataset = PretrainDataset(
        VAL_DATA,
        tokenizer,
        max_length=MAX_SEQ_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    results = {}

    for stage, info in CHECKPOINTS.items():
        print()
        print("=" * 80)
        print(
            f"{stage}: "
            f"micro_step={info['step']}, "
            f"optimizer_step={info['optimizer_step']}"
        )
        print("Checkpoint:", info["path"])

        model = load_model(info["path"])

        train_loss = evaluate_loss(
            model,
            train_loader,
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
        )

        print(f"Train Eval Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        generations = {}

        for prompt in PROMPTS:
            response = generate_text(
                model,
                tokenizer,
                prompt,
            )

            generations[prompt] = response

            print(f"\nPrompt: {prompt}")
            print(f"Generation: {response}")

        results[stage] = {
            "step": info["step"],
            "optimizer_step": info["optimizer_step"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "generations": generations,
        }

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    git_commit = get_git_commit()

    write_results(
        results,
        git_commit,
    )

    print()
    print("=" * 80)
    print("Evaluation finished.")
    print(f"Results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()