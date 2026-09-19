# Week 4 - Fixed Prompt Evaluation

## Experiment Setup

- Git commit: `518e9a563a073ee95374ca047242ccbf55bb1334`
- Model: MiniMind Pretrain
- hidden_size: `768`
- num_hidden_layers: `8`
- max_seq_len: `768`
- Train samples: `8000`
- Validation samples: `2000`
- Split seed: `42`

### Generation Config

- decoding: `greedy`
- do_sample: `False`
- max_new_tokens: `64`
- temperature: `1.0`
- top_p: `1.0`
- top_k: `0`
- repetition_penalty: `1.0`

## Loss Comparison

| Checkpoint | Micro Step | Optimizer Step | Fixed Train Eval Loss | Validation Loss |
| --- | ---: | ---: | ---: | ---: |
| Early | 100 | 25 | 7.2222 | 7.2279 |
| Middle | 500 | 125 | 6.0104 | 6.0425 |
| Late | 1000 | 250 | 5.4029 | 5.4747 |

## Generation Results

### Prompt: `中国的首都是`

#### Early Checkpoint

```text
，
```

#### Middle Checkpoint

```text
我，我在我是我是我，我是我是我，我是我是我，我是我是我。我，我是我，我是我，我是我，我是我，我是我，我是我，我是我，我是我，我是
```

#### Late Checkpoint

```text
夏天的诗歌。秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
秋天，
```

### Prompt: `机器学习是`

#### Early Checkpoint

```text
，
```

#### Middle Checkpoint

```text
我你的。我我在我你是我你是我，我在我在我，我你是我是我，我是我是我，我是我是我，我是我是我，我是我是我，我是我是我，我是我是我，
```

#### Late Checkpoint

```text
““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““““
```

### Prompt: `1 + 1 =`

#### Early Checkpoint

```text
，
```

#### Middle Checkpoint

```text
 































































```

#### Late Checkpoint

```text
 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 = 3 =
```

## Analysis

Across the three checkpoints, both fixed train evaluation loss and
validation loss decreased consistently:

- Early: Train 7.2222 / Validation 7.2279
- Middle: Train 6.0104 / Validation 6.0425
- Late: Train 5.4029 / Validation 5.4747

However, generation quality did not improve monotonically with loss.

The Early checkpoint mostly generated punctuation, indicating that the
model had not yet learned useful language structure.

The Middle checkpoint began producing common Chinese tokens and short
local patterns, but suffered from severe repetition and lacked coherent
semantics.

The Late checkpoint produced more recognizable textual structures, but
generation was still unstable and often incorrect. For example,
`1 + 1 =` generated repeated `3 = 3 = ...`, despite the substantially
lower validation loss.

Therefore, lower token-level cross-entropy loss does not necessarily
imply better factual correctness, reasoning ability, fluency, or
sequence-level generation quality.

This experiment also uses a very small pretraining run with only 8,000
training samples and 250 optimizer updates. The poor generations are
therefore expected and should not be interpreted as evidence that
pretraining itself is ineffective.

Because this is a pretrained causal language model rather than an
instruction-tuned model, the generations should primarily be interpreted
as text continuation behavior rather than instruction-following ability.