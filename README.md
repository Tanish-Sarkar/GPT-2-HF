# GPT-2 Next Token Prediction — From Scratch

> Built without tutorial code. Documentation only. Every line written by hand.

---

## The Problem I Was Fixing

I was assembling code, not engineering it. I could copy a GPT-2 script from a tutorial and make it run. But I couldn't explain why each line existed. This project was built under one rule: understand the concept first, write the code yourself, no copying.

---

## What This Project Covers

1. Load GPT-2, tokenize input, get logits, print top 5 next token probabilities
2. Build a manual generation loop token by token without `.generate()`
3. Experiment with temperature and top-k sampling to see how outputs change

---

## Part 1 — The Forward Pass

### The Pipeline

To predict the next token after a phrase like `"The sky is"`, three things happen in order:

1. **Tokenize** — convert the string into a sequence of integer ids
2. **Forward pass** — run those ids through the model
3. **Decode** — convert the output scores back into readable tokens

### The Two Classes

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
```

**Why `GPT2LMHeadModel` and not just `GPT2Model`?**

The base `GPT2Model` outputs hidden state vectors — rich internal representations of meaning, but not directly readable as token predictions. The `LMHead` (Language Model Head) is a linear layer added on top that maps those hidden states to **logits** — one score per token in the vocabulary (50,257 tokens for GPT-2). Higher score = more likely next token.

### Tokenization

```python
inputs = tokenizer("The sky is", return_tensors="pt")
```

This returns a `BatchEncoding` dictionary with two keys:
- `input_ids` — integer ids, one per token. Each id maps to a specific token in GPT-2's vocabulary.
- `attention_mask` — a tensor of 1s, same length as `input_ids`. Tells the model which tokens are real vs padding.

`return_tensors="pt"` returns PyTorch tensors because that's what the model expects.

### The Forward Pass

```python
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
```

`**inputs` unpacks the dictionary so `input_ids` and `attention_mask` are passed as keyword arguments.

`torch.no_grad()` disables gradient tracking — we're doing inference, not training. Saves memory and compute.

`outputs.logits` has shape `(batch, sequence_length, vocab_size)` → `(1, 3, 50257)` for our input.

### Why the Last Position?

The model produces predictions at every token position. Position 0 predicts what comes after token 0, position 1 predicts what comes after token 1, and so on. To predict what comes *after the full input*, we want the last position:

```python
last_token_logits = logits[0, -1, :]  # batch 0, last position, all vocab scores
```

### Top 5 Predictions

```python
probs = torch.softmax(last_token_logits, dim=0)
top_5 = torch.topk(probs, 5)

for token_id in top_5.indices:
    print(tokenizer.decode([token_id]))
```

`torch.topk` returns two things:
- `.values` — the actual probability numbers
- `.indices` — positions in the vocabulary, which decode to readable tokens

### Output

```
' the'    → 15.2%
' blue'   → 8.6%
' falling'→ 7.3%
' a'      → 5.1%
' full'   → 3.4%
```

The remaining ~60% of probability mass is distributed across the other 50,252 tokens — each tiny, but they add up.

---

## Part 2 — Manual Generation Loop

No `.generate()`. Every token produced by hand.

### The Logic

At each step:
1. Run a forward pass on the current sequence
2. Get logits at the last position
3. Pick the next token
4. Append it to `input_ids` and `attention_mask`
5. Repeat

```python
N = 10
original_input_ids = inputs["input_ids"]
original_attention_mask = inputs["attention_mask"]

input_ids = original_input_ids.clone()
attention_mask = original_attention_mask.clone()

for step in range(N):
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits

    last_token_logits = logits[0, -1, :]
    probs = torch.softmax(last_token_logits, dim=-1)
    next_token = torch.argmax(probs, dim=-1)

    input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long)], dim=1)

print(tokenizer.decode(input_ids[0]))
```

### Why `.unsqueeze(0).unsqueeze(0)`?

`torch.argmax` returns a scalar tensor with shape `()`. `input_ids` has shape `(1, sequence_length)`. To concatenate along `dim=1`, both tensors need to be 2D. `.unsqueeze(0).unsqueeze(0)` reshapes the scalar to `(1, 1)`.

### Why update `attention_mask`?

Every new token is real — not padding. The mask needs a new `1` appended every iteration, otherwise the model would ignore the new token.

### Output (greedy)

```
The sky is the limit.
The sky is the limit.
The sky is the limit.
...
```

**Greedy decoding always picks the highest probability token.** It gets stuck in loops. "The sky is the limit" is the most probable continuation, and once the model generates it, it becomes the most probable continuation again. Forever.

This is why sampling exists.

---

## Part 3 — Temperature and Top-K Sampling

### Temperature

Before applying softmax, divide logits by a temperature value T:

```python
probs = torch.softmax(last_token_logits / temperature, dim=-1)
```

**T < 1 (e.g. 0.5)** — dividing by a small number magnifies differences. High logits get much higher, low logits get much lower. The distribution becomes **sharper** — the model is more confident, outputs are more predictable and repetitive.

**T > 1 (e.g. 1.5)** — dividing by a large number compresses differences. The distribution becomes **flatter** — probability spreads out across more tokens. Outputs are more creative but can become incoherent at high values.

**The key insight:** temperature changes the *shape* of the distribution. But if you still use `argmax`, temperature has no effect — argmax always picks the top token regardless. Temperature only matters when you *sample*.

### Top-K Sampling

Instead of sampling from all 50,257 tokens (most of which have near-zero probability), restrict sampling to the top k most probable tokens:

```python
top_probs, top_indices = torch.topk(probs, k)
sampled_position = torch.multinomial(top_probs, 1)
next_token = top_indices[sampled_position]
```

**How `torch.multinomial` works:**
- Takes a probability distribution and returns a sampled *position* within it
- If `top_probs` has k elements, multinomial returns a number between 0 and k-1
- That position indexes into `top_indices` to get the actual vocabulary id

### Full Loop

```python
temperatures = [0.5, 1.0, 1.5]
top_ks = [10, 50, 100]

for t, k in zip(temperatures, top_ks):
    input_ids = original_input_ids.clone()
    attention_mask = original_attention_mask.clone()

    for step in range(N):
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

        last_token_logits = logits[0, -1, :]
        last_token_logits = last_token_logits / t

        probs = torch.softmax(last_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k)
        next_token = top_indices[torch.multinomial(top_probs, 1)]

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long)], dim=1)

    print(f"\nTemperature={t}, top_k={k}")
    print(tokenizer.decode(input_ids[0]))
```

### Outputs

```
Temperature=0.5, top_k=10
The sky is blue and the sky is blue.  The

Temperature=1.0, top_k=50
The sky is not blue and we cannot see it as it can

Temperature=1.5, top_k=100
The sky is obviously no better, and it's absolutely horrible-
```

### What This Proves

| Setting | Distribution | Output |
|---|---|---|
| Low T, small k | Sharp, restricted | Coherent but repetitive |
| Medium T, medium k | Balanced | Natural, varied |
| High T, large k | Flat, wide | Creative but loses coherence |

**The tradeoff:** predictability vs creativity. Production systems tune these values for the use case. Code generators use low temperature. Creative writing assistants use higher temperature. There is no universally correct value.

---

## Key Intuitions

**On logits vs probabilities:** Logits are raw unnormalized scores. Softmax converts them to a probability distribution that sums to 1. Always apply softmax before sampling or comparing probabilities.

**On the LM Head:** The base transformer outputs hidden states — dense vectors encoding meaning. The LM Head is a single linear layer that projects those vectors into vocabulary space. Without it, you can't produce token predictions.

**On greedy decoding:** Always picking the most probable token sounds optimal but isn't. It collapses into repetitive loops because the most probable token at each step reinforces itself. Real generation requires sampling.

**On temperature as confidence:** Temperature is literally how confident you let the model be in its own predictions. Low temperature = overconfident = boring. High temperature = underconfident = chaotic. The right temperature depends on what you're building.

---

## What I Actually Learned

The difference between assembling and engineering is whether you can explain every line. I can now explain every line in this script — what shape each tensor is, why each operation exists, what breaks if you remove it.

That's the only metric that matters at this stage.

---

*Built by Tanish | 3rd year Computer Engineering 
*Part of a series: understanding LLMs from the inside out*
