# Adapt the model for a message-routing assistant

A small model can help an assistant decide how to handle a message. For example, it might propose one of three labels:

| Label | Example | Possible next step |
|---|---|---|
| `chat` | “Good morning!” | Reply conversationally. |
| `explain` | “What kinds of things can you help with?” | Show a reviewed explanation. |
| `lookup` | “What time does the museum close today?” | Use an approved information source. |

These are example labels. Choose names and definitions that match your own application. A model trained only on these labels cannot reproduce a more complex output schema without additional examples and evaluation.

A useful architecture is:

```text
Message → fixed command checks → model proposes a label → application validates it → approved next step
```

Handle simple fixed commands directly in code. Treat the model's answer as a proposal: check it against a list of allowed labels and let application code choose the action. If a response is invalid or the model is unavailable, use a fallback appropriate to your application, such as asking for clarification. A label alone is not evidence that an answer is correct or current.

## 1. Design examples

Write a short, precise definition for each label. Gather examples of normal requests, different wording, typos, unclear messages, and messages that combine categories. Agree on the correct label for ambiguous cases. Include whatever context the real classifier will see, such as a short conversation history, in the training input. Do not include fields at training time that will not be available at inference time.

Use fictional data or data you have permission to use. Remove sensitive details before preparing examples. If a larger model drafts labels, review them: teacher labels can be wrong. Keep private datasets out of the repository; the `local-data/` directory is ignored by Git.

The included dataset is a tiny illustration. For a real project, collect a much larger and more varied set and review it for quality and coverage. The right amount depends on the task; measure performance instead of assuming a particular example count is sufficient.

## 2. Split the data

Create three separate sets:

- **Training:** examples used to update the adapter.
- **Validation:** examples used during development to monitor training and choose settings.
- **Test:** examples set aside until you are ready to evaluate the chosen approach.

Keep related messages, paraphrases, and turns from the same conversation together in one split. Otherwise, nearly identical examples can inflate results. Keep the final test set out of prompts, training, and model selection; if you repeatedly use it to make decisions, reserve another untouched test set for the final evaluation.

## 3. Format your training data

After following the setup steps in the main README, create `local-data/my-assistant/train.jsonl` and `local-data/my-assistant/valid.jsonl`. Each line is one complete JSON object. Use the same system instruction for training and inference. For example, this is a single line of training data:

```json
{"messages":[{"role":"system","content":"Choose exactly one label: chat, explain, or lookup."},{"role":"user","content":"What time does the museum close today?"},{"role":"assistant","content":"lookup"}]}
```

Make the last message the expected assistant answer. Keep your own test set separately; the trainer below reads only `train.jsonl` and `valid.jsonl`. If you need additional structured fields, define the expected schema in the prompt, supply representative examples of the complete output, and test that every output can be parsed and validated. More complex structures may require a different training setup.

## 4. Train a new adapter

From the repository root, with the virtual environment activated:

```sh
python -m mlx_lm lora \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --train --mask-prompt \
  --data local-data/my-assistant \
  --adapter-path adapters/my-assistant-v1 \
  --iters 100 --batch-size 1 --max-seq-length 512 \
  --val-batches 3 --steps-per-eval 50
```

Choose a new adapter path for each experiment to preserve earlier runs. This is a starting configuration, not a guarantee of accuracy. Check tokenized input lengths and increase the sequence limit if necessary: truncating long inputs can omit context or the expected answer. Longer sequences take more time and memory. Save the data version, model version, training settings, and results so comparisons are reproducible.

## 5. Try the adapter

The following example runs locally. Use the same system instruction as the one used for your dataset:

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/Qwen3.5-0.8B-4bit",
    adapter_path="adapters/my-assistant-v1",
)
messages = [
    {"role": "system", "content": "Choose exactly one label: chat, explain, or lookup."},
    {"role": "user", "content": "What time does the museum close today?"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
label = generate(model, tokenizer, prompt=prompt, max_tokens=16, verbose=False).strip()
allowed = {"chat", "explain", "lookup"}
print(label if label in allowed else "clarify")
```

This prints a proposed label. It does not look anything up or execute an action. Your application should also handle model-loading failures, timeouts, and invalid output explicitly.

## 6. Evaluate before connecting it

Compare the base model and the tuned model using the **same** examples and scoring rules. For the final evaluation, use the untouched test set. Record:

- Valid-output rate and accuracy for each label, not just an overall average.
- Errors on ambiguous messages and messages that need an external lookup.
- Behavior when output is malformed, the model times out, or context is missing.
- End-to-end latency and resource use on the target machine, including any network or application overhead.

Inspect the actual mistakes. A short local demo and lower training loss do not prove that the model will generalize. If results are promising, run the model in **shadow mode**: record its proposed labels alongside the existing system's decisions without letting the new model control the user-facing action. Compare outcomes on appropriately reviewed examples before deciding whether to use it.
