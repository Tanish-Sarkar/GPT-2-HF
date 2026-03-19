
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("The sky is", return_tensors="pt")
input_ids = inputs["input_ids"]
print(inputs)

# with torch.no_grad():
#   outputs = model(**inputs)
#   logits = outputs.logits

# # get the last squene in the logits
# last_token_logits = logits[0, -1, :]
# print(last_token_logits)

# # logit vector -> probabilities
# probs = torch.softmax(last_token_logits, dim=0)
# print(probs)

# # top 5 probabilities
# top_5 = torch.topk(probs, 5)
# print(top_5)

# # convert to tokens
# tokens = []
# for token in top_5.indices:
#   tokens.append(tokenizer.decode([token]))
# print(tokens)

## **Step 02 : generations loop
print(tokenizer.decode(input_ids[0]))

'''
N = 10
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
for step in range(N):
  with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits

  last_token_logits = logits[0, -1, :]
  probs = torch.softmax(last_token_logits, dim=-1);
  next_token = torch.argmax(probs, dim=-1)
  input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
  # attention mask 👇
  attention_mask = torch.cat([attention_mask, torch.ones(1,1, dtype=torch.long)], dim=1)

generated_text = tokenizer.decode(input_ids[0])
print(generated_text)
'''

"""## Generation Loop but with temperatures"""



N = 10
original_input_ids = inputs["input_ids"]
original_attention_mask = inputs["attention_mask"]
temperature = [0.5, 1, 1.5]
top_k = [10, 50, 100]

for t,k in zip(temperature, top_k):
  input_ids = original_input_ids.clone()
  attention_mask = original_attention_mask.clone()
  for step in range(N):
    with torch.no_grad():
      output = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = output.logits

    last_token_logits = logits[0, -1, :]
    last_token_logits = last_token_logits / t

    probs = torch.softmax(last_token_logits, dim=-1);
    top_probs, top_indices = torch.topk(probs,k)
    next_token = top_indices[torch.multinomial(top_probs,1)]
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(1,1, dtype=torch.long)], dim=1)

  generated_text = tokenizer.decode(input_ids[0])
  print(f"\nTemperature={t}, top_k={k}")
  print(generated_text)

