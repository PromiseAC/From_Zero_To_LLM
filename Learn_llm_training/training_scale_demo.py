import torch

micro_batch_size = 4
gradient_accumulation_steps = 8
world_size = 2

sequence_length = 2048
training_steps = 10000

dataset_tokens = 2_000_000_000

global_batch_size = (
    micro_batch_size
    * gradient_accumulation_steps
    * world_size
)

tokens_per_step = global_batch_size * sequence_length

total_training_tokens = tokens_per_step * training_steps

estimated_epochs = total_training_tokens / dataset_tokens

print("Global batch size:", global_batch_size)
print("Tokens per step:", tokens_per_step)
print("Total training tokens:", total_training_tokens)
print("Estimated epochs:", estimated_epochs)


# optimizer.zero_grad()

# for step, (x, y) in enumerate(dataloader):
#     output = model(x)
#     loss = criterion(output, y)

#     loss = loss / gradient_accumulation_steps
#     loss.backward()

#     if (step+1) % gradient_accumulation_steps == 0:
#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(),
#             max_norm=1.0
#         )

#         optimizer.step()
#         optimizer.zero_grad()
