import torch
from transformers import AutoModel, AutoTokenizer

# Use the smaller model that we know is already cached
model_name = "Qwen/Qwen2-0.5B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Model: {model_name}")
print(f"Layers: {model.config.num_hidden_layers}")
print(f"Hidden dim: {model.config.hidden_size}")
print(f"Device: {next(model.parameters()).device}")

# Test with a longer passage
text = """Abraham Lincoln was an American lawyer, politician, and statesman who served as 
the 16th president of the United States from 1861 until his assassination in 1865. 
Lincoln led the United States through the American Civil War, defending the nation as a 
constitutional union, defeating the insurgent Confederacy, playing a major role in the 
abolition of slavery, expanding the power of the federal government, and modernizing the 
U.S. economy."""

inputs = tokenizer(text, return_tensors="pt").to(model.device)
seq_len = inputs["input_ids"].shape[1]
print(f"\nInput: {seq_len} tokens")

with torch.no_grad():
    outputs = model(**inputs)

hs = outputs.hidden_states
print(f"Hidden states: {len(hs)} tensors (embedding + {len(hs)-1} layers)")
print(f"Each shape: [batch=1, seq={seq_len}, hidden={model.config.hidden_size}]")

# Analyze different layers
print("\nLayer analysis (mean-pooled vector norms):")
for i in [0, len(hs)//4, len(hs)//2, 3*len(hs)//4, len(hs)-1]:
    layer_hs = hs[i].squeeze(0)  # [seq_len, hidden_dim]
    mean_vec = layer_hs.mean(dim=0)  # [hidden_dim]
    norm = torch.norm(mean_vec).item()
    var = layer_hs.var().item()
    print(f"  Layer {i:2d}: norm={norm:.2f}, variance={var:.4f}")

# Simulate engram extraction (4 tokens)
mid_layer = len(hs) // 2
hidden = hs[mid_layer].squeeze(0)  # [seq_len, hidden_dim]
num_engram_tokens = 4
chunk_size = seq_len // num_engram_tokens

engram_vectors = []
for i in range(num_engram_tokens):
    start = i * chunk_size
    end = start + chunk_size if i < num_engram_tokens - 1 else seq_len
    chunk_mean = hidden[start:end].mean(dim=0)
    engram_vectors.append(chunk_mean)

engram = torch.stack(engram_vectors)
print(f"\nEngram extracted from layer {mid_layer}!")
print(f"  Shape: {engram.shape}")
print(f"  Compression: {seq_len} tokens -> {num_engram_tokens} vectors ({seq_len//num_engram_tokens}x)")

norms = [f"{torch.norm(v).item():.2f}" for v in engram]
print(f"  Vector norms: {norms}")

print("\n=== SUCCESS: We can access activation space! ===")
