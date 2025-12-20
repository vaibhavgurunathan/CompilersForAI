import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. Configuration
# ==========================================
DATA_FILE = "moe_trajectories.jsonl"
TARGET_LAYER = 12   
NUM_EXPERTS = 60    
EXPERTS_PER_TOK = 4 

# --- TRANSFORMER CONFIG ---
WINDOW_SIZE = 10    # Increased window: Transformers handle longer context better
D_MODEL = 128       # Internal dimension of the Transformer
N_HEAD = 4          # Number of attention heads
N_LAYERS = 2        # Number of Transformer blocks
DROPOUT = 0.1
# --------------------------

EPOCHS = 100
BATCH_SIZE = 128    
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = f"expert_transformer_layer_{TARGET_LAYER}.pth"

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
print(f"Loading {DATA_FILE}...")

if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Please run the generation script first.")
    exit()

sequences = []
with open(DATA_FILE, 'r') as f:
    for line in f:
        try:
            entry = json.loads(line)
            traj = entry['router_trajectories'].get(str(TARGET_LAYER))
            if traj:
                sequences.append(traj)
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(sequences)} sequences for Layer {TARGET_LAYER}.")

# ==========================================
# 3. Data Structuring (3D Sequences)
# ==========================================
print(f"Structuring dataset with Context Window of {WINDOW_SIZE}...")

X_list = []
y_list = []

# Fit MLB on all potential experts
mlb = MultiLabelBinarizer(classes=range(NUM_EXPERTS))
mlb.fit([[i for i in range(NUM_EXPERTS)]])

for seq in sequences:
    if len(seq) <= WINDOW_SIZE:
        continue

    # Slide window
    for t in range(WINDOW_SIZE, len(seq)):
        # Input: The previous WINDOW_SIZE steps
        context_window = seq[t-WINDOW_SIZE : t]
        
        # Target: The current step experts
        target_experts = seq[t]
        
        # 1. Encode each step -> Matrix of shape (WINDOW_SIZE, NUM_EXPERTS)
        encoded_window = mlb.transform(context_window)
        
        # CRITICAL CHANGE: We do NOT flatten anymore. We keep the sequence structure.
        # X shape will be (N_Samples, WINDOW_SIZE, NUM_EXPERTS)
        X_list.append(encoded_window)
        y_list.append(target_experts)

# Transform targets
y_encoded = mlb.transform(y_list)

# Convert to numpy arrays
X_encoded = np.array(X_list)

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.15, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print(f"Dataset Shape: {X_train_tensor.shape}") # (Batch, Window, Experts)
print(f"Input Sequence Length: {WINDOW_SIZE}")

# ==========================================
# 4. Transformer Model Definition
# ==========================================
class ExpertTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_head, num_layers, output_dim, max_len=50, dropout=0.1):
        super(ExpertTransformer, self).__init__()
        
        # 1. Input Projection: Map generic expert vector (60) to Model Dimension (128)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding: Learnable vector to tell the model "this is step T-1", "this is T-2"
        # Shape: (1, max_len, d_model) broadcastable to batch
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        
        # Embed inputs
        x = self.embedding(x) # -> (Batch, Seq_Len, D_Model)
        
        # Add Positional Encoding (Sliced to current sequence length)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # We only care about the prediction from the LAST token in the sequence
        # (This represents the state after seeing the full history)
        last_token_state = x[:, -1, :]
        
        # Predict next experts
        out = self.fc_out(self.dropout(last_token_state))
        return out

# Custom Dataset Wrapper
class ExpertDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ExpertDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ExpertDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. Training Loop
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ExpertTransformer(
    input_dim=NUM_EXPERTS, 
    d_model=D_MODEL, 
    n_head=N_HEAD, 
    num_layers=N_LAYERS, 
    output_dim=NUM_EXPERTS,
    max_len=WINDOW_SIZE + 5
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print(f"\nTraining Transformer on {device}...")

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# Save the trained model
print(f"\nSaving model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# ==========================================
# 6. Evaluation (Strict Recall & Jaccard)
# ==========================================
print("\nEvaluating...")
model.eval()

total_recall = 0
total_jaccard = 0
perfect_matches = 0
count = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # Get top K predicted experts
        _, top_indices = torch.topk(outputs, k=EXPERTS_PER_TOK, dim=1)
        
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            true_indices = (targets[i] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
            pred_indices = top_indices[i].cpu().numpy()
            
            # Intersection
            intersection = np.intersect1d(true_indices, pred_indices)
            hits = len(intersection)
            
            # Recall: Hits / Total True Experts
            recall = hits / len(true_indices) if len(true_indices) > 0 else 0
            total_recall += recall
            
            # Jaccard
            union = len(np.union1d(true_indices, pred_indices))
            jaccard = hits / union if union > 0 else 0
            total_jaccard += jaccard
            
            # Perfect Match
            if hits == len(true_indices) and len(true_indices) == EXPERTS_PER_TOK:
                perfect_matches += 1
            
            count += 1

avg_recall = total_recall / count
avg_jaccard = total_jaccard / count
perfect_rate = perfect_matches / count

print(f"-"*30)
print(f"RESULTS (Layer {TARGET_LAYER}, Window {WINDOW_SIZE})")
print(f"-"*30)
print(f"Recall@4: {avg_recall:.2%} (Avg % of active experts correctly found)")
print(f"Jaccard Score: {avg_jaccard:.2%} (Intersection over Union accuracy)")
print(f"Perfect Match Rate: {perfect_rate:.2%} (All 4 experts predicted correctly)")
print(f"-"*30)

# ==========================================
# 7. Visualization
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')
plt.title(f'Transformer Predictor (Window={WINDOW_SIZE})')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.savefig('predictor_training.png')
