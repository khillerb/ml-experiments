import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load data
data = pd.read_csv("superstore_sales_train.csv")

# Handle missing values (if any)
data.fillna(data.mode(), inplace=True)

# Handle dates as separate features
data["Order Date"] = pd.to_datetime(data["Order Date"], dayfirst=True)
data["Ship Date"] = pd.to_datetime(data["Ship Date"], dayfirst=True)
data["Order Year"] = data["Order Date"].dt.year
data["Order Month"] = data["Order Date"].dt.month
data["Order Day"] = data["Order Date"].dt.day
data["Ship Year"] = data["Ship Date"].dt.year
data["Ship Month"] = data["Ship Date"].dt.month
data["Ship Day"] = data["Ship Date"].dt.day
data = data.drop(columns=["Order Date", "Ship Date"])

# Label encode categorical features, excluding numerical fields
categorical_cols = [
    "Ship Mode",
    "Segment",
    "Country",
    "City",
    "Category",
    "Sub-Category",
    "Product Name",
]  # Exclude Row ID
encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    encoders[col] = encoder

# Convert numerical fields to NumPy arrays
numerical_cols = [
    "Row ID",
    "Order Year",
    "Order Month",
    "Order Day",
    "Ship Year",
    "Ship Month",
    "Ship Day",
    "Sales",
]

# Combine numerical and encoded categorical data
data_with_labels = np.hstack(
    (data[numerical_cols].values, data[categorical_cols].values)
)


import torch
from torch import nn

learning_rate = 1e-2  # Lower learning rate for smaller steps
epochs = 1800  # Increase epochs for more training iterations


# Xavier initialization for weights
def weight_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


# Early stopping (optional)
class EarlyStopping(object):
    def __init__(self, mode="min", patience=5):
        self.mode = mode
        self.patience = patience
        self.best_loss = None
        self.best_epoch = None
        self.wait = 0

    def step(self, loss):
        if self.mode == "min" and (self.best_loss is None or loss < self.best_loss):
            self.best_loss = loss
            self.best_epoch = epoch
            self.wait = 0
        elif self.mode == "max" and (self.best_loss is None or loss > self.best_loss):
            self.best_loss = loss
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            return True


early_stopping = EarlyStopping(patience=5)


# Regularization (L1/L2 penalty)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(data_with_labels.shape[1], 512),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(p=0.01),  # Dropout after first activation
            nn.BatchNorm1d(512),  # Batch norm after dropout
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(p=0.01),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),  # Bottleneck layer (embedding size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(p=0.01),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Dropout(p=0.01),
            nn.BatchNorm1d(512),
            nn.Linear(
                512, data_with_labels.shape[1]
            ),  # No batch norm in the last layer
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create model instance with L1/L2 penalty (optional)
model = Autoencoder()
layer_inv = False
model.apply(
    weight_init
)  # Apply weight initialization to all modules  # Instantiate the model
criterion = nn.MSELoss()  # Reconstruction loss
l1_lambda = 0.00005  # Regularization parameter (adjust as needed)
l1_loss = nn.L1Loss(reduction="sum")  # L1 penalty
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, factor=0.18, patience=8, verbose=True, min_lr=1e-9
)  # Reduce LR on plateau
data_tensor = torch.from_numpy(data_with_labels).float()
# Train model
for epoch in range(epochs):
    # Forward pass
    outputs = model(data_tensor)
    recon_loss = criterion(outputs, data_tensor)
    total_loss = recon_loss
    # Early stopping check (optional)
    if epoch > 1500:
        if early_stopping.step(total_loss.item()):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Add L1 penalty to total loss (optional)
    if l1_lambda > 0:
        for param in model.parameters():
            total_loss += l1_lambda * l1_loss(param, torch.zeros_like(param))

    # Backward pass and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss)  # Update learning rate after each step
    if layer_inv:
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(
                    f"Layer: {name}, Gradient mean: {param.grad.mean():.4f}, Gradient std: {param.grad.std():.4f}"
                )

    # Print loss (optional)
    print(f"Epoch: {epoch+1}, loss: {total_loss.item():.4f}")


from sklearn.linear_model import LogisticRegression


embeddings = model.encoder(data_tensor)  # Get embeddings from the encoder
y_train = data.index.values  # Row IDs as labels
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(
    embeddings.detach().numpy()
)  # Scale embeddings


from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")  # or any other checkpoint


# Define your linear mapping model
class LinearMappingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearMappingModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


encoder_output_dim = 64  # Dimensions of autoencoder encoder embeddings
gpt2_embedding_dim = 768  # Dimensions of GPT-2 embeddings

# Initialize linear mapping model
linear_mapping_model = LinearMappingModel(encoder_output_dim, gpt2_embedding_dim)

criterion_mapping = nn.MSELoss()
optimizer_mapping = torch.optim.Adam(
    linear_mapping_model.parameters(), lr=learning_rate
)

# Move the model and data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
linear_mapping_model.to(device)
gpt2_model.to(device)
data_tensor = data_tensor.to(device)

for epoch in range(epochs):
    # Forward pass
    encoder_embeddings = model.encoder(data_tensor)

    # Map embeddings to 768 dimensions
    mapped_embeddings = linear_mapping_model(encoder_embeddings)

    loss_mapping = criterion_mapping(mapped_embeddings, closest_gpt2_embeddings)

    # Backward pass and optimize
    optimizer_mapping.zero_grad()
    loss_mapping.backward()
    optimizer_mapping.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Mapping Loss: {loss_mapping.item():.4f}")

# Extract closest matching token
input_embeddings = embeddings[0].unsqueeze(0)  # Using first embedding as an example
mapped_embedding = linear_mapping_model(input_embeddings)
closest_token = tokenizer.convert_ids_to_tokens(mapped_embedding)

print(f"Decoded Token: {closest_token}")
print(data.iloc[[0]])

print(f"Decoded Token: {closest_token}")
print(data.iloc[[0]])
