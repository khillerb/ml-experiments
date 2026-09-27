import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
from torch_geometric.data import Data
from datasets import load_dataset, load_metric
import numpy as np
from torch_geometric.utils import subgraph
from tqdm import tqdm
import time
from multiprocessing import Pool


class GraphDataset(Dataset):

    def __init__(
        self,
        knowledge_graph_path,
        corpus_path,
        e_alias_path,
        r_alias_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
        print("Loading knowledge graph...")
        (
            self.knowledge_graph,
            self.entity_mapping,
            self.relation_mapping,
        ) = self.load_knowledge_graph(knowledge_graph_path)
        print("Loading corpus...")
        self.corpus = self.load_corpus(corpus_path)
        print("Loading entity aliases...")
        self.e_aliases = self.load_entity_aliases(e_alias_path)
        print("Loading relation aliases...")
        self.r_aliases = self.load_relation_aliases(r_alias_path)
        print("Creating graph...")
        self.create_graph()
        print("Generating sequences...")
        self.graph_data, self.input_sequence, self.target_sequence = (
            self.generate_sequences_PLL()
        )

    def load_knowledge_graph(self, path):
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["source", "relation", "target"],
        )
        entity_ids = pd.concat([df["source"], df["target"]]).unique()
        relation_ids = df["relation"].unique()
        entity_mapping = {id: i for i, id in enumerate(entity_ids)}
        relation_mapping = {id: i for i, id in enumerate(relation_ids)}
        df["source"] = df["source"].map(entity_mapping)
        df["target"] = df["target"].map(entity_mapping)
        df["relation"] = df["relation"].map(relation_mapping)
        edge_index = torch.tensor(df[["source", "target"]].values.T, dtype=torch.long)
        edge_type = torch.tensor(df["relation"].values, dtype=torch.long)
        kg_data = Data(edge_index=edge_index, edge_attr=edge_type)
        return kg_data, entity_mapping, relation_mapping

    def load_corpus(self, path):
        # Initialize a dict to hold filtered corpus data
        filtered_corpus = {}
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.split("\t", 1)
                entity_id, description = parts
                filtered_corpus[entity_id] = description.strip()
        return filtered_corpus

    def load_aliases(self, file_path, id_column_name):
        # Initialize a list to hold the rows as dictionaries
        rows_list = []

        # Open the file and process each line
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # Split the line into parts using tabs
                parts = line.strip().split("\t")
                # The first part is the ID, the rest are aliases
                id_value = parts[0]
                aliases = parts[1:]
                # Append to list as a dictionary
                rows_list.append({id_column_name: id_value, "aliases": aliases})

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(rows_list)

        # Normalize the DataFrame if needed (optional)
        # This step is optional and can be adjusted based on how you want to use the aliases

        # Create a dictionary mapping from ID to a list of its aliases
        aliases_dict = df.set_index(id_column_name)["aliases"].to_dict()

        return aliases_dict

    def load_entity_aliases(self, file_path):
        return self.load_aliases(file_path, "entity_id")

    def load_relation_aliases(self, file_path):
        return self.load_aliases(file_path, "relation_id")

    def create_graph(self):
        entity_alias_to_index = {
            alias: i
            for i, (entity_id, aliases) in enumerate(self.e_aliases.items())
            for alias in aliases
        }
        relation_alias_to_index = {
            alias: i
            for i, (relation_id, aliases) in enumerate(self.r_aliases.items())
            for alias in aliases
        }
        node_features = torch.full(
            (len(self.entity_mapping), 1), -1
        )  # Placeholder for no alias
        edge_features = torch.full(
            (self.knowledge_graph.edge_index.size(1), 1), -1
        )  # Placeholder for no relation alias

        # Update node features with alias indices
        for entity_id, aliases in self.e_aliases.items():
            if entity_id in self.entity_mapping:  # Ensure the entity is in the graph
                node_index = self.entity_mapping[entity_id]
                # Use the first alias as the representative alias (or implement a different strategy as needed)
                alias_index = entity_alias_to_index[aliases[0]] if aliases else -1
                node_features[node_index] = alias_index
        for i, (source, target, relation) in enumerate(
            zip(
                self.knowledge_graph.edge_index[0],
                self.knowledge_graph.edge_index[1],
                self.knowledge_graph.edge_attr,
            )
        ):
            relation_id = (
                relation.item()
            )  # Assuming edge_attr stores relation IDs directly
            relation_aliases = self.r_aliases.get(relation_id, [])
            alias_index = (
                relation_alias_to_index[relation_aliases[0]] if relation_aliases else -1
            )
            edge_features[i] = alias_index

        # Step 3: Attach Encoded Features to the Graph
        self.knowledge_graph.x = (
            node_features  # Node features representing entity aliases
        )
        self.knowledge_graph.edge_attr = (
            edge_features  # Edge features representing relation aliases
        )

    def extract_entities_from_text(self, text):
        # Create a reverse mapping from alias to entity ID
        alias_to_entity_id = {
            alias: entity_id
            for entity_id, aliases in self.e_aliases.items()
            for alias in aliases
        }

        # Tokenize the text and look for potential entity mentions by aliases
        tokens = (
            text.split()
        )  # Simple split; consider more sophisticated tokenization as needed
        entities_mentioned = set()

        for token in tokens:
            if token in alias_to_entity_id:
                entities_mentioned.add(alias_to_entity_id[token])

        return list(entities_mentioned)

    def extract_graph_data(self, entities_mentioned):
        # Convert entity mentions (which could include aliases) to graph node IDs
        entity_ids = [
            self.entity_mapping[entity]
            for entity in entities_mentioned
            if entity in self.entity_mapping
        ]

        # Collect node indices for the mentioned entities
        subgraph_nodes = torch.tensor(list(set(entity_ids)), dtype=torch.long)

        # Extract the subgraph for the mentioned entities
        subgraph_edge_index, subgraph_edge_attr = subgraph(
            subgraph_nodes,
            self.knowledge_graph.edge_index,
            edge_attr=self.knowledge_graph.edge_attr,
            relabel_nodes=True,
            num_nodes=self.knowledge_graph.num_nodes,
        )

        # Create a new graph data object for the subgraph
        subgraph_data = Data(
            edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr
        )
        return subgraph_data

    def generate_sequences(self, sequence_length=1024, max_model_sequence_length=4096):
        input_sequence = []
        masks = []
        graph_data = []
        target_sequence = []
        for entity_id, description in tqdm(self.corpus.items()):
            tokens = self.tokenizer.encode(
                description,
                add_special_tokens=True,
                max_length=max_model_sequence_length,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            mask = tokens["attention_mask"]
            tokens = tokens["input_ids"]
            for i in range(0, len(tokens), sequence_length):
                window = tokens[i : i + sequence_length]
                if len(window) < sequence_length:
                    break  # Break if the window is too short
                input_sequence.append(window[:-1])
                target_sequence.append(window[1:])
                masks.append(mask[i : i + sequence_length])
                subgraph_data = self.extract_graph_data(
                    self.extract_entities_from_text(description)
                )
                graph_data.append(subgraph_data)
        return graph_data, masks, input_sequence, target_sequence

    def worker_function(self, item):
        entity_id, description = item
        tokens = self.tokenizer.encode(
            description, add_special_tokens=True, max_length=4096, truncation=True
        )
        mask = tokens["attention_mask"]
        tokens = tokens["input_ids"]
        sequence_length = 1024
        input_sequence = []
        masks = []
        graph_data = []
        target_sequence = []
        for i in range(0, len(tokens), sequence_length):
            window = tokens[i : i + sequence_length]
            if len(window) < sequence_length:
                break  # Skip the last window if it's shorter than the sequence_length
            input_sequence.append(window[:-1])
            target_sequence.append(window[1:])
            masks.append(mask[i : i + sequence_length])
            subgraph_data = self.extract_graph_data(
                self.extract_entities_from_text(description)
            )
            graph_data.append(subgraph_data)
        return graph_data, masks, input_sequence, target_sequence

    def generate_sequences_PLL(self, num_processes=5):
        # Prepare data for multiprocessing
        items = list(self.corpus.items())

        # Initialize multiprocessing Pool
        with Pool(processes=num_processes) as pool:
            results = list(
                tqdm(pool.imap(self.worker_function, items), total=len(items))
            )

        # Initialize containers for the combined results
        combined_graph_data = []
        combined_masks = []
        combined_input_sequence = []
        combined_target_sequence = []

        # Combine results from all processes
        for graph_data, masks, input_sequence, target_sequence in results:
            combined_graph_data.extend(graph_data)
            combined_masks.extend(masks)
            combined_input_sequence.extend(input_sequence)
            combined_target_sequence.extend(target_sequence)

        return (
            combined_graph_data,
            combined_masks,
            combined_input_sequence,
            combined_target_sequence,
        )

    def __len__(self):
        return len(self.input_sequence)

    def __getitem__(self, idx):
        # Returns the graph data, input sequence, and target sequence for a given index
        return self.input_sequence[idx], self.graph_data[idx], self.target_sequence[idx]


def scaled_dot_product_attention(query, key, value, mask=None, attention_mods=None):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # Apply modifications to attention logits if provided
    if attention_mods is not None:
        scaled_attention_logits += attention_mods

    attention_weights = nn.functional.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class ParamGenerator(nn.Module):
    def __init__(self, d_model):
        super(ParamGenerator, self).__init__()
        self.gat = GATConv(d_model, d_model)
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, graph_data):
        attention_data = self.gat(graph_data.x, graph_data.edge_index)
        return self.network(attention_data)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, param_mods=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Pass param_mods as attention_mods to scaled_dot_product_attention
        attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, attention_mods=param_mods
        )
        attention = attention.permute(0, 2, 1, 3).contiguous()
        out = attention.view(batch_size, -1, self.d_model)
        out = self.dropout(out)
        out = self.dense(out)
        return out, attention_weights


class GPTLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(GPTLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )
        self.param_gen = ParamGenerator(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        # GNN within transformer layer
        # self.gnn_within = GCNConv(d_model, d_model)
        # self._init_weights()

    def _init_weights(self):
        # Using Xavier initialization as an example
        for m in [self.gnn_within]:  # Add other layers as needed, e.g., self.gnn_after
            if isinstance(m, GCNConv):
                # Initialize weights for GCNConv layers
                if hasattr(m, "weight"):
                    torch.nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, mask, graph_data):
        # Generate parameter modifications from graph data
        param_mods = self.param_gen(
            graph_data
        )  # Assuming param_gen is added to __init__

        attn_output, _ = self.mha(x, x, x, mask, param_mods)
        x = self.norm1(x + self.dropout(attn_output))

        # GNN within transformer layer
        # x = self.gnn_within(graph_data.x, graph_data.edge_attr) + x

        ffn_output = self.dropout(self.ffn(x))
        x = self.norm2(x + self.dropout(ffn_output))


class GPTGNN(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super(GPTGNN, self).__init__()
        self.layers = nn.ModuleList(
            [GPTLayer(d_model, num_heads) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, graph_data):
        for layer in self.layers:
            x = layer(x, mask, graph_data)
        return self.final_layer_norm(x)


def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    _start = time.time()
    # Initialize your model
    print("Initializing GPT-GNN model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt_gnn = GPTGNN(d_model=1296, num_heads=24, num_layers=12).bfloat16().to(device)
    print(f"Total trainable parameters: {get_parameter_count(gpt_gnn)}")
    param_list = list(gpt_gnn.parameters())
    learning_rate = 1e-4
    optimizer = optim.AdamW(param_list, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    _end = time.time()
    elapsed_time = _end - _start
    _start = _end
    print(f"Segment took {elapsed_time} seconds.")
    num_epochs = 4
    kg_path = (
        "C:/Users/thata/OneDrive/Desktop/mushroomID/wikidata5m_inductive_train.txt"
    )
    corpus_path = "C:/Users/thata/OneDrive/Desktop/mushroomID/wikidata5m_text.txt"
    entity_aliases_path = (
        "C:/Users/thata/OneDrive/Desktop/mushroomID/wikidata5m_entity.txt"
    )
    relation_aliases_path = (
        "C:/Users/thata/OneDrive/Desktop/mushroomID/wikidata5m_relation.txt"
    )
    print("Loading dataset")
    dataset = GraphDataset(
        kg_path, corpus_path, entity_aliases_path, relation_aliases_path
    )
    pickle.dump(
        dataset, open("C:/Users/thata/OneDrive/Desktop/mushroomID/dataset.pkl", "wb")
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    _end = time.time()
    elapsed_time = _end - _start
    _start = _end
    print(f"Segment took {elapsed_time} seconds.")
    print("Dataset loaded")
    # Pre-training loop
    print("Pre-training GPT-GNN model")
    gpt_gnn.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for input_seq, graph_data, targets in progress_bar:
            input_seq, targets = input_seq.to(device), targets.to(
                device
            )  # Move data to GPU

            optimizer.zero_grad()

            # Forward pass and loss computation
            outputs = gpt_gnn(input_seq, mask=None, graph_data=graph_data)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save checkpoint after each epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": gpt_gnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"C:/Users/thata/OneDrive/Desktop/mushroomID/checkpoint_epoch_{epoch+1}.pth",
        )
        _end = time.time()
        elapsed_time = _end - _start
        _start = _end
        print(f"Segment took {elapsed_time} seconds.")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

        # Optional: Evaluate the model on a validation set

    def evaluate_model(
        model, tokenizer, dataset_name, split="validation", num_samples=100
    ):
        model.eval()

        # Load dataset and metric
        dataset = load_dataset(dataset_name, split=split)
        metric = load_metric("accuracy")

        # Sample from the dataset
        dataset = dataset.shuffle().select(range(num_samples))

        # Evaluation loop
        for item in dataset:
            inputs = tokenizer(item["context"], return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_length=inputs.input_ids.size(-1) + 1
                )

            pred = tokenizer.decode(output[0], skip_special_tokens=True)
            metric.add_batch(predictions=[pred], references=[item["label"]])

        return metric.compute()

    """
    # Evaluate Mistral-7B on HellaSwag and MMLU
    mistral_hellaswag = evaluate_model(mistral, tokenizer, "Rowan/hellaswag")
    mistral_mmlu = evaluate_model(mistral, tokenizer, "cais/mmlu")

    print("Mistral-7B HellaSwag Accuracy:", mistral_hellaswag)
    print("Mistral-7B MMLU Accuracy:", mistral_mmlu)
    # Assuming gpt_gnn_model is your model instance and gpt_gnn_tokenizer is your tokenizer

    gpt_gnn_hellaswag = evaluate_model(gpt_gnn, tokenizer, "Rowan/hellaswag")
    gpt_gnn_mmlu = evaluate_model(gpt_gnn, tokenizer, "cais/mmlu")

    print("GPT-GNN HellaSwag Accuracy:", gpt_gnn_hellaswag)
    print("GPT-GNN MMLU Accuracy:", gpt_gnn_mmlu)


    import matplotlib.pyplot as plt

    # Data for plotting
    models = ["Mistral-7B", "GPT-GNN"]
    hellaswag_scores = [mistral_hellaswag["accuracy"], gpt_gnn_hellaswag["accuracy"]]
    mmlu_scores = [mistral_mmlu["accuracy"], gpt_gnn_mmlu["accuracy"]]

    # Plotting
    fig, ax = plt.subplots()
    index = np.arange(len(models))
    bar_width = 0.35

    rects1 = ax.bar(index, hellaswag_scores, bar_width, label="HellaSwag")
    rects2 = ax.bar(index + bar_width, mmlu_scores, bar_width, label="MMLU")

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison on HellaSwag and MMLU")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(models)
    ax.legend()

    plt.show()
    """
