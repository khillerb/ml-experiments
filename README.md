# ml-experiments

This is a collection of machine learning experiments from 2023 and 2024. They
used to live inside the `mushroomID` repository and were moved here so that
project could stand on its own.

None of these are finished pipelines. Each script was written to answer one
question, runs from top to bottom, and often has its paths written right into
the code. It's best to think of them as notebooks saved as `.py` files.

Large data and anything based on private data are kept out of git (see
[Not in git](#not-in-git)).

## What's inside

### `llm_architectures/`: custom model building blocks

- **`MoE.py`** is a mixture-of-experts layer in PyTorch. It has multi-head
  attention, a feed-forward block, and a set of linear "experts" picked by a
  softmax gate. It also has a transformer decoder layer that uses the experts
  instead of the usual feed-forward block. It only defines the pieces and has
  no training loop.
- **`SNN.py`** is a spiking neural network. It uses leaky integrate-and-fire
  layers, which pass signals as on/off spikes over time, and a "surrogate
  gradient" trick so the spikes can still be trained with backpropagation. It
  has a small demo that checks the output shapes.
- **`customslm.py`** is a copy of Hugging Face's RWKV implementation (Apache
  2.0, original headers kept). It's here for reference only. Its relative
  imports (`from ...modeling_utils`) only work inside the `transformers`
  package itself.

### `graph-llm/`: language modeling with a knowledge graph

- **`customLLM.py`** is the biggest experiment. `GPTGNN` is a GPT-style
  decoder whose attention is shaped by graph features. Those features come
  from a graph attention (GAT) layer run over a piece of the Wikidata5m
  knowledge graph. A GCN layer is in the code too, but it's commented out. `GraphDataset` joins the graph, the text corpus,
  and the entity and relation names. The `__main__` block builds a 12-layer,
  24-head model (hidden size 1296, bf16) and trains it. **The data paths still
  point to the old `mushroomID` folder and need to be changed to
  `graph-llm/data/`.**
- **`tokenizer_map.py`** builds a lookup from tokens to GPT-2 embeddings over
  the IMDB dataset.
- **`data/`** holds the Wikidata5m files (about 3.2 GB): entities, relations,
  training triples, and the text corpus.

### `tabular/`: sales and market modeling

- **`dataenc.py`** runs a full pipeline on `superstore_sales_train.csv`. It
  pulls features out of the dates, encodes the labels, and scales the
  numbers. Next, it trains a dense autoencoder that compresses each row down
  to 64 values, with early stopping. Finally, it maps that 64-value embedding
  into GPT-2's 768-value space.
- **`mock_data_gen.py`** creates synthetic foodservice market data: operator
  types, cuisine types, sales split by metro area, and metrics like void, gap,
  and opportunity. It reads a source spreadsheet whose path is set in
  `mock_data_config.yaml`. To get started, copy `mock_data_config.yaml.example`
  and fill in your path.
- **`mock_data_3.csv`** and **`mock_data*.zip`** are generated outputs from
  several runs. They're not in git, because the numbers are made up but the
  product names come from the real source spreadsheet.
- **`test_trainer/denester.py`** flattens columns that hold JSON into regular
  columns. Its before and after CSVs are not in git, for the same reason.
- **`reward_model_output/`** is a leftover TensorBoard run from April 2024.
  The script that made it is gone.

### `agents/`: LLM agents over the sales data

- **`agent_4.py`** runs Mistral-7B-Instruct in 4-bit behind LangChain, aimed
  at questions about product and location opportunities.
- **`sql_agent.py`** turns questions into SQL with LangChain and runs them on
  AWS Athena through SQLAlchemy. Query results are staged in S3, and table
  info comes from a Glue metastore.

  To set it up, copy `athena_config.yaml.example` to `athena_config.yaml` and
  fill in your region, database, S3 staging folder, and workgroup. The real
  config is git-ignored. AWS keys come from the `AWS_ACCESS_KEY_ID` and
  `AWS_SECRET_ACCESS_KEY` environment variables.

### `calculus-regression/`: numerical methods from scratch

- **`linreg.py`** does linear regression in plain NumPy. It uses hand-written
  gradient descent, BCE and MSE losses, and matplotlib plots.
- **`linreg_ext.py`** extends it to quadratic regression and standardizes the
  features first.
- **`nonlinereg.py`** covers nonlinear and logistic versions, including a
  sigmoid that stays stable with very large or small inputs.
- **`Exploration of Basic Calculus and Linear Regression.pdf`** is the
  write-up that goes with these scripts.
- **`proj.zip`** is a zipped copy of the folder.

### `scratch/`: odds and ends

- **`autoclicker.py`** sends one prompt to `AdaptLLM/finance-LLM`.
- **`krimclicker.py`** is a PyAutoGUI loop that holds and releases the mouse.
- **`robotrader.py`** is a scratch file for a trading script and only has
  comments so far.

## Dependencies

There's no requirements file, because each experiment needs different
packages. Across all the scripts, they use `torch`, `torch_geometric`,
`transformers`, `datasets`, `bitsandbytes`, `langchain`,
`langchain_community`, `sqlalchemy`, `boto3`, `PyAthena`, `pyyaml`, `pandas`,
`numpy`, `scikit-learn`, `matplotlib`, `openpyxl`, `tqdm`, and `pyautogui`.
It's easier to install them one experiment at a time.

## Not in git

`.gitignore` leaves out:

- `graph-llm/data/`: the Wikidata5m files (about 3.2 GB). You can download
  them from the
  [Wikidata5m project page](https://deepgraphlearning.github.io/project/wikidata5m).
- `tabular/mock_data*` and the `test_trainer/` CSVs, because they're based on
  a private source spreadsheet.
- `agents/athena_config.yaml` and `tabular/mock_data_config.yaml`, the real
  configs. Copy the `.example` files to make your own.
