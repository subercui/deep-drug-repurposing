# Drug Repurposing

## Installation

`pip install -r requirements.txt`

## Predict drug candidates for covid

- Download the data [here](https://drive.google.com/drive/folders/1W9G2Zxq385FlJSWaB3-wxsmBXTpfrPl2?usp=sharing) and set the data directories in `config.json`.

- Use `python predict_drug.py -c config.json` to generate a drug list and store in a output file assigned in `config.json`.

It is recommended to put all settings in a config file, i.e. `config.json`.

## Use the biological graph object (msi)

```python
from predict_drug import run_predict
msi = run_predict()

# retrieve the nodes in graph
msi.graph.nodes

# check the connections of a node
msi.graph['DB04865']
```

## Train the hierarchical GCN (protein and pathway level)

```python
python tran_dual_gcn.py
```

Currently we are using [networkx](https://networkx.github.io/documentation/stable/). Click and find more interfaces from the documents.
