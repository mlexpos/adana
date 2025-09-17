# Codebase mostly taken from: "Benchmarking Optimizers for Large Language Model Pretraining"
[![arXiv](https://img.shields.io/badge/arXiv-2401.06766-b31b1b.svg)](https://arxiv.org/abs/2509.01440)
[![BibTeX](https://img.shields.io/badge/BibTeX-Citation-green)](https://arxiv.org/bibtex/2509.01440)

## Quickstart 

Load modules (tamia cluster):

```
module load arrow/21.0.0
module load python/3.10.13
module load httpproxy
```

Create a uv environment and install dependencies:

```
uv venv llm
source llm/bin/activate
uv pip install -r requirements.txt
```

Then download 10B tokens from fineweb (takes around 1h with 16 CPUs to download and tokenize)

```
uv run python src/data/fineweb.py
```

and launch one of 

```
sbatch mila.sh
sbatch tamia.sh
```