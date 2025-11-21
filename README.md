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


FOR RORQUAL

module load gcc
module load arrow/21.0.0
module load python/3.13

python -m venv ~/danastarenv
source ~/danastarenv/bin/activate
pip install -r rorqualrequirements.txt


#THE FOLLOWING IS NEEDED FOR OFFLINE USE OF TIKTOKEN
mkdir -p ~/tiktoken_cache

source ~/danastarenv/bin/activate
python -c "
import hashlib
vocab_bpe_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe'
encoder_json_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json'
vocab_bpe_key = hashlib.sha1(vocab_bpe_url.encode()).hexdigest()
encoder_json_key = hashlib.sha1(encoder_json_url.encode()).hexdigest()
print('vocab.bpe cache key:', vocab_bpe_key)
print('encoder.json cache key:', encoder_json_key)
"

python -c "
import requests
import os

vocab_bpe_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe'
encoder_json_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json'

print('Downloading vocab.bpe...')
r1 = requests.get(vocab_bpe_url)
with open('$HOME/tiktoken_cache/6d1cbeee0f20b3d9449abfede4726ed8212e3aee', 'wb') as f:
    f.write(r1.content)

print('Downloading encoder.json...')
r2 = requests.get(encoder_json_url)
with open('$HOME/tiktoken_cache/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6', 'wb') as f:
    f.write(r2.content)

print('Files downloaded successfully!')
"

#ADD THIS TO ANY SCRIPT THAT NEEDS TO USE TIKTOKEN
export TIKTOKEN_CACHE_DIR=$HOME/tiktoken_cache

#############

Now we need to download and tokenize the data

Go to the danastar folder:
source $HOME/danastarenv/bin/activate
cd $HOME/danastar

python src/data/download_fineweb_100bt.py --local-dir=$HOME/links/scratch/fineweb/

mkdir $HOME/danastar/logs

sbatch scripts/rorqual_get_fineweb100.sh

