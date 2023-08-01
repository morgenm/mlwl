# mlwl

**_Generate wordlists for hash cracking using machine learning (Word2Vec)._**

## About
This program will generate a wordlist for hash cracking. The output wordlist is determined by user-defined keywords. Word2Vec is used to determine the most similar words for each word in the wordlist.

## Usage
Rust and [maturin](https://github.com/PyO3/maturin) are required to run mlwl. Additionally several Python libraries need to be installed to run. Run the following commands to prepare the program:
```bash
git clone https://github.com/morgenm/mlwl
cd mlwl
pip install -r requirements.txt
make
```

To run the program and see the available arguments:
```bash
python3 python/mlwl/mlwl.py
```

On first run, mlwl will download and save all supported Word2Vec models. Subsequent runs will be much faster than the initial run because of this.