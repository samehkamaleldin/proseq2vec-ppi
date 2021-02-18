# ProSeq2Vec
Protein sequence embeddings for predicting protein-protein interactions.


### Downloading the benchmarking datasets
to download the benchmarking datasets you need to run the download script as follows:
``` shell
sh ./scripts/download_data.sh
```

This will download and extract the `hamp15`, `you14` and `swissprot-ppi` benchmarking datasets to the `data` directory.


### Usage
> Before using the `proseq2vec-ppi` model you will need to install the `proseq2vec` package. Also, if you want to use 
any of the benchmarking datasets wou need to download these benchmarks as shown in the previous sections.

There are multiple examples that demonstrate simple use cases for using the `proseq2vec-ppi` model to predict 
protein-protein interaction data in the `examples` directory, please check the basic example at `./examples/basic_example.py`.