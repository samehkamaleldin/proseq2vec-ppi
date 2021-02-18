mkdir -p ./data
wget https://github.com/samehkamaleldin/benchmarks/releases/download/p2v-ppi/ppi_datasets.zip -O ./data/ppi_datasets.zip
cd ./data
unzip ./ppi_datasets.zip
rm ./ppi_datasets.zip
cd ..