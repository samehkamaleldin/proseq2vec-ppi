# make essential directories
log_dir="../logs/txt_logs/scale_test"
mkdir -p  $log_dir

# ------------------------------------------------------------------------------------------------
# Scalability in respect to the grow of the ppi size
# ------------------------------------------------------------------------------------------------
data_size_list=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
rm -f $log_dir/data_size.txt
touch $log_dir/data_size.txt
for data_size in "${data_size_list[@]}"
do
  python ./report_runtime.py --data_size "${data_size}" --tag "data=${data_size}" >> $log_dir/data_size.txt
done
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Scalability in respect to the grow of the fixed sequence size
# ------------------------------------------------------------------------------------------------
seq_size_list=( 300 600 900 1200 1500 1800 2100 2400 2700 3000)
rm -f $log_dir/seq_size.txt
touch $log_dir/seq_size.txt
for seq_size in "${seq_size_list[@]}"
do
  python ./report_runtime.py --seq "${seq_size}" --tag "seq_size=${seq_size}" >> $log_dir/seq_size.txt
done
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Scalability in respect to the grow of the model's number of layers
# ------------------------------------------------------------------------------------------------
nb_layer_list=( 2 3 4 5 6 7 8 9 10 11)
rm -f $log_dir/nb_layers.txt
touch $log_dir/nb_layers.txt
for nb_layers in "${nb_layer_list[@]}"
do
  python ./report_runtime.py --nb_layers "${nb_layers}" --tag "nb_layers=${nb_layers}" >> $log_dir/nb_layers.txt
done
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Scalability in respect to the grow of the model's sequence embedding size
# ------------------------------------------------------------------------------------------------
seq_k_list=( 32 64 96 128 160 192 224 256 288 320)
rm -f $log_dir/seq_k.txt
touch $log_dir/seq_k.txt
for seq_k in "${seq_k_list[@]}"
do
  python ./report_runtime.py --seq_k "${seq_k}" --tag "seq_k=${seq_k}" >> $log_dir/seq_k.txt
done
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Scalability in respect to the grow of the model's amino acid embedding size
# ------------------------------------------------------------------------------------------------
aa_k_list=( 2 4 6 8 10 12 14 16 18 20)
rm -f $log_dir/aa_k.txt
touch $log_dir/aa_k.txt
for aa_k in "${aa_k_list[@]}"
do
  python ./report_runtime.py --aa_k "${aa_k}" --tag "aa_k=${aa_k}" >> $log_dir/aa_k.txt
done
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Scalability in respect to the grow of the model's LSTM hidden size
# ------------------------------------------------------------------------------------------------
hidden_list=( 32 64 96 128 160 192 224 256 288 320)
rm -f $log_dir/hidden.txt
touch $log_dir/hidden.txt
for hidden in "${hidden_list[@]}"
do
  python ./report_runtime.py --hidden "${hidden}" --tag "hidden=${hidden}" >> $log_dir/hidden.txt
done
# ------------------------------------------------------------------------------------------------