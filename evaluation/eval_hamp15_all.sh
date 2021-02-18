mkdir -p ../logs/txt_logs/hamp15/

echo "+-----------------------------+----------------+" >> ../logs/txt_logs/hamp15/human_eval.txt
echo "+-----------------------------+----------------+" >> ../logs/txt_logs/hamp15/yeast_eval.txt

for run in {0..9}
  do
    for fold in {0..9}
      do
         python ./hamp15_experiment.py -s human -r $run -f $fold >> ../logs/txt_logs/hamp15/human_eval.txt
         python ./hamp15_experiment.py -s yeast -r $run -f $fold >> ../logs/txt_logs/hamp15/yeast_eval.txt
      done
  done

echo "+-----------------------------+----------------+" >> ../logs/txt_logs/hamp15/human_eval.txt
echo "+-----------------------------+----------------+" >> ../logs/txt_logs/hamp15/yeast_eval.txt