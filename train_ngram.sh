CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7
learning_rate=5e-5  # bert:5e-5  transformer:5e-3  swag(bert):2e-5
num_epochs=4    # bert:4 transformer:10
batch_size=16  
model_type='bert'  # bert, transformer
data_type='yelp'  # yelp,swag,amazon     
window_size=32
seeds=3047
# accumulation_steps=1
num_classes=5  # mutipleChoice(swag):4, classifier:5

#'shuffle' 'mjp_unk_drloc' 'mjp_unk' 'wope'/ 'shuffle_dynamic'    

for minlen in 180
do
  maxlen=$((minlen+220))
  for train_type in 'wope' 
  do
      if [ "$train_type" = "shuffle" ] || [ "$train_type" = "wope" ]; then
          use_unk=0
          use_drloc=0
      elif [ "$train_type" = "mjp_unk" ]; then
          use_unk=1
          use_drloc=0
      elif [ "$train_type" = "mjp_unk_drloc" ]; then
          use_unk=1
          use_drloc=1
      elif [ "$train_type" = "shuffle_dynamic" ]; then
          use_unk=0
          use_drloc=1
      else
          echo "Unknown train_type: $train_type"
          exit 1
      fi

      for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
      do
          command="python3 -m torch.distributed.run --nproc_per_node 7 --master_port 12345 BERT/main_ngram.py \
                   -learning_rate $learning_rate \
                   -batch_size $batch_size \
                   -use_unk $use_unk \
                   -use_drloc $use_drloc \
                   -model_type $model_type \
                   -data_type $data_type \
                   -shuffle_ratio $ratio \
                   -minlen $minlen \
                   -maxlen $maxlen \
                   -num_epochs $num_epochs \
                   -window_size $window_size \
                   -seeds $seeds \
                   -num_classes $num_classes \
                   -train_type $train_type"
          echo "Executing command: $command"
          $command
      done
  done
done









