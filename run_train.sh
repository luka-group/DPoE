
data='ag'
poison_method='addsent'
clean_path='data/clean/agnews'
poison_path='data/poisoned/'$poison_method'/agnews'
path='results/agnews/'$poison_method'/poe_w_rdrop'


for l in 2 3 4 5
do
  for small_lr in 0.00008
  do
    for ra in 3.0
    do

python poe_not_fix_rdrop.py \
  --data $data \
  --poison_data_path $poison_path \
  --clean_data_path $clean_path \
  --model_name "bert-base-uncased" \
  --epoch 3 \
  --lr 2e-5 \
  --gpu 1 \
  --batch_size 16 \
  --poe_alpha 1.0 \
  --rdrop_alpha $ra \
  --temperature 1.0 \
  --do_PoE True \
  --rdrop_mode_2 True \
  --small_lr $small_lr \
  --num_hidden_layers $l \
  --dropout_prob 0.1 \
  --result_path $path \

done
done
done

