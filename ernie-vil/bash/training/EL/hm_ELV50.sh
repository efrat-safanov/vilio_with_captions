#bash -x ./env.sh

### VGATTR 50

mv ./data/hm/hm_vg10100.tsv ./data/hm/HM_gt_img.tsv
mv ./data/hm/hm_vg5050.tsv ./data/hm/HM_img.tsv

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./data/ernielarge/params \
train \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_2500train \
./data/log \
dev_seen ELV50 False

# Save Space

rm -r ./data/hm/img

# SUB 1

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250train \
trains1 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250trains1 \
./data/log \
dev_seens1 ELV50 False

# SUB2

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250train \
trains2 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250trains2 \
./data/log \
dev_seens2 ELV50 False

# SUB3

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250train \
trains3 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250trains3 \
./data/log \
dev_seens3 ELV50 False

##################### TRAINDEV


bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./data/ernielarge/params \
traindev \
2500

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_2500traindev \
./data/log \
test_seen ELV50 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_2500traindev \
./data/log \
test_unseen ELV50 False

# Midsave

#cp -r ./output_hm/step_1250 ./data/

# SUB 1

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindev \
traindevs1 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs1 \
./data/log \
test_seens1 ELV50 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs1 \
./data/log \
test_unseens1 ELV50 False

# SUB2

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindev \
traindevs2 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs2 \
./data/log \
test_seens2 ELV50 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs2 \
./data/log \
test_unseens2 ELV50 False

# SUB3

bash run_finetuning.sh hm conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindev \
traindevs3 \
1250

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs3 \
./data/log \
test_seens3 ELV50 False

bash run_inference.sh hm "" val conf/hm/model_conf_hm \
./data/ernielarge/vocab.txt \
./data/ernielarge/ernie_vil.large.json \
./output_hm/step_1250traindevs3 \
./data/log \
test_unseens3 ELV50 True