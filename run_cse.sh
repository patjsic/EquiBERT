python multitask_classifier.py --mode default --train_canonical --option finetune --epochs 5 --use_gpu --batch_size 64 --canonical_epochs 10 \
                                --sst_dev_out "canonical_pred/sst-dev-output.csv" --sst_test_out "canonical_pred/sst-test-output.csv" \
                                --para_dev_out "canonical_pred/para-dev-output.csv" --para_test_out "canonical_pred/para-test-output.csv"\
                                --sts_dev_out "canonical_pred/sts-dev-output.csv" --sts_test_out "canonical_pred/sts-test-output.csv"

python multitask_classifier.py --mode default --option finetune --epochs 5 --use_gpu --batch_size 64  \
                                --sst_dev_out "vanilla_pred/sst-dev-output.csv" --sst_test_out "vanilla_pred/sst-test-output.csv" \
                                --para_dev_out "vanilla_pred/para-dev-output.csv" --para_test_out "vanilla_pred/para-test-output.csv"\
                                --sts_dev_out "vanilla_pred/sts-dev-output.csv" --sts_test_out "vanilla_pred/sts-test-output.csv"

python multitask_classifier.py --mode simcse --option finetune --epochs 5 --use_gpu --batch_size 64 --lambda1 0.25 \
                                --sst_dev_out "simcse_pred/sst-dev-output.csv" --sst_test_out "simcse_pred/sst-test-output.csv" \
                                --para_dev_out "simcse_pred/para-dev-output.csv" --para_test_out "simcse_pred/para-test-output.csv"\
                                --sts_dev_out "simcse_pred/sts-dev-output.csv" --sts_test_out "simcse_pred/sts-test-output.csv"
                  
python multitask_classifier.py --mode simcse --train_canonical --option finetune --epochs 5 --use_gpu --batch_size 64 --lambda1 0.25 --canonical_epochs 10\
                                --sst_dev_out "simcse_canon_pred/sst-dev-output.csv" --sst_test_out "simcse_canon_pred/sst-test-output.csv" \
                                --para_dev_out "simcse_canon_pred/para-dev-output.csv" --para_test_out "simcse_pred/para-test-output.csv"\
                                --sts_dev_out "simcse_canon_pred/sts-dev-output.csv" --sts_test_out "simcse_canon_pred/sts-test-output.csv"         
#--lambda1 0.25
