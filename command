3-1

CUDA_VISIBLE_DEVICES=0 nohup python -u run.py --data WN18RR --batch 256 \
--hid_drop 0.5 --feat_drop 0.1 --lr 0.00125 --inp_drop 0.2 \
--gpu 0 --name wn18rr_t --way t --train_strategy one_to_n \
--epoch_c 3000 > res/31_wn18rr.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --data FB15k-237 --batch 256 \
--hid_drop 0.5 --feat_drop 0.2 --lr 0.001 --inp_drop 0.3 \
--gpu 1 --name fb15k237_t --way t --trai
n_strategy one_to_x \
--epoch_c 3000 --outfolder ./checkpoints/fb/ --batch_cagnn 272115 > res/31_fb15k237.log &

试一试encoder500 decoder500
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --data WN18RR --batch 256 \
--hid_drop 0.5 --feat_drop 0.1 --lr 0.001 --inp_drop 0.2 \
--way t --train_strategy one_to_n \
--epoch_c 500 > res/34_wn18rr.log &


