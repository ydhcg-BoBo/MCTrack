cd src
python -u main.py --task tracking --modal RGB-T --save_all  --exp_id MCTrack --dataset mot_rgbt --dataset_version mot_rgbt --batch_size 48 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --num_epochs 20 --lr_step 10 --lr 1.25e-4

cd ..