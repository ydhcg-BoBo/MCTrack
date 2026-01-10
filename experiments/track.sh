cd src

python test_rgbt.py --task tracking --modal RGB-T --test_mot_rgbt True --exp_id og_result --dataset mot_rgbt --dataset_version mot_rgbt --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model model.pth

cd ..