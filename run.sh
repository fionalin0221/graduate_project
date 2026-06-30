cd dataset/
python3 crop_wsi_new.py
python3 choose_region_new.py
python3 combine_label_csv.py
cd ..
python3 -m trainer.run