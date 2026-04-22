mkdir -p data
rm -rf /home/ubuntu/FAR/holosoma/data/ds_box_data
mkdir -p data/ds_box_data
rsync -avh /nfs/zzzihanw/ds_box_data/scale_mix_all data/ds_box_data