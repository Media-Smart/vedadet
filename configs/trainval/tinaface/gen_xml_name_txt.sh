
ls -l data/WIDERFace/WIDER_train/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/WIDERFace/WIDER_train/train.txt
ls -l data/WIDERFace/WIDER_val/Annotations | grep "^-" | awk '{print $9}' | cut -d '.' -f 1 > data/WIDERFace/WIDER_val/val.txt
