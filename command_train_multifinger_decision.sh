# source /disk1/anaconda/bin/activate
# conda activate mink38
python train_multifinger.py --dataset_root=logs/data/decision_model/allegro  --log_dir=logs/model/allegro_model/allegro_obj140/1_class_1_model/ --gripper_type=Allegro --train_multifinger_type=9
# python train_multifinger.py --dataset_root=logs/data/decision_model/dh3/obj140 --log_dir=logs/model/dh3_model/obj140/1_class_1_model/ --gripper_type=DH3 --train_multifinger_type=1
# python train_multifinger.py --dataset_root=logs/data/decision_model/inspire/obj140 --log_dir=logs/model/inspire_model/obj140/1_class_1_model/ --gripper_type=Inspire --train_multifinger_type=8
