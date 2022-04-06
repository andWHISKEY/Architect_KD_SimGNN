python ./src/edit_main1.py --batch-size 128 --dropout 0.2 --epoch 100 --experiment-name GCN1 --save-path ./saved_student_models
python ./src/edit_main3.py --batch-size 128 --dropout 0.2 --epoch 100 --experiment-name GCN2_13 --save-path ./saved_student_models
python ./src/edit_main5.py --batch-size 128 --dropout 0.2 --epoch 100 --experiment-name GCN2_23 --save-path ./saved_student_models
python ./src/edit_main1.py --batch-size 128 --dropout 0.2 --epoch 100 --experiment-name GCN1 --load-path ./saved_teacher_models
python ./src/edit_main1.py --batch-size 128 --dropout 0.2 --epoch 100 --experiment-name GCN1 --load-path ./saved_student_models
python ./src/edit_main3.py --batch-size 128 --dropout 0.2 --epoch 100 --experiment-name GCN2_13 --load-path ./saved_student_models
python ./src/edit_main5.py --batch-size 128 --dropout 0.2 --epoch 100 --experiment-name GCN2_23 --load-path ./saved_student_models