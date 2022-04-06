python ./src/main.py --batch-size 64 --dropout 0.2 --epoch 200 --experiment-name GCN2_13 --save-path ./saved_teacher_models
@REM python ./src/main.py --batch-size 64 --dropout 0.2 --epoch 200 --experiment-name GCN2_13 --save-path ./saved_student_models
python ./src/main.py --batch-size 64 --dropout 0.2 --epoch 200 --experiment-name GCN2_13 --load-path ./saved_teacher_models
python ./src/main.py --batch-size 64 --dropout 0.2 --epoch 200 --experiment-name GCN2_13 --load-path ./saved_student_models