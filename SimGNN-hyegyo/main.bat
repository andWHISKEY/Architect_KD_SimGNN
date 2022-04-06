@REM python ./src/main.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name KDmeanmodel --load-path ./saved_teacher_models
@REM python ./src/main.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name KDmeanmodel --load-path ./saved_student_models
@REM python ./src/origin_main.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name KDconv2 --load-path ./saved_teacher_models
@REM python ./src/origin_main.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name KDconv2 --load-path ./saved_student_models
@REM python ./src/edit_main.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN1relu --save-path ./saved_student_models
@REM python ./src/edit_main1.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN1 --save-path ./saved_student_models
@REM python ./src/edit_main2.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_13relu --save-path ./saved_student_models
python ./src/edit_main3.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_13 --save-path ./saved_student_models
python ./src/edit_main3.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_13 --save-path ./saved_teacher_models
@REM python ./src/edit_main4.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_23relu --save-path ./saved_student_models
@REM python ./src/edit_main5.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_23 --save-path ./saved_student_models
python ./src/edit_main.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN1relu --load-path ./saved_teacher_models
@REM python ./src/edit_main.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN1relu --load-path ./saved_student_models
@REM python ./src/edit_main1.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN1 --load-path ./saved_student_models
@REM python ./src/edit_main2.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_13relu --load-path ./saved_student_models
python ./src/edit_main3.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_13 --load-path ./saved_student_models
@REM python ./src/edit_main4.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_23relu --load-path ./saved_student_models
@REM python ./src/edit_main5.py --batch-size 32 --dropout 0.2 --epoch 200 --experiment-name GCN2_23 --load-path ./saved_student_models