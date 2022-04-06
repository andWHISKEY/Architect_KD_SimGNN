import torch
import time
from tqdm import tqdm,trange
from param_parser import parameter_parser
from tensorboardX import SummaryWriter,writer
from torch.nn import L1Loss,MSELoss
from torch.utils.data import DataLoader
# from parallel import DataParallelModel, DataParallelCriterion
from torchsummaryX import summary

from utils import *
from edit_simgnn3 import *
from datas import *


def main():
    
    set_seed(42)

    print('-----------------------------------------------------')
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    print('학습을 진행하는 기기:',device)
    print('cuda index:', torch.cuda.current_device())
    print('gpu 개수:', torch.cuda.device_count())
    print('graphic name:', torch.cuda.get_device_name())
    print('-----------------------------------------------------')

    args=parameter_parser()
    BATCH_SIZE=args.batch_size
    EPOCHS=args.epochs
    EXPERIMENT_NAME=args.experiment_name
    LEARNING_RATE=args.learning_rate
    WEIGHT_DECAY=args.weight_decay
    SAVE_NAME=f'batch{BATCH_SIZE}_drop{args.dropout}_epoch{EPOCHS}_{EXPERIMENT_NAME}_realdata'

    train_dataset=GraphDataset(is_train=True)
    test_dataset=GraphDataset(is_train=False)
    number_of_labels,global_labels=test_dataset.initial_label_enumeration()
    train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)
    test_dataloader=DataLoader(test_dataset,batch_size=1)

    teacher_model=SimGNN(args,number_of_labels,device).to(device)
    # teacher_model=DataParallelModel(teacher_model).to(device)
    student_model=Student(args,number_of_labels,device).to(device)
    # student_model=DataParallelModel(student_model).to(device)

    #torch summary
    summary(teacher_model, {"features_1": torch.zeros((4,7)), "features_2":torch.zeros((16,7)) , "edge_index_1": torch.zeros((2,6), dtype=torch.int64) ,"edge_index_2": torch.zeros((2,30), dtype=torch.int64) })
    summary(student_model, {"features_1": torch.zeros((4,7)), "features_2":torch.zeros((16,7)) , "edge_index_1": torch.zeros((2,6), dtype=torch.int64) ,"edge_index_2": torch.zeros((2,30), dtype=torch.int64) })

    #MAE_LOSS=L1Loss()
    LOSS_FN=MSELoss()
    # LOSS_FN=DataParallelCriterion(LOSS_FN)

    # model 불러와서 결과만 출력
    if args.load_path:
        teacher_weight=torch.load(f'./saved_teacher_models/{SAVE_NAME}.pth')
        student_weight=torch.load(f'./saved_student_models/{SAVE_NAME}.pth')
        
        
        teacher_model.load_state_dict(teacher_weight)    
        student_model.load_state_dict(student_weight, strict=False)   
        
        with torch.no_grad():
            student_model.tensor_network.weight_matrix.copy_(teacher_weight['tensor_network.weight_matrix'])
            student_model.tensor_network.weight_matrix_block.copy_(teacher_weight['tensor_network.weight_matrix_block'])
            student_model.tensor_network.bias.copy_(teacher_weight['tensor_network.bias'])
            student_model.fully_connected_first.weight.copy_(teacher_weight['fully_connected_first.weight'])
            student_model.fully_connected_first.bias.copy_(teacher_weight['fully_connected_first.bias'])
            student_model.scoring_layer.weight.copy_(teacher_weight['scoring_layer.weight'])
            student_model.scoring_layer.bias.copy_(teacher_weight['scoring_layer.bias'])

        ged_predict_list=[]
        ged_gt_list=[]
        origin_ged=[]
        ged_file_name=[]
        
        time_teacher=[]
        time_student=[]

        test_graphs = glob.glob("./test_dataset/" + "*.json")
        for graph_pair in (test_graphs):
            data = process_pair(graph_pair)
            origin_ged.append(data['ged'])
            data = transfer_to_torch(data,global_labels)
            start = time.time()
            if args.load_path=='./saved_teacher_models':
                prediction,embedding1,embedding2 = teacher_model(data)
                time_teacher.append(time.time()-start)
            elif args.load_path=='./saved_student_models':
                temp = []
                embedding_vector1,embedding_vector2=student_model(data)
                now = time.time()
                temp.append(now-start)
                prediction= student_model.embedded_forward(embedding_vector1,embedding_vector2)    
                temp.append(time.time()-now)
                time_student.append(temp)
            end=time.time()
            ged_file_name.append(graph_pair.split(sep='\\')[1])
            ged_gt_list.append(data['target'].item())
            ged_predict_list.append(prediction.item())
        print(f'time_teacher: {time_teacher}')
        print(f'time_student: {time_student}')
        print(f'total time: {end-start}')
        print(ged_file_name)
        print(ged_predict_list)
        print(ged_gt_list)
        print(origin_ged)
        infos = zip(ged_file_name, ged_predict_list, ged_gt_list, origin_ged)
        infos = sorted(infos, key= lambda x: x[1])
        print(infos[:4])
    else:
        writer=SummaryWriter(log_dir=f"./runs/batch{BATCH_SIZE}_drop{args.dropout}_epoch{EPOCHS}_{EXPERIMENT_NAME}_realdata")
        # model fit(train)
        optimizer=torch.optim.Adam(teacher_model.parameters(),
                                            lr=LEARNING_RATE,
                                            weight_decay=WEIGHT_DECAY)
        student_optimizer = torch.optim.Adam(student_model.parameters(),
                                            lr=LEARNING_RATE,
                                            weight_decay=WEIGHT_DECAY)                                   

        teacher_model.train()
        student_model.train()
        teacher_score_max=0
        student_score_max=0
        epochs=trange(EPOCHS, leave=True, desc="Epoch")
        for epoch in epochs:
            loss_sum=0
            # batches=create_batches(train_dataloader,BATCH_SIZE)
            # print(f'batch: {batches}')
            main_index=0
            indices = np.random.choice(len(train_dataset), len(train_dataset))
            for step in tqdm(range(0,len(train_dataset),BATCH_SIZE),total=len(train_dataset), desc="Batches"):
                optimizer.zero_grad()
                student_optimizer.zero_grad()
                losses=0
                KD_losses=0
                cnt=0
                batch_scores = []
                ground_truth = []
                batch = []
                for idx in indices[step:step+BATCH_SIZE]:
                    batch.append(train_dataset.__getitem__(idx))

                for graph_pair,norm_ged in batch:
                    ground_truth.append(norm_ged)
                    target = graph_pair["target"] 
                    
                    # 여기서 prediction이 simgnn score일텐데, 여기서 embedding vector불러와서 비교 
                    prediction,teacher_vector1,teacher_vector2 =teacher_model(graph_pair)
                    # Student model data넣어서 embedding vector뽑기
                    # print(f'{cnt}----------------------------------------------------')
                    # cnt=cnt+1
                    student_vector1,student_vector2=student_model(graph_pair)
                    # print(f'CHECK_KD_vector: //teacher={teacher_vector1},{teacher_vector2}')
                    # print(f'CHECK_KD_vector: //student={student_vector1},{student_vector2}')
                    # print('----------------------------------------------------')
                    KD_losses=KD_losses+l1_loss(student_vector1.to(device),teacher_vector1.to(device).detach())+l1_loss(student_vector2.to(device),teacher_vector2.to(device).detach())

                    batch_scores.append(calculate_loss(prediction,target.to(device)).detach().cpu().numpy())
                    losses = losses + l1_loss(prediction,target.to(device))

                losses.backward(retain_graph=True)
                optimizer.step()
                if epoch>50:
                    KD_losses.backward(retain_graph=True)
                    student_optimizer.step()
                loss_score = losses.item()
                train_score= 1 - print_evaluation(ground_truth,batch_scores)
                main_index = main_index + len(batch)
                loss_sum = loss_sum + loss_score * len(batch)
                loss = loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            
            #test_data에 대한 score계산
            teacher_model.eval()
            student_model.eval()
            teacher_scores=[]
            student_scores=[]
            ground_truth=[]
            with torch.no_grad():
                for step in tqdm(range(0,len(test_dataset))):
                    graph_pair,norm_ged=train_dataset.__getitem__(step)
                    ground_truth.append(norm_ged)
                    target=graph_pair['target']
                    teacher_prediction,teacher_embedding1,teacher_embedding2 = teacher_model(graph_pair)
                    student_embedding1, student_embedding2 = student_model(graph_pair)
                    student_prediction = teacher_model.embedded_forward(student_embedding1, student_embedding2)
                    print(f'teacherprediction:{teacher_prediction}------target:{target}')
                    teacher_scores.append(LOSS_FN(teacher_prediction.to(device),target.to(device)).detach().cpu().numpy())
                    teacher_test_score = 1 - print_evaluation(ground_truth,teacher_scores)

                    student_scores.append(LOSS_FN(student_prediction.to(device), target.to(device)).detach().cpu().numpy())
                    student_test_score = 1 - print_evaluation(ground_truth,student_scores)
                
                if epoch==99:
                    torch.save(teacher_model.state_dict(),f'./saved_teacher_models/{SAVE_NAME}.pth')
                    torch.save(student_model.state_dict(),f'./saved_student_models/{SAVE_NAME}.pth')

                if(writer!=None):
                    writer.add_scalar("loss/train",loss_sum,epoch)
                    writer.add_scalar("score/train",train_score,epoch)
                    writer.add_scalar("score/test_teacher",teacher_test_score,epoch)
                    writer.add_scalar("score/test_student",student_test_score,epoch)
                 

if __name__=="__main__":
    main()