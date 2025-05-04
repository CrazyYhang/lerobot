----30 episodes camera fixed


#找端口
python lerobot/scripts/find_motors_bus_port.py



#每次任务名称
TASK_NAME=A1234-B-C_mvA2B  ok
TASK_NAME=A123-B4-C_mvA2C  
TASK_NAME=A12-B4-C3_mvB2C  
TASK_NAME=A12-B-C34_mvA2B  
TASK_NAME=A1-B2-C34_mvC2A  
TASK_NAME=A14-B2-C3_mvC2B  
TASK_NAME=A14-B23-C_mvA2B  
TASK_NAME=A1-B234-C_mvA2C  
TASK_NAME=A-B234-C1_mvB2C  
TASK_NAME=A-B23-C14_mvB2A  
TASK_NAME=A3-B2-C14_mvC2A  
TASK_NAME=A34-B2-C1_mvB2C  
TASK_NAME=A34-B-C12_mvA2B  
TASK_NAME=A3-B4-C12_mvA2C  
TASK_NAME=A-B4-C123_mvB2C  

#登陆hugging face
huggingface-cli login --token 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 --add-to-git-credential

#赋值环境变量
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER


#设备权限
xhost +

#遥操不带摄像头
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate

#带摄像头
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=teleoperate

#查看相机id
python lerobot/common/robot_devices/cameras/opencv.py \
  --images-dir outputs/images_test
















































record
#记录数据，30集

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="4 disk hanoi solution" \
  --control.repo_id=${HF_USER}/${TASK_NAME} \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=100 \
  --control.episode_time_s=45 \
  --control.reset_time_s=100 \
  --control.num_episodes=30 \
  --control.push_to_hub=true

#后面继续接，或者数据没传到huggingface可以在hf上删除后接0集自动重新上传
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="4 disk hanoi solution" \
  --control.repo_id=${HF_USER}/${TASK_NAME} \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=100 \
  --control.episode_time_s=45 \
  --control.reset_time_s=100 \
  --control.num_episodes=0 \
  --control.push_to_hub=true \
  --control.resume=true


#删除记录数据
  rm -rf /root/.cache/huggingface/lerobot/CrazyYhang/$TASK_NAME


#本地可视化记录
  python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/${TASK_NAME}


#重播某一集
  python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/${TASK_NAME} \
  --control.episode=25

#学长写的代码，已废除
  python test/ReConsolidate.py --dataset_repo_id ${HF_USER}/${TASK_NAME}































#连接到老师电脑，开内网穿透，wandb
TASK_NAME=  #记得换
ssh -p 53769 star@ld.frp.one    #内网穿透




#翻墙登huggingface后训练

huggingface-cli login --token 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 --add-to-git-credential

HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER














#训练数据，平时用这个
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/${TASK_NAME} \
  --policy.type=act \
  --output_dir=outputs/train/${TASK_NAME} \
  --job_name=${TASK_NAME} \
  --policy.device=cuda \
  --wandb.enable=false

#写论文用这个，db可视化有图表
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/${TASK_NAME} \
  --policy.type=act \
  --output_dir=outputs/train/${TASK_NAME} \
  --job_name=${TASK_NAME} \
  --policy.device=cuda \
  --wandb.enable=true
























































#模型验证
  python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/eval_${TASK_NAME} \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/${TASK_NAME}/checkpoints/last/pretrained_model