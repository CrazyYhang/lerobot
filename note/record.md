----30 episodes camera fixed


#找端口
python lerobot/scripts/find_motors_bus_port.py



#登陆hugging face
huggingface-cli login --token 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 --add-to-git-credential

#赋值环境变量
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER


#设备权限
xhost +
sudo chmod 777 /dev/ttyACM*

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





































#登陆wandb
wandb login --relogin
930851e1dfb63071dd45062bfa41b4b16ec425e0



#每次任务名称
TASK_NAME=precision-assembly

record
#记录数据，30集

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="precision-assembly" \
  --control.repo_id=${HF_USER}/${TASK_NAME} \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=100 \
  --control.episode_time_s=45 \
  --control.reset_time_s=100 \
  --control.num_episodes=1 \
  --control.push_to_hub=true

#后面继续接，或者数据没传到huggingface可以在hf上删除后接0集自动重新上传
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="precision-assembly" \
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
































#连接到老师电脑，开内网穿透，wandb
TASK_NAME=precision-assembly  #记得换
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

#db可视化数据
https://wandb.ai/home 
























































#模型验证
  python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="precision-assembly" \
  --control.repo_id=${HF_USER}/eval_${TASK_NAME} \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/${TASK_NAME}/checkpoints/last/pretrained_model

rm -rf ~/.cache/huggingface/lerobot/CrazyYhang/eval_${TASK_NAME}