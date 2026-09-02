# 1. 日志logs zsy
- 添加机械臂的摆动，用于上肢遥操，下肢locomotion

- 增加机械臂观测值

- 2025 3 25 增加机械臂的扰动项，增加机械臂的课程学习部分，类似于terrian的curriculumn的写法，具体见代码

# 2. 文件目录说明
## envs
- cowa_dual 被继承，所有机器人继承这个类，这个类继承legged robot

- cowa_w_arm 为cowa机器人with机械臂的缩写，用于测试机械臂扰动外部情况下双足机器人的自身稳定性，其中机械臂没过policy，只作为外部的扰动项，类似于抬高做sin跟随

- cowa_w_arm_est 为增加estimator网络的，编码器输出为速度和高度

- 为方便起见同时不干扰其他环境的继承，这里有些文件的legged_robot.cfg 和env等会单独塞进一个文件夹，周知

.
├── base
├── buff_bin
├── cowa
├── cowa_dual
├── cowa_dual_6dof
├── cowa_dual_fix
├── cowa_dual_zsy
├── cowa_est
├── cowa_rma
├── cowa_stages_jump    ## 分階段機器人跳躍，沒有被使用，部署不了
├── cowa_vae
├── cowa_w_arm          ##带机械臂课程干扰
├── cowa_w_arm_est      ##带机械臂课程干扰estimator方案
├── cowa_w_arm_vae      ##带机械臂课程干扰vae方案
├── __init__.py
├── __pycache__
└── readme.md
## scripts
- train.py
- arms_play/train.py 用于带机械臂操作play/train
- stages_play/train.py 用于分阶段动作训练play/train（用不到）

## 注意
- 由于添加的机械臂的序号在轮子前，所以12个joint ,0~6是机械臂的关节  7~12是个是hip、轮子的关节,在init_buffers里面的处理，e.g. dof_pos_all是所iyou关节的，dof_pos_arm是机械臂的，dof_pos的下肢的。



git commit \
  -m "- 新增机械臂-四足机器人训练环境" \
  -m "- resources/robots/cowa_quadruped_arm_v1: 新增带机械臂四足机器人的完整URDF" \
  -m "- wheel_legged_gym/envs/quadruped_wtw_arm_fix: 带机械臂四足机器人-静止模型训练配置" \
  -m "- wheel_legged_gym/scripts/train_wtw_arm_fix: 带机械臂四足机器人-静止模型训练启动" \
  -m "- sim2sim-v1/: 新增MuJoCo部署脚本与多个机器人的仿真描述文件"

git config --global user.name "刘一威"
git config --global user.email "ev@cowarobot.com"