# 项目结构

## 1. cowa_w_arm_est_add_arm 
添加机械臂观测的，6+6 dof
Estimator网络
## 2. cowa_w_arm_him_add_arm 
添加机械臂观测的，6+6 dof
HIM网络
## 3. cowa_w_arm_roa_add_arm 
添加机械臂观测的，6+6 dof
ROA网络
暂时可以用，主要用来初步验证算法
## 4. cowa_wo_arm_est_terrain
无机械臂观测的，6 dof, 主要训练阶梯地形上5cm；楼梯 2rad/s nee joints
Estimator网络
可以实现上楼梯，添加了雷达输入的高层图编码器
## 5. cowa_wo_arm_est_terrain_2
无机械臂观测的，6 dof, 主要训练阶梯地形上5cm；楼梯 2rad/s nee joints
Estimator网络
不同的只是观测添加了轮子的相对位置
添加了雷达输入的高层图编码器
## 6. cowa_wo_arm_ppo_terrain
无机械臂观测的，6 dof, 主要训练阶梯地形上5cm；楼梯 2rad/s nee joints
最原始的PPO网络
添加了雷达输入的高层图编码器

## 7. cowa_wo_arm_roa_terrain_origin
for latent_height --> history_encoder 

## 8. cowa_wo_arm_roa_terrain
for latent_height --> actor directly 



## 9. cowa_wo_arm_ts_terrain
两阶段的师生网络蒸馏

