import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 开启交互后端
plt.switch_backend('TkAgg')
plt.ion()

def visualize_top3_pt_data(pt_file_path, joint_num=6):
    try:
        data = torch.load(pt_file_path)
        if data.is_cuda:
            data = data.cpu().numpy()
        else:
            data = data.numpy()
        print(f"成功加载数据，形状：{data.shape}（样本数×特征数）")
        
        # ========== 优化1：采样（避免点数过多） ==========
        sample_size = min(10000, len(data))  # 最多显示1000个点
        np.random.seed(42)
        sample_idx = np.random.choice(len(data), sample_size, replace=False)
        data = data[sample_idx]
        
    except Exception as e:
        print(f"加载失败：{e}")
        return
    
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax1 = fig.add_subplot(111, projection='3d')
    
    # ========== 优化2：点的显示（增大尺寸+加轮廓+按值上色） ==========
    z_values = data[:,2]
    sc = ax1.scatter(data[:,0], data[:,1], data[:,2],
                     c=z_values,          # 按Z值上色，区分分布
                     cmap='viridis',      # 清晰的配色
                     s=20,                # 增大点尺寸
                     alpha=0.3,           # 透明避免遮挡
                     edgecolors='black',  # 轮廓区分单个点
                     linewidths=0.2)
    
    # ========== 优化3：调整坐标轴范围（聚焦数据） ==========
    x_min, x_max = data[:,0].min()*1.1, data[:,0].max()*1.1
    y_min, y_max = data[:,1].min()*1.1, data[:,1].max()*1.1
    z_min, z_max = data[:,2].min()*1.1, data[:,2].max()*1.1
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    
    # ========== 其他优化 ==========
    ax1.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax1.set_zlabel('Z (m)', fontsize=12, labelpad=10)
    ax1.set_title('ee pos command 3D', fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax1, label='Z Value (m)')  # 颜色条说明
    
    # 激活交互
    ax1.mouse_init()
    plt.tight_layout(pad=3)
    plt.ioff()
    plt.show(block=True)

# 调用
if __name__ == "__main__":
    pt_file_path = "wheel_legged_gym/scripts/command_set/command_set.pt"
    visualize_top3_pt_data(pt_file_path, joint_num=6)