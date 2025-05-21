import matplotlib.pyplot as plt
import numpy as np

# 原始Loss数据
epochs = list(range(50))
# loss_values = [
#     0.022739, 0.017793, 0.014531, 0.012367, 0.010749, 
#     0.009498, 0.008680, 0.007959, 0.007381, 0.006979, 
#     0.006648, 0.006386, 0.006275, 0.006111, 0.005952, 
#     0.005859, 0.005694, 0.005447, 0.005520, 0.005264, 
#     0.005258, 0.005256, 0.005255, 0.005117, 0.005102, 
#     0.005099, 0.005125, 0.005189, 0.004966, 0.005164, 
#     0.004994, 0.005372, 0.005123, 0.005012, 0.005176, 
#     0.005140, 0.005086, 0.005459, 0.005160, 0.005271, 
#     0.005400, 0.005252, 0.005220, 0.005165, 0.005221, 
#     0.005373, 0.005159, 0.005464, 0.005184, 0.005038
# ]


loss_values = [
    0.001825, 0.001472, 0.001238, 0.001062, 0.000945,
    0.000851, 0.000788, 0.000738, 0.000697, 0.000668,
    0.000645, 0.000626, 0.000608, 0.000592, 0.000579,
    0.000571, 0.000565, 0.000555, 0.000548, 0.000542,
    0.000538, 0.000533, 0.000529, 0.000525, 0.000521,
    0.000521, 0.000517, 0.000514, 0.000512, 0.000513,
    0.000511, 0.000508, 0.000507, 0.000506, 0.000507,
    0.000506, 0.000505, 0.000502, 0.000503, 0.000503,
    0.000503, 0.000500, 0.000501, 0.000498, 0.000499,
    0.000496, 0.000497, 0.000498, 0.000496, 0.000495
]


# 将Loss乘以10
scaled_loss = [loss * 10 for loss in loss_values]

# 创建美观的图表
# plt.figure(figsize=(12, 6))

# 绘制Loss曲线
plt.plot(epochs, scaled_loss, 
         color='blue',  # 珊瑚红色
         linewidth=2.0, 
        )

# 添加标题和标签
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)

# # 设置坐标轴
# plt.xlim(-1, 50)
# plt.xticks(np.arange(0, 51, 5))
# plt.ylim(0.04, 0.25)
# plt.yticks(np.arange(0.04, 0.26, 0.02))

# 添加网格和图示
# plt.legend(fontsize=12)

# 高亮显示最低Loss点
# min_loss = min(scaled_loss)
# min_epoch = scaled_loss.index(min_loss)
# plt.scatter(min_epoch, min_loss, 
#             color='#4ECDC4', 
#             s=150, 
#             zorder=5,
#             label=f'Min Loss: {min_loss:.3f} at Epoch {min_epoch}')

# # 添加注释
# plt.annotate(f'Minimum: {min_loss:.3f}', 
#              xy=(min_epoch, min_loss),
#              xytext=(min_epoch+5, min_loss+0.02),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图表
# plt.savefig('fc1_loss_curve.png', dpi=300)
plt.savefig('mat_qkv_loss_curve.png', dpi=300)
