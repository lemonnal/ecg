import random
import time
from collections import deque
import matplotlib.pyplot as plt

# 创建队列
signal_queue = deque([0], maxlen=100)

# 创建图形
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([])
ax.set_ylim(-3, 3)

# 循环生成随机数并输入队列
try:
    while True:
        # 生成随机数
        value = random.uniform(-2.0, 2.0)

        # 添加到队列
        signal_queue.append(value)

        # 更新图形
        line.set_ydata(signal_queue)
        line.set_xdata(range(len(signal_queue)))

        # 清除之前的红圈
        for artist in ax.lines[1:]:
            artist.remove()

        # 在超过1.9的点画红圈
        for i, v in enumerate(signal_queue):
            if v > 1.9:
                ax.plot(i, v, 'ro', markersize=8)

        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\n停止")
    plt.ioff()
    plt.show()
