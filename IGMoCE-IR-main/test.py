from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pyplot as plt

# 修改为你自己的 events 文件路径
event_file = "logs/2025_05_26_05_49_20/lightning_logs/version_0/events.out.tfevents.1747988814.ccit-a100-9li.2624301.0"

losses = []
epochs = []

for summary in summary_iterator(event_file):
    for value in summary.summary.value:
        if value.tag == 'Train_Loss':
            # Assuming each step corresponds to an epoch, use summary.step as epoch.
            epochs.append(summary.step)
            losses.append(value.simple_value)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 显示图形
plt.show()

#  nohup python src/train.py --model MyModel_S --batch_size 8 --trainset CDD11_all --num_gpus 4 --loss_type FFT --ckpt_dir /home/aiswjtu/hl/IGMoCE-IR-main/checkpoints_cdd11_all_old  --balance_loss_weight 0.01 --fft_loss_weight 0.1 --de_type denoise_15 denoise_25 denoise_50 dehaze derain --data_file_dir /home/aiswjtu/hl/Data/ > train.log 2>&1 &