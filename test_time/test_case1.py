import time
import datetime
from tqdm import tqdm

def simulate_finetuning(data_size):
    # Record the start time
    start_time = datetime.datetime.now()
    print(f"微调启动时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Simulate loading the model and data
    time.sleep(1)  # Simulating loading time
    print("模型加载成功")
    
    time.sleep(1)  # Simulating data loading time
    print(f"数据加载成功，数据量：{data_size}")
    
    # Simulate the fine-tuning process with a progress bar
    for _ in tqdm(range(100), desc="低秩微调模型进程", ncols=100):
        time.sleep(0.05)  # Simulating fine-tuning steps
    
    # Record the end time
    end_time = datetime.datetime.now()
    print(f"微调结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate the total time taken in hours
    total_time = (end_time - start_time).total_seconds() / 3600
    print(f"微调总耗时：{total_time:.2f}小时")
    print("="*50)
    
    return total_time

# Simulate three rounds of fine-tuning with different data sizes
times = []
times.append(simulate_finetuning("xxx"))
times.append(simulate_finetuning("yyy"))
times.append(simulate_finetuning("zzz"))

# Calculate and print the average fine-tuning time
average_time = sum(times) / len(times)
print(f"三次微调平均耗时：{average_time:.2f}小时")
