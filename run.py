import subprocess
import pandas as pd
import time
import re

# 修正后的实验配置
experiments = [
    # ml-1m数据集
    ["BPRMF", "ml-1m", "--batch_size 2048 --epoch 30 --num_neg 1"],
    ["LightGCN", "ml-1m", "--n_layers 3 --batch_size 2048 --epoch 30 --num_neg 1"],
    ["AHNS", "ml-1m", "--alpha 0.5 --beta 1.0 --p -2.0 --candidate_M 32 --num_neg 32"],
    ["AHNS", "ml-1m", "--alpha 1.0 --beta 0.8 --p -1.0 --candidate_M 16 --num_neg 16"],
    
    # Grocery数据集 - 稀疏数据需要不同配置
    ["BPRMF", "Grocery_and_Gourmet_Food", "--batch_size 1024 --epoch 60 --num_neg 4"],
    ["LightGCN", "Grocery_and_Gourmet_Food", "--n_layers 2 --batch_size 1024 --epoch 60 --num_neg 4"],
    ["AHNS", "Grocery_and_Gourmet_Food", "--alpha 0.3 --beta 0.5 --p -1.5 --candidate_M 16 --num_neg 16"],
    ["AHNS", "Grocery_and_Gourmet_Food", "--alpha 0.2 --beta 0.3 --p -1.0 --candidate_M 8 --num_neg 8 --hardness_weight 1 --hardness_temp 0.5"]
]

results = []

for i, (model, dataset, args) in enumerate(experiments, 1):
    print(f"\n{'='*60}")
    print(f"实验 {i}/8: {model} on {dataset}")
    print(f"参数: {args}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    start = time.time()
    
    # 构建命令 - 只包含必要的参数
    base_cmd = f"python src/main.py --model_name {model} --dataset {dataset}"
    
    # 添加基础参数
    if "ml-1m" in dataset:
        base_cmd += " --batch_size 2048 --emb_size 64 --lr 0.001 --l2 0"
    else:  # Grocery数据集
        base_cmd += " --batch_size 1024 --emb_size 128 --lr 0.005 --l2 1e-5"
    
    # 添加训练和评估参数
    base_cmd += " --gpu 0 --train 1 --load 0"
    
    # 添加模型特定参数
    if args:
        base_cmd += f" {args}"
    
    print(f"命令: {base_cmd}")
    
    try:
        # 运行命令并捕获输出
        process = subprocess.Popen(
            base_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 收集输出
        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)
        
        process.wait()
        runtime = time.time() - start
        output = ''.join(output_lines)
        
        # 解析指标 - 从完整的训练日志中查找
        ndcg20 = "N/A"
        final_epoch_loss = "N/A"
        dev_ndcg5 = "N/A"
        
        # 查找最终测试结果
        for line in output.split('\n'):
            if 'Test After Training' in line:
                # 提取NDCG@20
                match = re.search(r'NDCG@20:(\d+\.\d+)', line)
                if match:
                    ndcg20 = float(match.group(1))
            
            # 查找训练过程中的loss和验证指标
            if 'loss=' in line and 'dev=' in line:
                # 提取loss和dev NDCG@5
                parts = line.split()
                for part in parts:
                    if part.startswith('loss='):
                        final_epoch_loss = float(part.split('=')[1])
                    if part.startswith('dev=(NDCG@5:'):
                        dev_ndcg5 = float(part.split(':')[1].rstrip(')'))
        
        results.append({
            'exp_id': i,
            'model': model,
            'dataset': dataset,
            'args': args[:80],
            'time': round(runtime, 1),
            'loss': final_epoch_loss,
            'dev_ndcg5': dev_ndcg5,
            'NDCG@20': ndcg20,
            'status': 'success' if process.returncode == 0 else f'failed({process.returncode})'
        })
        
        print(f"\n完成! 用时: {runtime:.1f}s")
        print(f"最终loss: {final_epoch_loss}")
        print(f"验证集NDCG@5: {dev_ndcg5}")
        print(f"测试集NDCG@20: {ndcg20}")
        
        # 根据结果给出建议
        if ndcg20 != "N/A" and isinstance(ndcg20, float):
            if dataset == "ml-1m" and ndcg20 < 0.1:
                print("⚠️  警告: ml-1m的NDCG@20应该 > 0.3")
            elif dataset == "Grocery_and_Gourmet_Food" and ndcg20 < 0.02:
                print("⚠️  警告: Grocery的NDCG@20可能偏低")
        
    except Exception as e:
        runtime = time.time() - start
        results.append({
            'exp_id': i,
            'model': model,
            'dataset': dataset,
            'args': args[:80],
            'time': round(runtime, 1),
            'NDCG@20': 'error',
            'status': str(e)[:50]
        })
        print(f"错误: {e}")

# 保存结果
df = pd.DataFrame(results)
csv_name = f"ahns_corrected_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(csv_name, index=False)
print(f"\n结果保存到: {csv_name}")

# 打印摘要
print("\n" + "="*80)
print("实验结果汇总:")
print("="*80)
print(df[['exp_id', 'model', 'dataset', 'NDCG@20', 'time', 'status']].to_string())

# 按数据集分析
print("\n" + "="*80)
print("数据集分析:")
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    print(f"\n{dataset}:")
    for _, row in subset.iterrows():
        print(f"  {row['model']}: NDCG@20 = {row['NDCG@20']}, 时间 = {row['time']}s")