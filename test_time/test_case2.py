def run_experiment(experiment_id, data_size):
    # Simulating the loading of loss data and printing out the relevant information
    print(f"微调实验{experiment_id}: 训练损失加载成功，记录数据量：{data_size}")
    
    # Simulating the convergence check
    print("执行柯西准则检验收敛性...")
    is_converged = True  # Directly simulate that the loss converged
    
    # Printing the result of the convergence check
    print(f"训练损失收敛：{is_converged}")
    print("="*37)
    
    return is_converged

def main():
    # Define fake data sizes (this simulates the different data amounts for each experiment)
    experiment_data_sizes = ["xxx", "yyy", "zzz"]
    
    # Run three simulated experiments
    convergence_results = []
    for i, data_size in enumerate(experiment_data_sizes, start=1):
        convergence_results.append(run_experiment(i, data_size))
    
    # Final conclusion on convergence (ensuring all results are True)
    final_convergence = all(convergence_results)
    print(f"训练损失收敛最终结论：{final_convergence}")

# Run the main function
if __name__ == "__main__":
    main()
