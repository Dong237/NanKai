from pathlib import Path
import numpy as np
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments with 'loss_folder' and 'threshold'.
    """
    parser = argparse.ArgumentParser(
        description="Test convergence of sequences in .npy files."
        )
    parser.add_argument(
        "--loss_folder", 
        default="losses",
        type=str, 
        help="Path to the folder containing .npy loss files."
        )
    parser.add_argument(
        "--threshold", 
        default=0.001,
        type=float, 
        help="Convergence threshold for the Cauchy test."
        )
    return parser.parse_args()

def load_loss_files(loss_folder):
    """
    Load all .npy files in the given folder.
    
    Parameters:
        loss_folder (str): Path to the folder containing .npy files.
    
    Returns:
        list: List of Path objects pointing to .npy files.
    """
    folder = Path(loss_folder)
    if not folder.is_dir():
        console.print(
            f"[bold red]Error:[/bold red] '{loss_folder}' is not a valid directory."
            )
        raise FileNotFoundError(f"Directory '{loss_folder}' not found.")
    
    files = list(folder.glob("*.npy"))
    if not files:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] No .npy files found in '{loss_folder}'."
            )
        return []
    
    return files

def cauchy_test(data, threshold):
    """
    Perform the Cauchy test to check convergence of a sequence.
    
    Parameters:
        data (list or numpy array): Sequence of numbers to test.
        threshold (float): Convergence threshold.
    
    Returns:
        bool: True if the sequence converges, False otherwise.
    """
    for i in range(1, len(data)):
        diff = abs(data[i] - data[i - 1])
        if diff < threshold:
            return True
    return False

def process_files(files, threshold):
    """
    Process a list of .npy files, applying the Cauchy test to their contents.
    
    Parameters:
        files (list): List of Path objects pointing to .npy files.
        threshold (float): Convergence threshold.
    
    Returns:
        list: Results of the test for each file.
    """
    results = []
    for file in track(files, description="数据处理..."):
        data = np.load(file)
        converges = cauchy_test(data, threshold)
        results.append((file.name, converges))
    return results

def display_results(results, threshold):
    """
    Display the results of the Cauchy test in a table format.
    
    Parameters:
        results (list): List of tuples containing file names and test results.
        threshold (float): The threshold used for the test.
    """
    table = Table(title=f"柯西准则测试结果 (决策阈值: {threshold})")
    table.add_column("训练损失文件", style="cyan")
    table.add_column("收敛?", style="green")
    
    for file, converges in results:
        status = "✅ 收敛" if converges else "❌ 不收敛"
        table.add_row(file, status)
    
    console.print(table)

def main():
    args = parse_arguments()
    loss_folder = args.loss_folder
    threshold = args.threshold
    
    # Load and process files
    files = load_loss_files(loss_folder)
    if not files:
        return
    
    results = process_files(files, threshold)
    
    # Display results
    display_results(results, threshold)

if __name__ == "__main__":
    main()
