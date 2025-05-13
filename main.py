import os
import asyncio
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import unifai
import argparse

data_dir = "."

class SearchServiceAnalyzer:
    def __init__(self, queries_file=None):
        """
        Initialize the analyzer with an API key and queries file
        
        Args:
            queries_file: Path to JSON file containing queries with expected actions
        """
        self.tools = unifai.Tools(api_key="")
        
        # Load queries from file
        if not queries_file or not os.path.exists(queries_file):
            raise ValueError("A valid queries file must be provided")
        
        self.queries_with_expected_actions = self.load_queries_from_file(queries_file)
    
    def load_queries_from_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load queries from a JSONL file (JSON Lines format - one JSON object per line)
        
        Args:
            file_path: Path to the JSONL file containing queries
            
        Returns:
            List of dictionaries with query and expected_action fields
        """
        queries = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        query = json.loads(line)
                        
                        # Validate the structure
                        if not isinstance(query, dict) or "query" not in query or "expected_action" not in query:
                            raise ValueError("Each query must be a dictionary with 'query' and 'expected_action' fields")
                        
                        queries.append(query)
            
            print(f"Loaded {len(queries)} queries from {file_path}")
            return queries
        except Exception as e:
            raise ValueError(f"Error loading queries from {file_path}: {e}")
    
    async def search_service(self, query: str, limit: int) -> List[Dict]:
        """
        Search for services using the unifai SDK
        """
        try:
            # Use the SDK to search for services
            results = await self.tools._api.search_tools(arguments={"query": query, "limit": limit})
            return results
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            return []
    
    async def find_position(self, query: str, expected_action: str, limit: int) -> Optional[int]:
        """
        Find the position of the expected action in the search results
        """
        results = await self.search_service(query, limit)
        
        # Look for the expected action in the results
        for i, service in enumerate(results):
            if service.get("action") == expected_action:
                return i + 1  # 1-based position
        
        # Not found
        return None
    
    async def analyze_positions(self, max_limit: int = 20, concurrency: int = 5) -> List[Optional[int]]:
        """
        Analyze the position of expected services in search results
        with different limits, using controlled concurrency
        
        Args:
            max_limit: Maximum number of results to retrieve
            concurrency: Maximum number of concurrent requests
            
        Returns:
            List of positions (1-based) where expected actions were found
        """
        # Initialize results list with placeholders to maintain order
        positions: List[Optional[int]] = [None for _ in range(len(self.queries_with_expected_actions))]
        
        # Create a semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        # Save data to a file as we go
        data_file = f"{data_dir}/search_service_positions.jsonl"
        with open(data_file, "w") as f:
            f.write("")  # Initialize empty file
        
        # Create a lock for file writing to avoid conflicts
        file_lock = asyncio.Lock()
        
        async def process_query(i: int, query_info: Dict[str, str]) -> None:
            query = query_info["query"]
            expected_action = query_info["expected_action"]
            
            async with semaphore:
                # Find position
                position = await self.find_position(query, expected_action, max_limit)
                positions[i] = position
                
                # Save result to file
                async with file_lock:
                    with open(data_file, "a") as f:
                        f.write(json.dumps({
                            "query": query,
                            "expected_action": expected_action,
                            "position": position
                        }) + "\n")
        
        print(f"Analyzing {len(self.queries_with_expected_actions)} queries with max_limit={max_limit}, concurrency={concurrency}")
        
        # Create and gather tasks
        tasks = []
        for i, query_info in enumerate(self.queries_with_expected_actions):
            task = asyncio.create_task(process_query(i, query_info))
            tasks.append(task)
        
        # Show progress
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await future
        
        return positions
    
    def calculate_recall_at_k(self, positions: List[Optional[int]], k: int) -> float:
        """Calculate recall@k metric"""
        found_within_k = sum(1 for pos in positions if pos is not None and pos <= k)
        return found_within_k / len(positions)
    
    def analyze_results(self, positions: List[Optional[int]], max_limit: int = 20) -> Dict[str, Any]:
        """
        Analyze the results and calculate metrics
        """
        results = {}
        
        # Calculate recall@k for different k values
        k_values = list(range(1, max_limit + 1))
        recall_values = [self.calculate_recall_at_k(positions, k) for k in k_values]
        
        results["k_values"] = k_values
        results["recall_values"] = recall_values
        
        # Calculate position distribution
        position_counts = {}
        for pos in positions:
            if pos is not None:
                position_counts[pos] = position_counts.get(pos, 0) + 1
        
        results["position_counts"] = position_counts

        
        
        # Calculate not found count
        not_found_count = sum(1 for pos in positions if pos is None)
        not_found_percentage = (not_found_count / len(positions)) * 100
        
        results["not_found_count"] = not_found_count
        results["not_found_percentage"] = not_found_percentage
        
        return results
    
    def plot_results(self, results: Dict[str, Any], positions: List[Optional[int]], save_path: str = f"{data_dir}/recall_at_k.png") -> None:
        """
        Plot the recall@k results and position histogram in a single figure
        """
        plt.figure(figsize=(15, 6))

        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 14,
            'figure.titlesize': 22
        })
        
        # Plot recall@k
        plt.subplot(1, 2, 1)
        plt.plot(results["k_values"], results["recall_values"], marker='o', linestyle='-', linewidth=2)
        plt.title('Recall@k for search_service Results', fontsize=20)
        plt.xlabel('k (Top k results)', fontsize=18)
        plt.ylabel('Recall@k', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Create custom x-ticks: show all values up to 10, then only even numbers
        custom_xticks = [k for k in results["k_values"] if k <= 10 or k % 2 == 0]
        plt.xticks(custom_xticks)
        
        # Add annotations for key points
        for i, k in enumerate(results["k_values"]):
            if k in [1, 3, 5, 10, 15, 20]:
                plt.annotate(f"{results['recall_values'][i]:.2f}", 
                            (k, results['recall_values'][i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')
        # Add arrow at k=10 to indicate default value
        if 10 in results["k_values"]:
            k10_index = results["k_values"].index(10)
            k10_recall = results["recall_values"][k10_index]
            plt.annotate("Default Search Limit",
                      xy=(10, k10_recall),
                      xytext=(10, k10_recall - 0.1),
                      arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8, edgecolor='red'),
                      ha='center',
                      fontsize=16,
                      color='red')

        # Create a position histogram as second subplot
        plt.subplot(1, 2, 2)
        filtered_positions = [pos for pos in positions if pos is not None]
        if filtered_positions:
            plt.hist(filtered_positions, bins=range(1, max(filtered_positions) + 2), alpha=0.7, edgecolor='black')
            plt.title('Distribution of Expected Action Positions', fontsize=20)
            plt.xlabel('Position', fontsize=18)
            plt.ylabel('Frequency', fontsize=18)
            
            # Create custom x-ticks for the histogram: show all values up to 10, then only even numbers
            max_pos = max(filtered_positions)
            custom_pos_xticks = [pos for pos in range(1, max_pos + 1) if pos <= 10 or pos % 2 == 0]
            plt.xticks(custom_pos_xticks)
            
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path)
    
    def print_summary(self, results: Dict[str, Any], positions: List[Optional[int]]) -> None:
        """
        Print a summary of the analysis
        """
        print(f"Total queries: {len(self.queries_with_expected_actions)}")
        print(f"Services found: {len(self.queries_with_expected_actions) - results['not_found_count']}")
        print(f"Services not found: {results['not_found_count']} ({results['not_found_percentage']:.2f}%)")
        
        print("\nPosition distribution:")
        for pos in sorted(results["position_counts"].keys()):
            count = results["position_counts"][pos]
            percentage = (count / len(self.queries_with_expected_actions)) * 100
            print(f"Position {pos}: {count} queries ({percentage:.2f}%)")
        
        print("\nRecall@k:")
        for i, k in enumerate(results["k_values"]):
            if k in [1, 3, 5, 10, 15, 20]:
                print(f"Recall@{k}: {results['recall_values'][i]:.4f}")
        
        # Calculate percentage of queries where the service was found in top positions
        filtered_positions = [pos for pos in positions if pos is not None]
        in_top_1 = sum(1 for pos in filtered_positions if pos == 1) / len(self.queries_with_expected_actions) * 100
        in_top_3 = sum(1 for pos in filtered_positions if pos <= 3) / len(self.queries_with_expected_actions) * 100
        in_top_5 = sum(1 for pos in filtered_positions if pos <= 5) / len(self.queries_with_expected_actions) * 100
        in_top_10 = sum(1 for pos in filtered_positions if pos <= 10) / len(self.queries_with_expected_actions) * 100
        
        print(f"\nPercentage of queries where service found in top 1: {in_top_1:.2f}%")
        print(f"Percentage of queries where service found in top 3: {in_top_3:.2f}%")
        print(f"Percentage of queries where service found in top 5: {in_top_5:.2f}%")
        print(f"Percentage of queries where service found in top 10: {in_top_10:.2f}%")
        
        # Determine optimal limit
        # Calculate the elbow point (where marginal improvements diminish)
        increases = [results["recall_values"][i] - results["recall_values"][i-1] for i in range(1, len(results["recall_values"]))]
        
        # Simple elbow detection - find where the increase falls below a threshold
        threshold = 0.01  # 1% increase in recall
        optimal_k = 1
        for i, increase in enumerate(increases):
            if increase < threshold:
                optimal_k = results["k_values"][i+1]
                break
        
        # If no clear elbow, use the point where we reach 95% of maximum recall
        if optimal_k == 1:
            target_recall = 0.95 * max(results["recall_values"])
            for i, recall in enumerate(results["recall_values"]):
                if recall >= target_recall:
                    optimal_k = results["k_values"][i]
                    break
        
        print(f"\nRecommended optimal limit for search_service: {optimal_k}")

async def main():
    """
    Main function to run the analysis
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze search service performance")
    parser.add_argument("--queries_file", type=str, default="search_queries.jsonl", 
                        help="Path to JSON file containing queries with expected actions")
    parser.add_argument("--max_limit", type=int, default=20, 
                        help="Maximum number of results to retrieve")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Maximum number of concurrent requests")
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SearchServiceAnalyzer(queries_file=args.queries_file)
    
    # Set maximum limit to test
    max_limit = args.max_limit
    concurrency = args.concurrency
    
    # Analyze positions
    positions = await analyzer.analyze_positions(max_limit=max_limit, concurrency=concurrency)
    
    # Analyze results
    results = analyzer.analyze_results(positions, max_limit=max_limit)
    
    # Plot results
    analyzer.plot_results(results, positions)
    
    # Print summary
    analyzer.print_summary(results, positions)
    
    # Save results to CSV
    df = pd.DataFrame({
        "k": results["k_values"],
        "recall": results["recall_values"]
    })
    df.to_csv(f"{data_dir}/recall_at_k.csv", index=False)
    
    print(f"Analysis complete. Results saved to {data_dir}/recall_at_k.png and {data_dir}/recall_at_k.csv")

if __name__ == "__main__":
    asyncio.run(main())
