"""
Benchmark script to measure prediction performance optimization.
"""
import time
import requests
import json
from statistics import mean, stdev

API_URL = "http://localhost:8000"

def benchmark_prediction(model_id: str, forecast_horizon: int, num_runs: int = 5):
    """Benchmark prediction for a specific model."""
    times = []

    payload = {
        "model_ids": [model_id],
        "forecast_horizon": forecast_horizon
    }

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_id}")
    print(f"Forecast horizon: {forecast_horizon} days ({forecast_horizon * 24} hours)")
    print(f"Number of runs: {num_runs}")
    print(f"{'='*60}\n")

    for i in range(num_runs):
        start = time.time()

        response = requests.post(
            f"{API_URL}/api/predict",
            json=payload,
            timeout=300
        )

        elapsed = time.time() - start
        times.append(elapsed)

        if response.status_code == 200:
            data = response.json()
            # API returns list of prediction objects
            if isinstance(data, list) and len(data) > 0:
                num_predictions = len(data[0]['forecast'])
                latency = data[0]['metadata'].get('latency_ms', 0)
                print(f"  Run {i+1}: {elapsed:.2f}s ({num_predictions} predictions, API latency: {latency:.0f}ms)")
            else:
                print(f"  Run {i+1}: {elapsed:.2f}s")
        else:
            print(f"  Run {i+1}: FAILED - {response.status_code}")
            return None

    avg_time = mean(times)
    std_time = stdev(times) if len(times) > 1 else 0

    print(f"\n{'─'*60}")
    print(f"  Average: {avg_time:.2f}s ± {std_time:.2f}s")
    print(f"  Min:     {min(times):.2f}s")
    print(f"  Max:     {max(times):.2f}s")
    print(f"{'─'*60}\n")

    return {
        "model_id": model_id,
        "forecast_horizon": forecast_horizon,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min(times),
        "max_time": max(times),
        "all_times": times
    }


def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{API_URL}/api/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ Server is running!")
            print(f"   Available models: {len(models)}")
            return True
    except Exception as e:
        print(f"   Error: {e}")

    print("❌ Server is not running!")
    print("   Please start the server with: uvicorn api.main:app --reload")
    return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" ENERGY FORECAST API - PREDICTION BENCHMARK")
    print("="*60)

    if not check_server():
        exit(1)

    # Benchmark configurations
    configs = [
        # DL models (should show the most improvement)
        {"model_id": "LSTM", "forecast_horizon": 7},
        {"model_id": "GRU", "forecast_horizon": 7},
        {"model_id": "Bidirectional_LSTM", "forecast_horizon": 7},

        # Shorter forecasts
        {"model_id": "LSTM", "forecast_horizon": 1},
        {"model_id": "GRU", "forecast_horizon": 1},
    ]

    results = []

    for config in configs:
        result = benchmark_prediction(**config, num_runs=3)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'Horizon':<10} {'Avg Time':<15} {'Throughput':<15}")
    print("─"*65)

    for r in results:
        hours = r['forecast_horizon'] * 24
        throughput = hours / r['avg_time']  # predictions per second
        print(f"{r['model_id']:<25} {r['forecast_horizon']}d ({hours}h){'':<3} "
              f"{r['avg_time']:.2f}s ± {r['std_time']:.2f}s{'':<3} "
              f"{throughput:.1f} pred/s")

    print("\n" + "="*60)
    print("\n💡 Expected improvements with optimizations:")
    print("   - 2-3x faster for DL models (LSTM, GRU, Bidirectional)")
    print("   - Reduced memory allocations")
    print("   - Better scaling with longer forecasts")
    print("="*60 + "\n")
