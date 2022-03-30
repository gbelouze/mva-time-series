from pathlib import Path
import glob

import pandas as pd  # type: ignore

base_dir = Path(__file__).resolve().parents[2]
data_dir = base_dir / "data" / "ydata-labeled-time-series-anomalies-v1_0"
benchmarks = {
    data_dir / "A1Benchmark": "real_",
    data_dir / "A2Benchmark": "synthetic_",
    data_dir / "A3Benchmark": "A3Benchmark-TS",
    data_dir / "A4Benchmark": "A4Benchmark-TS",
}


def read(benchmark_index: int, dataset_index: int) -> pd.DataFrame:
    benchmark = data_dir / f"A{benchmark_index}Benchmark"
    base = benchmarks[benchmark]
    return pd.read_csv(benchmark / f"{base}{dataset_index}.csv")


class BenchmarkDataset():
    def __init__(self, benchmark_index : int):
        benchmark = data_dir / f"A{benchmark_index}Benchmark"
        base = benchmarks[benchmark]

        s = str(benchmark / f"{base}*.csv")
        self.files = glob.glob(s)
        self.len = len(self.files)

    def read(self, i : int):
        if i < 0 or i >= self.len:
            raise IndexError('index out of dataset')
        df = pd.read_csv(self.files[i])
        df = df.rename(columns={"anomaly": "is_anomaly", "timestamps": "timestamp"})

        return df