## Data Loading from Raw CSV

import pandas as pd
import yaml
import os

def project_root() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current, '..', '..'))


def load_config() -> dict:
    root = project_root()
    config_path = os.path.join(root, 'configs', 'paths.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config, root


def _to_snake_case(name: str) -> str:
    """
    Convert Camel case columsn names to snake_case
    """

    import re
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()

def load_raw_data(filepath: str = None) -> pd.DataFrame:

    if filepath is None:
        config, root = load_config()
        filepath = os.path.join(root, config["data"]["raw"])

    df = pd.read_csv(filepath)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    df.columns = [_to_snake_case(col) for col in df.columns]

    if "customer_id" in df.columns:
        df = df.set_index("customer_id")

    
    return df

if __name__ == "__main__":
    ### Quick check
    df = load_raw_data()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")


    


    
