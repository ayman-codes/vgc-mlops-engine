import pandas as pd
import pytest

GOLD_PATH = "data/processed/gold_tensors.parquet"

@pytest.fixture
def gold_df():
    return pd.read_parquet(GOLD_PATH)

def test_continuous_numerical_space(gold_df):
    object_cols = gold_df.select_dtypes(include=['object']).columns
    assert len(object_cols) == 0, f"Failure: Non-numerical columns detected -> {object_cols}"

def test_null_eradication(gold_df):
    null_count = gold_df.isna().sum().sum()
    assert null_count == 0, f"Failure: Matrix contains {null_count} unhandled null values."

def test_tensor_type_coercion(gold_df):
    invalid_dtypes = [dtype for dtype in gold_df.dtypes if dtype not in ['float32']]
    assert len(invalid_dtypes) == 0, "Failure: Columns failed PyTorch float32 coercion."

def test_matrix_dimensionality(gold_df):
    assert gold_df.shape[0] > 0
    assert gold_df.shape[1] > 20, "Failure: OHE expansion failed to execute."