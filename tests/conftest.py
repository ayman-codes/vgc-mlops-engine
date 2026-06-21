from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def gold_df() -> pd.DataFrame:
    return pd.DataFrame({"feat_1": [0.5, 0.3], "feat_2": [0.2, 0.7]}).astype("float32")


@pytest.fixture
def mock_battle() -> MagicMock:
    return MagicMock()
