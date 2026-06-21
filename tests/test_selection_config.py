from src.config.selection_model import SelectionConfig


def test_default_instantiation() -> None:
    config = SelectionConfig()
    assert config.procedural_variance == 0.20
    assert config.timeout_limit_sec == 60.0
    assert config.async_batch_size == 4


def test_custom_values() -> None:
    config = SelectionConfig(procedural_variance=0.10, timeout_limit_sec=30.0, async_batch_size=8)
    assert config.procedural_variance == 0.10
    assert config.timeout_limit_sec == 30.0
    assert config.async_batch_size == 8


def test_partial_override() -> None:
    config = SelectionConfig(async_batch_size=2)
    assert config.procedural_variance == 0.20
    assert config.timeout_limit_sec == 60.0
    assert config.async_batch_size == 2


def test_type_validation() -> None:
    config = SelectionConfig(procedural_variance=0.20, timeout_limit_sec=60.0, async_batch_size=4)
    assert isinstance(config.procedural_variance, float)
    assert isinstance(config.timeout_limit_sec, float)
    assert isinstance(config.async_batch_size, int)


def test_field_names_match_spec() -> None:
    fields = set(SelectionConfig.model_fields.keys())
    assert fields == {"procedural_variance", "timeout_limit_sec", "async_batch_size"}


def test_load_unknown_fields_filtered() -> None:
    config = SelectionConfig(procedural_variance=0.30, timeout_limit_sec=45.0, async_batch_size=6)
    assert config.procedural_variance == 0.30
    assert config.timeout_limit_sec == 45.0
    assert config.async_batch_size == 6
