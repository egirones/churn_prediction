import pytest

def df_plugin():
    return None

def pytest_configure():
    pytest.df = df_plugin()