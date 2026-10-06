"""AgentForge-style evaluation toolkit: forge, test and compare AI agents."""
from .client import Client
from .comparison import ComparisonResult
from .errors import (
    AdapterError,
    AgentForgeError,
    ConfigError,
    DatasetError,
    MetricError,
)
from .models import RunResult, Sample, SampleResult

__version__ = "0.1.0"
__all__ = [
    "Client", "ComparisonResult", "RunResult", "Sample", "SampleResult",
    "AgentForgeError", "AdapterError", "ConfigError", "DatasetError", "MetricError",
    "__version__",
]
