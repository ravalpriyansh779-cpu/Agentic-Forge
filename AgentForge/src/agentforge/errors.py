"""Exception hierarchy for AgentForge."""


class AgentForgeError(Exception):
    """Base class for all AgentForge errors."""


class ConfigError(AgentForgeError):
    """Bad configuration (unknown adapter, missing API key, ...)."""


class AdapterError(AgentForgeError):
    """An agent adapter failed to produce an answer."""


class MetricError(AgentForgeError):
    """A metric could not be computed."""


class DatasetError(AgentForgeError):
    """A dataset could not be loaded or is malformed."""
