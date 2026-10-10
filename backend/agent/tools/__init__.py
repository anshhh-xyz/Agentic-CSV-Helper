"""Importing this package registers every tool (see each module's @tool decorators)."""

from agent.tools import (  # noqa: F401
    mathematical_operations,
    data_manipulation,
    data_summary,
    graphs,
    correlation_analysis,
    outlier_analysis,
    memory,
)
