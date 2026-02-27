"""
agent_pipeline.pipeline_integration
------------------------------------
Compat shim -- re-exports ChatAgent as PipelineAgent so
bridge.py works without import changes.
"""

from .agent import ChatAgent

PipelineAgent = ChatAgent
