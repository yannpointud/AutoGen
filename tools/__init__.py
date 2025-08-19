"""
Module tools pour AutoGen.
Contient les implémentations des outils pour tous les agents.
"""

from .base_tools import (
    tool_search_context,
    tool_send_message_to_agent,
    tool_share_discovery,
    tool_report_to_supervisor
)

from .developer_tools import (
    tool_implement_code,
    tool_create_tests,
    tool_create_project_file
)

from .analyst_tools import (
    tool_create_document,
    tool_generate_architecture_diagrams,
    tool_generate_configuration_files
)

from .supervisor_tools import (
    tool_assign_agents_to_milestone,
    tool_get_progress_report,
    tool_add_milestone,
    tool_modify_milestone,
    tool_remove_milestone
)

__all__ = [
    # Base tools (common to all agents)
    'tool_search_context',
    'tool_send_message_to_agent',
    'tool_share_discovery',
    'tool_report_to_supervisor',
    
    # Developer tools
    'tool_implement_code',
    'tool_create_tests',
    'tool_create_project_file',
    
    # Analyst tools
    'tool_create_document',
    'tool_generate_architecture_diagrams',
    'tool_generate_configuration_files',
    
    # Supervisor tools
    'tool_assign_agents_to_milestone',
    'tool_get_progress_report',
    'tool_add_milestone',
    'tool_modify_milestone',
    'tool_remove_milestone',
]