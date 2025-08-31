# Software Architecture Documentation

This document provides a detailed overview of the software architecture of the AutoGen platform, including descriptions of all classes and methods in the `agents`, `tools`, and `core` directories.

## `agents` Directory

### `analyst.py`

*   **`Analyst` class**: This agent is responsible for analysis, architecture, and quality. It uses tools to create documents and configurations.
    *   **`__init__(self, project_name: str, supervisor: BaseAgent, rag_engine: Optional[Any] = None)`**: Initializes the Analyst agent with its specific tools.
    *   **`_register_analyst_tools(self) -> None`**: Registers the tools specific to the analyst, such as creating documents, generating architecture diagrams, and creating configuration files.
    *   **`communicate(self, message: str, recipient: Optional[BaseAgent] = None) -> str`**: Handles communication for the analyst, providing structured and analytical responses.

### `base_agent.py`

*   **`ToolResult` class**: Represents the standardized result of a tool execution.
    *   **`__init__(self, status: str, result: Any = None, artifact: Optional[str] = None, error: Optional[str] = None)`**: Initializes the `ToolResult` with a status, result, optional artifact path, and optional error message.
    *   **`to_dict(self) -> Dict[str, Any]`**: Converts the `ToolResult` to a dictionary.
*   **`Tool` class**: Represents a tool that an agent can use.
    *   **`__init__(self, name: str, description: str, parameters: Dict[str, str])`**: Initializes the `Tool` with a name, description, and a dictionary of its parameters.
    *   **`to_dict(self) -> Dict[str, Any]`**: Converts the `Tool` to a dictionary.
*   **`BaseAgent` class (ABC)**: The abstract base class for all agents in the platform.
    *   **`__init__(self, name: str, role: str, personality: str, llm_config: Dict[str, Any], project_name: str, supervisor: Optional['BaseAgent'] = None, rag_engine: Optional[Any] = None)`**: Initializes the `BaseAgent` with a name, role, personality, LLM configuration, project name, supervisor, and RAG engine.
    *   **`register_tool(self, tool: Tool, implementation: Callable) -> None`**: Registers a new tool for the agent to use.
    *   **`_register_common_tools(self) -> None`**: Registers tools that are common to all agents, such as searching the RAG, sending messages, sharing discoveries, and reporting to the supervisor.
    *   **`_parse_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]`**: Parses tool calls from the LLM's response.
    *   **`_robust_json_parse(self, json_content: str) -> List[Dict[str, Any]]`**: Robustly parses JSON content using multiple strategies.
    *   **`execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult`**: Executes a given tool with the provided parameters.
    *   **`think(self, task: Dict[str, Any]) -> Dict[str, Any]`**: The agent's cognitive cycle, where it analyzes a task and creates a plan.
    *   **`act(self, plan: Dict[str, Any]) -> Dict[str, Any]`**: Executes the plan created in the `think` phase.
    *   **`_generate_structured_report(self, plan: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]`**: Generates a structured report of the agent's actions and their results.
    *   **`_format_tools_for_prompt(self) -> str`**: Formats the list of available tools for inclusion in the LLM prompt.
    *   **`answer_colleague(self, asking_agent: str, question: str) -> str`**: Responds to a question from another agent.
    *   **`communicate(self, message: str, recipient: Optional['BaseAgent'] = None) -> str`**: An abstract method for handling communication with other agents or the user.
    *   **`_load_guidelines(self) -> List[str]`**: Loads the agent's guidelines from the configuration file.
    *   **`update_state(self, **kwargs) -> None`**: Updates the agent's internal state.
    *   **`reset_exchange_counter(self, task_id: Optional[str] = None) -> None`**: Resets the communication exchange counter for a given task.
    *   **`get_agent(self, agent_name: str) -> Optional['BaseAgent']`**: Retrieves another agent by its name through the supervisor.
    *   **`receive_report(self, agent_name: str, report: Dict[str, Any]) -> None`**: Receives a report from another agent.
    *   **`receive_message(self, sender: str, message: str) -> None`**: Receives a message from another agent.
    *   **`log_interaction(self, interaction_type: str, content: Dict[str, Any]) -> None`**: Logs an interaction in the agent's history.
    *   **`add_message_to_memory(self, role: str, content: str) -> None`**: Adds a message to the agent's conversational memory.
    *   **`get_conversation_context(self) -> List[Dict[str, str]]`**: Retrieves the agent's conversation context for use in LLM prompts.
    *   **`get_agent_context(self) -> Dict[str, Any]`**: Retrieves the agent's context for logging purposes.
    *   **`generate_with_context(self, prompt: str, **kwargs) -> str`**: Generates a response from the LLM using the agent's conversation history and RAG context.
    *   **`_parse_json_from_llm_response(self, response: str) -> Dict[str, Any]`**: Parses a JSON object from the LLM's response.
    *   **`generate_with_context_enriched(self, clean_prompt: str, strategic_context: str = None, **kwargs) -> str`**: Generates a response from the LLM with temporarily enriched context.
    *   **`_calculate_final_prompt_size(self, messages: List[Dict[str, str]], rag_context: Optional[str] = None) -> int`**: Calculates the final size of the prompt to be sent to the LLM.
    *   **`_get_smart_rag_context(self, prompt: str) -> Optional[str]`**: Retrieves relevant context from the RAG engine.
    *   **`_extract_search_keywords(self, prompt: str) -> Optional[str]`**: Extracts search keywords from a prompt using a lightweight LLM.
    *   **`_get_project_charter_from_file(self) -> Optional[str]`**: Retrieves the project charter from its file.
    *   **`generate_json_with_context(self, prompt: str, **kwargs) -> Dict[str, Any]`**: Generates a JSON response from the LLM with context.

### `developer.py`

*   **`Developer` class**: This agent is responsible for implementing code, writing tests, and handling technical documentation.
    *   **`__init__(self, project_name: str, supervisor: BaseAgent, rag_engine: Optional[Any] = None)`**: Initializes the Developer agent with its specific tools.
    *   **`_register_developer_tools(self) -> None`**: Registers the tools specific to the developer, such as implementing code, creating tests, and creating project files.
    *   **`communicate(self, message: str, recipient: Optional[BaseAgent] = None) -> str`**: Handles communication for the developer, providing technical and code-oriented responses.

### `supervisor.py`

*   **`_UnifiedMilestoneManager` class**: Manages the project milestones.
    *   **`__init__(self, supervisor)`**: Initializes the milestone manager.
    *   **`_create_milestone_structure(self, **data)`**: Creates the basic structure for a milestone.
    *   **`_assign_sequential_ids(self)`**: Assigns sequential IDs to all milestones.
    *   **`create_initial_milestones(self, milestones_data)`**: Creates the initial set of project milestones.
    *   **`insert_correction_after_current(self, **data)`**: Inserts a correction milestone after the current one.
    *   **`replace_future_milestones(self, new_milestones_data)`**: Replaces all future milestones with a new set.
    *   **`complete_current_and_advance(self, status='completed')`**: Completes the current milestone and advances to the next one.
    *   **`get_current_milestone(self)`**: Returns the current milestone.
    *   **`find_milestone(self, milestone_id)`**: Finds a milestone by its ID.
*   **`Supervisor` class**: This agent orchestrates the project, manages milestones, and coordinates the other agents.
    *   **`__init__(self, project_name: str, project_prompt: str, rag_engine: Optional[Any] = None)`**: Initializes the Supervisor agent.
    *   **`_register_supervisor_tools(self) -> None`**: Registers the tools specific to the supervisor, such as assigning agents to milestones, getting progress reports, and managing milestones.
    *   **`think(self, task: Dict[str, Any]) -> Dict[str, Any]`**: The supervisor's cognitive cycle, where it analyzes the project and plans the milestones.
    *   **`act(self, plan: Dict[str, Any]) -> Dict[str, Any]`**: Prepares the project for execution by creating the necessary agents and sharing the plan.
    *   **`create_agents(self) -> Dict[str, BaseAgent]`**: Creates the Analyst and Developer agents.
    *   **`orchestrate(self) -> Dict[str, Any]`**: Orchestrates the entire project from start to finish, executing each milestone in sequence.
    *   **`_generate_pure(self, prompt: str, **kwargs) -> str`**: Generates a response from the LLM without using the RAG context.
    *   **`_execute_milestone(self, milestone: Dict[str, Any]) -> Dict[str, Any]`**: Executes a single project milestone.
    *   **`_create_milestones_from_analysis(self, analysis: str, project_prompt: str, use_pure_generation: bool = False) -> List[Dict[str, Any]]`**: Creates the project milestones based on the initial analysis.
    *   **`_get_default_milestones(self) -> List[Dict[str, Any]]`**: Returns a default set of milestones to be used as a fallback.
    *   **`_create_fallback_plan(self, project_prompt: str) -> Dict[str, Any]`**: Creates a fallback plan if the initial planning fails.
    *   **`_generate_project_summary(self, orchestration_result: Dict[str, Any]) -> None`**: Generates a summary of the completed project.
    *   **`get_agent(self, agent_name: str) -> Optional[BaseAgent]`**: Retrieves an agent by its name.
    *   **`get_all_agents_with_roles(self) -> Dict[str, str]`**: Returns a dictionary of all agents and their roles.
    *   **`handle_escalation(self, from_agent: str, issue: str) -> str`**: Handles an escalation from another agent.
    *   **`get_progress_report(self) -> Dict[str, Any]`**: Generates a progress report for the project.
    *   **`communicate(self, message: str, recipient: Optional[BaseAgent] = None) -> str`**: Handles communication for the supervisor.
    *   **`_update_plan_in_rag(self, change_description: str) -> None`**: Updates the project plan in the RAG.
    *   **`_evaluate_plan_after_interaction(self, interaction_type: str, content: str) -> None`**: Evaluates the project plan after an interaction with another agent.
    *   **`_format_milestones_for_evaluation(self, milestones: List[Dict[str, Any]]) -> str`**: Formats the milestones for evaluation.
    *   **`_apply_plan_changes(self, suggested_changes: List[Dict[str, Any]]) -> None`**: Applies changes to the project plan.
    *   **`_update_journal_de_bord(self, interaction_type: str, content: str, evaluation: Dict[str, Any]) -> None`**: Updates the project's logbook.
    *   **`receive_report(self, agent_name: str, report: Dict[str, Any]) -> None`**: Receives a report from another agent.
    *   **`_extract_milestone_from_report(self, agent_name: str, report: Dict[str, Any]) -> str`**: Extracts the milestone ID from a report.
    *   **`_reset_milestone_buffer(self, new_milestone_id: str) -> None`**: Resets the buffer for milestone reports.
    *   **`_evaluate_milestone_with_context(self) -> None`**: Evaluates a milestone with the full context of all reports.
    *   **`_trigger_plan_evaluation_with_context(self, automatic_report: Dict[str, Any], manual_reports: List[Dict[str, Any]]) -> None`**: Triggers a plan evaluation with full context.
    *   **`_create_journal_entry(self, entry_type: str, content: str, details: Dict[str, Any] = None) -> None`**: Creates an entry in the project's logbook.
    *   **`_request_human_validation(self, reason: str, recommended_action: str, milestone_details: Dict[str, Any] = None, agent_reports: List[Dict[str, Any]] = None, verification_info: Dict[str, Any] = None) -> Dict[str, Any]`**: Requests human validation for a critical decision.
    *   **`_analyze_user_instruction_for_plan_adjustment(self, instruction: str) -> str`**: Analyzes a user's instruction for adjusting the project plan.
    *   **`_generate_milestone_summary(self, milestone_result: Dict[str, Any]) -> str`**: Generates a summary of a completed milestone.
    *   **`_get_project_charter_from_file(self) -> str`**: Retrieves the project charter from its file.
    *   **`_verify_milestone_completion(self, milestone: Dict[str, Any], milestone_result: Dict[str, Any]) -> Dict[str, Any]`**: Verifies the completion of a milestone.
    *   **`_deep_milestone_evaluation(self, milestone: Dict[str, Any], milestone_result: Dict[str, Any], structured_reports: List[Dict[str, Any]]) -> Dict[str, Any]`**: Performs a deep evaluation of a milestone.
    *   **`_apply_verification_decision(self, verification: Dict[str, Any], current_milestone: Dict[str, Any]) -> None`**: Applies a verification decision to a milestone.
    *   **`_force_milestone_approval(self, milestone: Dict[str, Any], reason: str) -> None`**: Forces the approval of a milestone.
    *   **`_mark_milestone_partially_completed(self, milestone: Dict[str, Any], reason: str) -> None`**: Marks a milestone as partially completed.
    *   **`adjust_plan(self, reason: str) -> None`**: Adjusts the project plan.
    *   **`halt_orchestration(self) -> None`**: Halts the project orchestration.
    *   **`_parse_json_from_llm_response(self, response: str) -> Dict[str, Any]`**: Parses a JSON object from the LLM's response.
    *   **`generate_json_with_context(self, prompt: str, **kwargs) -> Dict[str, Any]`**: Generates a JSON response from the LLM with context.
    *   **`generate_with_context_enriched(self, clean_prompt: str, strategic_context: str = None, **kwargs) -> str`**: Generates a response from the LLM with temporarily enriched context.
    *   **`_calculate_final_prompt_size(self, messages: List[Dict[str, str]], rag_context: Optional[str] = None) -> int`**: Calculates the final size of the prompt to be sent to the LLM.
    *   **`_get_smart_rag_context(self, prompt: str) -> Optional[str]`**: Retrieves relevant context from the RAG engine.
    *   **`_extract_search_keywords(self, prompt: str) -> Optional[str]`**: Extracts search keywords from a prompt using a lightweight LLM.

## `tools` Directory

### `analyst_tools.py`

*   **`tool_create_document(agent, params)`**: Creates a documentation file (e.g., specifications, guides).
*   **`tool_generate_architecture_diagrams(agent, params)`**: Generates architecture diagrams in Mermaid format.
*   **`tool_generate_configuration_files(agent, params)`**: Generates configuration files for the project.

### `base_tools.py`

*   **`tool_search_context(agent, params)`**: Searches for relevant context in the RAG.
*   **`tool_send_message_to_agent(agent, params)`**: Sends a message or question to another agent.
*   **`tool_share_discovery(agent, params)`**: Shares an important discovery in the working memory.
*   **`tool_report_to_supervisor(agent, params)`**: Sends a report to the supervisor.

### `developer_tools.py`

*   **`tool_implement_code(agent, params)`**: Implements source code and places it in the `src/` directory.
*   **`tool_create_tests(agent, params)`**: Creates test files and places them in the `src/tests/` directory.
*   **`tool_create_project_file(agent, params)`**: Creates a project file (e.g., README, `package.json`) at the root of the project.

### `supervisor_tools.py`

*   **`tool_assign_agents_to_milestone(supervisor, params)`**: Assigns agents to a specific milestone.
*   **`tool_get_progress_report(supervisor, params)`**: Generates a progress report for the project.
*   **`tool_add_milestone(supervisor, params)`**: Adds a new milestone to the project plan.
*   **`tool_modify_milestone(supervisor, params)`**: Modifies an existing milestone.
*   **`tool_remove_milestone(supervisor, params)`**: Removes a milestone from the project plan.
*   **`tool_add_correction(supervisor, params)`**: Adds a correction milestone to the project plan.

## `core` Directory

### `cli_interface.py`

*   **`CLIInterface` class**: Manages the command-line interface for user interaction.
    *   **`__init__(self)`**: Initializes the `rich` console.
    *   **`display_welcome(self) -> None`**: Displays a welcome message.
    *   **`get_project_name(self) -> str`**: Prompts the user for and validates the project name.
    *   **`get_project_prompt(self) -> str`**: Prompts the user for the project description.
    *   **`display_project_created(self, project_name: str, project_path: str) -> None`**: Displays a confirmation that the project has been created.
    *   **`display_info(self, message: str) -> None`**: Displays an informational message.
    *   **`display_error(self, message: str) -> None`**: Displays an error message.
    *   **`display_warning(self, message: str) -> None`**: Displays a warning message.
    *   **`ask_confirmation(self, question: str, default: bool = True) -> bool`**: Asks the user for confirmation.
    *   **`display_progress(self, message: str) -> None`**: Displays a progress message.

### `communication.py`

*   **`MessagePriority` enum**: Defines the priority levels for messages.
*   **`MessageType` enum**: Defines the types of messages that can be sent.
*   **`CommunicationProtocol` enum**: Defines the communication protocols that can be used.
*   **`Message` class**: Represents a message exchanged between agents.
    *   **`to_dict(self) -> Dict[str, Any]`**: Converts the message to a dictionary.
    *   **`from_dict(cls, data: Dict[str, Any]) -> 'Message'`**: Creates a `Message` object from a dictionary.
*   **`Channel` class**: Represents a communication channel.
    *   **`__init__(self, name: str, protocol: CommunicationProtocol)`**: Initializes the channel.
    *   **`subscribe(self, agent_name: str) -> None`**: Subscribes an agent to the channel.
    *   **`unsubscribe(self, agent_name: str) -> None`**: Unsubscribes an agent from the channel.
    *   **`add_filter(self, filter_func: Callable[[Message], bool]) -> None`**: Adds a filter to the channel.
    *   **`should_deliver(self, message: Message) -> bool`**: Checks if a message should be delivered based on the channel's filters.
    *   **`publish(self, message: Message) -> None`**: Publishes a message to the channel.
    *   **`get_messages(self, agent_name: str, limit: int = 10) -> List[Message]`**: Retrieves messages for a subscribed agent.
*   **`MessageBroker` class**: The central message broker for the platform.
    *   **`__init__(self)`**: Initializes the message broker.
    *   **`_create_default_channels(self)`**: Creates the default communication channels.
    *   **`create_channel(self, name: str, protocol: CommunicationProtocol) -> Channel`**: Creates a new communication channel.
    *   **`register_agent(self, agent_name: str, endpoint: 'AgentEndpoint') -> None`**: Registers an agent with the broker.
    *   **`unregister_agent(self, agent_name: str) -> None`**: Unregisters an agent from the broker.
    *   **`send_message(self, message: Message) -> bool`**: Sends a message through the broker.
    *   **`_handle_broadcast(self, message: Message) -> bool`**: Handles a broadcast message.
    *   **`_handle_direct(self, message: Message) -> bool`**: Handles a direct message.
    *   **`_handle_request(self, message: Message) -> bool`**: Handles a request message.
    *   **`_handle_response(self, message: Message) -> bool`**: Handles a response message.
    *   **`_handle_channel_message(self, message: Message) -> bool`**: Handles a message sent to a specific channel.
    *   **`_timeout_request(self, request_id: str) -> None`**: Handles a request that has timed out.
    *   **`get_message_stats(self) -> Dict[str, Any]`**: Retrieves statistics about the messages sent through the broker.
*   **`AgentEndpoint` class**: The communication endpoint for an individual agent.
    *   **`__init__(self, agent_name: str, callback: Callable[[Message], None])`**: Initializes the agent endpoint.
    *   **`receive_message(self, message: Message) -> None`**: Receives a message and calls the agent's callback.
    *   **`get_inbox(self, limit: int = 10) -> List[Message]`**: Retrieves messages from the agent's inbox.
    *   **`clear_inbox(self) -> None`**: Clears the agent's inbox.
*   **`CommunicationManager` class**: Manages communication for an agent.
    *   **`__init__(self, agent_name: str, broker: Optional[MessageBroker] = None)`**: Initializes the communication manager.
    *   **`_register(self) -> None`**: Registers the agent with the message broker.
    *   **`_handle_incoming_message(self, message: Message) -> None`**: Handles an incoming message.
    *   **`register_handler(self, message_type: MessageType, handler: Callable[[Message], None]) -> None`**: Registers a handler for a specific message type.
    *   **`subscribe_to_channel(self, channel_name: str) -> None`**: Subscribes the agent to a communication channel.
    *   **`send_message(self, content: Dict[str, Any], recipient: Optional[str] = None, message_type: MessageType = MessageType.DIRECT, priority: MessagePriority = MessagePriority.NORMAL, protocol: CommunicationProtocol = CommunicationProtocol.ASYNC, correlation_id: Optional[str] = None, ttl: Optional[int] = None) -> str`**: Sends a message.
    *   **`broadcast(self, content: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> str`**: Broadcasts a message to all agents.
    *   **`request(self, recipient: str, content: Dict[str, Any], timeout: int = 30) -> str`**: Sends a request to another agent.
    *   **`respond(self, request_message: Message, response_content: Dict[str, Any]) -> str`**: Responds to a request from another agent.
    *   **`alert(self, content: Dict[str, Any]) -> str`**: Sends an alert.
    *   **`share_discovery(self, discovery: str, metadata: Optional[Dict[str, Any]] = None) -> str`**: Shares a discovery with other agents.
    *   **`get_messages(self, limit: int = 10, message_type: Optional[MessageType] = None) -> List[Message]`**: Retrieves messages from the agent's inbox.
    *   **`get_stats(self) -> Dict[str, Any]`**: Retrieves communication statistics for the agent.
*   **`get_global_broker() -> MessageBroker`**: Returns the global instance of the message broker.

### `global_rate_limiter.py`

*   **`GlobalRateLimiter` class**: Manages the global rate limit for all API calls to prevent exceeding quotas.
    *   **`__new__(cls)`**: Implements the singleton pattern to ensure only one instance of the rate limiter exists.
    *   **`__init__(self)`**: Initializes the rate limiter.
    *   **`_get_rate_limit_interval(self) -> float`**: Retrieves the rate limit interval from the configuration file.
    *   **`enforce_rate_limit(self, connector_name: str = "Unknown") -> None`**: Enforces the rate limit, pausing execution if necessary.
    *   **`get_statistics(self) -> dict`**: Returns statistics about the rate limiter's activity.
    *   **`reset_statistics(self) -> None`**: Resets the rate limiter's statistics.
    *   **`update_rate_limit_interval(self, new_interval: float) -> None`**: Updates the rate limit interval.
    *   **`get_instance(cls) -> 'GlobalRateLimiter'`**: Returns the singleton instance of the rate limiter.

### `json_parser.py`

*   **`RobustJSONParser` class**: A centralized and robust JSON parser with multiple fallback strategies.
    *   **`__init__(self, logger_name: str = "JSONParser")`**: Initializes the JSON parser.
    *   **`parse_universal(self, content: str, return_type: str = 'auto') -> Any`**: A universal parsing method that tries all available strategies to parse a JSON object.
    *   **`parse_tool_array(self, content: str) -> List[Dict[str, Any]]`**: Parses a JSON array of tools.
    *   **`parse_llm_response(self, content: str) -> Dict[str, Any]`**: Parses a simple JSON response from an LLM.
    *   **`parse_config_object(self, content: str) -> Dict[str, Any]`**: Parses a JSON configuration object.
    *   **`_execute_strategies(self, content: str, strategies: List[Callable], return_list: bool = True) -> Any`**: Executes a list of parsing strategies in cascade until one succeeds.
*   **`get_json_parser(logger_name: str = "JSONParser") -> RobustJSONParser`**: Returns an instance of the `RobustJSONParser`.

### `lightweight_llm_service.py`

*   **`LightweightLLMService` class**: A service for performing quick and lightweight LLM tasks, such as keyword extraction and summarization.
    *   **`__init__(self, project_name: str = "System")`**: Initializes the lightweight LLM service.
    *   **`extract_keywords(self, prompt: str) -> str`**: Extracts relevant keywords from a prompt.
    *   **`summarize_context(self, context: str) -> str`**: Summarizes a long context.
    *   **`summarize_conversation(self, conversation_text: str) -> str`**: Summarizes a conversation history.
    *   **`self_evaluate_mission(self, objective: str, artifacts: list, issues: list) -> dict`**: Evaluates the success of a mission based on its objective and results.
    *   **`summarize_constraints(self, project_charter: str, task_description: str) -> str`**: Summarizes the project constraints relevant to a specific task.
*   **`get_lightweight_llm_service(project_name: str = "System") -> LightweightLLMService`**: Returns an instance of the `LightweightLLMService`.

### `llm_connector.py`

*   **`TimeoutError` exception**: A custom exception raised when an LLM call times out.
*   **`timeout_handler(signum, frame)`**: A signal handler for LLM timeouts.
*   **`LLMConnector` class (ABC)**: The abstract base class for all LLM connectors.
    *   **`generate(self, prompt: str, system_prompt: Optional[str] = None, agent_context: Optional[Dict[str, Any]] = None, **kwargs) -> str`**: An abstract method for generating a response from a prompt.
    *   **`generate_with_messages(self, messages: List[Dict[str, str]], agent_context: Optional[Dict[str, Any]] = None, **kwargs) -> str`**: An abstract method for generating a response from a history of messages.
    *   **`generate_json(self, prompt: str, system_prompt: Optional[str] = None, schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]`**: An abstract method for generating a JSON response.
*   **`MistralConnector` class**: A connector for the Mistral API.
    *   **`__init__(self, api_key: Optional[str] = None, model: Optional[str] = None)`**: Initializes the Mistral connector.
    *   **`_enforce_rate_limit(self)`**: Enforces the global rate limit for API calls.
    *   **`generate(self, prompt: str, system_prompt: Optional[str] = None, agent_context: Optional[Dict[str, Any]] = None, **kwargs) -> str`**: Generates a response using the Mistral API.
    *   **`generate_with_messages(self, messages: List[Dict[str, str]], agent_context: Optional[Dict[str, Any]] = None, **kwargs) -> str`**: Generates a response from a history of messages using the Mistral API.
    *   **`generate_json(self, prompt: str, system_prompt: Optional[str] = None, schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]`**: Generates a JSON response using the Mistral API.
    *   **`_validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None`**: Validates a JSON object against a given schema.
*   **`MistralEmbedConnector` class**: A connector for the Mistral embedding API.
    *   **`__init__(self, api_key: Optional[str] = None)`**: Initializes the Mistral embedding connector.
    *   **`_enforce_rate_limit(self)`**: Enforces the global rate limit for API calls.
    *   **`embed_texts(self, texts: List[str]) -> np.ndarray`**: Generates embeddings for a list of texts.
*   **`DeepSeekConnector` class**: A connector for the DeepSeek API.
    *   **`__init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat")`**: Initializes the DeepSeek connector.
    *   **`_enforce_rate_limit(self)`**: Enforces the global rate limit for API calls.
    *   **`generate(self, prompt: str, system_prompt: Optional[str] = None, agent_context: Optional[Dict[str, Any]] = None, **kwargs) -> str`**: Generates a response using the DeepSeek API.
    *   **`generate_with_messages(self, messages: List[Dict[str, str]], agent_context: Optional[Dict[str, Any]] = None, **kwargs) -> str`**: Generates a response from a history of messages using the DeepSeek API.
    *   **`generate_json(self, prompt: str, system_prompt: Optional[str] = None, schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]`**: Generates a JSON response using the DeepSeek API.
*   **`LLMFactory` class**: A factory for creating LLM connectors.
    *   **`create(cls, provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> LLMConnector`**: Creates or retrieves an instance of an LLM connector.
    *   **`_detect_provider(cls, model_name: str) -> str`**: Detects the LLM provider from the model name.
    *   **`register_connector(cls, name: str, connector_class: type)`**: Registers a new connector with the factory.
    *   **`clear_cache(cls)`**: Clears the cache of connector instances.
    *   **`get_instance_count(cls) -> int`**: Returns the number of cached connector instances.

### `metrics_visualizer.py`

*   **`ModernMetricsCollector` class**: Collects metrics from the project's logs.
    *   **`__init__(self, project_name: str)`**: Initializes the metrics collector.
    *   **`collect_all_metrics(self) -> Dict[str, Any]`**: Collects all available metrics from the logs.
    *   **`_collect_from_main_logs(self, logs_path: Path) -> Dict[str, Any]`**: Collects metrics from the main log files.
    *   **`_collect_from_llm_debug(self, llm_debug_path: Path) -> Dict[str, Any]`**: Collects metrics from the LLM debug logs.
    *   **`_process_main_log_entry(self, entry: Dict[str, Any], metrics: Dict[str, Any])`**: Processes a single entry from the main logs.
    *   **`_process_llm_log_entry(self, entry: Dict[str, Any], metrics: Dict[str, Any])`**: Processes a single entry from the LLM debug logs.
    *   **`_calculate_final_metrics(self, main_logs: Dict[str, Any], llm_logs: Dict[str, Any]) -> Dict[str, Any]`**: Calculates the final metrics from the collected data.
    *   **`_empty_metrics(self) -> Dict[str, Any]`**: Returns an empty metrics dictionary.
*   **`ModernMetricsVisualizer` class**: Visualizes the collected metrics in an HTML dashboard.
    *   **`__init__(self, project_name: str)`**: Initializes the metrics visualizer.
    *   **`generate_dashboard(self, rag_singleton=None) -> str`**: Generates a complete, self-contained HTML dashboard.
    *   **`_generate_modern_html_dashboard(self, metrics: Dict[str, Any]) -> str`**: Generates the HTML for the dashboard.
    *   **`_generate_bar_chart_html(self, data: List[Dict[str, Any]], label_key: str, value_key: str) -> str`**: Generates the HTML for a bar chart.
    *   **`_generate_llm_agent_stats_html(self, llm_stats: Dict[str, Dict[str, Any]]) -> str`**: Generates the HTML for the LLM statistics per agent.
    *   **`_generate_timeline_html(self, timeline: List[Dict[str, Any]]) -> str`**: Generates the HTML for the project timeline.
*   **`get_metrics_collector(project_name: str) -> ModernMetricsCollector`**: Returns an instance of the `ModernMetricsCollector`.
*   **`generate_dashboard(project_name: str) -> str`**: Generates a dashboard for a given project.

### `project_manager.py`

*   **`ProjectManager` class**: Manages the creation and structure of projects.
    *   **`__init__(self, base_path: Optional[Path] = None)`**: Initializes the project manager.
    *   **`create_project_structure(self, project_name: str) -> Optional[Path]`**: Creates the complete directory structure for a new project.
    *   **`_create_readme(self, project_path: Path, project_name: str) -> None`**: Creates the `README.md` file for the project.
    *   **`_create_project_config(self, project_path: Path, project_name: str) -> None`**: Creates the configuration file for the project.
    *   **`_create_gitignore(self, project_path: Path) -> None`**: Creates the `.gitignore` file for the project.
    *   **`load_project(self, project_name: str) -> Optional[Dict[str, Any]]`**: Loads the configuration of an existing project.
    *   **`list_projects(self) -> list[str]`**: Lists all existing projects.

### `rag_engine.py`

*   **`CompressionManager` class**: Manages the compression of the RAG index to save space and improve performance.
    *   **`__init__(self, rag_engine: 'RAGEngine')`**: Initializes the compression manager.
    *   **`compression_context(self)`**: A context manager to prevent recursion during compression.
    *   **`should_compress(self) -> bool`**: Determines if the RAG index needs to be compressed.
    *   **`compress(self) -> Dict[str, Any]`**: Compresses the RAG index by summarizing related entries.
    *   **`_group_entries_for_compression(self) -> Dict[Tuple[str, str], List[Tuple[int, Dict]]]`**: Groups index entries by agent and milestone for compression.
    *   **`_is_project_file(self, metadata: Dict[str, Any]) -> bool`**: Checks if an index entry is a project file that should be preserved.
    *   **`_create_summary(self, agent: str, milestone: str, entries: List[Tuple[int, Dict]]) -> Optional[Dict]`**: Creates a summary of a group of index entries.
    *   **`_rebuild_index(self, preserved_entries: List[Tuple[int, Dict]], new_summaries: List[Dict])`**: Rebuilds the RAG index with the preserved entries and new summaries.
*   **`RAGEngine` class**: The Retrieval-Augmented Generation engine, responsible for indexing and searching for contextual information.
    *   **`__init__(self, project_name: str, embedding_model: Optional[str] = None)`**: Initializes the RAG engine.
    *   **`_init_embedding_model(self)`**: Initializes the embedding model.
    *   **`_init_index(self)`**: Initializes the FAISS index.
    *   **`_create_new_index(self)`**: Creates a new FAISS index.
    *   **`_init_working_memory(self)`**: Initializes the working memory.
    *   **`_create_new_working_memory(self)`**: Creates a new working memory.
    *   **`_load_metadata(self)`**: Loads the index metadata from a file.
    *   **`_save_metadata(self)`**: Saves the index metadata to a file.
    *   **`_save_index(self)`**: Saves the FAISS index to a file.
    *   **`_save_working_memory(self)`**: Saves the working memory to a file.
    *   **`_chunk_text(self, text: str) -> List[str]`**: Chunks a text document into smaller pieces for indexing.
    *   **`_index_raw(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]], to_working_memory: bool = False) -> int`**: Indexes embeddings and metadata without triggering compression.
    *   **`index_document(self, content: str, metadata: Dict[str, Any], to_working_memory: bool = False) -> int`**: Indexes a document.
    *   **`trigger_compression_if_needed(self)`**: Triggers compression of the RAG index if needed.
    *   **`index_log_entry(self, log_entry: Dict[str, Any]) -> bool`**: Indexes a log entry.
    *   **`index_to_working_memory(self, content: str, metadata: Dict[str, Any]) -> int`**: Indexes content to the working memory.
    *   **`index_project_files(self) -> int`**: Indexes the project's source files.
    *   **`search(self, query: str, top_k: Optional[int] = None, filter_metadata: Optional[Dict[str, Any]] = None, include_working_memory: bool = True) -> List[Dict[str, Any]]`**: Searches the RAG index for relevant information.
    *   **`_apply_confidence_scoring(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]`**: Applies a confidence score to the search results.
    *   **`get_proactive_context(self, task_description: str, deliverables: List[str] = None, agent_name: str = None) -> str`**: Retrieves proactive context for a given task.
    *   **`merge_working_memory_to_main(self, milestone_id: str = None)`**: Merges the working memory into the main RAG index.
    *   **`clear_working_memory(self)`**: Clears the working memory.
    *   **`get_memory_usage(self) -> Dict[str, Any]`**: Returns the memory usage of the RAG engine.
    *   **`create_summary(self) -> Dict[str, Any]`**: Creates a summary of the content in the RAG index.
    *   **`save_all(self)`**: Saves all RAG data to disk.
    *   **`clear_index(self)`**: Clears the RAG index.

## `utils` Directory

### `logger.py`

*   **`setup_logger(project_name: str, log_level: int = logging.INFO) -> logging.Logger`**: Sets up a structured JSON logger for the project.
*   **`get_logger(project_name: str) -> logging.Logger`**: Retrieves an already configured logger.
*   **`log_with_context(logger: logging.Logger, level: int, message: str, **kwargs)`**: Logs a message with additional context.

## `root` Directory

### `main.py`

*   **`main()`**: The main entry point of the application. It handles command-line arguments, sets up the project, and starts the orchestration process.
*   **`run_project(project_name: str, project_prompt: str)`**: Initializes and runs a new project.
*   **`load_and_run_project(project_name: str)`**: Loads and runs an existing project.
*   **`list_existing_projects()`**: Lists all existing projects.
*   **`show_project_dashboard(project_name: str)`**: Generates and displays the project dashboard.
