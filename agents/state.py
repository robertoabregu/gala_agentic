from typing import Any, Dict, List, Optional, TypedDict, NotRequired


class AgentState(TypedDict):
    session_id: str
    memory: Dict[str, Any]
    pending_route: str
    question: str
    original_question: str
    standalone_question: str
    is_followup: bool
    route: str
    search_query: str
    documents: List[Dict[str, Any]]
    context: str
    answer: str
    final_answer: str
    error: Optional[str]
    topic: NotRequired[str]
    tool_name: NotRequired[str]
    tool_input: NotRequired[Dict[str, Any]]
    tool_output: NotRequired[Dict[str, Any]]
    needs_clarification: NotRequired[bool]
    missing_fields: NotRequired[List[str]]
    user_location: NotRequired[Dict[str, Any]]
    media: NotRequired[Dict[str, Any]]
    credit_card_statement: NotRequired[Dict[str, Any]]
    send_csat: NotRequired[bool]
    csat_template_sid: NotRequired[str]
