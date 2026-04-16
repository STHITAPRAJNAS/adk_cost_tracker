import sys
from unittest.mock import MagicMock

# Mock google.adk for tests
mock_google = MagicMock()
sys.modules["google"] = mock_google
sys.modules["google.adk"] = MagicMock()
sys.modules["google.adk.agents"] = MagicMock()
sys.modules["google.adk.agents.callback_context"] = MagicMock()
sys.modules["google.adk.models"] = MagicMock()
sys.modules["google.adk.models.llm_response"] = MagicMock()
sys.modules["google.adk.plugins"] = MagicMock()
sys.modules["google.adk.plugins.base_plugin"] = MagicMock()
