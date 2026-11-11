"""API tests for simulated and normal CloudFlow chat modes."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import api, rag_pipeline
from app.rag_pipeline import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_TIMEOUT_MS,
    get_gemini_model_name,
)
from app.semantic_search import SearchResult


class TestChatAPIModes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.original_app_state = api.app.state._state.copy()

    async def asyncTearDown(self) -> None:
        api.app.state._state.clear()
        api.app.state._state.update(self.original_app_state)

    async def test_gemini_model_setting_uses_environment_or_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_gemini_model_name(), DEFAULT_GEMINI_MODEL)
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-test-override"}):
            self.assertEqual(get_gemini_model_name(), "gemini-test-override")
        with patch.dict(os.environ, {"GEMINI_MODEL": "   "}):
            self.assertEqual(get_gemini_model_name(), DEFAULT_GEMINI_MODEL)

    async def test_cli_passes_environment_model_to_rag(self) -> None:
        with (
            patch.dict(os.environ, {"GEMINI_MODEL": "gemini-cli-override"}),
            patch("sys.argv", ["rag_pipeline.py", "A test question"]),
            patch.object(rag_pipeline, "create_gemini_client", return_value=object()),
            patch.object(rag_pipeline, "load_search_context", return_value=(None, None, None, 0)),
            patch.object(rag_pipeline, "answer_question", return_value="Simulated answer") as answer,
            patch("builtins.print"),
        ):
            rag_pipeline.main()

        self.assertEqual(answer.call_args.kwargs["model_name"], "gemini-cli-override")

    async def test_mock_mode_skips_model_initialization_and_keeps_history(self) -> None:
        with (
            patch.dict(os.environ, {api.MOCK_MODE_ENV_VAR: "1"}),
            patch("app.api.load_search_context") as load_search_context,
            patch("app.api.create_gemini_client") as create_gemini_client,
            patch("app.api.answer_question") as answer_question,
        ):
            async with api.lifespan(api.app):
                self.assertTrue(api.app.state.mock_mode)
                self.assertIsNone(api.app.state.gemini_client)
                self.assertIsNone(api.app.state.search_context)

                history = []
                for number in range(4):
                    response = api.chat(
                        api.ChatRequest(
                            question=f"Simulated question {number}",
                            conversation_history=history,
                        )
                    )
                    self.assertIn("Risposta simulata", response.answer)
                    self.assertIn("FAQ — faq.txt", response.answer)
                    history = response.conversation_history

                self.assertEqual(len(history), api.MAX_CONVERSATION_MESSAGES)
                self.assertEqual(
                    [message.role for message in history],
                    ["user", "assistant"] * 3,
                )

            load_search_context.assert_not_called()
            create_gemini_client.assert_not_called()
            answer_question.assert_not_called()

    async def test_normal_mode_uses_rag_with_simulated_gemini(self) -> None:
        fake_context = (object(), object(), object(), 384)
        fake_result = SearchResult(
            text="CloudFlow offers Starter, Growth, and Scale.",
            source_page="Pricing",
            source_file="pricing.txt",
            chunk_index=0,
            distance=0.2,
        )

        class SimulatedGeminiModels:
            def __init__(self) -> None:
                self.requests: list[dict[str, object]] = []

            def generate_content(self, **kwargs: object) -> SimpleNamespace:
                self.requests.append(kwargs)
                return SimpleNamespace(text="CloudFlow offers three plans. [S1]")

        simulated_models = SimulatedGeminiModels()
        simulated_client = SimpleNamespace(models=simulated_models)

        with (
            patch.dict(
                os.environ,
                {
                    api.MOCK_MODE_ENV_VAR: "0",
                    "GEMINI_MODEL": "gemini-test-override",
                },
            ),
            patch("app.api.load_search_context", return_value=fake_context) as load_search_context,
            patch("app.api.create_gemini_client", return_value=simulated_client) as create_gemini_client,
            patch("app.rag_pipeline.search", return_value=[fake_result]) as search,
        ):
            async with api.lifespan(api.app):
                self.assertFalse(api.app.state.mock_mode)
                self.assertEqual(api.app.state.gemini_model_name, "gemini-test-override")
                first = api.chat(api.ChatRequest(question="What plans does CloudFlow offer?"))
                second = api.chat(
                    api.ChatRequest(
                        question="How much does the second one cost?",
                        conversation_history=first.conversation_history,
                    )
                )

            load_search_context.assert_called_once_with()
            create_gemini_client.assert_called_once_with()
            self.assertEqual(search.call_count, 2)

        self.assertIn("CloudFlow offers three plans", first.answer)
        self.assertIn("[S1] Pricing — pricing.txt", first.answer)
        self.assertEqual(len(second.conversation_history), 4)
        self.assertEqual(len(simulated_models.requests), 2)
        self.assertEqual(simulated_models.requests[0]["model"], "gemini-test-override")
        self.assertEqual(
            simulated_models.requests[0]["config"].max_output_tokens,
            512,
        )
        self.assertEqual(GEMINI_TIMEOUT_MS, 60_000)
        self.assertIn("What plans does CloudFlow offer?", simulated_models.requests[1]["contents"])


if __name__ == "__main__":
    unittest.main()
