"""Automatically discovered tests for the CloudFlow RAG pipeline."""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import patch

from app.rag_pipeline import (
    DEFAULT_GEMINI_MODEL,
    INITIAL_CANDIDATE_COUNT,
    MAX_CONVERSATION_MESSAGES,
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_CHARS,
    GEMINI_TIMEOUT_MS,
    answer_question,
    create_gemini_client,
    format_context,
    main,
    rerank_results,
    update_conversation_history,
)
from app.semantic_search import DEFAULT_TOP_K, SearchResult, load_search_context, search


def _run_test_payment_methods_faq_chunk_in_top_three() -> None:
    """Regression check: payment-method question retains FAQ chunk 0 in top 3.

    This test uses local semantic search only and does not call Gemini.
    """
    question = "Which payment methods are accepted?"
    collection, model, numpy, embedding_dimension = load_search_context()
    candidates = search(
        question,
        collection,
        model,
        numpy,
        embedding_dimension,
        top_k=INITIAL_CANDIDATE_COUNT,
    )
    top_results = rerank_results(question, candidates, top_k=DEFAULT_TOP_K)
    formatted_context, _ = format_context(top_results)
    assert any(
        result.source_page == "FAQ"
        and result.source_file == "faq.txt"
        and result.chunk_index == 0
        for result in top_results
    ), "FAQ chunk 0 must rank among the top three payment-method results"
    assert len(formatted_context) <= MAX_PROMPT_CHARS


def _run_test_gemini_request_limits_with_simulated_client() -> None:
    """Check request limits and timeout/retry settings without a network call."""
    from google.genai import types

    with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-placeholder"}):
        with patch("google.genai.Client") as client_factory:
            create_gemini_client()
    http_options = client_factory.call_args.kwargs["http_options"]
    assert http_options.timeout == GEMINI_TIMEOUT_MS
    assert http_options.retry_options.attempts == 1

    result = SearchResult(
        text="Visa, Mastercard, and American Express are accepted.",
        source_page="FAQ",
        source_file="faq.txt",
        chunk_index=0,
        distance=0.2,
    )

    class SimulatedModels:
        def __init__(self) -> None:
            self.request: dict[str, Any] = {}

        def generate_content(self, **kwargs: Any) -> Any:
            self.request = kwargs
            return type("Response", (), {"text": "Sono accettate queste carte. [S1]"})()

    simulated_models = SimulatedModels()
    simulated_client = type("SimulatedClient", (), {"models": simulated_models})()
    history = [
        {"role": "user", "content": "What is the Growth plan?"},
        {"role": "assistant", "content": "A plan with advanced reporting. [S1]"},
    ]
    with patch.dict(os.environ, {"GEMINI_MODEL": DEFAULT_GEMINI_MODEL}):
        with patch("app.rag_pipeline.search", return_value=[result]) as search_mock:
            answer_question(
                "Which payment methods are accepted?",
                client=simulated_client,
                search_context=(None, None, None, 0),
                conversation_history=history,
            )

    search_mock.assert_called_once()
    assert search_mock.call_args.kwargs["top_k"] == INITIAL_CANDIDATE_COUNT
    request = simulated_models.request
    assert request["model"] == DEFAULT_GEMINI_MODEL
    assert isinstance(request["config"], types.GenerateContentConfig)
    assert request["config"].max_output_tokens == MAX_OUTPUT_TOKENS
    assert request["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
    prompt_context = request["contents"].split("PASSAGGI RECUPERATI:\n", 1)[1].split(
        "\n\nDOMANDA CORRENTE:", 1
    )[0]
    assert len(request["contents"]) <= MAX_PROMPT_CHARS
    assert "[S1] Pagina: FAQ" in prompt_context
    prompt = request["contents"]
    assert "CRONOLOGIA CONVERSAZIONALE (solo contesto dialogico):" in prompt
    assert "Utente: What is the Growth plan?" in prompt
    assert "PASSAGGI RECUPERATI:" in prompt
    assert prompt.index("CRONOLOGIA CONVERSAZIONALE") < prompt.index("PASSAGGI RECUPERATI:")


def _run_test_plan_and_followup_price_prompts_with_simulated_client() -> None:
    """Verify plan references, price details, and citations in both prompts."""
    plan_result = SearchResult(
        text=(
            "CloudFlow offers Starter, Growth, and Scale. Growth costs $24 USD "
            "per user/month billed annually or $29 USD per user/month billed monthly."
        ),
        source_page="FAQ",
        source_file="faq.txt",
        chunk_index=0,
        distance=0.2,
    )
    price_result = SearchResult(
        text=(
            "Growth is $24 per user per month billed annually, or $29 per user "
            "per month billed monthly. All prices are in USD."
        ),
        source_page="Pricing",
        source_file="pricing.txt",
        chunk_index=0,
        distance=0.2,
    )

    class SimulatedModels:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_content(self, **kwargs: Any) -> Any:
            self.prompts.append(kwargs["contents"])
            return type("Response", (), {"text": "Risposta di test. [S1]"})()

    simulated_models = SimulatedModels()
    simulated_client = type("SimulatedClient", (), {"models": simulated_models})()
    history = [
        {"role": "user", "content": "What plans does CloudFlow offer?"},
        {"role": "assistant", "content": "Starter, Growth, and Scale. [S1]"},
    ]
    with patch(
        "app.rag_pipeline.search", side_effect=[[plan_result], [price_result]]
    ):
        answer_question(
            "What plans does CloudFlow offer?",
            client=simulated_client,
            search_context=(None, None, None, 0),
        )
        answer_question(
            "How much does the second one cost?",
            client=simulated_client,
            search_context=(None, None, None, 0),
            conversation_history=history,
        )

    assert len(simulated_models.prompts) == 2
    for prompt in simulated_models.prompts:
        assert "Usa esclusivamente i fatti" in prompt
        assert "risolvere riferimenti" in prompt
        assert "importo, la valuta" in prompt
        assert "periodo di fatturazione" in prompt
        assert "citazione valida" in prompt
        assert "Usa solo gli ID dei PASSAGGI RECUPERATI in questo turno" in prompt

    first_prompt, followup_prompt = simulated_models.prompts
    assert "Starter, Growth, and Scale" in first_prompt
    assert "Growth, and Scale" in followup_prompt
    assert "Growth is $24 per user per month billed annually" in followup_prompt
    assert "$29 per user per month billed monthly" in followup_prompt


def _assert_mocked_gemini_api_error(api_error: Exception, expected_text: str) -> str:
    """Run answer_question against mocked retrieval and a simulated API error."""
    result = SearchResult(
        text="CloudFlow plans are Starter, Growth, and Scale.",
        source_page="FAQ",
        source_file="faq.txt",
        chunk_index=0,
        distance=0.2,
    )

    class FailingModels:
        def generate_content(self, **kwargs: Any) -> Any:
            raise api_error

    simulated_client = type(
        "SimulatedClient", (), {"models": FailingModels()}
    )()
    with patch("app.rag_pipeline.search", return_value=[result]):
        try:
            answer_question(
                "What plans are available?",
                client=simulated_client,
                search_context=(None, None, None, 0),
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("The simulated Gemini API error should be surfaced")

    assert expected_text in message
    return message


def _run_test_gemini_503_reports_temporary_unavailability() -> None:
    from google.genai.errors import APIError

    message = _assert_mocked_gemini_api_error(
        APIError(503, {"error": {"status": "UNAVAILABLE", "message": "temporary"}}),
        "Gemini è temporaneamente non disponibile (HTTP 503). Riprova più tardi.",
    )
    assert "temporary" not in message


def _run_test_gemini_504_reports_timeout() -> None:
    from google.genai.errors import APIError

    _assert_mocked_gemini_api_error(
        APIError(504, {"error": {"status": "DEADLINE_EXCEEDED", "message": "timeout"}}),
        "Gemini non ha risposto entro il timeout (HTTP 504). Riprova più tardi.",
    )


def _run_test_generic_gemini_api_error_reports_http_status() -> None:
    from google.genai.errors import APIError

    message = _assert_mocked_gemini_api_error(
        APIError(400, {"error": {"status": "INVALID_ARGUMENT", "message": "bad request"}}),
        "Errore API Gemini (HTTP 400 INVALID_ARGUMENT).",
    )
    assert "bad request" not in message


def _run_test_each_turn_runs_fresh_retrieval_with_history() -> None:
    """Each question searches again and gets its own independent RAG context."""
    first_result = SearchResult(
        text="Growth plan has advanced reporting.",
        source_page="Pricing",
        source_file="pricing.txt",
        chunk_index=1,
        distance=0.2,
    )
    second_result = SearchResult(
        text="Starter plan supports CSV export.",
        source_page="FAQ",
        source_file="faq.txt",
        chunk_index=0,
        distance=0.3,
    )

    class SimulatedModels:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate_content(self, **kwargs: Any) -> Any:
            self.prompts.append(kwargs["contents"])
            return type("Response", (), {"text": "Risposta di test. [S1]"})()

    simulated_models = SimulatedModels()
    simulated_client = type("SimulatedClient", (), {"models": simulated_models})()
    history = [
        {"role": "user", "content": "Tell me about Growth."},
        {"role": "assistant", "content": "Growth has advanced reporting. [S1]"},
    ]
    with patch(
        "app.rag_pipeline.search", side_effect=[[first_result], [second_result]]
    ) as search_mock:
        answer_question(
            "What about reporting?",
            client=simulated_client,
            search_context=(None, None, None, 0),
            conversation_history=history,
        )
        answer_question(
            "Can Starter export CSV?",
            client=simulated_client,
            search_context=(None, None, None, 0),
            conversation_history=history,
        )

    assert search_mock.call_count == 2
    assert "Growth plan has advanced reporting." in simulated_models.prompts[0]
    assert "Starter plan supports CSV export." in simulated_models.prompts[1]
    assert "Growth plan has advanced reporting." not in simulated_models.prompts[1]


def _run_test_interactive_chat_passes_and_updates_history() -> None:
    """The interactive loop supplies prior turns and stores each completed turn."""
    from copy import deepcopy

    seen: list[tuple[str, list[dict[str, str]], bool]] = []

    def simulated_answer(
        question: str,
        client: Any,
        search_context: Any,
        *,
        top_k: int,
        model_name: str,
        conversation_history: list[dict[str, str]],
        debug: bool,
    ) -> str:
        seen.append((question, deepcopy(conversation_history), debug))
        return f"Answer to {question}\n\nTempi: 0.01 s"

    with (
        patch("sys.argv", ["rag_pipeline.py"]),
        patch("app.rag_pipeline.create_gemini_client", return_value=object()),
        patch("app.rag_pipeline.load_search_context", return_value=(None, None, None, 0)),
        patch("app.rag_pipeline.answer_question", side_effect=simulated_answer),
        patch("builtins.input", side_effect=["First question", "Second question", "exit"]),
        patch("builtins.print"),
    ):
        main()

    assert seen[0] == ("First question", [], False)
    assert seen[1] == (
        "Second question",
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer to First question"},
        ],
        False,
    )


def _run_test_debug_prints_raw_response_only_when_enabled() -> None:
    """Verify DEBUG emits raw response text and normal mode remains quiet."""
    from contextlib import redirect_stdout
    from io import StringIO

    result = SearchResult(
        text="CloudFlow offers Starter, Growth, and Scale.",
        source_page="FAQ",
        source_file="faq.txt",
        chunk_index=0,
        distance=0.2,
    )
    raw_response = "CloudFlow offers three plans. [S1]"

    class SimulatedModels:
        def generate_content(self, **kwargs: Any) -> Any:
            return type("Response", (), {"text": raw_response})()

    simulated_client = type(
        "SimulatedClient", (), {"models": SimulatedModels()}
    )()

    outputs: dict[bool, str] = {}
    for debug in (False, True):
        captured = StringIO()
        with patch("app.rag_pipeline.search", return_value=[result]):
            with redirect_stdout(captured):
                answer_question(
                    "What plans does CloudFlow offer?",
                    client=simulated_client,
                    search_context=(None, None, None, 0),
                    debug=debug,
                )
        outputs[debug] = captured.getvalue()

    assert raw_response not in outputs[False]
    assert "[RAG DEBUG] Raw Gemini response:" in outputs[True]
    assert raw_response in outputs[True]


def _run_test_prompt_limit_drops_oldest_complete_turns_with_mocked_client() -> None:
    """Trim old history turns while preserving the latest turn and RAG text."""
    from copy import deepcopy

    history = [
        {"role": "user", "content": "OLD_USER " + "u" * 9_000},
        {"role": "assistant", "content": "OLD_ASSISTANT " + "a" * 7_000},
        {"role": "user", "content": "Recent question: tell me about Growth."},
        {"role": "assistant", "content": "Recent answer: Growth includes reporting."},
    ]
    original_history = deepcopy(history)
    result = SearchResult(
        text="Growth has advanced reporting.",
        source_page="Pricing",
        source_file="pricing.txt",
        chunk_index=0,
        distance=0.2,
    )

    class SimulatedModels:
        def __init__(self) -> None:
            self.prompt = ""

        def generate_content(self, **kwargs: Any) -> Any:
            self.prompt = kwargs["contents"]
            return type("Response", (), {"text": "Growth has reporting. [S1]"})()

    simulated_models = SimulatedModels()
    simulated_client = type("SimulatedClient", (), {"models": simulated_models})()
    with patch("app.rag_pipeline.search", return_value=[result]):
        answer_question(
            "What does Growth include?",
            client=simulated_client,
            search_context=(None, None, None, 0),
            conversation_history=history,
        )

    prompt = simulated_models.prompt
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert "OLD_USER" not in prompt
    assert "OLD_ASSISTANT" not in prompt
    assert "Recent question: tell me about Growth." in prompt
    assert "Recent answer: Growth includes reporting." in prompt
    assert "What does Growth include?" in prompt
    assert "Growth has advanced reporting." in prompt
    assert history == original_history


def _run_test_oversized_context_is_rejected_without_truncation() -> None:
    """Ensure an over-budget passage is never silently cut before generation."""
    result = SearchResult(
        text="x" * (MAX_PROMPT_CHARS + 1),
        source_page="FAQ",
        source_file="faq.txt",
        chunk_index=0,
        distance=0.2,
    )

    class UnexpectedModels:
        def generate_content(self, **kwargs: Any) -> Any:
            raise AssertionError("Gemini must not be called for oversized context")

    simulated_client = type(
        "SimulatedClient", (), {"models": UnexpectedModels()}
    )()
    with patch("app.rag_pipeline.search", return_value=[result]):
        try:
            answer_question(
                "What is this?",
                client=simulated_client,
                search_context=(None, None, None, 0),
            )
        except ValueError as exc:
            assert "prompt completo supera il limite" in str(exc)
        else:
            raise AssertionError("Oversized context should be rejected")


def _run_test_conversation_history_inserts_user_and_assistant_messages() -> None:
    history: list[dict[str, str]] = []
    update_conversation_history(history, "user", "Hello")
    update_conversation_history(history, "assistant", "Hi there")
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def _run_test_conversation_history_retains_only_latest_six_messages() -> None:
    history: list[dict[str, str]] = []
    for turn in range(4):
        update_conversation_history(history, "user", f"Question {turn}")
        update_conversation_history(history, "assistant", f"Answer {turn}")
    assert len(history) == MAX_CONVERSATION_MESSAGES
    assert history[0] == {"role": "user", "content": "Question 1"}
    assert history[-1] == {"role": "assistant", "content": "Answer 3"}


def _run_test_conversation_history_enforces_role_alternation() -> None:
    history: list[dict[str, str]] = []
    update_conversation_history(history, "user", "First")
    try:
        update_conversation_history(history, "user", "Second")
    except ValueError as exc:
        assert "alternare" in str(exc)
    else:
        raise AssertionError("Repeated roles should be rejected")

    update_conversation_history(history, "assistant", "Reply")
    assert [message["role"] for message in history] == ["user", "assistant"]


class TestRagPipeline(unittest.TestCase):
    def test_payment_methods_faq_chunk_in_top_three(self) -> None:
        _run_test_payment_methods_faq_chunk_in_top_three()

    def test_gemini_request_limits_with_simulated_client(self) -> None:
        _run_test_gemini_request_limits_with_simulated_client()

    def test_plan_and_followup_price_prompts_with_simulated_client(self) -> None:
        _run_test_plan_and_followup_price_prompts_with_simulated_client()

    def test_gemini_503_reports_temporary_unavailability(self) -> None:
        _run_test_gemini_503_reports_temporary_unavailability()

    def test_gemini_504_reports_timeout(self) -> None:
        _run_test_gemini_504_reports_timeout()

    def test_generic_gemini_api_error_reports_http_status(self) -> None:
        _run_test_generic_gemini_api_error_reports_http_status()

    def test_each_turn_runs_fresh_retrieval_with_history(self) -> None:
        _run_test_each_turn_runs_fresh_retrieval_with_history()

    def test_interactive_chat_passes_and_updates_history(self) -> None:
        _run_test_interactive_chat_passes_and_updates_history()

    def test_debug_prints_raw_response_only_when_enabled(self) -> None:
        _run_test_debug_prints_raw_response_only_when_enabled()

    def test_prompt_limit_drops_oldest_complete_turns_with_mocked_client(self) -> None:
        _run_test_prompt_limit_drops_oldest_complete_turns_with_mocked_client()

    def test_oversized_context_is_rejected_without_truncation(self) -> None:
        _run_test_oversized_context_is_rejected_without_truncation()

    def test_conversation_history_inserts_user_and_assistant_messages(self) -> None:
        _run_test_conversation_history_inserts_user_and_assistant_messages()

    def test_conversation_history_retains_only_latest_six_messages(self) -> None:
        _run_test_conversation_history_retains_only_latest_six_messages()

    def test_conversation_history_enforces_role_alternation(self) -> None:
        _run_test_conversation_history_enforces_role_alternation()
