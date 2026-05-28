"""
Test that models not in the cost map do NOT bypass budget enforcement.

Regression test for the bug where unmapped models got fallback costs of 0,
causing _is_model_cost_zero() to return True and skip all budget checks.

See: https://github.com/BerriAI/litellm/issues/24770
"""

import copy

import litellm
from litellm.proxy.auth.auth_checks import _is_model_cost_zero
from litellm.proxy.auth.user_api_key_auth import _should_skip_budget_checks
from litellm.router import Router


class TestUnmappedModelBudgetEnforcement:
    """Unmapped models must NOT bypass budget checks."""

    def setup_method(self):
        """Snapshot litellm.model_cost before each test."""
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        """Restore litellm.model_cost after each test."""
        litellm.model_cost = self._saved_model_cost

    def test_unmapped_model_enforces_budget(self):
        """A model not in litellm.model_cost should have budget enforced."""
        router = Router(
            model_list=[
                {
                    "model_name": "custom-model",
                    "litellm_params": {
                        "model": "openai/totally-nonexistent-model-xyz",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="custom-model", llm_router=router)
        assert result is False, (
            "Unmapped model should enforce budget (return False), "
            "not bypass it (return True)"
        )

    def test_explicitly_free_model_bypasses_budget(self):
        """A model with explicit cost=0 in model_info should bypass budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "free-model-id",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="free-model", llm_router=router)
        assert (
            result is True
        ), "Explicitly free model should bypass budget (return True)"

    def test_known_paid_model_enforces_budget(self):
        """A model in the cost map with non-zero costs should enforce budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "paid-model",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="paid-model", llm_router=router)
        assert result is False, "Known paid model should enforce budget (return False)"

    def test_unmapped_model_with_litellm_params_pricing(self):
        """A model with cost=0 in litellm_params (not model_info) should bypass budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-via-params",
                    "litellm_params": {
                        "model": "openai/nonexistent-but-free-model",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="free-via-params", llm_router=router)
        assert (
            result is True
        ), "Model with explicit cost=0 in litellm_params should bypass budget"

    def test_zero_cost_model_with_paid_router_fallback_enforces_budget(self):
        """A free primary must not bypass budgets when router fallback is paid."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-fallback",
                    "litellm_params": {
                        "model": "openai/paid-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[{"free-primary": ["paid-fallback"]}],
        )

        assert _is_model_cost_zero(model="free-primary", llm_router=router) is True
        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_zero_cost_router_fallback_skips_budget(self):
        """All-zero primary/fallback chain keeps the existing skip behavior."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-fallback",
                    "litellm_params": {
                        "model": "openai/free-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ],
            fallbacks=[{"free-primary": ["free-fallback"]}],
        )

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is True
        )

    def test_zero_cost_model_with_transitive_paid_fallback_enforces_budget(self):
        """A paid fallback later in the router chain keeps budget checks enabled."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-fallback",
                    "litellm_params": {
                        "model": "openai/free-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-fallback",
                    "litellm_params": {
                        "model": "openai/paid-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[
                {"free-primary": ["free-fallback"]},
                {"free-fallback": ["paid-fallback"]},
            ],
        )

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_paid_request_fallback_enforces_budget(self):
        """Client-provided paid fallbacks also keep budget checks enabled."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-fallback",
                    "litellm_params": {
                        "model": "openai/paid-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ]
        )

        assert (
            _should_skip_budget_checks(
                request_data={
                    "model": "free-primary",
                    "fallbacks": [{"free-primary": ["paid-fallback"]}],
                },
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_request_fallback_with_transitive_paid_router_fallback_enforces_budget(
        self,
    ):
        """Request fallback models are also checked through router fallback chains."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-request-fallback",
                    "litellm_params": {
                        "model": "openai/free-request-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-router-fallback",
                    "litellm_params": {
                        "model": "openai/paid-router-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[
                {"free-request-fallback": ["paid-router-fallback"]},
            ],
        )

        assert (
            _should_skip_budget_checks(
                request_data={
                    "model": "free-primary",
                    "fallbacks": [{"free-primary": ["free-request-fallback"]}],
                },
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_paid_generic_router_fallback_enforces_budget(self):
        """Generic router fallback lists are checked beyond the first entry."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-generic-fallback",
                    "litellm_params": {
                        "model": "openai/free-generic-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-generic-fallback",
                    "litellm_params": {
                        "model": "openai/paid-generic-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
        )
        router.fallbacks = ["free-generic-fallback", "paid-generic-fallback"]

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_paid_generic_dict_fallback_enforces_budget(self):
        """Generic router fallback dict entries are included in budget checks."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-dict-fallback",
                    "litellm_params": {
                        "model": "openai/paid-dict-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[{"model": "paid-dict-fallback"}],
        )

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )
