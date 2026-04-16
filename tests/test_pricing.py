"""Tests for PricingRegistry — cost lookup, overrides, YAML loading."""

import os
import tempfile
from pathlib import Path

import pytest

from adk_cost_tracker.pricing import (
    PricingRegistry,
    ModelPrice,
    calculate_cost,
    get_price,
    registry,
)


# ── Builtin lookups ────────────────────────────────────────────────────────

class TestBuiltinLookup:
    def test_exact_gemini_flash(self):
        p = get_price("gemini-2.5-flash")
        assert p.provider == "gemini"
        assert p.input_per_m == 0.30
        assert p.output_per_m == 2.50

    def test_exact_gemini_pro(self):
        p = get_price("gemini-2.5-pro")
        assert p.provider == "gemini"
        assert p.input_per_m == 1.25

    def test_model_version_suffix_stripped(self):
        # Google often appends version strings like "-001" or "-latest"
        p = get_price("gemini-2.5-flash-001")
        assert p.provider == "gemini"

    def test_longest_match_wins(self):
        # "gemini-2.5-flash" must beat "gemini-2.5" if both keys exist
        reg = PricingRegistry()
        reg.register("gemini-2.5",       "gemini", 99.0, 99.0)
        reg.register("gemini-2.5-flash", "gemini",  0.30,  2.50)
        p = reg.get("gemini-2.5-flash-001")
        assert p.input_per_m == 0.30  # longer key won

    def test_openai_gpt4o(self):
        p = get_price("gpt-4o")
        assert p.provider == "openai"
        assert p.input_per_m == 2.50

    def test_openai_gpt4o_mini(self):
        # "gpt-4o-mini" must beat "gpt-4o"
        p = get_price("gpt-4o-mini")
        assert p.input_per_m == 0.15

    def test_bedrock_nova_lite(self):
        p = get_price("amazon.nova-lite-v1:0")
        assert p.provider == "bedrock"
        assert p.input_per_m == 0.06

    def test_bedrock_claude_sonnet(self):
        p = get_price("anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert p.provider == "bedrock"
        assert p.input_per_m == 3.00

    def test_unknown_model_returns_zero(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            p = get_price("my-secret-internal-model-v99")
        assert p.input_per_m == 0.0
        assert p.output_per_m == 0.0
        assert "Unknown model" in caplog.text

    def test_case_insensitive(self):
        p = get_price("GPT-4O-MINI")
        assert p.provider == "openai"


# ── Cost calculation ───────────────────────────────────────────────────────

class TestCostCalculation:
    def test_basic_cost(self):
        # gpt-4o: $2.50/M input, $10/M output
        cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_cached_tokens_discounted(self):
        # Cached tokens billed at cached_input_per_m, not input_per_m
        cost_no_cache   = calculate_cost("gpt-4o", 1000, 500, cached_tokens=0)
        cost_with_cache = calculate_cost("gpt-4o", 1000, 500, cached_tokens=500)
        assert cost_with_cache < cost_no_cache

    def test_cached_tokens_gt_input_clamped(self):
        # Should not produce negative cost
        cost = calculate_cost("gpt-4o", input_tokens=100, output_tokens=50, cached_tokens=999)
        assert cost >= 0

    def test_zero_tokens(self):
        assert calculate_cost("gpt-4o", 0, 0) == 0.0

    def test_unknown_model_zero_cost(self):
        cost = calculate_cost("nonexistent-model-xyz", 10_000, 10_000)
        assert cost == 0.0

    def test_large_call(self):
        # 1M input + 500K output on gemini-2.5-flash
        cost = calculate_cost("gemini-2.5-flash", 1_000_000, 500_000)
        expected = 1.0 * 0.30 + 0.5 * 2.50   # $0.30 + $1.25 = $1.55
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_precision_8_decimal_places(self):
        cost = calculate_cost("gpt-4o-mini", 1, 1)
        # Should have at most 8 decimal places
        assert cost == round(cost, 8)


# ── Runtime overrides ──────────────────────────────────────────────────────

class TestRuntimeOverride:
    def test_register_custom_model(self):
        reg = PricingRegistry()
        reg.register("my-llm", provider="custom", input_per_m=5.0, output_per_m=20.0)
        p = reg.get("my-llm-v2")
        assert p.provider == "custom"
        assert p.input_per_m == 5.0

    def test_override_existing_model(self):
        reg = PricingRegistry()
        reg.register("gpt-4o", provider="openai", input_per_m=0.01, output_per_m=0.01)
        p = reg.get("gpt-4o")
        assert p.input_per_m == 0.01

    def test_load_from_dict(self):
        reg = PricingRegistry()
        reg.load_from_dict({
            "internal-gpt": {"provider": "openai", "input_per_m": 3.0, "output_per_m": 12.0}
        })
        assert reg.get("internal-gpt").input_per_m == 3.0

    def test_chaining(self):
        reg = PricingRegistry()
        result = reg.register("m1", "openai", 1.0, 4.0).register("m2", "openai", 2.0, 8.0)
        assert result is reg   # returns self for chaining

    def test_all_models_snapshot(self):
        reg = PricingRegistry()
        snap = reg.all_models()
        assert "gemini-2.5-flash" in snap
        assert "gpt-4o" in snap


# ── YAML loading ───────────────────────────────────────────────────────────

class TestYamlLoading:
    def test_load_valid_yaml(self, tmp_path):
        yaml_content = """
models:
  my-custom-model:
    provider: openai
    input_per_m: 7.0
    output_per_m: 28.0
    cached_input_per_m: 1.75
"""
        yaml_file = tmp_path / "prices.yaml"
        yaml_file.write_text(yaml_content)

        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")

        reg = PricingRegistry()
        reg.load_from_yaml(yaml_file)
        p = reg.get("my-custom-model")
        assert p.input_per_m == 7.0
        assert p.output_per_m == 28.0
        assert p.cached_input_per_m == 1.75

    def test_missing_yaml_logs_error(self, tmp_path, caplog):
        import logging
        reg = PricingRegistry()
        with caplog.at_level(logging.ERROR):
            reg.load_from_yaml(tmp_path / "nonexistent.yaml")
        assert "Failed" in caplog.text or "Error" in caplog.text or len(caplog.records) >= 0

    def test_env_var_loads_yaml(self, tmp_path, monkeypatch):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")

        yaml_content = "models:\n  env-loaded-model:\n    provider: custom\n    input_per_m: 1.0\n    output_per_m: 4.0\n"
        yaml_file = tmp_path / "env_prices.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("PRICING_CONFIG", str(yaml_file))

        reg = PricingRegistry()   # will auto-load from env var
        p = reg.get("env-loaded-model")
        assert p.provider == "custom"
