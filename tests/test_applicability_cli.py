"""Tests for the applicability CLI entry point and figure generation."""
import json
import pathlib
import pytest

from tensorphysics.applicability.__main__ import main, build_artifact, _parse_queries
from tensorphysics.applicability.figures import make_figures
from tensorphysics.applicability.capability import CapabilityRegistry
from tensorphysics.applicability.probe import probe_registry


CONFIG_PATH = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorphysics"
               / "applicability" / "configs" / "default_capabilities.v1.json")


class TestCLI:
    """Test the command-line interface."""

    def test_main_produces_artifact(self, tmp_path):
        """CLI should write a valid JSON artifact."""
        out_file = tmp_path / "envelopes.json"
        result = main(["--out", str(out_file), "--quiet"])
        assert result == 0
        assert out_file.exists()
        artifact = json.loads(out_file.read_text())
        assert artifact["schema"] == "tensorphysics.applicability.envelopes.v1"
        assert "suite_digest" in artifact
        assert "config_digest" in artifact
        assert "capabilities" in artifact

    def test_main_with_figures(self, tmp_path):
        """CLI should generate figures when --figures is specified."""
        fig_dir = tmp_path / "figures"
        result = main(["--figures", str(fig_dir), "--quiet"])
        assert result == 0
        assert fig_dir.exists()
        assert (fig_dir / "fig1_envelopes.png").exists()
        assert (fig_dir / "fig2_acceptance_window.png").exists()

    def test_main_with_query(self, tmp_path):
        """CLI should handle queries and report out-of-domain."""
        out_file = tmp_path / "envelopes.json"
        query = '{"multigrid-contraction-vs-damping": {"damping": 1.5}}'
        result = main(["--out", str(out_file), "--query", query, "--quiet"])
        assert result == 0
        artifact = json.loads(out_file.read_text())
        assert "queries" in artifact
        assert len(artifact["queries"]) == 1
        assert artifact["query_summary"]["total"] == 1
        assert artifact["query_summary"]["out_of_domain"] == 1

    def test_main_summary_output(self, capsys):
        """CLI should print summary when not quiet."""
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "multigrid-contraction-vs-damping" in captured.out
        assert "stationary-iteration-stability" in captured.out
        assert "stationary-solve-error-vs-grid" in captured.out

    def test_main_no_refine(self, tmp_path):
        """CLI should skip refinement when --no-refine is specified."""
        out_file = tmp_path / "envelopes.json"
        result = main(["--out", str(out_file), "--no-refine", "--quiet"])
        assert result == 0
        artifact = json.loads(out_file.read_text())
        assert artifact["schema"] == "tensorphysics.applicability.envelopes.v1"

    def test_parse_queries_inline(self):
        """_parse_queries should handle inline JSON."""
        queries = _parse_queries('{"cap1": {"axis1": 1.0}}')
        assert len(queries) == 1
        assert queries[0]["capability_id"] == "cap1"
        assert queries[0]["point"]["axis1"] == 1.0

    def test_parse_queries_file(self, tmp_path):
        """_parse_queries should handle file path."""
        query_file = tmp_path / "query.json"
        query_file.write_text('{"cap1": {"axis1": 1.0}}')
        queries = _parse_queries(str(query_file))
        assert len(queries) == 1
        assert queries[0]["capability_id"] == "cap1"

    def test_parse_queries_none(self):
        """_parse_queries should return empty list for None."""
        queries = _parse_queries(None)
        assert queries == []


class TestFigures:
    """Test the figure generation module."""

    def test_make_figures_produces_two_files(self, tmp_path):
        """make_figures should produce exactly two PNG files."""
        registry = CapabilityRegistry.from_config_file(CONFIG_PATH)
        suite = probe_registry(registry)
        paths = make_figures(suite, registry, tmp_path)
        assert len(paths) == 2
        assert all(p.suffix == ".png" for p in paths)
        assert all(p.exists() for p in paths)

    def test_make_figures_creates_directory(self, tmp_path):
        """make_figures should create the output directory if it doesn't exist."""
        fig_dir = tmp_path / "nested" / "figures"
        registry = CapabilityRegistry.from_config_file(CONFIG_PATH)
        suite = probe_registry(registry)
        paths = make_figures(suite, registry, fig_dir)
        assert fig_dir.exists()
        assert len(paths) == 2

    def test_make_figures_file_sizes_reasonable(self, tmp_path):
        """Generated figures should have reasonable file sizes."""
        registry = CapabilityRegistry.from_config_file(CONFIG_PATH)
        suite = probe_registry(registry)
        paths = make_figures(suite, registry, tmp_path)
        for path in paths:
            size = path.stat().st_size
            assert size > 1000, f"Figure {path.name} is too small ({size} bytes)"
            assert size < 10_000_000, f"Figure {path.name} is too large ({size} bytes)"


class TestBuildArtifact:
    """Test the build_artifact function."""

    def test_build_artifact_with_dict_queries(self):
        """build_artifact should accept dict-style queries."""
        queries = {"multigrid-contraction-vs-damping": {"damping": 0.5}}
        artifact, suite, registry = build_artifact(CONFIG_PATH, queries=queries)
        assert len(artifact["queries"]) == 1
        assert artifact["queries"][0]["verdict"]["domain_status"] == "IN"

    def test_build_artifact_with_list_queries(self):
        """build_artifact should accept list-style queries."""
        queries = [{
            "label": "test-query",
            "capability_id": "multigrid-contraction-vs-damping",
            "point": {"damping": 0.5}
        }]
        artifact, suite, registry = build_artifact(CONFIG_PATH, queries=queries)
        assert len(artifact["queries"]) == 1
        assert artifact["queries"][0]["label"] == "test-query"

    def test_build_artifact_no_refine(self):
        """build_artifact should skip refinement when refine=False."""
        artifact, suite, registry = build_artifact(CONFIG_PATH, refine=False)
        assert artifact["schema"] == "tensorphysics.applicability.envelopes.v1"
        # Boundaries should still exist but may be less refined
        for cap_id in registry.ids():
            report = suite.reports[cap_id]
            for axis_name, axis_env in report.envelope.axes.items():
                assert axis_env.boundaries is not None

    def test_build_artifact_multiple_queries(self):
        """build_artifact should handle multiple queries."""
        queries = {
            "multigrid-contraction-vs-damping": {"damping": 0.5},
            "stationary-iteration-stability": {"relaxation": 0.8},
        }
        artifact, suite, registry = build_artifact(CONFIG_PATH, queries=queries)
        assert len(artifact["queries"]) == 2
        assert artifact["query_summary"]["total"] == 2
