"""Basic unit tests for the Tube Magic Helper CLI functions.
Run with: pytest -q
"""
import os
from pathlib import Path
import importlib
import pytest

# Ensure module import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

cli = importlib.import_module("tube_magic_helper")

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def set_api_keys(monkeypatch):
    # Provide dummy keys so that API checks pass. Actual network calls will be mocked.
    monkeypatch.setenv("YOUTUBE_API_KEY", "DUMMY")
    monkeypatch.setenv("OPENAI_API_KEY", "DUMMY")
    # Reload module to pick up env vars.
    import importlib
    importlib.reload(cli)


# -----------------------------------------------------------------------------
# Tests for helper functions (pure logic)
# -----------------------------------------------------------------------------

def test_get_video_id_url_formats():
    assert cli.get_video_id("https://youtu.be/abc123") == "abc123"
    assert cli.get_video_id("https://www.youtube.com/watch?v=xyz789") == "xyz789"
    assert cli.get_video_id("xyz789") == "xyz789"


def test_get_video_id_invalid():
    assert cli.get_video_id("not a url") == "not a url"  # returns input fallback


def test_generate_video_ideas_parsing(monkeypatch):
    sample_response = "1. Idea one\n2. Idea two\n3. Idea three"
    monkeypatch.setattr(cli, "openai_chat", lambda *a, **k: sample_response)
    ideas = cli.generate_video_ideas("test niche", 3)
    assert ideas == ["Idea one", "Idea two", "Idea three"]


def test_generate_video_ideas_empty(monkeypatch):
    monkeypatch.setattr(cli, "openai_chat", lambda *a, **k: "")
    res = cli.generate_video_ideas("niche", 5)
    assert res == []


def test_optimize_metadata_json(monkeypatch):
    mock_json = {
        "title": "Best Title",
        "description": "A great description",
        "tags": ["tag1", "tag2"]
    }
    monkeypatch.setattr(cli, "openai_chat", lambda *a, **k: __import__("json").dumps(mock_json))
    meta = cli.optimize_metadata("some topic")
    assert meta == mock_json


def test_optimize_metadata_raw(monkeypatch):
    monkeypatch.setattr(cli, "openai_chat", lambda *a, **k: "Not JSON")
    meta = cli.optimize_metadata("topic")
    assert "raw" in meta


def test_openai_chat_no_client(monkeypatch):
    monkeypatch.setattr(cli, "openai_client", None)
    res = cli.openai_chat("prompt")
    assert res is None


# -----------------------------------------------------------------------------
# Edge case tests for keyword research logic (with mocked API)
# -----------------------------------------------------------------------------

def test_keyword_research_handles_no_client(monkeypatch):
    monkeypatch.setattr(cli, "youtube_client", None)
    res = cli.keyword_research(["keyword"], 5)
    assert res == []


def test_keyword_research_zero_competition(monkeypatch):
    # Mock YouTube client methods
    class DummyClient:
        def search(self):
            return self
        def list(self, **kwargs):
            return self
        def execute(self):
            if "search" in DummyClient.call_type:
                return {
                    "items": [{"id": {"videoId": "abc"}}],
                    "pageInfo": {"totalResults": 0}
                }
            else:
                return {"items": [{"statistics": {"viewCount": "1000"}}]}
        def videos(self):
            DummyClient.call_type = "videos"
            return self
    DummyClient.call_type = "search"
    monkeypatch.setattr(cli, "youtube_client", DummyClient())
    res = cli.keyword_research(["test"], 5)
    assert res[0]["competition"] == 0
    assert res[0]["score"] > 0
