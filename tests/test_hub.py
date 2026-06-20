from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from mlxvm.hub import HubClient


def test_search_filters_multimodal_results(monkeypatch, tmp_path: Path) -> None:
    text = SimpleNamespace(
        id="mlx-community/text",
        sha="abc",
        downloads=2,
        likes=1,
        pipeline_tag="text-generation",
        last_modified=datetime.now(timezone.utc),
        used_storage=100,
        gated=False,
    )
    image = SimpleNamespace(
        id="mlx-community/vision",
        sha="def",
        downloads=3,
        likes=1,
        pipeline_tag="image-text-to-text",
        last_modified=None,
        used_storage=200,
        gated=False,
    )

    class Api:
        def list_models(self, **kwargs):
            return [image, text]

    monkeypatch.setattr(
        HubClient,
        "_imports",
        staticmethod(lambda: (Api, None, None, OSError, OSError)),
    )
    results = HubClient(tmp_path).search("model", limit=10)
    assert [result.repo_id for result in results] == ["mlx-community/text"]
    assert results[0].size_bytes == 100
