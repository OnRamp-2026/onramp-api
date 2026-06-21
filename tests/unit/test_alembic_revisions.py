from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_alembic_has_one_head_and_unique_revision_ids() -> None:
    root = Path(__file__).resolve().parents[2]
    config = Config(root / "alembic.ini")
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)

    revisions = list(script.walk_revisions())

    assert len(script.get_heads()) == 1
    assert len({revision.revision for revision in revisions}) == len(revisions)
