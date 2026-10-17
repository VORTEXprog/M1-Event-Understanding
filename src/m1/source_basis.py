"""Content identities for this portable inference implementation."""
from pathlib import Path


def runtime_source_paths():
    """List the actual installed implementation, without external evidence files."""
    return tuple(Path(path.name)
                 for path in sorted(Path(__file__).parent.glob('*.py')))
