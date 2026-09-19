"""Expose upstream line-comment contracts to Doxygen without editing headers."""
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
for line in source.splitlines(keepends=True):
    # Keeping one output line per source line preserves Doxygen source locations.
    sys.stdout.write(re.sub(r"^(\s*)//(?![/!]) ?", r"\1/// ", line))
