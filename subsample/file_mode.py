"""The mode Subsample gives a file it writes through a temporary file.

tempfile.mkstemp creates its file 0600, readable only by its owner, and the
rename that puts it in place keeps that mode.  Sidecars, recordings and preview
images are ordinary data that a user browses, shares or serves over SMB or NFS,
so each writer gives its file the mode a plain open() would: 0666 less the
process umask (0644 under the usual 022).
"""

import os
import typing


# Reading the umask means setting it and setting it back, which is not
# thread-safe; this runs at import, before any thread starts.
_UMASK: typing.Final[int] = os.umask(0)
os.umask(_UMASK)

DATA_FILE_MODE: typing.Final[int] = 0o666 & ~_UMASK
