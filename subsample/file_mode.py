"""How Subsample keeps the files it writes: the mode they get, and making a recording outlast a power cut.

tempfile.mkstemp creates its file 0600, readable only by its owner, and the
rename that puts it in place keeps that mode.  Sidecars, recordings and preview
images are ordinary data that a user browses, shares or serves over SMB or NFS,
so each writer gives its file the mode a plain open() would: 0666 less the
process umask (0644 under the usual 022).

A recording is also flushed to the disk before it is named, and its folder
after (sync_folder, sync_file): one a power cut catches in the disk's cache can
otherwise come back empty or cut short under its new name (#4704, L28).
Sidecars, previews and rendered variants are not, as each is made again from
the audio, and a re-analysis of a whole library writes thousands.
"""

import os
import pathlib
import typing


# Reading the umask means setting it and setting it back, which is not
# thread-safe; this runs at import, before any thread starts.
_UMASK: typing.Final[int] = os.umask(0)
os.umask(_UMASK)

DATA_FILE_MODE: typing.Final[int] = 0o666 & ~_UMASK


def sync_folder (folder: pathlib.Path) -> None:

	"""Flush a folder's entries to the disk, so a file just named in it stays named.

	Best effort: a file system that cannot open or flush a folder (some
	network ones) is left as it is, as the file itself was flushed.
	"""

	try:
		handle = os.open(folder, os.O_RDONLY)
	except OSError:
		return

	try:
		os.fsync(handle)
	except OSError:
		pass
	finally:
		os.close(handle)


def sync_file (path: pathlib.Path) -> None:

	"""Flush a file written in place, and its folder, to the disk.

	Raises:
		OSError: When the file cannot be flushed, so its bytes may not be on
		         the disk.
	"""

	handle = os.open(path, os.O_RDONLY)

	try:
		os.fsync(handle)
	finally:
		os.close(handle)

	sync_folder(path.parent)
