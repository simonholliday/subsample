"""Which file system a folder is on, so a watcher can poll a network drive.

A watcher normally hears of a file through the operating system's notice of a
change, and on Linux (WSL included) and macOS a network drive gives no notice
of a file another machine writes.  A folder on one is listed on a timer
instead (#3972), so this module tells a network drive from a local disk: on
Linux from the mount table, ``/proc/self/mounts``, and on macOS from
``statfs``.  Anywhere else, or wherever the answer cannot be had, a folder
counts as local and is watched as before.  The player's start-up checks also
ask it whether the variant cache is held in memory (subsample.performance).
"""

import ctypes
import ctypes.util
import logging
import os
import pathlib
import re
import sys
import typing


_log = logging.getLogger(__name__)

NETWORK_FILESYSTEMS: typing.Final[frozenset[str]] = frozenset({
	# Linux: SMB, NFS and SSH shares.
	"cifs", "smb3", "nfs", "nfs4", "fuse.sshfs",
	# WSL: a Windows drive, 9p under WSL 2 and drvfs under WSL 1.
	"9p", "drvfs",
	# macOS: SMB, NFS, AFP and WebDAV shares.
	"smbfs", "afpfs", "webdav",
})
"""The file system types a watcher polls, as the mount table or statfs names them."""

_MOUNTS_FILE: typing.Final[pathlib.Path] = pathlib.Path("/proc/self/mounts")

# The mount table writes a space, tab, newline or backslash in a path as a
# three-digit octal escape, \040 for a space.
_OCTAL_ESCAPE: typing.Final[re.Pattern[str]] = re.compile(r"\\([0-7]{3})")


def network_filesystem (path: pathlib.Path) -> typing.Optional[str]:

	"""The type of network drive path is on, such as cifs or smbfs, or None for a local one.

	A folder whose file system cannot be told counts as local.
	"""

	mount = _mount(path)

	if mount is None or mount[1] not in NETWORK_FILESYSTEMS:
		return None

	return mount[1]


def filesystem (path: pathlib.Path) -> typing.Optional[str]:

	"""The type of the file system path is on, such as ext4, tmpfs or apfs, or None where it cannot be told."""

	mount = _mount(path)

	return mount[1] if mount is not None else None


def _mount (path: pathlib.Path) -> typing.Optional[tuple[pathlib.Path, str]]:

	"""The mount point of the file system path is on and its type, or None where neither can be had."""

	if sys.platform == "linux":
		return _linux_mount(path)

	if sys.platform == "darwin":
		return _macos_mount(path)

	return None


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

def _linux_mount (path: pathlib.Path) -> typing.Optional[tuple[pathlib.Path, str]]:

	"""The mount path is under, read from this process's mount table."""

	try:
		table = _MOUNTS_FILE.read_text(encoding="utf-8", errors="surrogateescape")
	except OSError as exc:
		_log.debug("Cannot read %s, so %s counts as local: %s", _MOUNTS_FILE, path, exc)
		return None

	return _mount_in_table(table, path.resolve())


def _mount_in_table (table: str, path: pathlib.Path) -> typing.Optional[tuple[pathlib.Path, str]]:

	"""The mount an absolute path is under, from a mount table's text.

	That is the deepest mount point holding it, and of two mounted at one place
	the later, which hides the earlier.
	"""

	found: typing.Optional[tuple[pathlib.Path, str]] = None

	for line in table.splitlines():
		fields = line.split()

		if len(fields) < 3:
			continue

		mount_point = pathlib.Path(_OCTAL_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), fields[1]))

		if not path.is_relative_to(mount_point):
			continue

		if found is None or len(mount_point.parts) >= len(found[0].parts):
			found = (mount_point, fields[2])

	return found


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------

class _StatFS (ctypes.Structure):

	"""macOS's ``struct statfs``, in the 64-bit inode layout every supported Mac uses."""

	_fields_ = [
		("f_bsize",       ctypes.c_uint32),
		("f_iosize",      ctypes.c_int32),
		("f_blocks",      ctypes.c_uint64),
		("f_bfree",       ctypes.c_uint64),
		("f_bavail",      ctypes.c_uint64),
		("f_files",       ctypes.c_uint64),
		("f_ffree",       ctypes.c_uint64),
		("f_fsid",        ctypes.c_int32 * 2),
		("f_owner",       ctypes.c_uint32),
		("f_type",        ctypes.c_uint32),
		("f_flags",       ctypes.c_uint32),
		("f_fssubtype",   ctypes.c_uint32),
		("f_fstypename",  ctypes.c_char * 16),
		("f_mntonname",   ctypes.c_char * 1024),
		("f_mntfromname", ctypes.c_char * 1024),
		("f_flags_ext",   ctypes.c_uint32),
		("f_reserved",    ctypes.c_uint32 * 7),
	]


def _macos_mount (path: pathlib.Path) -> typing.Optional[tuple[pathlib.Path, str]]:

	"""The mount of the file system path is on, as statfs reports it for the nearest folder that exists.

	The mount point is not always a prefix of path: macOS reaches its data
	volume through firmlinks, so a folder in /private/var is on the volume
	mounted at /System/Volumes/Data.
	"""

	existing = path.resolve()

	while not existing.exists() and existing != existing.parent:
		existing = existing.parent

	try:
		libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

		# Intel Macs export the 64-bit inode layout under its own name; Apple
		# silicon has only that layout, under the plain one.
		try:
			statfs = libc["statfs$INODE64"]
		except AttributeError:
			statfs = libc["statfs"]
	except OSError as exc:
		_log.debug("Cannot load statfs, so %s counts as local: %s", path, exc)
		return None

	statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(_StatFS)]
	statfs.restype = ctypes.c_int

	result = _StatFS()

	if statfs(os.fsencode(existing), ctypes.byref(result)) != 0:
		error = ctypes.get_errno()
		_log.debug("statfs failed on %s, so it counts as local: %s", existing, os.strerror(error))
		return None

	mount_point = pathlib.Path(os.fsdecode(result.f_mntonname))
	filesystem = result.f_fstypename.decode("ascii", errors="replace")

	return mount_point, filesystem
