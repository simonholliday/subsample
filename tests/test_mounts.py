"""Tests for subsample.mounts: telling a network drive from a local disk (#3972)."""

import pathlib
import sys

import pytest

import subsample.mounts


_TABLE = """\
sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0
/dev/nvme0n1p2 / ext4 rw,relatime 0 0
//nas/music /mnt/music cifs rw,relatime,vers=3.1.1 0 0
nas:/export/samples /mnt/music/samples nfs4 rw,relatime 0 0
/dev/sdb1 /mnt/data ext4 rw,relatime 0 0
//nas/my\\040share /mnt/my\\040share cifs rw 0 0
"""


class TestMountInTable:

	def test_the_deepest_mount_holding_a_path_is_its_own (self) -> None:

		"""A share mounted inside another is the one a folder under it is on."""

		assert subsample.mounts._mount_in_table(_TABLE, pathlib.Path("/mnt/music/samples/kicks")) == (
			pathlib.Path("/mnt/music/samples"), "nfs4",
		)
		assert subsample.mounts._mount_in_table(_TABLE, pathlib.Path("/mnt/music/loops")) == (
			pathlib.Path("/mnt/music"), "cifs",
		)
		assert subsample.mounts._mount_in_table(_TABLE, pathlib.Path("/home/si/samples")) == (
			pathlib.Path("/"), "ext4",
		)

	def test_a_mount_point_is_on_its_own_mount (self) -> None:

		"""A library set to the share's own mount point is on the share."""

		assert subsample.mounts._mount_in_table(_TABLE, pathlib.Path("/mnt/music")) == (
			pathlib.Path("/mnt/music"), "cifs",
		)

	def test_a_name_that_only_starts_the_same_is_another_folder (self) -> None:

		"""/mnt/data2 is not under /mnt/data."""

		assert subsample.mounts._mount_in_table(_TABLE, pathlib.Path("/mnt/data2/kit")) == (
			pathlib.Path("/"), "ext4",
		)

	def test_a_later_mount_at_one_place_hides_the_earlier (self) -> None:

		"""A share mounted over a local folder is what a path there reaches."""

		table = _TABLE + "//nas/data /mnt/data cifs rw 0 0\n"

		assert subsample.mounts._mount_in_table(table, pathlib.Path("/mnt/data/kit")) == (
			pathlib.Path("/mnt/data"), "cifs",
		)

	def test_a_space_in_a_mount_point_is_read_from_its_escape (self) -> None:

		"""The table writes a space as \\040."""

		assert subsample.mounts._mount_in_table(_TABLE, pathlib.Path("/mnt/my share/kit")) == (
			pathlib.Path("/mnt/my share"), "cifs",
		)

	def test_short_lines_are_skipped (self) -> None:

		"""A line without a type does not stop the rest being read."""

		table = "garbage\n\n/dev/sda1 /\n" + _TABLE

		assert subsample.mounts._mount_in_table(table, pathlib.Path("/mnt/music")) == (
			pathlib.Path("/mnt/music"), "cifs",
		)

	def test_no_mount_is_none (self) -> None:

		"""An empty table names no mount."""

		assert subsample.mounts._mount_in_table("", pathlib.Path("/mnt/music")) is None


class TestNetworkFilesystem:

	@pytest.mark.parametrize("filesystem", sorted(subsample.mounts.NETWORK_FILESYSTEMS))
	def test_each_network_type_is_named (self, filesystem: str, monkeypatch: pytest.MonkeyPatch) -> None:

		"""A folder on any listed type is on a network drive of that type."""

		monkeypatch.setattr(subsample.mounts, "_mount", lambda _path: (pathlib.Path("/mnt/share"), filesystem))

		assert subsample.mounts.network_filesystem(pathlib.Path("/mnt/share/kit")) == filesystem

	@pytest.mark.parametrize("filesystem", ["ext4", "apfs", "btrfs", "tmpfs", "hfs", "fuse.rclone"])
	def test_a_local_type_is_none (self, filesystem: str, monkeypatch: pytest.MonkeyPatch) -> None:

		"""A folder on any other type counts as local."""

		monkeypatch.setattr(subsample.mounts, "_mount", lambda _path: (pathlib.Path("/"), filesystem))

		assert subsample.mounts.network_filesystem(pathlib.Path("/kit")) is None

	def test_a_folder_whose_mount_cannot_be_told_counts_as_local (self, monkeypatch: pytest.MonkeyPatch) -> None:

		"""No answer means the operating system's observer, as before polling existed."""

		monkeypatch.setattr(subsample.mounts, "_mount", lambda _path: None)

		assert subsample.mounts.network_filesystem(pathlib.Path("/kit")) is None

	def test_an_unreadable_mount_table_is_no_answer (self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:

		"""A Linux without /proc counts every folder as local."""

		monkeypatch.setattr(subsample.mounts, "_MOUNTS_FILE", tmp_path / "missing")

		assert subsample.mounts._linux_mount(tmp_path) is None


class TestThisMachine:

	"""The real answers, which the CI runs check on Linux and on macOS."""

	@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/mounts")
	def test_linux_names_the_mount_a_folder_is_on (self, tmp_path: pathlib.Path) -> None:

		"""The test's own folder is under the mount named, which has a type."""

		mount = subsample.mounts._mount(tmp_path)

		assert mount is not None
		assert tmp_path.resolve().is_relative_to(mount[0])
		assert mount[1]

	@pytest.mark.skipif(sys.platform != "darwin", reason="calls macOS's statfs")
	def test_macos_names_the_mount_a_folder_is_on (self, tmp_path: pathlib.Path) -> None:

		"""statfs's answer is read from the right places in its record.

		A layout read wrongly would give a mount point of another file system,
		or a type that is not a Mac's local disk.  The mount point is compared
		by device, not by path: macOS reaches its data volume through firmlinks,
		so a temporary folder in /private/var is on the volume mounted at
		/System/Volumes/Data without being under that path.
		"""

		mount = subsample.mounts._mount(tmp_path)

		assert mount is not None
		assert mount[0].stat().st_dev == tmp_path.stat().st_dev
		assert mount[1] in {"apfs", "hfs"}
		assert subsample.mounts.network_filesystem(tmp_path) is None

	@pytest.mark.skipif(sys.platform != "darwin", reason="calls macOS's statfs")
	def test_macos_answers_for_a_folder_not_made_yet (self, tmp_path: pathlib.Path) -> None:

		"""A folder that does not exist yet is on the mount of the nearest one that does."""

		assert subsample.mounts._mount(tmp_path / "not" / "yet") == subsample.mounts._mount(tmp_path)
