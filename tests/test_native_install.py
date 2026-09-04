"""Exercise environment recovery and installer error reporting without CUDA downloads."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import venv

import pytest


spec = importlib.util.spec_from_file_location(
    "native_worldmodels", Path(__file__).parents[1] / "scripts/native_worldmodels.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
check_spec = importlib.util.spec_from_file_location(
    "check_native_dependencies", Path(__file__).parents[1] / "scripts/check_native_dependencies.py")
checks = importlib.util.module_from_spec(check_spec)
check_spec.loader.exec_module(checks)


def make_environment(path, inherit=False):
    venv.EnvBuilder(with_pip=False, system_site_packages=inherit).create(path)


def test_rejects_inherited_packages_and_wrong_python(tmp_path, monkeypatch):
    env = tmp_path / ".venv"
    make_environment(env, inherit=True)
    assert not native.isolated_python_ready(env)
    make_environment(env)
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    monkeypatch.setattr(native, "PYTHON_VERSION", current)
    assert native.isolated_python_ready(env)
    monkeypatch.setattr(native, "PYTHON_VERSION", "0.0")
    assert not native.isolated_python_ready(env)


def test_replaces_incompatible_environment_and_preserves_files(tmp_path, monkeypatch):
    args = SimpleNamespace(work=str(tmp_path), method="lewm")
    env = tmp_path / "lewm/.venv"
    make_environment(env, inherit=True)
    (env / "keep.txt").write_text("previous environment")
    monkeypatch.setattr(native, "PYTHON_VERSION",
                        f"{sys.version_info.major}.{sys.version_info.minor}")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/test/uv")

    def create_environment(command, log_path, env=None):
        assert command[1:3] == ["venv", "--python"]
        assert "--seed" in command
        make_environment(Path(command[-1]))

    monkeypatch.setattr(native, "logged_install_command", create_environment)
    native.ensure_python_environment(args, tmp_path / "install.log")
    assert native.isolated_python_ready(env)
    backups = list(env.parent.glob(".venv.previous-*"))
    assert len(backups) == 1
    assert (backups[0] / "keep.txt").read_text() == "previous environment"

    def unexpected_install(*args, **kwargs):
        pytest.fail("A compatible environment should be reused")

    monkeypatch.setattr(native, "logged_install_command", unexpected_install)
    native.ensure_python_environment(args, tmp_path / "install.log")


def test_installer_error_includes_stderr_and_persistent_log(tmp_path):
    log = tmp_path / "install.log"
    with pytest.raises(RuntimeError) as error:
        native.logged_install_command(
            [sys.executable, "-c", "import sys; print('resolver failure', file=sys.stderr); sys.exit(7)"],
            log)
    assert "exit 7" in str(error.value)
    assert "resolver failure" in str(error.value)
    assert str(log) in str(error.value)
    assert "resolver failure" in log.read_text()


@pytest.fixture
def decord_metadata_warning(monkeypatch):
    result = SimpleNamespace(returncode=1, stdout=checks.DECORD_WARNING + "\n")
    monkeypatch.setattr(checks.subprocess, "run", lambda *a, **kw: result)
    monkeypatch.setattr(checks.platform, "system", lambda: "Linux")
    monkeypatch.setattr(checks.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(checks.metadata, "distribution", lambda name: SimpleNamespace(
        version="0.6.0", read_text=lambda name: checks.DECORD_WHEEL_TAG + "\n"))
    return result


def test_known_decord_warning_requires_successful_decode(decord_metadata_warning, monkeypatch):
    decoded = []
    monkeypatch.setattr(checks, "decode_test_video", lambda: decoded.append(True))
    checks.check_dependencies()
    assert decoded == [True]

    def broken_decoder():
        raise RuntimeError("Cannot load libdecord.so")

    monkeypatch.setattr(checks, "decode_test_video", broken_decoder)
    with pytest.raises(RuntimeError, match="Cannot load libdecord"):
        checks.check_dependencies()


@pytest.mark.parametrize("extra_error", ["foo requires bar, which is not installed.",
                                       "other 0.6.0 is not supported on this platform"])
def test_decord_exception_does_not_hide_other_errors(
        decord_metadata_warning, monkeypatch, extra_error):
    decord_metadata_warning.stdout += extra_error + "\n"
    monkeypatch.setattr(checks, "decode_test_video", lambda: pytest.fail("Must fail before decode"))
    with pytest.raises(RuntimeError, match="dependency errors"):
        checks.check_dependencies()


def test_decord_exception_rejects_other_platforms(decord_metadata_warning, monkeypatch):
    monkeypatch.setattr(checks.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="Unsupported decord platform"):
        checks.check_dependencies()
