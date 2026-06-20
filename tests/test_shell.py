from mlxvm.shell import completion_script, environment_commands, shell_init


def test_posix_environment_is_shell_escaped() -> None:
    output = environment_commands(
        "zsh",
        {
            "MLXVM_MODEL": "repo/model@abc; echo unsafe",
            "MLXVM_REVISION": "abc",
            "MLXVM_PROFILE": None,
        },
    )
    assert "export MLXVM_MODEL='repo/model@abc; echo unsafe';" in output
    assert "unset MLXVM_PROFILE;" in output


def test_all_supported_shell_assets_render() -> None:
    for shell in ("bash", "zsh", "fish"):
        assert "mlxvm" in shell_init(shell)
        assert "doctor" in completion_script(shell)
