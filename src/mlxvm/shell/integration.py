from __future__ import annotations

import shlex
from typing import Dict, Optional

COMMANDS = [
    "doctor",
    "search",
    "ls-remote",
    "install",
    "ls",
    "current",
    "use",
    "deactivate",
    "alias",
    "unalias",
    "uninstall",
    "run",
    "chat",
    "serve",
    "exec",
    "cache",
    "upgrade",
    "shell-init",
    "completions",
]


def environment_commands(shell: str, values: Dict[str, Optional[str]]) -> str:
    if shell not in {"bash", "zsh", "fish"}:
        raise ValueError(f"unsupported shell: {shell}")
    lines = []
    for name in ("MLXVM_MODEL", "MLXVM_REVISION", "MLXVM_PROFILE"):
        value = values.get(name)
        if shell == "fish":
            lines.append(f"set -gx {name} {shlex.quote(value)};" if value else f"set -e {name};")
        else:
            lines.append(f"export {name}={shlex.quote(value)};" if value else f"unset {name};")
    return "\n".join(lines)


def shell_init(shell: str) -> str:
    if shell in {"bash", "zsh"}:
        return f"""mlxvm() {{
  if [ "$1" = "use" ] || [ "$1" = "deactivate" ]; then
    local _mlxvm_action="$1"
    shift
    eval "$(command mlxvm env "$_mlxvm_action" --shell {shell} "$@")"
  else
    command mlxvm "$@"
  fi
}}
"""
    if shell == "fish":
        return """function mlxvm
  if test (count $argv) -gt 0; and contains -- $argv[1] use deactivate
    set -l action $argv[1]
    set -e argv[1]
    command mlxvm env $action --shell fish $argv | source
  else
    command mlxvm $argv
  end
end
"""
    raise ValueError(f"unsupported shell: {shell}")


def completion_script(shell: str) -> str:
    words = " ".join(COMMANDS)
    if shell == "bash":
        return f'''_mlxvm_complete() {{
  if [ "$COMP_CWORD" -eq 1 ]; then
    COMPREPLY=($(compgen -W "{words}" -- "${{COMP_WORDS[COMP_CWORD]}}"))
  fi
}}
complete -F _mlxvm_complete mlxvm
'''
    if shell == "zsh":
        return f"""#compdef mlxvm
_mlxvm() {{
  local -a commands
  commands=({words})
  _describe 'command' commands
}}
compdef _mlxvm mlxvm
"""
    if shell == "fish":
        return "\n".join(f"complete -c mlxvm -f -a {shlex.quote(command)}" for command in COMMANDS)
    raise ValueError(f"unsupported shell: {shell}")
