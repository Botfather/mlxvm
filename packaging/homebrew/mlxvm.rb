class Mlxvm < Formula
  include Language::Python::Virtualenv

  desc "nvm-style local model manager for MLX"
  homepage "https://github.com/Botfather/mlxvm"
  url "https://github.com/Botfather/mlxvm/archive/refs/tags/v0.1.0.tar.gz"
  # Placeholder until the first tagged release; the release workflow's
  # auto-bump job rewrites url + sha256 on every `v*` tag.
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/Botfather/mlxvm.git", branch: "main"

  # mlxvm targets Apple Silicon only (MLX is arm64 macOS only).
  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.12"

  def install
    venv = virtualenv_create(libexec, "python3.12")
    # mlxvm depends on MLX / MLX-LM, which ship as large native wheels.
    # Install mlxvm from this source tree and let pip resolve those
    # dependencies from PyPI rather than vendoring every resource here.
    system venv.root/"bin/pip", "install", "--no-cache-dir", buildpath
    bin.install_symlink libexec/"bin/mlxvm"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/mlxvm --version")
  end
end
