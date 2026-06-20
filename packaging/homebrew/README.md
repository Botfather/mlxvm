# Homebrew tap for mlxvm

This directory holds the canonical Homebrew formula for `mlxvm`. The formula is
*served* from a separate tap repository so users can run:

```sh
brew install botfather/tap/mlxvm
```

## One-time tap bootstrap

Homebrew taps live in a repo named `homebrew-<tap>`. To create the
`botfather/tap` tap:

1. Create a public GitHub repo named **`homebrew-tap`** under your account.
2. Add the formula at `Formula/mlxvm.rb`:

   ```sh
   git clone https://github.com/Botfather/homebrew-tap.git
   mkdir -p homebrew-tap/Formula
   cp packaging/homebrew/mlxvm.rb homebrew-tap/Formula/mlxvm.rb
   cd homebrew-tap && git add Formula/mlxvm.rb && git commit -m "Add mlxvm formula" && git push
   ```

   (The seeded `sha256` is a placeholder; the first release fills it in — see
   below.)

## Automated updates

The `homebrew` job in [`.github/workflows/release.yml`](../../.github/workflows/release.yml)
runs on every `v*` tag, after the PyPI publish succeeds. It uses
[`mislav/bump-homebrew-formula-action`](https://github.com/mislav/bump-homebrew-formula-action)
to recompute the release tarball's `url` + `sha256` and commit the bump to
`Formula/mlxvm.rb` in the tap repo.

### Required secret

The action needs a token with push access to the **tap** repo (the default
`GITHUB_TOKEN` cannot write to another repository). Create one and add it to the
`mlxvm` repo:

- A fine-grained PAT scoped to `Botfather/homebrew-tap` with **Contents:
  read/write**, or a classic PAT with `repo` scope.
- Store it as the repo secret **`HOMEBREW_TAP_TOKEN`**
  (Settings → Secrets and variables → Actions).

## Verifying locally

```sh
brew install --build-from-source ./packaging/homebrew/mlxvm.rb
brew test mlxvm
brew audit --strict --formula ./packaging/homebrew/mlxvm.rb
```
