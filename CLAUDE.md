# Claude Development Guidelines

## Project Conventions

This project follows development conventions documented in `zz-chop-conventions/`. Please refer to these files for guidance:

### Development Setup

- **General workflow rules**: `zz-chop-conventions/dev-setup/workflow-rules.md`
- **Git hooks**: `zz-chop-conventions/dev-setup/githooks.md`
- **GitHub Actions**: `zz-chop-conventions/dev-setup/github-actions-setup.md`
- **Justfile usage**: `zz-chop-conventions/dev-setup/justfile.md`

### Development Inner Loop

- **Getting started**: `zz-chop-conventions/dev-inner-loop/a_readme_first.md`
- **Clean code practices**: `zz-chop-conventions/dev-inner-loop/clean-code.md`
- **Commit guidelines**: `zz-chop-conventions/dev-inner-loop/clean-commits.md`
- **Running commands**: `zz-chop-conventions/dev-inner-loop/running-commands.md`

### Python-Specific

- **CLI development**: `zz-chop-conventions/python/python-cli.md`
- **UV shebang dependencies**: `zz-chop-conventions/python/uv-shebang-deps.md`

## Commands

When working on this project, use the justfile for common operations:

```bash
just --list  # Show available commands
```

Refer to `zz-chop-conventions/dev-setup/justfile.md` for detailed justfile usage patterns.

## Git Guidelines

**IMPORTANT: NEVER use git push --force or git push -f. Always use regular git push to preserve commit history.**
