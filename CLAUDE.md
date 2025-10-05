# Claude Development Guidelines

## Debugging changes.py Hangs

When `changes.py` hangs, check these resource exhaustion points:

**Resources that could exhaust:**

1. File descriptors - pipes, sockets, files
2. Threads/processes - asyncio creates threads for I/O
3. Memory - swapping kills performance
4. Network connections - TCP connections to APIs
5. Ephemeral ports - client-side port exhaustion
6. Kernel limits - per-process limits
7. DNS resolution - hung DNS lookups
8. SSL connections - hung handshakes
9. Semaphore deadlock - though unlikely

**Diagnostic commands (replace PID with actual process ID):**

```bash
# Find the process
pgrep -f "changes --after"

# 1. Process limits
cat /proc/PID/limits  # Linux
ulimit -a  # macOS

# 2. Thread count
/bin/ps -M PID | wc -l  # macOS
ps -o nlwp PID  # Linux

# 3. Memory usage
/bin/ps -o rss,vsz -p PID

# 4. All open files by type
lsof -p PID | awk '{print $5}' | sort | uniq -c | sort -rn

# 5. Network connections specifically
lsof -p PID -i
netstat -an | grep PID

# 6. Ephemeral port usage (system-wide)
netstat -an | grep ESTABLISHED | wc -l

# 7. Check if stuck in syscall
sample PID 1 -f /tmp/sample.txt  # macOS
strace -p PID  # Linux

# 8. Python thread dump (SIGQUIT won't kill process)
kill -QUIT PID

# 9. System-wide resource check
sysctl kern.maxproc kern.maxprocperuid  # macOS
cat /proc/sys/kernel/threads-max  # Linux
```

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
