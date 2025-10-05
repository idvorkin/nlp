# changes.py Hang Investigation

## Symptom

The `changes.py` script hangs indefinitely when running with `--md-only` flag. The process becomes unresponsive and requires SIGQUIT or SIGKILL to terminate.

**Observable behavior:**

- Process starts normally
- Gets stuck after processing some files
- CPU usage drops to near-zero
- No output or progress
- Must be killed manually

## Debugging Commands

When the process hangs, use these commands to diagnose:

```bash
# 1. Find the hung process
pgrep -f "changes"

# 2. Get process details (replace PID with actual process ID)
/bin/ps -fp PID

# 3. Check resource usage
/bin/ps -o rss,vsz,%cpu,%mem,etime -p PID

# 4. Count threads
/bin/ps -M PID | wc -l

# 5. Check open files by type
lsof -p PID | awk '{print $5}' | sort | uniq -c | sort -rn

# 6. Get stack trace (macOS)
sample PID 1 -f /tmp/stack_trace.txt && cat /tmp/stack_trace.txt

# 7. Alternative: Send SIGQUIT to dump Python stack (terminates process)
kill -QUIT PID
```

**Key things to look for:**

- **Thread count**: Should be ~3 threads. If 20+, indicates gRPC issue
- **Open files**: Look for excessive IPv4/unix sockets
- **Stack trace**: Shows what system call the main thread is stuck in
  - `poll` in SSL = HTTP timeout issue
  - `__psynch_cvwait` with gRPC = fork deadlock issue
  - `kevent` = waiting on async I/O

## Root Cause Analysis

### Stack Trace Analysis (2025-10-05)

Crash report from PID 14340 shows a **fork deadlock** between gRPC and subprocess operations.

**Thread 4 (The Fork Attempt):**

```
fork()
  └─ _pthread_atfork_prepare_handlers
     └─ grpc_event_engine::experimental::PrepareFork()
        └─ LivingThreadCount::BlockUntilThreadCount()
           └─ WAITING FOREVER (stuck in __psynch_cvwait)
```

**Thread 11 (The Deadlock):**

```
grpc DNS lookup
  └─ ares_init_sysconfig_macos
     └─ dyld4::APIs::dlclose()
        └─ _os_unfair_lock_lock_slow
           └─ DEADLOCKED (stuck in __ulock_wait2)
```

**The Problem:**

1. `asyncio.create_subprocess_exec()` calls `fork()` internally
2. gRPC's fork preparation handler tries to wait for all threads to be in a safe state
3. Thread 11 is stuck in DNS initialization, holding a dyld lock
4. Thread 4 waits forever for Thread 11 to finish
5. Fork never completes → script hangs

### Resource State at Hang

From diagnostic commands on PID 14340:

- **Thread count**: 20 threads spawned
- **gRPC threads**: Timer threads, polling threads, work-stealing thread pool
- **Subprocess usage**: Multiple `asyncio.create_subprocess_exec()` calls for git operations

**Key subprocess calls in changes.py:**

- Line 235: `is_binary_cmd` - checking if file is binary
- Line 268: `size_cmd` - getting file size
- Line 293: `which_process` - finding nbdiff command
- Line 310: `nbdiff_process` - diffing notebooks
- Line 342: `git_diff_process` - main git diff operation
- Line 494: Git log command
- Line 520: First diff for revision detection
- Line 544: Another git diff subprocess

## Attempted Fixes

### Attempt 1: Add gRPC Fork Support ❌ FAILED

**Change:** Added `grpc.experimental.fork_support()` at script startup

```python
# Enable gRPC fork support to prevent deadlocks when using subprocess
try:
    import grpc.experimental
    grpc.experimental.fork_support()
except (ImportError, AttributeError):
    pass
```

**Location:** `/Users/idvorkin/gits/nlp/changes.py:30-37`

**Rationale:**

- gRPC documentation recommends this for fork-heavy applications
- Should configure gRPC to handle fork() calls safely
- Prevents blocking on thread cleanup during fork preparation

**Result:** Still hangs - the DNS deadlock in Thread 11 occurs before fork_support can help

### Attempt 2: Disable gRPC for Google GenAI ⏳ TESTING

**Change:** Force all `ChatGoogleGenerativeAI` instances to use REST transport instead of gRPC

```python
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    transport="rest",  # Use REST instead of gRPC
)
```

**Location:** `/Users/idvorkin/gits/nlp/langchain_helper.py:286-367`

**Affected models:**

- `gemini-2.5-pro` (line 286)
- `gemini-2.5-flash` (line 293)
- `gemini-2.5-flash` with thinking (LOW) (line 300, 318)
- `gemini-2.5-flash` with thinking (MEDIUM) (line 336)
- `gemini-2.5-flash` with thinking (HIGH) (line 354)

**Rationale:**

- `langchain-google-genai` supports `transport="rest"` parameter
- REST transport avoids gRPC's threading model entirely
- No fork handlers, no DNS resolver threads, no dyld deadlocks
- Should eliminate the root cause completely

**Result:** ✅ FIXED gRPC deadlock! But revealed a new issue: HTTP timeout hang

**New symptom after fix:**

- No more gRPC threads (down from 20 threads to 3)
- No more DNS resolver deadlock
- No more dyld lock contention
- **BUT** now hangs in SSL socket read waiting for HTTP response

**Stack trace after fix (PID 20142):**

```
_ssl__SSLSocket_read
  └─ PySSL_select
     └─ poll (waiting indefinitely for network I/O)
```

The script is now stuck waiting for an HTTP response that never arrives. This is likely:

1. A timeout issue - no timeout set on HTTP requests
2. A connection that's waiting for data that will never come
3. An API endpoint that's hanging or rate-limiting

### Attempt 3: Add HTTP Timeout to Google GenAI ⏳ IN PROGRESS

**Root cause of hang discovered:**

The issue is **semaphore starvation**, not a simple timeout:

```python
async with max_parallel:  # Semaphore acquired (line 957)
    result = await llm.ainvoke({})  # HTTP call holds semaphore
```

**What happens:**

1. Each LLM call acquires a semaphore slot (limit: MAX_CONCURRENT_LLM_CALLS = 100)
2. The HTTP request happens **while holding the semaphore**
3. If the HTTP call hangs indefinitely, the semaphore slot is **never released**
4. Eventually all 100 slots fill with hung HTTP calls
5. All remaining tasks wait forever for a semaphore that never frees
6. **Process deadlocks**

**Fix in progress:**
Adding `timeout=60` to all ChatGoogleGenerativeAI instances to ensure HTTP calls fail after 60 seconds and release their semaphore slots.

**Status:** ✅ WORKAROUND IMPLEMENTED - Google/Gemini models disabled by default

**Changes made:**

1. Commented out all Google model additions in `get_models()` function (langchain_helper.py:181-198)
2. Changed CLI default from `google: bool = True` to `google: bool = False` (changes.py:417)
3. Added comment: "Disabled - causes resource contention with other threads"
4. Added kimi, deepseek, and llama to defaults for better model coverage

**Current enabled models (defaults):**

- ✅ Claude (Anthropic)
- ✅ Llama
- ✅ Kimi
- ✅ DeepSeek
- ✅ Grok4 Fast
- ❌ Google/Gemini (disabled - can re-enable with `--google` flag if needed)
- ❌ OpenAI GPT (off by default)

## Technical Details

### gRPC Architecture Issues

The issue stems from gRPC's complex threading model:

1. **Timer threads** - manage timeouts and scheduled tasks
2. **Polling threads** - handle network I/O via completion queues
3. **Work-stealing thread pool** - execute callbacks and DNS lookups
4. **DNS resolver** - uses c-ares library with platform-specific initialization

The macOS c-ares initialization (`ares_init_sysconfig_macos`) performs dynamic library operations that can deadlock with dyld's internal locks during fork preparation.

### The Deadlock Chain

```
Main thread                   gRPC DNS thread          Fork preparation
    |                              |                         |
    | spawn subprocess             |                         |
    |----------------------------->|                         |
    |                              | init DNS resolver       |
    |                              |---> dlclose()           |
    |                              |     [LOCKS dyld]        |
    | fork()                       |                         |
    |------------------------------|------------------------>|
    |                              |                         | wait for threads
    |                              |     [DEADLOCK]          | [BLOCKED]
    |     [WAITING]                |     [STUCK]             | [WAITING]
```

### Why fork_support() Didn't Help

The `grpc.experimental.fork_support()` API configures gRPC to:

- Use pthread_atfork handlers properly
- Reset state in child processes
- Avoid some threading issues

BUT it doesn't prevent:

- DNS initialization deadlocks in c-ares
- dyld lock contention during library loading
- Race conditions in platform-specific code paths

The deadlock occurs **during** DNS initialization, which is a synchronous operation in gRPC's network stack. Fork support can't interrupt or prevent this.

## Next Steps to Try

### Option 1: Avoid Fork Entirely

Replace `asyncio.create_subprocess_exec()` with synchronous `subprocess.run()` or use a pre-fork subprocess pool.

### Option 2: Defer gRPC Initialization

Initialize langchain/gRPC clients lazily, after all subprocess operations complete.

### Option 3: Use Process Pool

Replace fork-based subprocess with a process pool that doesn't trigger gRPC's fork handlers.

### Option 4: Disable DNS in gRPC

Set environment variables to prevent gRPC from using system DNS resolver:

```python
os.environ["GRPC_DNS_RESOLVER"] = "native"
```

### Option 5: Network Isolation

Run subprocess operations before any network calls, or in separate processes entirely.

## References

- gRPC fork issue: https://github.com/grpc/grpc/issues/21895
- c-ares threading: https://c-ares.org/docs.html
- Python subprocess + async: https://docs.python.org/3/library/asyncio-subprocess.html
- macOS dyld locking: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/dlopen.3.html
