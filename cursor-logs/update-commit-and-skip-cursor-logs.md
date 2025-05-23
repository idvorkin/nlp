## SpecStory

## Update Commit and Skip Cursor-Logs (2/15/2025, 5:11:39 AM)

_**User**_

Update commit and changes to skip files in cursor-logs

---

_**Assistant**_

I'll help you update the code to skip files in the cursor-logs directory. First, let me check the contents of `changes.py` since it was mentioned in your request.

---

_\*\*\*\*_

Read file: changes.py

---

_\*\*\*\*_

I see that both files need to be updated to skip files in the cursor-logs directory. I'll update both files to add this functionality. Let me make the changes:

1. First, I'll update `changes.py` to add cursor-logs to the skip conditions:

---

_\*\*\*\*_

```diff
          return True

+     # Skip cursor-logs directory
+     if "cursor-logs" in str(file_path):
+         if verbose:
+             ic(f"Skip cursor-logs file: {file}")
+         return True
+
      # Skip binary and media files
      binary_extensions = {
                  result = await (
                      prompt_summarize_diff(
-                         file, diff_content, repo_path=repo_info.url, end_rev=last
+                         file, diff_content, repo_path=repo_path, end_rev=last
                      )
                      | llm
```

---

_\*\*\*\*_

2. Now, I'll update `commit.py` to also skip cursor-logs files. I'll add a function to check for files to skip and use it when processing the git diff:

---

_\*\*\*\*_

```diff
  import langchain_helper
  from openai_wrapper import num_tokens_from_string
+ from pathlib import Path

  # Disable Rich's line wrapping
  console = Console(width=10000)

+
+ def should_skip_file(file_path: str) -> bool:
+     """Check if a file should be skipped in the diff processing."""
+     if "cursor-logs" in file_path:
+         return True
+     return False
+
+
+ def filter_diff_content(diff_output: str) -> str:
+     """Filter out diffs from files that should be skipped."""
+     lines = diff_output.splitlines()
+     filtered_lines = []
+     skip_current_file = False
+
+     for line in lines:
+         if line.startswith('diff --git'):
+             # Extract file path from diff header
+             file_path = line.split(' b/')[-1]
+             skip_current_file = should_skip_file(file_path)
+
+         if not skip_current_file:
+             filtered_lines.append(line)
+
+     return '\n'.join(filtered_lines)
+

  def prompt_summarize_diff(diff_output, oneline=False):
```

---

_\*\*\*\*_

Now I need to update the `a_build_commit` function to use the new filtering:

---

_\*\*\*\*_

```diff
  async def a_build_commit(oneline: bool = False, fast: bool = False):
      user_text = "".join(sys.stdin.readlines())
+     # Filter out diffs from files that should be skipped
+     filtered_text = filter_diff_content(user_text)

      if fast:
      else:
          llms = langchain_helper.get_models(openai=True, claude=True, google=True)
-         tokens = num_tokens_from_string(user_text)
+         tokens = num_tokens_from_string(filtered_text)
          if tokens < 8000:
              llms += [langchain_helper.get_model(llama=True)]

      def describe_diff(llm: BaseChatModel):
-         return prompt_summarize_diff(user_text, oneline) | llm
+         return prompt_summarize_diff(filtered_text, oneline) | llm

      describe_diffs = await langchain_helper.async_run_on_llms(describe_diff, llms)
```

---

_\*\*\*\*_

I've made the following changes to both files:

1. In `changes.py`:

   - Added a new condition in `is_skip_file` to skip any files that contain "cursor-logs" in their path
   - The skip will be logged if verbose mode is enabled

2. In `commit.py`:
   - Added a new `should_skip_file` function to check if a file should be skipped (currently just checking for cursor-logs)
   - Added a new `filter_diff_content` function that processes the git diff output and removes diffs for files that should be skipped
   - Updated `a_build_commit` to use the filtered diff content instead of the raw input

These changes will ensure that both the `changes` and `commit` commands skip any files in the cursor-logs directory. The filtering is done early in the process, so it will reduce the amount of text being processed by the language models and keep the commit messages focused on relevant changes.

Would you like me to make any adjustments to these changes?

---

---
