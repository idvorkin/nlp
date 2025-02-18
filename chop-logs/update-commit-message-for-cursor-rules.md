## SpecStory

## Update Commit Message for Cursor Rules (2/18/2025, 8:12:26 AM)

_**User**_

Update commit message to have a special section on cursor rules if there are edits to cursor rules

---

_**Assistant**_

I'll help you modify the `commit.py` file to include a special section for cursor rules changes. First, I'll check the existing code and then make the necessary modifications.

Let me first fetch the cursor rules to understand their format and requirements.

---

_\*\*\*\*_

---

_\*\*\*\*_

Now I'll modify the commit message generation to include a special section for cursor rules changes. I'll update the prompt in the code.

---

_\*\*\*\*_

```diff
      * Put them in the order of importance
      * Use unnumbered lists as the user will want to reorder them
+ * If any changes involve Cursor rules (files in .cursor/rules/ or with .mdc extension), include a special section:
+     **Cursor Rules Changes**
+     * List each rule change with its version update (if applicable)
+     * Include the impact of the rule change
  * If you see any of the following, skip them, or at most list them last as a single line
      * Changes to formatting/whitespace
      * List breaking changes

+ **Cursor Rules Changes:** (include only for cursor rule changes)
+     * Update rule-name (v1.2.0 -> v1.3.0): Brief description of change and impact
+
  **Reason for change**
      * reason 1
```

---

_\*\*\*\*_

I've updated the `prompt_summarize_diff` function to include a special section for cursor rules changes. The changes include:

1. Added instructions to check for cursor rule changes (files in `.cursor/rules/` or with `.mdc` extension)
2. Added a new section "Cursor Rules Changes" in the commit message format
3. Specified that rule changes should include:
   - The rule name
   - Version changes (if applicable)
   - Brief description of the change and its impact

Now when generating commit messages, if there are any changes to cursor rules files, they will be highlighted in their own section with proper versioning and impact information. This will make it easier to track and review changes to cursor rules.

The changes maintain the existing functionality while adding the new cursor rules section. The oneline format remains unchanged since it's meant to be a single-line summary.

Would you like me to make any adjustments to how the cursor rules changes are formatted or what information is included in this section?

---

---
