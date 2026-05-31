# ferray Claude skills

Repo-local skills auto-loaded by Claude Code (and any compatible agent
runtime) to provide context for ferray-specific tasks. Each `*.md` file
in this directory is a single skill with frontmatter describing when it
should fire.

## Available skills

| Skill | Triggers on |
|-------|-------------|
| **`ferray-python-binding`** | Adding/modifying PyO3 bindings in `ferray-python/src/*.rs`, exposing new functions to Python, working with the dispatch macros, registering submodules, writing pyclasses |
| **`ferray-numpy-parity`** | Verifying NumPy compatibility, writing pytest parity tests, troubleshooting dtype/shape/numerical discrepancies, dtype promotion rules |
| **`ferray-rust-conventions`** | Writing Rust code in any of the workspace crates (ferray-core, ferray-ufunc, ferray-stats, ferray-linalg, etc.), Element trait, FerrayError variants, SIMD with pulp |

## How agents use these

Claude (and other compatible runtimes) automatically inject the
relevant skill body into the working context when the user prompt
matches the skill's `description` field. So when a user says
"add `numpy.ravel_multi_index` to ferray-python", the
`ferray-python-binding` and `ferray-numpy-parity` skills both load
without an explicit `/skill` command.

## Adding a new skill

Create a new `<skill-name>.md` in this directory with frontmatter:

```yaml
---
description: Use this skill when ... [precise trigger conditions, including
  what NOT to trigger on, and how it relates to other skills]
---

# Skill title

[Body — guidance, code patterns, common pitfalls. Aim for concrete
example code over abstract prose. Reference specific files and line
patterns where useful.]
```

Keep skill bodies under ~200 lines. If you find yourself writing more,
the topic probably needs splitting (e.g. one skill per crate or per
feature area).

## Maintaining

- When a binding pattern changes (e.g. PyO3 deprecation forces a different idiom), update the relevant skill alongside the code change.
- When a new common gotcha is found, add it to the "Common pitfalls" or "Common parity gotchas" section of the matching skill rather than scattering tribal knowledge in commit messages.
- The skills aim to be evergreen guidance, not changelogs — if you find yourself writing "as of 2026-05" timestamps, refactor toward stable advice that won't go stale.
