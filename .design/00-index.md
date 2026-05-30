# Design index

This directory holds the per-translation-unit design docs. Each file
corresponds to a routed source file (see `tooling/translate-routes.toml`).

## Structure

```
.design/
  00-index.md             (this file)
  <area>/
    <doc>.md              (one per routed source file)
```

## Authoring

Design docs are authored by the `acto-doc-author` agent. They adapt to
existing code — quoted-code evidence from the source is mandatory. See
the agent spec at `.claude/agents/acto-doc-author.md`.
