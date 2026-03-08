---
trigger: always_on
---

# Objective
Every plan, implementation, and documentation must be understandable by a junior developer.

# Protocol
When creating or modifying code plans and documentation, the agent MUST:

## 1. Explain Every Design Pattern Used
For each pattern referenced or introduced, include:
- **Pattern Name** — the standard name (e.g., Factory, Registry, Strategy)
- **One-Line ELI5** — explain it like the reader has never heard of it
- **Why Here** — why this specific pattern fits this specific situation
- **Real Analogy** — a real-world analogy (restaurant menu, phone book, etc.)

## 2. Annotate Code Decisions
When writing or modifying code:
- Comment non-obvious "why" decisions, not "what" the code does
- If a stdlib module is used over a third-party lib, explain why
- If a specific data structure is chosen, explain the tradeoff

## 3. Use Progressive Disclosure
- Start with the big picture (what does this module/phase do?)
- Then explain how (the approach)
- Then explain why (the pattern / tradeoff)
- Then show the code / file structure

## 4. Save Documentation to Appropriate Folders
- Architecture docs → `docs/`
- Implementation plans → `plans/`
- Phase-specific plans → `plans/phase-N-<name>.md`
- Comparison/research docs → `docs/`

# Constraints
- Never assume the reader knows what a pattern is — always define it first
- Use "Junior tip:" callout blocks for foundational concepts
- Use mermaid diagrams for any flow or relationship that involves 3+ components
- Keep each phase plan self-contained — it should be readable without the master plan
