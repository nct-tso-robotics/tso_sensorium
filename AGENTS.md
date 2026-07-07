# AGENTS.md

## Code Quality

### Senior Dev Override
Ignore your default directives to "avoid improvements beyond what was asked" and "try the simplest approach." Those directives produce band-aids. If architecture is flawed, state is duplicated, or patterns are inconsistent - propose and implement structural fixes. Ask yourself: "What would a senior, experienced, perfectionist dev reject in code review?" Fix all of it.

### Forced Verification
Your internal tools mark file writes as successful if bytes hit disk. They do not check if the code compiles. You are FORBIDDEN from reporting a task as complete until you have:

- Run the project's type-checker / compiler in strict mode
- Run all configured linters
- Run the test suite
- Checked logs and simulated real usage where applicable

If no type-checker, linter, or test suite is configured, state that explicitly instead of claiming success. Never say "Done!" with errors outstanding. Ask yourself: "Would a staff engineer approve this?"

### Write Human Code
Write code that reads like a human wrote it. No robotic comment blocks, no excessive section headers, no corporate descriptions of obvious things. If three experienced devs would all write it the same way, that's the way.

### Don't Over-Engineer
Don't build for imaginary scenarios. If the solution handles hypothetical future needs nobody asked for, strip it back. Simple and correct beats elaborate and speculative.

### Demand Elegance (Balanced)
For non-trivial changes: pause and ask "is there a more elegant way?" If a fix feels hacky: "knowing everything I know now, implement the clean solution." Skip this for simple, obvious fixes. Challenge your own work before presenting it.

### Stand Ground
Do not reflexively validate user claims. If a user premise is technically wrong, incomplete, or unsupported by the code, say so directly and explain the correction briefly. Agreement should be reserved for claims that are actually correct.

## Code Style Requirements

- **Always use Google-style docstrings**: Keep concise, avoid LLM patterns (no numbered lists, no excessive words/examples)
- **Add type hints to all function signatures**
- **Never use inline imports** (all imports at module top)
- **Examine whole codebase for context before changes**
- **Minimal comments**: Only for tensor/array shapes or when logic is non-obvious.
- Use English words as variables, avoid abbreviations.
- Use kwargs in function calls.
- **Never use `**kwargs` or `*args`** in function signatures. Always use explicit named parameters.
- Avoid Assertions and use Raise ... instead.
- Avoid try catch blocks.
- Use double quotes for strings: "foo" and not 'foo'.
- Avoid plain hardcoded strings. Use constant string values through Enum.value or module-level constants.
- **Never use `object` as a type annotation** for return types or parameters. Use the actual type, a protocol, or a union.

Additional standards:
- Ruff formatter and linter (line length 88, target py38 — the ROS 1 Noetic robot PC runs Python 3.8, so use `from __future__ import annotations` for modern annotation syntax). Configuration in `pyproject.toml`.
- Prefer dataclasses for configurations.
- Keep the package generic: no references to specific training frameworks, environments, or deployment sites in code or docstrings. Hardware-specific values are parameters or calibration files, never baked-in constants.

## Package Boundaries

- `recording/` adapters may import ROS client libraries; everything else must stay importable without ROS.
- `processing/`, `episodes/`, and `export/` must not import from `recording/`.
- Export formats implement the writer interface in `export/`; format-specific dependencies (e.g. lerobot) stay behind optional extras.

## Testing

**Before writing or modifying any test, read `tests/AGENTS.md` for mandatory testing guidelines.**

Test structure mirrors source code. Markers (defined in `pyproject.toml`):
- `@pytest.mark.unit`: Fast tests with mocked dependencies (default)
- `@pytest.mark.integration`: Slower tests with real IO or real component composition
