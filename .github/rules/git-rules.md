# Git Rules

## Objective
This file defines mandatory patterns for commits, branches, and pull requests for FIAP TECH 02.

## Project Context

**Project:** FIAP Tech Challenge 02  
**Type:** Python ML / MLOps (MovieLens 20M)

These Git rules apply to all contributions to this project.

## Scope
- Commit patterns
- Branch structure
- PR process
- Code review

---

## Commit Messages

### Format
Use Conventional Commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat:** New feature
- **fix:** Bug fix
- **docs:** Documentation changes
- **style:** Formatting, no code change
- **refactor:** Code refactoring
- **test:** Test addition or fix
- **chore:** Maintenance tasks

### Examples
```
feat(auth): add JWT authentication

Implement JWT-based authentication with refresh tokens.
Add login and logout endpoints.

Closes #123
```

```
fix(api): handle null values in user endpoint

Prevent null pointer exception when user data is missing.
```

---

## Branch Strategy

### Branch Naming
- **Feature:** `feature/description`
- **Bugfix:** `fix/description`
- **Hotfix:** `hotfix/description`
- **Release:** `release/version`

### Main Branches
- **main/master:** Production-ready code
- **develop:** Integration branch for features

---

## Pull Request Process

### Before Creating PR
- [ ] Code follows style guide
- [ ] All tests passing

- [ ] Tests written before code (TDD followed)
- [ ] Test coverage meets requirements

- [ ] No console.logs or debug code
- [ ] Documentation updated if needed
- [ ] Self-review completed

### PR Title
- Use same format as commit messages
- Be descriptive and clear

### PR Description
Must include:
- **What:** What changes were made
- **Why:** Why these changes were made
- **How:** How to test the changes
- **Screenshots:** If UI changes

### PR Size
- **Preferred:** Small, focused PRs (< 400 lines)
- **Maximum:** 1000 lines (requires justification)
- Split large features into multiple PRs

---

## Code Review

### Reviewer Responsibilities
- Check code quality and style
- Verify tests are adequate
- Ensure business logic is correct
- Check for security issues
- Approve or request changes

### Author Responsibilities
- Address all review comments
- Update PR description if needed
- Re-request review after changes

### Review Rules
- At least one approval required
- No self-approval
- Address all blocking comments
- Resolve conversations before merging

---

## Merge Rules

### Merge Methods
- **Squash and merge:** Preferred for feature branches
- **Merge commit:** For release branches
- **Rebase and merge:** Not allowed

### Before Merging
- [ ] All CI checks passing
- [ ] At least one approval
- [ ] No merge conflicts
- [ ] Branch is up to date

---

## Prohibited Practices

### Do NOT
- Commit directly to main/master
- Force push to shared branches
- Commit secrets or credentials
- Commit large binary files
- Skip hooks (--no-verify)
- Write vague commit messages

---

## Git Hooks

### Pre-commit
- Run `ruff check` and `ruff format --check`
- Run `pytest` (or subset on large changes)
- Check for secrets (`.env`, API keys)
- Block commit of `data/raw/`, `mlruns/`, large artifacts

### Pre-push
- Run full test suite
- Check branch is up to date

---

## Related Documentation

- **Code Style:** `.cursor/rules/code-style.md` (see section on lint and git hooks)
- **Testing Rules:** `.cursor/rules/testing-rules.md`
- **AI Usage Rules:** `.cursor/rules/ai-usage-rules.md`
