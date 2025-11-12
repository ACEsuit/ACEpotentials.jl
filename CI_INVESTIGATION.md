# CI Investigation: Why CI is Not Running

## Summary

**Finding**: CI is correctly configured but won't run on feature branches without a Pull Request.

**Reason**: The GitHub Actions workflow is configured to only run on:
1. Pushes to `main` or `v0.6.x` branches
2. Pull requests (any branch)
3. Manual workflow dispatch

**Current Branch**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe` (feature branch)

**Result**: ✅ CI configuration is correct - it's **designed not to run** on feature branch pushes

---

## CI Configuration Analysis

### From `.github/workflows/CI.yml` (lines 2-9):

```yaml
on:
  push:
    branches:
      - main         # Only runs on push to main
      - v0.6.x       # Only runs on push to v0.6.x
    tags: '*'
  pull_request:     # Runs on ANY pull request
  workflow_dispatch: # Allows manual triggering
```

### What This Means

| Event | Triggers CI? | Why |
|-------|-------------|-----|
| Push to `main` | ✅ Yes | Explicitly configured |
| Push to `v0.6.x` | ✅ Yes | Explicitly configured |
| Push to feature branch | ❌ No | Not in branches list |
| Pull request created | ✅ Yes | Always runs on PRs |
| Manual trigger | ✅ Yes | Via workflow_dispatch |

### Current Situation

- **Branch**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
- **Commits pushed**: ✅ Yes (2 commits)
- **Pull request created**: ❌ No
- **CI runs**: ❌ No (expected behavior)

---

## How to Trigger CI

### Option 1: Create a Pull Request (Recommended)

This is the standard approach and will trigger CI automatically:

```bash
# Via GitHub CLI (if available)
gh pr create --title "Migrate to EquivariantTensors.jl" \
             --body "See MIGRATION_README.md for details" \
             --base main

# Or via GitHub web interface:
# Navigate to: https://github.com/jameskermode/ACEpotentials.jl/pull/new/claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe
```

**Once PR is created**, CI will run automatically on:
- Initial PR creation
- Every subsequent push to the branch
- Manual re-runs via GitHub UI

### Option 2: Manual Workflow Dispatch

If you have write access, you can manually trigger CI via GitHub UI:

1. Go to the "Actions" tab
2. Select "CI" workflow
3. Click "Run workflow"
4. Select the branch: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
5. Click "Run workflow"

### Option 3: Temporarily Modify CI Config (Not Recommended)

You could temporarily add the feature branch to the CI trigger, but this is not recommended:

```yaml
# NOT RECOMMENDED - for emergency testing only
on:
  push:
    branches:
      - main
      - v0.6.x
      - claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe  # Temporary
```

**Downsides**:
- Requires modifying CI config
- Needs to be reverted later
- Not a standard workflow

---

## CI Test Matrix

When CI runs (via PR), it will test:

```yaml
matrix:
  version: ['1.11']           # Julia 1.11 only (1.10 commented out)
  python-version: ['3.8']     # Python 3.8
  os: [ubuntu-latest]         # Ubuntu only
  arch: [x64]                 # x64 only
```

### CI Steps (What Will Run)

1. **Checkout code** (`actions/checkout@v4`)
2. **Setup Julia 1.11** (`julia-actions/setup-julia@v2`)
3. **Setup Python 3.8** (`actions/setup-python@v2`)
4. **Add ACEregistry** (custom Julia registry)
5. **Cache dependencies** (`julia-actions/cache@v2`)
6. **Build package** (`julia-actions/julia-buildpkg@v1`)
7. **Run tests** (`julia-actions/julia-runtest@v1`)

**Timeout**: 120 minutes

---

## Expected CI Behavior for This Migration

### What CI Will Test

✅ **Should Pass**:
- Package instantiation with new EquivariantTensors.jl dependency
- Import statements don't error
- Model construction works
- Existing test suite passes

⚠️ **Potential Issues**:
- EquivariantTensors.jl v0.3 is "work in progress"
- Dependency resolution might have conflicts
- Numerical differences (though unlikely)

### CI Test Coverage

The CI will run:
```bash
# Standard Julia test command
julia --project=. -e 'using Pkg; Pkg.test()'
```

This includes:
- `test/runtests.jl` (main test suite)
- All test files in `test/` directory
- **INCLUDING** our new `test/test_migration.jl`

---

## Recommendations

### Immediate Action: Create Pull Request

**Why?**
1. Triggers CI automatically ✅
2. Enables code review ✅
3. Shows CI status in GitHub UI ✅
4. Standard GitHub workflow ✅

**How?**
```bash
# If gh CLI is available
gh pr create --fill

# Or use the URL from the git push output:
# https://github.com/jameskermode/ACEpotentials.jl/pull/new/claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe
```

### Alternative: Request Maintainer to Trigger CI

If you don't have permissions to create a PR or trigger workflows:

1. **Notify maintainer** that branch is ready for testing
2. **Provide branch name**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
3. **Request**: Either create PR or manually trigger workflow_dispatch

### What to Expect After CI Runs

**If CI passes** ✅:
- Green checkmark in GitHub
- Migration is validated
- Ready to merge

**If CI fails** ❌:
- Review failure logs
- Common issues:
  - Dependency resolution (EquivariantTensors.jl compatibility)
  - Import errors
  - Test failures
  - Numerical differences
- Fix issues and push updates (CI will re-run)

---

## CI Configuration Notes

### Why This Configuration?

This is a **best practice** setup:

**Pros**:
- ✅ Saves CI minutes (don't run on every feature branch push)
- ✅ Runs on all PRs (where it matters)
- ✅ Allows manual triggering when needed
- ✅ Tests main branch on every merge

**Cons**:
- ❌ Can't validate feature branches before PR creation
- ❌ Requires PR creation or manual trigger

### Comparison with Other Setups

| Setup | Pros | Cons |
|-------|------|------|
| **Current** (PR only) | Saves CI minutes | No pre-PR validation |
| Run on all branches | Immediate feedback | Wastes CI minutes |
| Run on push with "ci/" prefix | Balanced | Requires branch naming convention |

---

## Additional CI Files

### `.github/workflows/TagBot.yml`

Handles automatic release tagging:
```yaml
on:
  issue_comment:
  workflow_dispatch:
```

Not relevant for this migration.

---

## Testing Without CI

Since CI requires a PR or manual trigger, you can validate locally:

```bash
# Install Julia (if available in environment)
curl -fsSL https://install.julialang.org | sh

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run our migration tests
julia --project=. test/test_migration.jl

# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'
```

See `MIGRATION_TESTING.md` for comprehensive local testing procedures.

---

## Conclusion

**Status**: ✅ **CI configuration is working correctly**

**Action Required**: **Create a Pull Request** to trigger CI

**No Issues Found**: The CI setup is standard and appropriate for this repository.

**Next Steps**:
1. Create PR (triggers CI automatically)
2. Wait for CI to complete (~10-30 minutes)
3. Review CI results
4. Address any failures if needed
5. Merge when CI passes

---

**Document Created**: 2025-11-12
**Branch**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
**Related**: Migration commits `7b5cda9` and `7936bea`
