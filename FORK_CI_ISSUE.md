# Fork CI Issue: Why PR #1 Has No CI Checks

## 🔍 Root Cause Identified

**PR #1 exists but shows "Checks 0" because this is a FORK repository.**

**Repository**: `jameskermode/ACEpotentials.jl` (fork)
**Upstream**: `ACEsuit/ACEpotentials.jl` (original)
**PR**: https://github.com/jameskermode/ACEpotentials.jl/pull/1

---

## The Fork Problem

### Why CI Isn't Running

GitHub Actions has **security restrictions for forks**:

1. **Workflows don't auto-run on forks** by default
2. **Require manual approval** from repository owner
3. **Actions must be enabled** in fork settings
4. **PR from fork to fork** doesn't trigger the same CI as upstream

### Current Situation

```
Upstream: ACEsuit/ACEpotentials.jl
            ↓ (forked)
Fork:     jameskermode/ACEpotentials.jl
            ↓ (PR #1)
PR:       claude/explore-repo-structure-... → main (in fork)
Status:   ✅ Open, ❌ No CI checks
```

---

## Why This Happens

### GitHub Actions Fork Security

From GitHub documentation:

> **Workflows in forked repositories**:
> - Workflows don't run automatically on forked repositories
> - Must be enabled by fork owner
> - First-time contributors require approval
> - Prevents malicious code execution
> - Protects CI minutes and secrets

### This Fork's Situation

| Item | Status | Impact |
|------|--------|--------|
| Fork status | ✅ Confirmed | CI restrictions apply |
| Workflows exist | ✅ Yes (.github/workflows/CI.yml) | But won't run |
| PR created | ✅ Yes (PR #1) | Doesn't trigger CI |
| Actions enabled? | ❓ Unknown | Needs verification |

---

## Solutions

### Solution 1: Enable GitHub Actions on Fork (Recommended)

**For repository owner** (`jameskermode`):

1. Go to fork settings: https://github.com/jameskermode/ACEpotentials.jl/settings
2. Navigate to **Actions** → **General**
3. Under "Actions permissions", select:
   - ✅ **Allow all actions and reusable workflows**
   - OR **Allow local actions only** (if you trust the workflows)
4. Under "Fork pull request workflows", check:
   - ✅ **Run workflows from fork pull requests**
5. Save changes

**After enabling**:
- CI will run on existing PR #1
- Future PRs will have CI automatically
- May need to close/reopen PR to trigger

### Solution 2: Manual Workflow Dispatch

If Actions are enabled but not triggering:

1. Go to **Actions** tab in fork
2. Select **CI** workflow
3. Click **Run workflow**
4. Select branch: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
5. Click **Run workflow**

### Solution 3: Create PR to Upstream (Alternative)

Instead of PR within the fork, create PR to upstream:

```bash
# Create PR from fork to upstream
gh pr create --repo ACEsuit/ACEpotentials.jl \
             --head jameskermode:claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe \
             --base main \
             --title "Migrate to EquivariantTensors.jl" \
             --body "See MIGRATION_README.md"
```

**Pros**:
- ✅ Upstream CI will run (if permissions allow)
- ✅ Direct path to merge
- ✅ Visibility to upstream maintainers

**Cons**:
- ⚠️ Requires coordination with upstream
- ⚠️ May need upstream maintainer approval

### Solution 4: Test Locally

Since CI is blocked, validate locally:

```bash
# Install Julia
curl -fsSL https://install.julialang.org | sh

# Run tests
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/test_migration.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

See `MIGRATION_TESTING.md` for comprehensive local testing.

---

## Detailed Investigation

### Fork Detection

**Source**: https://github.com/jameskermode/ACEpotentials.jl

**Evidence**:
```
"forked from ACEsuit/ACEpotentials.jl"
```

### PR Status

**URL**: https://github.com/jameskermode/ACEpotentials.jl/pull/1

**Current State**:
- ✅ Open
- ✅ 3 commits
- ✅ 7 files changed (+1,145, -9)
- ❌ Checks: 0
- ❌ No CI runs
- ❌ No reviews

### Workflow File Exists

**File**: `.github/workflows/CI.yml`

**Configuration**:
```yaml
on:
  push:
    branches: [main, v0.6.x]
  pull_request:  # Should trigger on PRs
  workflow_dispatch:
```

**Expected**: Should trigger on PR
**Actual**: Not triggering (fork restriction)

---

## Comparison: Fork vs Upstream

| Aspect | Upstream Repo | Fork Repo (Current) |
|--------|--------------|---------------------|
| Location | ACEsuit/ACEpotentials.jl | jameskermode/ACEpotentials.jl |
| CI Auto-runs | ✅ Yes | ❌ No (disabled by default) |
| Actions enabled | ✅ Yes | ❓ Unknown |
| PR CI | ✅ Automatic | ❌ Requires enabling |
| Workflow files | ✅ Present | ✅ Present (but disabled) |

---

## How to Verify Actions Status

### Check if Actions are Enabled

Visit: https://github.com/jameskermode/ACEpotentials.jl/actions

**What to look for**:

1. **If disabled**: Message like "Workflows aren't being run on this forked repository"
2. **If enabled**: List of workflow runs (may be empty)
3. **If restricted**: "Some workflows may not run due to fork restrictions"

### Check Fork Settings

Visit: https://github.com/jameskermode/ACEpotentials.jl/settings/actions

**Key settings**:
- Actions permissions (allow/disable)
- Fork pull request workflows (enable/disable)
- Workflow approval requirements

---

## Expected Behavior After Fix

Once Actions are enabled on the fork:

### PR #1 Should Show

```
✅ CI / Julia 1.11 - ubuntu-latest - x64 - pull_request
   ├─ Setup Julia
   ├─ Add ACEregistry
   ├─ Build package
   └─ Run tests
      └─ test/test_migration.jl ✅
      └─ Full test suite ✅
```

### Time to Complete

- **Typical duration**: 10-30 minutes
- **Timeout**: 120 minutes (configured)
- **Matrix**: Julia 1.11, Python 3.8, Ubuntu

---

## Workflow on Forks: GitHub Best Practices

### Why GitHub Restricts Fork Workflows

1. **Security**: Prevent malicious code in workflows
2. **Resource protection**: Avoid CI minute abuse
3. **Secrets protection**: Don't expose upstream secrets
4. **Cost control**: Limit compute on forked repos

### What This Means for Contributors

- ✅ Workflows can run, but need enabling
- ✅ Fork owner controls Actions
- ✅ Safe by default
- ⚠️ Extra step required

---

## Recommended Action Plan

### Immediate (For Repository Owner)

1. **Enable Actions** in fork settings
2. **Allow fork PR workflows**
3. **Rerun/reopen PR #1** to trigger CI
4. **Wait for CI results** (~10-30 min)

### Alternative (If Can't Enable)

1. **Test locally** using Julia
2. **Share test results** in PR comments
3. **Request upstream maintainer review**
4. **Consider PR to upstream** directly

### Long-term

1. **Keep fork in sync** with upstream
2. **Enable Actions** for all future PRs
3. **Monitor CI results** on PRs
4. **Coordinate with upstream** for merging

---

## Related Documentation

- `CI_INVESTIGATION.md` - Why CI doesn't run on feature branches
- `MIGRATION_TESTING.md` - How to test locally
- `MIGRATION_README.md` - Migration overview

---

## GitHub Documentation References

- [GitHub Actions and Forks](https://docs.github.com/en/actions/managing-workflow-runs/approving-workflow-runs-from-public-forks)
- [Enabling Actions on Forks](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository)
- [Fork Pull Request Workflows](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#pull_request)

---

## Summary

**Problem**: PR #1 shows no CI checks
**Root Cause**: Repository is a fork with Actions likely disabled
**Solution**: Enable GitHub Actions in fork settings
**Alternative**: Test locally or create PR to upstream
**Status**: Awaiting repository owner to enable Actions

---

**Document Created**: 2025-11-12
**PR Affected**: https://github.com/jameskermode/ACEpotentials.jl/pull/1
**Branch**: `claude/explore-repo-structure-011CV4PfXPf4MHS4ceHdoMQe`
**Related Commits**: `7b5cda9`, `7936bea`, `8380f3d`
