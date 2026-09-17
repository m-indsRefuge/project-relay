# ================================================================
# Project Relay
# M0 Validation / Failure-Handling Update
#
# Purpose:
#   1. Fix the current Ruff import-order findings.
#   2. Harden m0-implement.ps1 so failed native commands stop the script.
#   3. Add FM-005 to docs/FAILURE_MAP.md.
#   4. Re-run all M0 validation gates.
#
# Safe to re-run.
# ================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = "C:\Users\nolan\AIProjects\relay"
$M0Script = Join-Path $Root "m0-implement.ps1"
$FailureMap = Join-Path $Root "docs\FAILURE_MAP.md"

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string]$Content
    )

    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Assert-NativeSuccess {
    param(
        [Parameter(Mandatory)]
        [string]$Operation
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

function Add-NativeGuardIfMissing {
    param(
        [Parameter(Mandatory)]
        [string]$Content,

        [Parameter(Mandatory)]
        [string]$Command,

        [Parameter(Mandatory)]
        [string]$Guard
    )

    if ($Content.Contains($Guard)) {
        return $Content
    }

    if (-not $Content.Contains($Command)) {
        throw "Could not find expected command in m0-implement.ps1: $Command"
    }

    return $Content.Replace(
        $Command,
        "$Command`r`n$Guard"
    )
}

Write-Host ""
Write-Host "================================================"
Write-Host " Project Relay - M0 Validation Update"
Write-Host "================================================"
Write-Host ""

# ------------------------------------------------
# Preconditions
# ------------------------------------------------

if (-not (Test-Path -LiteralPath $Root)) {
    throw "Relay repository not found: $Root"
}

Set-Location $Root

if (-not (Test-Path -LiteralPath ".git")) {
    throw "Relay is not initialized as a Git repository."
}

if (-not (Test-Path -LiteralPath $M0Script)) {
    throw "M0 implementation script not found: $M0Script"
}

if (-not (Test-Path -LiteralPath $FailureMap)) {
    throw "Failure map not found: $FailureMap"
}

Write-Host "Current worktree:"
git status --short
Assert-NativeSuccess "git status"

# The parent shell currently points at Batch-87's environment.
# Remove only the environment variable from this script process.
# This does NOT delete or modify that virtual environment.
if ($env:VIRTUAL_ENV) {
    Write-Host ""
    Write-Host "Clearing inherited VIRTUAL_ENV for this update process:"
    Write-Host "  $env:VIRTUAL_ENV"
    Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
}

# ------------------------------------------------
# 1. Fix the known Ruff findings
# ------------------------------------------------

Write-Host ""
Write-Host "Fixing known Ruff import-order findings..."

uv run ruff check `
    src/relay/graph.py `
    src/relay/state.py `
    --fix

Assert-NativeSuccess "Ruff auto-fix"

# ------------------------------------------------
# 2. Harden m0-implement.ps1
# ------------------------------------------------

Write-Host ""
Write-Host "Hardening m0-implement.ps1 native-command validation..."

$M0Content = Get-Content -LiteralPath $M0Script -Raw

$HelperMarker = "function Assert-NativeSuccess"

if (-not $M0Content.Contains($HelperMarker)) {
    $PreconditionsMarker = @'
# ------------------------------------------------
# Preconditions
# ------------------------------------------------
'@

    if (-not $M0Content.Contains($PreconditionsMarker)) {
        throw "Could not locate the Preconditions marker in m0-implement.ps1."
    }

    $Helper = @'
function Assert-NativeSuccess {
    param(
        [Parameter(Mandatory)]
        [string]$Operation
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

'@

    $M0Content = $M0Content.Replace(
        $PreconditionsMarker,
        "$Helper$PreconditionsMarker"
    )
}

$M0Content = Add-NativeGuardIfMissing `
    -Content $M0Content `
    -Command 'uv sync --extra dev' `
    -Guard 'Assert-NativeSuccess "uv sync"'

$M0Content = Add-NativeGuardIfMissing `
    -Content $M0Content `
    -Command 'uv run ruff check .' `
    -Guard 'Assert-NativeSuccess "Ruff"'

$M0Content = Add-NativeGuardIfMissing `
    -Content $M0Content `
    -Command 'uv run pytest' `
    -Guard 'Assert-NativeSuccess "pytest"'

$M0Content = Add-NativeGuardIfMissing `
    -Content $M0Content `
    -Command 'uv run python -m compileall -q src tests' `
    -Guard 'Assert-NativeSuccess "compileall"'

Write-Utf8NoBom -Path $M0Script -Content $M0Content

# ------------------------------------------------
# 3. Add FM-005
# ------------------------------------------------

Write-Host ""
Write-Host "Updating FAILURE_MAP.md..."

$FailureContent = Get-Content -LiteralPath $FailureMap -Raw

if (-not $FailureContent.Contains("FM-005")) {
    $Fm005 = @'

---

## FM-005 — Validation command fails but milestone script continues

**Component:** Engineering / milestone validation
**Code location:** `m0-implement.ps1`
**Responsibility:** Prevent an invalid milestone from being reported as successfully applied.

### Failure point

A native validation command such as Ruff, pytest, compileall, `uv`, or Git
returns a non-zero process exit code, but PowerShell continues running because
the exit code was not explicitly converted into a terminating script error.

### Observable symptoms

A validation command reports failure, but later validation steps continue and
the script may still print a successful completion message.

### Likely causes

`$ErrorActionPreference = "Stop"` handles PowerShell terminating errors, but
native executable exit codes must still be checked explicitly by the script.

### First check

Inspect:

`$LASTEXITCODE`

immediately after the native command that reported failure.

### Evidence

The original M0 implementation run demonstrated this failure mode:

- Ruff returned two `I001` violations.
- pytest still ran and passed.
- compileall still ran.
- the script still printed `M0 implementation applied.`

### Propagation

This failure can create a false-positive milestone result: the software may
contain a failed validation gate even though the automation reports successful
completion.

### Recovery

1. Correct the underlying validation failure.
2. Add an explicit native exit-code guard.
3. Re-run every milestone validation gate from the beginning.
4. Do not accept or commit the milestone until all gates pass.

### Retry safety

Safe after correcting the original validation failure.

### Data risk

Low for validation-only commands, but operational risk is high because a bad
build could otherwise be accepted or released.

### DO NOT

Do not treat the script's final success message as authoritative when an
earlier validation command reported failure.

### Related protection

Critical native commands in milestone automation must be followed by:

`Assert-NativeSuccess "<operation>"`

### Related tests / verification

This failure is verified operationally by forcing or observing a native
command with a non-zero exit code and confirming the script terminates before
later gates execute.
'@

    $FailureContent = $FailureContent.TrimEnd() + $Fm005 + "`r`n"
    Write-Utf8NoBom -Path $FailureMap -Content $FailureContent
}
else {
    Write-Host "FM-005 already exists; leaving it unchanged."
}

# ------------------------------------------------
# 4. Full M0 validation
# ------------------------------------------------

Write-Host ""
Write-Host "Synchronizing dependencies..."
uv sync --extra dev
Assert-NativeSuccess "uv sync"

Write-Host ""
Write-Host "Running Ruff..."
uv run ruff check .
Assert-NativeSuccess "Ruff"

Write-Host ""
Write-Host "Running M0 test suite..."
uv run pytest
Assert-NativeSuccess "pytest"

Write-Host ""
Write-Host "Running compileall..."
uv run python -m compileall -q src tests
Assert-NativeSuccess "compileall"

Write-Host ""
Write-Host "Running git diff --check..."
git diff --check
Assert-NativeSuccess "git diff --check"

# ------------------------------------------------
# Final report
# ------------------------------------------------

Write-Host ""
Write-Host "Git status:"
git status --short
Assert-NativeSuccess "git status"

Write-Host ""
Write-Host "================================================"
Write-Host " M0 validation update completed successfully."
Write-Host " All validation gates passed."
Write-Host "================================================"
Write-Host ""
Write-Host "Next review commands:"
Write-Host "  git diff --check"
Write-Host "  git diff --stat"
Write-Host "  git diff"
Write-Host ""
