# GitHub Repository Settings

## Repository

- Repository: `nirajptl321/ai4i-predictive-maintenance-project`
- Visibility: public
- Protected branch: `main`
- Configuration method: GitHub CLI and GitHub REST API through `gh`

## Current Verified Branch Protection

Verified with:

```bash
gh api /repos/nirajptl321/ai4i-predictive-maintenance-project/branches/main/protection
```

Relevant settings:

- `required_status_checks`: `null`
- `enforce_admins`: `false`
- `required_pull_request_reviews`: enabled
- `required_approving_review_count`: `0`
- `restrictions`: `null`
- `allow_force_pushes`: `false`
- `allow_deletions`: `false`
- `required_linear_history`: `false`
- `required_conversation_resolution`: `false`

Practical effect:

- `main` is protected.
- Teammates should make changes on branches and open pull requests.
- Force pushes to `main` are disabled.
- Deletion of `main` is disabled.
- CI/status checks are not required because the project currently has no CI workflow.
- Owner/admin bypass is not enforced because `enforce_admins` is `false`.

## Unsupported Owner-Only Push Restriction

The requested owner-only direct push allowlist was attempted with:

```bash
PUT /repos/nirajptl321/ai4i-predictive-maintenance-project/branches/main/protection
```

Requested restriction:

```json
"restrictions": {
  "users": ["nirajptl321"],
  "teams": [],
  "apps": []
}
```

GitHub rejected that setting for this repository with:

```text
gh: Validation Failed (HTTP 422)
{"message":"Validation Failed","errors":["Only organization repositories can have users and team restrictions"],"documentation_url":"https://docs.github.com/rest/branches/branch-protection#update-branch-protection","status":"422"}
```

This means the exact "only this username can push to main" restriction is not available through the API for this personal repository. The supported protection currently requires pull requests and blocks force pushes/deletions.

## Manual GitHub UI Steps

If GitHub later exposes push restrictions for this repository/account, configure them manually:

1. Open the repository on GitHub.
2. Go to `Settings`.
3. Go to `Branches`.
4. Select `Add branch protection rule` or edit the existing `main` rule.
5. Set branch name pattern to:

```text
main
```

6. Enable `Require a pull request before merging` if the team should use PRs.
7. Enable `Block force pushes`.
8. Enable `Restrict who can push to matching branches`, if available.
9. Add only the repository owner as an allowed pusher.
10. Do not allow deletions.
11. Save changes.

Notes:

- Public users who are not collaborators cannot push directly anyway.
- Only collaborators with write/admin access can push branches or attempt direct pushes.
- Branch protection mainly prevents collaborators from directly modifying `main`.
- Teammates should create feature branches and open pull requests.
- The repository owner should review and merge final changes.

