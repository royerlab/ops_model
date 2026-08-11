# DiffEx viewer — Argus/S3 deploy

Review copies of the infra PR to `chanzuckerberg/sfbiohub-infra` (write access granted).
Argus is stateless → viewer assets live in S3, the deployment downloads on boot.

- [diffex-viewer-dev.tf](diffex-viewer-dev.tf) — the one new file the PR adds: `diffex-viewer-dev`
  S3 bucket + read-only IRSA role for the nonprod Argus cluster (`argus-diffex-viewer-rdev`/`diffex-viewer`)
  + a readwrite uploader role for our `aws s3 sync`. Mirrors `proteohub-argus-s3-reader-dev.tf`.
- [PR_BODY.md](PR_BODY.md) — PR description (requests a 1 TB ceiling).

These are copies for in-workspace review. The actual PR working copy (branch `diffex-viewer-dev`) is a
clone of the infra repo at `/hpc/mydata/gav.sturm/sfbiohub-infra` — push from there:

```bash
cd /hpc/mydata/gav.sturm/sfbiohub-infra
git add terraform/accounts/biohub-nonprod/diffex-viewer-dev.tf
git commit -m "Add diffex-viewer-dev S3 bucket + Argus read-only role"
git push -u origin diffex-viewer-dev
gh pr create --repo chanzuckerberg/sfbiohub-infra --base main \
  --title "diffex-viewer-dev: S3 bucket + Argus read-only" --body-file .diffex_pr_body.md
```

Confirm the Argus namespace/SA (`argus-diffex-viewer-rdev`/`diffex-viewer`) matches the registered
deployment before pushing — that comes from the app-readiness step (`czbiohub-sf/biohub-argus-example-app`).
