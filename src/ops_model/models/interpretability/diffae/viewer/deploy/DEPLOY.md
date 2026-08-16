# OPSin / DiffEx viewer — deploy runbook (staging &/or prod)

## Repos & where things live
- **Edit the viewer here (source of truth):**
  `ops_mono → ops_model/src/ops_model/models/attention/diffex/viewer/webapp/{app.js,index.html,methods.js,style.css}`
- **Deploy repo:** `czbiohub-sf/diffex-viewer` — Dockerfile, GH Actions, and the pins
  `.infra/{rdev,staging,prod}/values.yaml` + `.argus-ci.yaml`. Clone: `/hpc/mydata/gav.sturm/diffex-viewer`.
- **Local preview only (NOT a deploy):** `/hpc/projects/icd.fast.ops/models/diffex/viewer_assets/` on :8805.

## Environments
| env | URL | auth | data source |
|---|---|---|---|
| rdev / staging | `*.apps.czbiohub.org` | **Okta-protected** | `diffex-viewer-dev` bucket (S3-CSI mount at `/data`) |
| **prod** | **opsin.apps.czbiohub.org** | **PUBLIC** (no Okta) | **CloudFront** (`d2xofoeogdsfrz.cloudfront.net`), public subset — *no* S3-CSI mount |

## The pipeline
1. Edit webapp in `ops_mono`, commit + push.
2. The sync mirrors `ops_mono` webapp → `diffex-viewer` `main` (auto commits `chore: sync webapp shell from monorepo (?v=N)`; `?v` bumps each sync — this is the cache-buster to watch).
3. Push to `diffex-viewer` `main` → **Argus Docker Build** builds `…:sha-<commit>`.
4. **release-please** maintains a release PR on branch `release-please--branches--main`
   (title `chore(main): release X.Y.Z`). **That PR repins `.infra/prod/values.yaml` AND
   `.infra/staging/values.yaml` `image.tag → sha-<latest built commit>`** (both together).
5. **Merge the release-please PR** → ArgoCD rolls **staging + prod** to the new image.

### Deploy command (staging + prod, the normal path) — **the Kyle way**
> Kyle: *"Don't touch that sha tag. The CI should do it for you. When you want to release to prod, just
> merge the PR — that's all. ~5–10 min to propagate. If it doesn't, lmk."*

**Never hand-edit `image.tag`.** release-please writes the fresh sha into the PR; you just merge it:
```bash
# merge commits are DISABLED on this repo → must squash
gh pr merge <release-please PR #> --repo czbiohub-sf/diffex-viewer --squash
```
Wait ~5–10 min for ArgoCD to propagate; verify (below). If it doesn't roll, ping Kyle.

> 📋 **ALWAYS (for prod deploys): hand the user the exact curl check with the expected `?v`.** After merging a
> release PR to prod, end your reply with the ready-to-paste command and the version it should print, e.g.:
> ```bash
> curl -s https://opsin.apps.czbiohub.org/ | grep -oE 'app.js\?v=[0-9]+'   # expect app.js?v=<N>
> ```
> Fill in `<N>` with the actual bumped version from this release so the user can confirm prod is live themselves.

> ⚠️ **Wait for the release PR's CI to finish before merging** (Kyle). Releasing frequently is fine, but the
> release PR only pins the *correct* sha once **Argus has finished building the latest commit** — you must make
> sure that last CI commit/build lands in the PR before merging. Merge too early and prod pins an image that
> isn't built yet (or a stale sha). Check the PR's `.infra/prod/values.yaml` `tag:` matches the newest **green**
> `Argus Docker Build`:
> ```bash
> gh run list --repo czbiohub-sf/diffex-viewer --workflow "Argus Docker Build" --status success --limit 1
> gh pr diff <release PR #> --repo czbiohub-sf/diffex-viewer | grep 'tag: sha'   # both should be the same sha
> ```

### Previewing a change WITHOUT touching prod
**"Don't touch the sha" is PROD-ONLY.** staging (and rdev) sha edits are fine — Kyle's rule is only about prod.
- **rdev:** PR branches auto-build (`branches: ["!main"]`) and deploy to an ephemeral **rdev** preview — good
  for quick per-PR checks.
- **staging (fine to hand-pin):** to preview a change on the shared staging site without waiting for a prod
  release, build the branch (open a PR → Argus builds `sha-<branch commit>`) and hand-edit **only**
  `.infra/staging/values.yaml` `image.tag → sha-<that>` in a small PR (leave prod). The next release-please
  merge will move staging+prod together to the released image (overwriting the manual staging pin — expected).

## Verify the deploy landed
```bash
curl -s https://opsin.apps.czbiohub.org/ | grep -oE 'app.js\?v=[0-9]+'   # should flip to the new ?v
gh release list --repo czbiohub-sf/diffex-viewer --limit 2               # new vX.Y.Z = Latest
git -C /hpc/mydata/gav.sturm/diffex-viewer show origin/main:.infra/prod/values.yaml | grep 'tag:'  # sha-<latest>
```
Then hard-reload (Cmd/Ctrl+Shift+R) to bust the browser cache.

## ⚠️ Public manifest / data prune — prod leak protection (DO NOT SKIP)
Prod is **public**. Hiding tabs in JS is **cosmetic only** (devtools can un-hide). The real protection is
**what data lives in the prod bucket / CloudFront**:
- **Prune the manifest:** run `diffex/viewer/prune_public_manifest.py` — drops grains not in
  `{geneKO, complex}` and strips `binder_prob` / `gene_target` metadata (it verifies after). This produces the
  **public manifest** that prod must serve.
- **Sync only the public data subset:** exclude `_montage/`, `attention_heads/`, `pcs/` (+ minibinder/PC grains)
  so any un-hidden tab **404s** instead of leaking.
- **rclone filter:** must copy the **pruned public `manifest.json`**, NOT the internal one — the filter is the
  thing that keeps the wrong manifest out of the prod bucket. **Re-check the filter before every prod data sync.**
- Client gate (cosmetic): `PUBLIC_HOSTS=["opsin.apps.czbiohub.org"]` + `VIEWER_ENV=public` hides
  Montage/PC/Attention tabs and trims the methods deck. Convenience, not security.

## Gotchas
- **`updatecli` PRs** (`update 'stack' helm chart version for diffex-viewer-*`) bump the **helm chart**, NOT the
  app image — merging them does **not** deploy your webapp. The image repin lives in the **release-please PR**.
- Old pins can be **orphaned `sha-` values** (not commits in current history). Always pin to a `sha-<commit>`
  that has a **successful "Argus Docker Build"** run (`gh run list … --workflow "Argus Docker Build" --status success`).
- **"Don't do it twice":** confirm the release branch contains `main` HEAD (`git merge-base --is-ancestor origin/main <release-branch>`) and that `app.js` matches, so the built image has the *latest* webapp. The release PR then pins staging+prod to the **same** latest sha → one merge, both current.
- Prod bucket + CloudFront + ingress already exist (Kyle/infra). Routine deploys are just the tag repin above — no IAM step.
