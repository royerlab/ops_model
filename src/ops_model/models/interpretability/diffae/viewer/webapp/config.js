// Runtime asset-origin config — OVERWRITTEN at container boot by
// docker-entrypoint.d/40-asset-base.sh (see diffex-viewer repo):
//   prod   → window.MANIFEST_URL = "<CloudFront domain>/manifest.json"
//   nonprod→ left empty; app.js then uses relative paths served off the /data S3 mount.
// This checked-in placeholder is empty so local/static serving doesn't 404.
