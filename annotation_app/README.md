# Bangla caption human-eval annotation app

React SPA for the §C3 human-evaluation pilot. Shows the blinded
(image, caption) pairs one at a time; collects reviewer info
(name/email for uniqueness, native-speaker status) and adequacy +
fluency ratings (1–5 Likert, anchors from the pilot README). Progress
persists in localStorage; at the end the reviewer downloads a
`pilot_ratings_<email>.csv` and sends it back. No backend — ratings
never leave the reviewer's machine until they export.

## Refresh data (after rebuilding the pilot pack)

```bash
python scripts/export_pilot_webapp_data.py
# -> annotation_app/public/{data.json, images/}   (gitignored)
```

## Run / build

```bash
cd annotation_app
npm install
npm run dev       # local dev
npm run build     # -> dist/ (static, works from any subpath)
```

Deploy `dist/` to any static host (Netlify/Vercel/GitHub Pages) or
share over LAN with `npm run preview`. The returned CSVs feed the
scoring script (metric-vs-human correlation table).
