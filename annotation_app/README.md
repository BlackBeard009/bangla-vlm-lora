# Bangla image-caption human-evaluation pilot

This directory contains a small React application for blinded, pairwise presentation of image--caption items in the Bangla captioning pilot. Reviewers score each caption for **adequacy** and **fluency** on 1--5 scales.

The application has no backend. Progress is kept in the browser's `localStorage`; at completion, the reviewer downloads a CSV file to return to the study team. The generated CSV may contain reviewer-provided identifying fields, so it should be handled as study data rather than committed to Git.

The pilot pack is configured as 20 sampled BAN-Cap validation images × 4 systems. It is a pilot instrument, not a validated benchmark.

## Prepare the app data

After building a pilot pack, export its static data:

```bash
python scripts/export_pilot_webapp_data.py
```

This writes `public/data.json` and `public/images/`. Both are ignored by Git because they are generated from the local pilot pack.

## Develop and build

```bash
cd annotation_app
npm install
npm run dev
npm run build
```

`npm run build` creates `dist/`, which can be hosted as a static site or served locally with `npm run preview`.
