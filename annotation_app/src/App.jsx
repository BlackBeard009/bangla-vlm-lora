import { useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "bangla-humaneval-pilot-v1";

const ADEQUACY = [
  [5, "Fully correct — covers the main content of the image"],
  [4, "Correct with minor omissions"],
  [3, "Partially correct — some right, some wrong or missing"],
  [2, "Mostly wrong but topically related"],
  [1, "Unrelated to the image"],
];

const FLUENCY = [
  [5, "Fluent, natural Bangla"],
  [4, "Minor awkwardness"],
  [3, "Understandable but clearly flawed"],
  [2, "Broken grammar, hard to understand"],
  [1, "Not understandable / not Bangla"],
];

function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? null;
  } catch {
    return null;
  }
}

function save(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function csvEscape(value) {
  const s = String(value ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export default function App() {
  const [rows, setRows] = useState(null);
  const saved = useMemo(loadSaved, []);
  const [reviewer, setReviewer] = useState(
    saved?.reviewer ?? {
      name: "",
      email: "",
      age: "",
      nativeBangla: "",
      occupation: "",
    }
  );
  const [phase, setPhase] = useState(saved?.phase ?? "intro");
  const [index, setIndex] = useState(saved?.index ?? 0);
  const [ratings, setRatings] = useState(saved?.ratings ?? {});

  useEffect(() => {
    fetch("./data.json")
      .then((r) => r.json())
      .then(setRows)
      .catch(() => setRows([]));
  }, []);

  useEffect(() => {
    save({ reviewer, phase, index, ratings });
  }, [reviewer, phase, index, ratings]);

  if (rows === null) return <div className="shell">Loading…</div>;
  if (!rows.length)
    return <div className="shell">data.json missing — rebuild the app data.</div>;

  const current = rows[index];
  const currentRating = ratings[current?.row_id] ?? {};
  const nDone = rows.filter(
    (r) => ratings[r.row_id]?.adequacy && ratings[r.row_id]?.fluency
  ).length;

  const startDisabled =
    !reviewer.name.trim() ||
    !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(reviewer.email) ||
    !reviewer.nativeBangla;

  function setScore(field, value) {
    setRatings((prev) => ({
      ...prev,
      [current.row_id]: { ...prev[current.row_id], [field]: value },
    }));
  }

  function downloadCsv() {
    const meta = [
      `#reviewer_name=${reviewer.name}`,
      `#reviewer_email=${reviewer.email}`,
      `#reviewer_age=${reviewer.age}`,
      `#native_bangla=${reviewer.nativeBangla}`,
      `#occupation=${reviewer.occupation}`,
      `#submitted_at=${new Date().toISOString()}`,
    ];
    const header = "row_id,image,caption,adequacy_1to5,fluency_1to5,comments";
    const body = rows.map((r) => {
      const v = ratings[r.row_id] ?? {};
      return [
        r.row_id,
        r.image,
        csvEscape(r.caption),
        v.adequacy ?? "",
        v.fluency ?? "",
        csvEscape(v.comment ?? ""),
      ].join(",");
    });
    const blob = new Blob(
      ["﻿" + [...meta, header, ...body].join("\n")],
      { type: "text/csv;charset=utf-8" }
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const slug = reviewer.email.replace(/[^a-z0-9]/gi, "_").toLowerCase();
    a.download = `pilot_ratings_${slug}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  if (phase === "intro") {
    return (
      <div className="shell">
        <h1>Bangla Image-Caption Evaluation</h1>
        <p>
          You will see {rows.length} image–caption pairs, one at a time. For
          each, rate <b>Adequacy</b> (does the caption describe this image?)
          and <b>Fluency</b> (is it good Bangla, ignoring the image?). Judge
          each caption on its own — scores are absolute, not relative to other
          captions of the same image. Your progress is saved in this browser;
          you can close the tab and continue later.
        </p>
        <div className="form">
          <label>
            Full name *
            <input
              value={reviewer.name}
              onChange={(e) => setReviewer({ ...reviewer, name: e.target.value })}
            />
          </label>
          <label>
            Email * <span className="hint">(used only to keep reviews unique)</span>
            <input
              type="email"
              value={reviewer.email}
              onChange={(e) => setReviewer({ ...reviewer, email: e.target.value })}
            />
          </label>
          <label>
            Age
            <input
              type="number"
              min="10"
              max="99"
              value={reviewer.age}
              onChange={(e) => setReviewer({ ...reviewer, age: e.target.value })}
            />
          </label>
          <label>
            Native Bangla speaker? *
            <select
              value={reviewer.nativeBangla}
              onChange={(e) =>
                setReviewer({ ...reviewer, nativeBangla: e.target.value })
              }
            >
              <option value="">— select —</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
          </label>
          <label>
            Occupation / field of study
            <input
              value={reviewer.occupation}
              onChange={(e) =>
                setReviewer({ ...reviewer, occupation: e.target.value })
              }
            />
          </label>
        </div>
        <button
          className="primary"
          disabled={startDisabled}
          onClick={() => setPhase("rating")}
        >
          {nDone > 0 ? `Continue (${nDone}/${rows.length} done)` : "Start"}
        </button>
      </div>
    );
  }

  if (phase === "done") {
    return (
      <div className="shell">
        <h1>Done — thank you!</h1>
        <p>
          All {rows.length} pairs rated. Download your ratings file and send it
          back to the study organizer.
        </p>
        <button className="primary" onClick={downloadCsv}>
          Download ratings CSV
        </button>
        <button className="ghost" onClick={() => setPhase("rating")}>
          Review my answers
        </button>
      </div>
    );
  }

  const canNext = currentRating.adequacy && currentRating.fluency;
  const isLast = index === rows.length - 1;

  return (
    <div className="shell">
      <div className="progress">
        <div
          className="progress-fill"
          style={{ width: `${(nDone / rows.length) * 100}%` }}
        />
      </div>
      <div className="counter">
        Pair {index + 1} / {rows.length} · {nDone} rated
      </div>

      <img
        className="photo"
        src={`./images/${current.image}`}
        alt="scene to caption"
      />
      <div className="caption bangla">{current.caption}</div>

      <div className="scales">
        <fieldset>
          <legend>Adequacy — does it describe this image?</legend>
          {ADEQUACY.map(([v, label]) => (
            <label key={v} className="option">
              <input
                type="radio"
                name={`adequacy-${current.row_id}`}
                checked={currentRating.adequacy === v}
                onChange={() => setScore("adequacy", v)}
              />
              <b>{v}</b> {label}
            </label>
          ))}
        </fieldset>
        <fieldset>
          <legend>Fluency — is it good Bangla?</legend>
          {FLUENCY.map(([v, label]) => (
            <label key={v} className="option">
              <input
                type="radio"
                name={`fluency-${current.row_id}`}
                checked={currentRating.fluency === v}
                onChange={() => setScore("fluency", v)}
              />
              <b>{v}</b> {label}
            </label>
          ))}
        </fieldset>
      </div>

      <input
        className="comment"
        placeholder="Optional comment"
        value={currentRating.comment ?? ""}
        onChange={(e) => setScore("comment", e.target.value)}
      />

      <div className="nav">
        <button
          className="ghost"
          disabled={index === 0}
          onClick={() => setIndex(index - 1)}
        >
          ← Back
        </button>
        <button
          className="primary"
          disabled={!canNext}
          onClick={() => (isLast ? setPhase("done") : setIndex(index + 1))}
        >
          {isLast ? "Finish" : "Next →"}
        </button>
      </div>
    </div>
  );
}
