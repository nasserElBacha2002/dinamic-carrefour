async function refreshHome() {
  const res = await fetch("/api/videos");
  const data = await res.json();

  // videos
  const vroot = document.getElementById("videos");
  if (vroot) {
    vroot.innerHTML = "";
    data.videos.forEach(v => {
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<span>${v}</span><button>▶ Correr</button>`;
      row.querySelector("button").onclick = async () => {
        const log = document.getElementById("log");
        log.textContent = "Corriendo...\n";
        const rr = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({video: v, guardar_crops: true})
        });
        const r = await rr.json();
        log.textContent += (r.stdout || "") + "\n" + (r.stderr || "");
        await refreshHome();
      };
      vroot.appendChild(row);
    });
  }

  // runs
  const rroot = document.getElementById("runs");
  if (rroot) {
    rroot.innerHTML = "";
    data.runs.forEach(run => {
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<a href="/run/${run}">${run}</a>`;
      rroot.appendChild(row);
    });
  }
}

async function loadRunReport(runId) {
  const res = await fetch(`/api/run/${runId}/report`);
  const data = await res.json();

  // actions
  const actions = document.getElementById("actions");
  if (actions) {
    actions.innerHTML = `
      <a class="btn" href="/api/run/${runId}/download_csv">Descargar CSV</a>
      ${data.has_learning ? `<a class="btn" href="/review/${runId}">Revisar</a>` : `<span class="muted">Sin learning/</span>`}
    `;
  }

  // csv table
  const csv = document.getElementById("csv");
  if (csv) {
    const rows = data.rows || [];
    if (!rows.length) {
      csv.innerHTML = `<p class="muted">No se encontró CSV.</p>`;
    } else {
      const cols = Object.keys(rows[0]);
      const thead = `<tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>`;
      const tbody = rows.map(r => `<tr>${cols.map(c => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`).join("");
      csv.innerHTML = `<h2>Inventario (CSV)</h2><table>${thead}${tbody}</table>`;
    }
  }

  // frames
  const framesEl = document.getElementById("frames");
  if (framesEl) {
    framesEl.innerHTML = "";
    (data.frames || []).slice(0, 80).forEach(name => {
      const img = document.createElement("img");
      img.src = `/media/run/${runId}/frame/${name}`;
      framesEl.appendChild(img);
    });
  }
}

(async () => {
  if (document.getElementById("videos")) await refreshHome();
  if (window.RUN_ID && document.getElementById("csv")) await loadRunReport(window.RUN_ID);
})();
