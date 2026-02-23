let ITEMS = [];
let CURRENT = 0;

function renderProgress(p) {
  const el = document.getElementById("progress");
  if (el) {
    el.innerHTML =
      `<b>Total:</b> ${p.total} | <b>Labeled:</b> ${p.labeled} | <b>Skipped:</b> ${p.skipped} | <b>Pending:</b> ${p.pending}`;
  }
}

function renderList() {
  const list = document.getElementById("list");
  if (!list) return;
  
  list.innerHTML = "";
  ITEMS.forEach(it => {
    const row = document.createElement("div");
    row.className = "item " + (it.idx === CURRENT ? "active" : "");
    row.textContent = `${it.idx} — ${it.status} — ${it.predicted_ean}`;
    row.onclick = () => loadItem(it.idx);
    list.appendChild(row);
  });
}

async function loadItems() {
  try {
    const res = await fetch(`/api/review/${RUN_ID}/items`);
    const data = await res.json();
    ITEMS = data.items;
    renderProgress(data.progress);
    renderList();
    if (ITEMS.length) await loadItem(0);
  } catch (e) {
    console.error("Error cargando items:", e);
    alert("Error cargando items: " + e.message);
  }
}

async function loadItem(idx) {
  CURRENT = idx;
  renderList();

  try {
    const res = await fetch(`/api/review/${RUN_ID}/item/${idx}`);
    const data = await res.json();

    const cropImg = document.getElementById("crop");
    if (cropImg) {
      cropImg.src = data.crop_url;
    }

    // candidates
    const cand = document.getElementById("candidates");
    if (cand) {
      const matches = data.top_matches || [];
      if (!matches.length) {
        cand.innerHTML = `<p class="muted">Sin top_matches en metadata.</p>`;
      } else {
        cand.innerHTML = `<h3>Candidatos</h3>` + matches.slice(0, 8).map(m => {
          const ean = m.ean ?? m.EAN ?? m.id ?? "—";
          const desc = m.descripcion ?? m.desc ?? m.description ?? "";
          const score = (m.score ?? m.sim ?? m.similitud ?? "").toString();
          return `<div class="cand" data-ean="${ean}"><b>${ean}</b> ${desc} <span class="muted">${score}</span></div>`;
        }).join("");
        cand.querySelectorAll(".cand").forEach(el => {
          el.onclick = () => {
            const input = document.getElementById("eanInput");
            if (input) input.value = el.dataset.ean;
          };
        });
      }
    }

    const eanInput = document.getElementById("eanInput");
    if (eanInput) {
      eanInput.value =
        (data.predicted_ean && data.predicted_ean !== "UNKNOWN") ? data.predicted_ean : "";
    }
  } catch (e) {
    console.error("Error cargando item:", e);
    alert("Error cargando item: " + e.message);
  }
}

async function assignEAN() {
  const eanInput = document.getElementById("eanInput");
  if (!eanInput) return;
  
  const ean = eanInput.value.trim();
  if (!ean) {
    alert("Ingresá un EAN.");
    return;
  }

  try {
    const res = await fetch(`/api/review/${RUN_ID}/set_ean`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({idx: CURRENT, ean})
    });
    const data = await res.json();
    await loadItems();
    // avanzar al próximo pendiente
    jumpNext();
  } catch (e) {
    console.error("Error asignando EAN:", e);
    alert("Error asignando EAN: " + e.message);
  }
}

async function skip() {
  try {
    const res = await fetch(`/api/review/${RUN_ID}/skip`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({idx: CURRENT})
    });
    await res.json();
    await loadItems();
    jumpNext();
  } catch (e) {
    console.error("Error saltando:", e);
    alert("Error saltando: " + e.message);
  }
}

function jumpNext() {
  const next = ITEMS.find(x => x.idx > CURRENT && (x.status !== "labeled" && x.status !== "skipped"));
  if (next) {
    loadItem(next.idx);
  } else {
    // Buscar desde el principio
    const firstPending = ITEMS.find(x => x.status !== "labeled" && x.status !== "skipped");
    if (firstPending) {
      loadItem(firstPending.idx);
    }
  }
}

async function searchDB(q) {
  if (!q || q.length < 2) {
    const dbResults = document.getElementById("dbResults");
    if (dbResults) dbResults.innerHTML = "";
    return;
  }
  
  try {
    const res = await fetch(`/api/db/search_products?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    const root = document.getElementById("dbResults");
    if (root) {
      const results = data.results || [];
      if (results.length === 0) {
        root.innerHTML = `<p class="muted">No se encontraron productos.</p>`;
      } else {
        root.innerHTML = `<h3>Búsqueda DB</h3>` + results.slice(0, 10).map(r =>
          `<div class="cand" data-ean="${r.ean}"><b>${r.ean}</b> ${r.descripcion || ""}</div>`
        ).join("");
        root.querySelectorAll(".cand").forEach(el => {
          el.onclick = () => {
            const input = document.getElementById("eanInput");
            if (input) input.value = el.dataset.ean;
          };
        });
      }
    }
  } catch (e) {
    console.error("Error buscando en DB:", e);
  }
}

async function absorb() {
  const log = document.getElementById("absorbLog");
  if (log) log.textContent = "Absorbiendo...\n";
  
  try {
    const res = await fetch(`/api/review/${RUN_ID}/absorb`, {method: "POST"});
    const data = await res.json();
    
    if (log) {
      log.textContent += (data.stdout || "") + "\n" + (data.stderr || "");
    }
    
    if (data.ok) {
      alert("Absorb OK");
    } else {
      alert("Absorb falló (ver log).");
    }
  } catch (e) {
    console.error("Error absorbiendo:", e);
    alert("Error absorbiendo: " + e.message);
  }
}

// Event listeners
const btnAssign = document.getElementById("btnAssign");
const btnSkip = document.getElementById("btnSkip");
const btnAbsorb = document.getElementById("btnAbsorb");
const eanInput = document.getElementById("eanInput");

if (btnAssign) btnAssign.onclick = assignEAN;
if (btnSkip) btnSkip.onclick = skip;
if (btnAbsorb) btnAbsorb.onclick = absorb;

if (eanInput) {
  eanInput.addEventListener("input", (e) => {
    const q = e.target.value.trim();
    if (q.length >= 2) searchDB(q);
  });
  
  // Enter para asignar
  eanInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      assignEAN();
    }
  });
}

// Cargar items al iniciar
loadItems();
