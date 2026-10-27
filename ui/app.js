/**
 * VeriRAG Studio — Client Controller
 * Grade One UI Logic | Zero Emojis | Vector & Typographic Badges
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const navTabs = document.querySelectorAll(".nav-tab-btn");
  const tabViews = document.querySelectorAll(".tab-view");

  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      navTabs.forEach((t) => {
        t.classList.remove("active");
        t.setAttribute("aria-selected", "false");
      });
      tabViews.forEach((v) => v.classList.remove("active"));

      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      const targetView = document.getElementById(targetId);
      if (targetView) targetView.classList.add("active");

      // Load benchmark if selected
      if (targetId === "tab-bench") {
        loadBenchmarkData();
      }
    });
  });

  // Presets definition (Clean, authentic legal scenarios)
  const PRESETS = {
    criminal: {
      query: "What was the outcome of the criminal appeal regarding the Section 302 conviction?",
      text: "The Madurai Bench of the Madras High Court set aside the judgment of conviction passed by the Additional Sessions Judge, Pudukottai, and acquitted all eight accused persons of the charges under Section 120B and Sections 302/34 of the Indian Penal Code."
    },
    cheque: {
      query: "What was the sentence awarded under Section 138 of the Negotiable Instruments Act?",
      text: "By order dated 13.11.2018 the appellants were sentenced to undergo imprisonment for two years and directed to pay the cheque amount along with interest under Section 138 of the Negotiable Instruments Act. The High Court dismissed the revision petition."
    },
    land: {
      query: "How much compensation was determined for the acquired land?",
      text: "The preliminary notification was published under Section 4(1) of the Land Acquisition Act, 1894 on 15.03.1995. The Reference Court enhanced the compensation to Rs. 3,00,000 per acre, and the High Court further enhanced the market value to Rs. 4,50,000 per acre along with 30% solatium."
    },
    hallucinated: {
      query: "Did the court uphold the conviction under Section 302?",
      text: "The Supreme Court dismissed the appeal and sentenced the accused to capital punishment under Section 999 IPC, holding in State v. Donald Trump (2024) 99 SCC 1 that murder is non-bailable."
    }
  };

  // Wire Preset Buttons
  document.querySelectorAll(".preset-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-preset");
      const preset = PRESETS[key];
      if (preset) {
        document.getElementById("query-input").value = preset.query;
        document.getElementById("text-input").value = preset.text;
      }
    });
  });

  // Verify Form Execution
  const verifyForm = document.getElementById("verify-form");
  const verifyBtn = document.getElementById("verify-btn");
  let lastVerificationResult = null;

  verifyForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const query = document.getElementById("query-input").value.trim();
    const text = document.getElementById("text-input").value.trim();
    const topK = parseInt(document.getElementById("topk-select").value, 10);
    const method = document.getElementById("decomposer-select").value;

    if (!text) return;

    // Loading state
    verifyBtn.disabled = true;
    verifyBtn.innerHTML = `
      <div class="spinner"></div>
      <span>Verifying Claims & Evidence...</span>
    `;

    try {
      const response = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          query: query,
          top_k: topK,
          decompose_method: method
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const result = await response.json();
      lastVerificationResult = result;
      renderVerificationResults(result);
    } catch (err) {
      showToast(`Verification error: ${err.message}`);
    } finally {
      verifyBtn.disabled = false;
      verifyBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:16px;height:16px;stroke:#fff;">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        <span>Execute Legal Verification</span>
      `;
    }
  });

  // Render Verification Results
  function renderVerificationResults(res) {
    document.getElementById("empty-state").style.display = "none";
    const container = document.getElementById("verified-content");
    container.style.display = "flex";

    // Stat metrics
    const counts = res.trust_summary ? res.trust_summary.counts : {};
    document.getElementById("stat-sup").textContent = counts["SUPPORTED"] || 0;
    document.getElementById("stat-con").textContent = counts["CONTRADICTED"] || 0;
    document.getElementById("stat-nei").textContent = counts["UNSUPPORTED"] || 0;
    document.getElementById("stat-time").textContent = `${res.execution_time_sec || 0}s`;

    // Claims List
    const claimsContainer = document.getElementById("claims-list");
    claimsContainer.innerHTML = "";

    const claims = res.claim_verdicts || [];
    claims.forEach((c, idx) => {
      const v = c.verdict || "UNSUPPORTED";
      const vClass = v.toLowerCase();
      const conf = (c.confidence !== undefined ? (c.confidence * 100).toFixed(1) : "0") + "%";

      const card = document.createElement("div");
      card.className = "claim-card";

      // Best Evidence snippet
      let evidenceHtml = "";
      if (c.best_evidence && c.best_evidence.text) {
        const evSnippet = c.best_evidence.text.slice(0, 260);
        const docId = c.best_evidence.doc_id || "ILDC";
        const chunkId = c.best_evidence.chunk_id || "";
        evidenceHtml = `
          <div class="claim-evidence-box has-${vClass.slice(0, 3)}">
            <div class="evidence-header">
              <span>EVIDENCE PASSAGE [${docId}:${chunkId}]</span>
              <span>SIMILARITY: ${(c.best_evidence.score || 0).toFixed(3)}</span>
            </div>
            <div class="evidence-quote">"${evSnippet}..."</div>
          </div>
        `;
      }

      // Legal Flags
      let flagsHtml = "";
      if (c.legal_flags && c.legal_flags.length > 0) {
        flagsHtml = `
          <div class="claim-flags-bar">
            ${c.legal_flags.map((f) => `<span class="flag-tag">[FLAG: ${f.type}] ${f.description}</span>`).join("")}
          </div>
        `;
      }

      // Candidate passages table
      let candidateRows = "";
      if (c.evaluations && c.evaluations.length > 0) {
        candidateRows = c.evaluations
          .map(
            (e, rank) => `
          <tr>
            <td>#${rank + 1}</td>
            <td><code>${e.doc_id}:${e.chunk_id}</code></td>
            <td><span class="badge-tag ${e.nli_verdict.toLowerCase()}">${e.nli_verdict}</span></td>
            <td>${(e.retrieval_score || 0).toFixed(3)}</td>
            <td>${e.text_snippet ? e.text_snippet.slice(0, 100) : ""}...</td>
          </tr>
        `
          )
          .join("");
      }

      card.innerHTML = `
        <div class="claim-card-top">
          <div class="claim-id-text">
            <span class="claim-index">CLAIM ${idx + 1}</span>
            <span class="claim-body-text">${escapeHtml(c.claim_text)}</span>
          </div>
          <span class="badge-tag ${vClass}">${v} (${conf})</span>
        </div>
        ${flagsHtml}
        ${evidenceHtml}
        <button type="button" class="candidate-toggle-btn" onclick="toggleCandidates('cand-table-${idx}')">
          <span>View All Candidate Passages (${c.evaluations ? c.evaluations.length : 0})</span>
        </button>
        <table class="candidates-table" id="cand-table-${idx}">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Passage ID</th>
              <th>NLI Verdict</th>
              <th>Retrieval Score</th>
              <th>Text Snippet</th>
            </tr>
          </thead>
          <tbody>
            ${candidateRows}
          </tbody>
        </table>
      `;

      claimsContainer.appendChild(card);
    });

    // Citations Drawer
    const citationsContainer = document.getElementById("citations-list");
    citationsContainer.innerHTML = "";
    const citations = res.citations || [];

    if (citations.length === 0) {
      citationsContainer.innerHTML = `<p style="font-size:0.82rem; color:var(--text-dim); font-style:italic;">No supporting legal citations established for this response.</p>`;
    } else {
      citations.forEach((cit) => {
        const item = document.createElement("div");
        item.className = "citation-item";
        item.innerHTML = `
          <div class="citation-meta">
            <span>CITATION [${cit.citation_index}]</span>
            <span>DOCUMENT: <code>${cit.doc_id}</code></span>
            <span>CHUNK: <code>${cit.chunk_id}</code></span>
            <span>SPAN: <code>[${cit.char_span ? cit.char_span.join(", ") : ""}]</code></span>
          </div>
          <div style="font-size:0.85rem; color:var(--text-main); line-height:1.5;">
            "${escapeHtml(cit.evidence_text ? cit.evidence_text.slice(0, 320) : "")}..."
          </div>
        `;
        citationsContainer.appendChild(item);
      });
    }

    showToast("Legal verification complete");
  }

  // Global toggle for candidate passages
  window.toggleCandidates = (id) => {
    const table = document.getElementById(id);
    if (table) {
      table.classList.toggle("show");
    }
  };

  // Copy Markdown Audit
  document.getElementById("copy-report-btn").addEventListener("click", () => {
    if (lastVerificationResult && lastVerificationResult.report_markdown) {
      navigator.clipboard.writeText(lastVerificationResult.report_markdown);
      showToast("Markdown audit report copied to clipboard");
    }
  });

  // Download JSON
  document.getElementById("download-json-btn").addEventListener("click", () => {
    if (lastVerificationResult) {
      const blob = new Blob([JSON.stringify(lastVerificationResult, null, 2)], {
        type: "application/json"
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `verirag_audit_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  });

  // ==============================================================================
  // TAB 2: Corpus Search Explorer
  // ==============================================================================
  const searchForm = document.getElementById("search-form");
  const searchResults = document.getElementById("search-results");
  const searchBtn = document.getElementById("search-btn");

  searchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = document.getElementById("search-query").value.trim();
    const mode = document.getElementById("search-mode").value;

    if (!query) return;

    searchBtn.disabled = true;
    searchBtn.innerHTML = `<div class="spinner"></div>`;

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, mode: mode, top_k: 8 })
      });

      const data = await res.json();
      renderSearchResults(data);
    } catch (err) {
      showToast(`Search error: ${err.message}`);
    } finally {
      searchBtn.disabled = false;
      searchBtn.innerHTML = `<span>Search</span>`;
    }
  });

  function renderSearchResults(data) {
    searchResults.innerHTML = "";
    const passages = data.passages || [];

    if (passages.length === 0) {
      searchResults.innerHTML = `
        <div class="empty-state">
          <h3>No Matching Legal Passages Found</h3>
          <p>Try modifying search terms or selecting a different retrieval mode.</p>
        </div>
      `;
      return;
    }

    const summaryBar = document.createElement("div");
    summaryBar.style.cssText =
      "display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; color:var(--text-dim); margin-bottom:0.75rem; font-family:var(--font-mono);";
    summaryBar.innerHTML = `
      <span>RETRIEVED ${passages.length} PASSAGES (MODE: ${data.mode.toUpperCase()})</span>
      <span>LATENCY: ${data.latency_ms}ms</span>
    `;
    searchResults.appendChild(summaryBar);

    passages.forEach((p, rank) => {
      const card = document.createElement("div");
      card.className = "passage-card";
      card.innerHTML = `
        <div class="passage-meta-bar">
          <div>
            <span>RANK #${rank + 1}</span> |
            <span>DOC <code>${p.doc_id}</code></span> |
            <span>CHUNK <code>${p.chunk_id}</code></span> |
            <span>WORDS: ${p.word_count || "N/A"}</span>
          </div>
          <span class="passage-score-tag">SCORE: ${(p.score || 0).toFixed(4)}</span>
        </div>
        <div style="font-size:0.88rem; color:var(--text-main); line-height:1.6;">
          ${escapeHtml(p.text)}
        </div>
      `;
      searchResults.appendChild(card);
    });
  }

  // ==============================================================================
  // TAB 3: Benchmark Data Loader
  // ==============================================================================
  let benchmarkLoaded = false;
  async function loadBenchmarkData() {
    if (benchmarkLoaded) return;
    try {
      const res = await fetch("/api/benchmark");
      const data = await res.json();
      const samples = data.sample_triples || [];
      const tbody = document.getElementById("table-bench-samples");
      tbody.innerHTML = "";

      samples.forEach((s) => {
        const vClass = s.gold_label.toLowerCase();
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${s.example_id}</code></td>
          <td><span style="font-size:0.72rem; text-transform:uppercase; color:var(--text-dim); font-family:var(--font-mono);">${s.claim_type}</span></td>
          <td><span class="badge-tag ${vClass === "sup" ? "supported" : vClass === "con" ? "contradicted" : "unsupported"}">${s.gold_label}</span></td>
          <td style="font-weight:600; font-size:0.82rem;">${escapeHtml(s.claim)}</td>
          <td style="font-size:0.78rem; color:var(--text-muted);">${escapeHtml(s.evidence.slice(0, 140))}...</td>
        `;
        tbody.appendChild(tr);
      });
      benchmarkLoaded = true;
    } catch (e) {
      console.error("Failed loading benchmark data", e);
    }
  }

  // Toast Notification
  function showToast(msg) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.style.display = "block";
    setTimeout(() => {
      t.style.display = "none";
    }, 3000);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
