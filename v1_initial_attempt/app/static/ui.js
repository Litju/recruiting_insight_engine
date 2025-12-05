(function () {
  const statusEl = document.getElementById("status");
  const btn = document.getElementById("submit-btn");

  const EDUCATION_OPTIONS = [
    "High School",
    "Associate’s",
    "Bachelor’s",
    "Master’s",
    "PhD",
  ];

  const JOB_OPTIONS = [
    "Software Engineer",
    "Data Analyst",
    "Data Scientist",
    "Product Manager",
    "Sales Associate",
    "Senior Manager",
    "Director",
  ];

  function populateSelect(id, options, defaultVal) {
    const sel = document.getElementById(id);
    sel.innerHTML = "";
    options.forEach((opt) => {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      if (opt === defaultVal) o.selected = true;
      sel.appendChild(o);
    });
  }

  populateSelect("education", EDUCATION_OPTIONS, "Bachelor’s");
  populateSelect("jobTitle", JOB_OPTIONS, "Software Engineer");

  function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.style.color = isError ? "#d93025" : "#6b7280";
  }

  function fmtCurrency(val) {
    if (val === null || val === undefined || isNaN(val)) return "—";
    return "$" + Number(val).toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  function fmtPct(val) {
    if (val === null || val === undefined || isNaN(val)) return "—";
    return Number(val).toFixed(1) + "%";
  }

  function safe(val, def = "—") {
    return val === null || val === undefined ? def : val;
  }

  function renderTag(band) {
    const tag = document.getElementById("market_band");
    tag.textContent = band || "—";
    tag.className = "tag " + (band || "");
  }

  function renderFlags(flags) {
    const el = document.getElementById("flags");
    if (!Array.isArray(flags) || flags.length === 0) {
      el.textContent = "";
      return;
    }
    el.textContent = flags.join(", ");
  }

  function renderDrivers(topDrivers) {
    const container = document.getElementById("drivers_list");
    container.innerHTML = "";
    if (!Array.isArray(topDrivers) || topDrivers.length === 0) {
      container.innerHTML = `<div class="empty-msg">No driver data available. This may be due to limited samples.</div>`;
      return;
    }
    topDrivers.slice(0, 4).forEach((d) => {
      const contrib = Number(d.contribution || 0);
      const width = Math.min(100, Math.abs(contrib)) + "%";
      const driver = document.createElement("div");
      driver.className = "driver";
      driver.innerHTML = `
        <div class="driver-line">
          <span>${d.feature}</span>
          <span>${fmtCurrency(contrib)}</span>
        </div>
        <div class="bar"><div class="bar-fill ${contrib < 0 ? "negative" : ""}" style="width:${width}"></div></div>
      `;
      container.appendChild(driver);
    });
  }

  function renderCohorts(insights) {
    const tbody = document.getElementById("cohort_rows");
    tbody.innerHTML = "";
    const rows = [
      { key: "by_job_title", label: "Job Title", delta: "delta_vs_job_mean", deltaPct: "delta_vs_job_mean_pct" },
      { key: "by_education_level", label: "Education", delta: "delta_vs_edu_mean", deltaPct: "delta_vs_edu_mean_pct" },
      { key: "by_gender", label: "Gender", delta: "delta_vs_gender_mean", deltaPct: "delta_vs_gender_mean_pct" },
    ];
    rows.forEach((r) => {
      const data = insights[r.key] || {};
      const cohort = data.cohort || {};
      const count = cohort.count || 0;
      const hasData = !!cohort.mean_salary && count > 0;
      const confidence = data.confidence || {};
      const tr = document.createElement("tr");
      if (!hasData) {
        tr.innerHTML = `
          <td class="dim-label">${r.label}</td>
          <td colspan="4" class="empty-msg">Insufficient data for this group</td>
          <td>very_low</td>
        `;
      } else {
        tr.innerHTML = `
          <td class="dim-label">${r.label}</td>
          <td>${fmtCurrency(cohort.mean_salary)}</td>
          <td>${fmtCurrency(data[r.delta])}</td>
          <td>${fmtPct(data[r.deltaPct])}</td>
          <td>${count}</td>
          <td>${confidence.level ? confidence.level : "—"}${confidence.score ? ` (${confidence.score})` : ""}</td>
        `;
      }
      tbody.appendChild(tr);
    });
  }

  function renderBias(biasAudit) {
    const container = document.getElementById("bias_sections");
    const flagsEl = document.getElementById("bias_flags");
    const clarity = document.getElementById("bias_clarity");
    container.innerHTML = "";
    clarity.textContent = "";
    if (!biasAudit) {
      flagsEl.textContent = "—";
      clarity.textContent = "Not enough data to compute bias metrics.";
      return;
    }
    const sections = [
      { key: "gender", title: "Gender" },
      { key: "education", title: "Education" },
      { key: "age_groups", title: "Age Groups" },
    ];
    let anyData = false;
    flagsEl.textContent = (biasAudit.bias_flags || []).join(", ") || "None";
    sections.forEach((s) => {
      const data = biasAudit[s.key] || {};
      const hasMeans = data.group_means && Object.keys(data.group_means).length > 0;
      const card = document.createElement("div");
      card.className = "audit-card";
      if (!hasMeans) {
        card.innerHTML = `
          <h4>${s.title}</h4>
          <div class="empty-msg">Not enough group diversity to compute bias metrics.</div>
        `;
      } else {
        anyData = true;
        card.innerHTML = `
          <h4>${s.title}</h4>
          <ul>
            <li><strong>Gap Ratio:</strong> ${safe(data.gap_ratio)}</li>
            <li><strong>Abs Diff:</strong> ${fmtCurrency(data.abs_diff)}</li>
            <li><strong>Group Means:</strong> ${JSON.stringify(data.group_means || {})}</li>
            <li><strong>Counts:</strong> ${JSON.stringify(data.group_counts || {})}</li>
          </ul>
        `;
      }
      container.appendChild(card);
    });
    if (!anyData) {
      clarity.textContent = "Not enough group diversity to compute bias metrics.";
    }
  }

  function renderNarrative(text) {
    const el = document.getElementById("narrative_text");
    el.textContent = text || "—";
  }

  function renderRaw(jsonObj) {
    document.getElementById("raw_json").textContent = JSON.stringify(jsonObj, null, 2);
  }

  function renderExecutive(bundle) {
    const gm = bundle.salary_insights?.global_market || {};
    document.getElementById("salary_prediction").textContent = fmtCurrency(bundle.salary_prediction);
    renderTag(gm.market_band);
    document.getElementById("fairness_score").textContent =
      gm.fairness_score !== undefined && gm.fairness_score !== null ? Number(gm.fairness_score).toFixed(1) : "—";
    const deltaPct = gm.delta_vs_mean_pct;
    document.getElementById("delta_vs_mean").textContent = fmtPct(deltaPct);
    const offer = gm.recommended_offer_range || {};
    document.getElementById("offer_range").textContent =
      offer.recommended_min !== undefined && offer.recommended_max !== undefined
        ? `${fmtCurrency(offer.recommended_min)} – ${fmtCurrency(offer.recommended_max)}`
        : "—";
    renderFlags(bundle.flags || []);

    const clarity = document.getElementById("exec_clarity");
    clarity.textContent = "";
    if (!gm.market_band || gm.market_band === "—") {
      clarity.textContent = "Market band is unavailable; underlying cohort data may be limited.";
    } else if (gm.fairness_score === null || gm.fairness_score === undefined) {
      clarity.textContent = "Fairness score not computed; sample size may be insufficient.";
    } else if (gm.delta_vs_mean_pct === null || gm.delta_vs_mean_pct === undefined) {
      clarity.textContent = "Δ vs mean not available; check dataset size.";
    }
  }

  function renderTopDrivers(bundle) {
    const top = bundle.interpretability?.top_drivers || [];
    renderDrivers(top);
  }

  function renderCohortSection(bundle) {
    renderCohorts(bundle.salary_insights || {});
  }

  function renderBiasSection(bundle) {
    renderBias(bundle.bias_audit);
  }

  async function fetchInsights(payload) {
    const res = await fetch("/api/insights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API Error ${res.status}: ${text}`);
    }
    return res.json();
  }

  function gatherPayload() {
    return {
      "Age": Number(document.getElementById("age").value),
      "Gender": document.getElementById("gender").value,
      "Education Level": document.getElementById("education").value,
      "Job Title": document.getElementById("jobTitle").value,
      "Years of Experience": Number(document.getElementById("experience").value),
    };
  }

  async function onSubmit() {
    try {
      btn.disabled = true;
      setStatus("Requesting insights...");
      const payload = gatherPayload();
      const data = await fetchInsights(payload);
      renderExecutive(data);
      renderTopDrivers(data);
      renderCohortSection(data);
      renderBiasSection(data);
      renderNarrative(data.narrative);
      renderRaw(data);
      setStatus("Insights loaded.");
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Error fetching insights", true);
    } finally {
      btn.disabled = false;
    }
  }

  btn.addEventListener("click", onSubmit);
  setStatus("Ready.");
})();
