function safe(value) {
    return value === null || value === undefined ? "N/A" : value;
}

function badgeClass(zone) {
    if (!zone) return "badge gray";
    if (zone === "GREEN") return "badge green";
    if (zone === "YELLOW") return "badge yellow";
    if (zone === "RED") return "badge red";
    return "badge gray";
}

// Main click handler for generating insights
document.getElementById("submitBtn").addEventListener("click", async () => {
    const candidate = {
        Age: Number(document.getElementById("age").value),
        Gender: document.getElementById("gender").value,
        EducationLevel: document.getElementById("education").value,
        JobTitle: document.getElementById("jobTitle").value,
        YearsOfExperience: Number(document.getElementById("experience").value),
    };

    document.getElementById("apiStatus").textContent = "Sending request...";

    try {
        const response = await fetch("/api/insights", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(candidate),
        });

        if (!response.ok) {
            const text = await response.text();
            document.getElementById("apiStatus").textContent =
                `Request failed: ${text}`;
            return;
        }

        const data = await response.json();
        document.getElementById("apiStatus").textContent = "Success";

        // ---- Populate UI ----

        // MHI
        const zone = data.mhi?.zone || "";
        document.getElementById("mhiZone").textContent = zone || "N/A";
        document.getElementById("mhiZone").className = badgeClass(zone);
        document.getElementById("mhiScore").textContent =
            data.mhi?.MHI ? data.mhi.MHI.toFixed(3) : "--";

        // Salary
        document.getElementById("predictedSalary").textContent =
            typeof data.prediction === "number"
                ? `$${data.prediction.toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                  })}`
                : "--";
        document.getElementById("offerBand").textContent = safe(data.offer_band);
        document.getElementById("marketBand").textContent = safe(data.market_band);

        // Drivers
        const driversList = document.getElementById("driversList");
        const drivers = Array.isArray(data.drivers) ? data.drivers : [];
        driversList.innerHTML =
            drivers.length > 0
                ? drivers
                      .map(
                          (d) => `
        <li>
          <strong>${d.feature ?? "N/A"}</strong><br>
          Contribution: ${d.contribution ?? "N/A"}<br>
          Direction: ${d.direction ?? "N/A"}<br>
          Reason: ${d.reason ?? "N/A"}
        </li>
      `,
                      )
                      .join("")
                : '<li class="muted">No drivers available yet.</li>';

        // Cohort
        document.getElementById("cohortSummary").textContent =
            safe(data.cohort?.summary);

        // Fairness
        document.getElementById("fairnessSummary").textContent =
            safe(data.fairness?.summary);

        // Narrative
        document.getElementById("narrativeText").textContent =
            safe(data.narrative);

        // Raw JSON
        document.getElementById("rawJson").textContent =
            JSON.stringify(data, null, 2);
    } catch (err) {
        document.getElementById("apiStatus").textContent =
            `Request failed: ${err.message}`;
    }
});

// Toggle raw JSON visibility
const toggleBtn = document.getElementById("toggleRawBtn");
toggleBtn.addEventListener("click", () => {
    const box = document.getElementById("rawJson");
    const isHidden = box.classList.toggle("hidden");
    toggleBtn.textContent = isHidden ? "Show Raw JSON" : "Hide Raw JSON";
});
