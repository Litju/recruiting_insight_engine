function safe(value) {
    return value === null || value === undefined ? "–" : value;
}

function badgeClass(zone) {
    if (!zone) return "badge gray";
    if (zone === "GREEN") return "badge green";
    if (zone === "YELLOW") return "badge yellow";
    if (zone === "RED") return "badge red";
    return "badge gray";
}

document.getElementById("generateBtn").addEventListener("click", async () => {
    const candidate = {
        Age: Number(document.getElementById("age").value),
        Gender: document.getElementById("gender").value,
        EducationLevel: document.getElementById("education").value,
        JobTitle: document.getElementById("jobTitle").value,
        YearsExperience: Number(document.getElementById("experience").value)
    };

    document.getElementById("statusMessage").textContent = "Sending request...";

    try {
        const response = await fetch("/api/insights", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(candidate)
        });

        if (!response.ok) {
            const text = await response.text();
            document.getElementById("statusMessage").textContent =
                `Request failed: ${text}`;
            return;
        }

        const data = await response.json();
        document.getElementById("statusMessage").textContent = "Success";

        // ---- Populate UI ----

        // MHI
        document.getElementById("mhiZone").textContent = data.mhi.zone;
        document.getElementById("mhiZone").className = badgeClass(data.mhi.zone);
        document.getElementById("mhiScore").textContent =
            data.mhi.MHI ? data.mhi.MHI.toFixed(3) : "–";

        // Salary
        document.getElementById("predSalary").textContent =
            data.prediction ? `$${data.prediction.toFixed(2)}` : "–";
        document.getElementById("offerBand").textContent = safe(data.offer_band);
        document.getElementById("marketBand").textContent = safe(data.market_band);

        // Drivers
        const driverList = document.getElementById("driversList");
        driverList.innerHTML = "";
        if (Array.isArray(data.drivers)) {
            data.drivers.forEach(d => {
                const li = document.createElement("li");
                li.textContent = `${d.feature}: ${d.importance}`;
                driverList.appendChild(li);
            });
        }

        // Cohort
        document.getElementById("cohortSummary").textContent =
            safe(data.cohort.summary);

        // Fairness
        document.getElementById("fairnessSummary").textContent =
            safe(data.fairness.summary);

        // Narrative
        document.getElementById("narrativeText").textContent =
            safe(data.narrative);

        // Raw JSON
        document.getElementById("rawJson").textContent =
            JSON.stringify(data, null, 2);

    } catch (err) {
        document.getElementById("statusMessage").textContent =
            `Request failed: ${err.message}`;
    }
});

// Toggle raw JSON
document.getElementById("toggleRaw").addEventListener("click", () => {
    const box = document.getElementById("rawJson");
    box.style.display = box.style.display === "none" ? "block" : "none";
});
