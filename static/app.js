// static/app.js

let startTime = null;
let aiSeen = false;
let aiFollowed = null;

document.addEventListener("DOMContentLoaded", function () {
  startTime = performance.now();

  const aiPanel = document.getElementById("ai-panel");
  if (aiPanel) aiSeen = true;

  document.getElementById("approve-btn")?.addEventListener("click", () => submitDecision("Approve"));
  document.getElementById("reject-btn")?.addEventListener("click", () => submitDecision("Reject"));
});

function submitDecision(decision) {
  const caseId = document.getElementById("case-id")?.value;
  if (!caseId) return;

  const timeMs = Math.round(performance.now() - startTime);

  const aiRecEl = document.getElementById("ai-recommendation");
  if (aiRecEl) {
    const rec = aiRecEl.dataset.recommendation;
    aiFollowed = (decision === rec);
  } else {
    aiFollowed = null;
  }

  disableButtons();

  fetch("/submit_decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      case_id: caseId,
      decision,
      time_ms: timeMs,
      ai_followed: aiFollowed,
      ai_seen: aiSeen
    })
  })
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        window.location.href = data.next || "/task";
      } else {
        alert(data.error || "Error");
        enableButtons();
      }
    })
    .catch(() => {
      alert("Network error");
      enableButtons();
    });
}

function disableButtons() {
  const a = document.getElementById("approve-btn");
  const r = document.getElementById("reject-btn");
  if (a) a.disabled = true;
  if (r) r.disabled = true;
}

function enableButtons() {
  const a = document.getElementById("approve-btn");
  const r = document.getElementById("reject-btn");
  if (a) a.disabled = false;
  if (r) r.disabled = false;
}