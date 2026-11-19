const tellsList = document.getElementById("tells-list");
const bpmText = document.getElementById("bpm-text");
const micText = document.getElementById("mic-text");
const confidencePct = document.getElementById("confidence-pct");
const truthMarker = document.getElementById("truth-marker");
const calibrationBanner = document.getElementById("calibration-banner");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const sessionStatus = document.getElementById("session-status");
const subjectInput = document.getElementById("subject-name");

const POLL_INTERVAL_MS = 500;

async function pollState() {
  try {
    const res = await fetch("/api/tells");
    const state = await res.json();
    renderState(state);
  } catch (err) {
    console.error("Failed to fetch state:", err);
  } finally {
    setTimeout(pollState, POLL_INTERVAL_MS);
  }
}

function renderState(state) {
  if (!state.calibrated) {
    calibrationBanner.classList.remove("hidden");
    const remaining = state.calibration_total - state.calibration_frames;
    calibrationBanner.textContent = `Calibrating... ${remaining} frames remaining`;
  } else {
    calibrationBanner.classList.add("hidden");
  }

  bpmText.textContent = state.bpm;

  if (state.mic_available) {
    if (state.mic_pitch > 0) {
      micText.textContent = state.mic_baseline
        ? `Mic: ${state.mic_pitch.toFixed(0)} Hz (baseline ${state.mic_baseline.toFixed(0)} Hz)`
        : `Mic: ${state.mic_pitch.toFixed(0)} Hz (calibrating baseline)`;
    } else {
      micText.textContent = "Mic: listening (no voice detected)";
    }
  } else {
    micText.textContent = "Mic: unavailable";
  }

  if (state.tells.length === 0) {
    tellsList.innerHTML = '<div class="empty">No active tells</div>';
  } else {
    tellsList.innerHTML = state.tells
      .map((t) => `<div class="tell-line">${t.text} (TTL: ${t.ttl})</div>`)
      .join("");
  }

  confidencePct.textContent = `${state.confidence}%`;
  truthMarker.style.left = `${state.confidence}%`;

  sessionStatus.textContent = state.session_active ? "Recording session..." : "No active session";
  startBtn.disabled = state.session_active;
  stopBtn.disabled = !state.session_active;
}

startBtn.addEventListener("click", async () => {
  const subject_name = subjectInput.value.trim();
  await fetch("/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject_name }),
  });
});

stopBtn.addEventListener("click", async () => {
  const res = await fetch("/session/stop", { method: "POST" });
  const result = await res.json();
  if (result.error) {
    alert(result.error);
    return;
  }
  alert(
    `Session ended.\nAverage confidence: ${result.avg_confidence.toFixed(1)}%\n` +
    `Max confidence: ${result.max_confidence.toFixed(1)}%`
  );
});

pollState();
