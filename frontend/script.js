/**
 * script.js — EduPredict AI Frontend Logic
 *
 * Responsibilities:
 *   1. Client-side form validation
 *   2. POST request to Flask /predict API
 *   3. Render prediction results:
 *      - Animated score ring
 *      - Performance category badge
 *      - Classification confidence bars
 *      - Feature importance bars
 *      - Personalised recommendation cards
 *   4. Loading state management
 *   5. Error handling (validation, API, network)
 *   6. Reset / clear form
 */

"use strict";

/* ── API Config ─────────────────────────────────────────────── */
const API_URL = "http://127.0.0.1:5000/predict";

/* ── Category metadata ──────────────────────────────────────── */
const CATEGORY_META = {
  "Excellent": {
    badgeClass: "badge-excellent",
    barClass:   "bar-excellent",
    desc:       "Outstanding academic performance. Keep it up!",
    ringColor:  "#16a34a",
  },
  "Good": {
    badgeClass: "badge-good",
    barClass:   "bar-good",
    desc:       "Above-average performance with room to reach excellence.",
    ringColor:  "#2563eb",
  },
  "Average": {
    badgeClass: "badge-average",
    barClass:   "bar-average",
    desc:       "Satisfactory performance. Consistent effort can improve this.",
    ringColor:  "#d97706",
  },
  "Needs Improvement": {
    badgeClass: "badge-needs",
    barClass:   "bar-needs",
    desc:       "Performance below target. See recommendations below.",
    ringColor:  "#dc2626",
  },
};

/* ── Human-readable feature labels ─────────────────────────── */
const FEATURE_LABELS = {
  attendance_pct:        "Attendance %",
  study_hours_per_week:  "Study Hours/Week",
  assignment_avg:        "Assignment Average",
  internal_marks:        "Internal Marks",
  prev_exam_score:       "Prev. Exam Score",
  assignment_completion: "Assignment Completion",
  participation_level:   "Participation Level",
  num_backlogs:          "No. of Backlogs",
};

/* ── Validation rules — must match Flask backend exactly ────── */
const FIELD_RULES = [
  { id: "attendance_pct",        label: "Attendance Percentage",     min: 0,  max: 100, isInt: false },
  { id: "study_hours_per_week",  label: "Study Hours Per Week",      min: 0,  max: 35,  isInt: false },
  { id: "assignment_avg",        label: "Assignment Average",         min: 0,  max: 100, isInt: false },
  { id: "internal_marks",        label: "Internal Assessment Marks", min: 0,  max: 50,  isInt: false },
  { id: "prev_exam_score",       label: "Previous Exam Score",       min: 0,  max: 100, isInt: false },
  { id: "assignment_completion", label: "Assignment Completion %",   min: 0,  max: 100, isInt: false },
  { id: "participation_level",   label: "Class Participation Level", min: 0,  max: 2,   isInt: true  },
  { id: "num_backlogs",          label: "Number of Backlogs",        min: 0,  max: 5,   isInt: true  },
];

/* ── DOM references ─────────────────────────────────────────── */
const form            = document.getElementById("predict-form");
const predictBtn      = document.getElementById("predict-btn");
const resetBtn        = document.getElementById("reset-btn");
const formError       = document.getElementById("form-error");
const resultsSection  = document.getElementById("results-section");
const loadingState    = document.getElementById("loading-state");
const apiError        = document.getElementById("api-error");
const apiErrorMsg     = document.getElementById("api-error-msg");
const resultContent   = document.getElementById("result-content");

// Score ring
const scoreNumber = document.getElementById("score-number");
const ringFill    = document.getElementById("ring-fill");

// Results panels
const categoryBadge    = document.getElementById("category-badge");
const categoryDesc     = document.getElementById("category-desc");
const confidenceBars   = document.getElementById("confidence-bars");
const importanceBars   = document.getElementById("importance-bars");
const recommendList    = document.getElementById("recommendations-list");

/* ── Score ring circumference (2π × r=50) ───────────────────── */
const RING_CIRCUMFERENCE = 2 * Math.PI * 50; // ≈ 314.16

/* ════════════════════════════════════════════════════════════════
   FORM VALIDATION
════════════════════════════════════════════════════════════════ */

/**
 * Validate all form fields.
 * Returns { valid: true, data: {...} } or { valid: false, errors: [...] }.
 * Field names in `data` exactly match Flask API expectations.
 */
function validateForm() {
  const errors = [];
  const data   = {};

  // Clear previous invalid states
  FIELD_RULES.forEach(rule => {
    const el = document.getElementById(rule.id);
    el.classList.remove("invalid");
  });

  FIELD_RULES.forEach(rule => {
    const el  = document.getElementById(rule.id);
    const raw = el.value.trim();

    // Empty check
    if (raw === "" || raw === null) {
      errors.push(`${rule.label} is required.`);
      el.classList.add("invalid");
      return;
    }

    const num = Number(raw);

    // Must be a number
    if (isNaN(num)) {
      errors.push(`${rule.label} must be a number.`);
      el.classList.add("invalid");
      return;
    }

    // Integer check
    if (rule.isInt && !Number.isInteger(num)) {
      errors.push(`${rule.label} must be a whole number (${rule.min}–${rule.max}).`);
      el.classList.add("invalid");
      return;
    }

    // Range check
    if (num < rule.min || num > rule.max) {
      errors.push(`${rule.label} must be between ${rule.min} and ${rule.max}.`);
      el.classList.add("invalid");
      return;
    }

    // Store as number (integer or float)
    data[rule.id] = rule.isInt ? parseInt(num, 10) : parseFloat(num);
  });

  return errors.length === 0
    ? { valid: true, data }
    : { valid: false, errors };
}

/* ════════════════════════════════════════════════════════════════
   UI STATE HELPERS
════════════════════════════════════════════════════════════════ */

function showLoading() {
  resultsSection.hidden      = false;
  loadingState.hidden        = false;
  loadingState.style.display = "";       // let .loading-card CSS take effect
  apiError.hidden            = true;
  resultContent.hidden       = true;
  predictBtn.disabled        = true;
  predictBtn.textContent     = "⏳  Predicting…";
}

function hideLoading() {
  loadingState.hidden       = true;
  loadingState.style.display = "none";   // override .loading-card { display:flex }
  predictBtn.disabled        = false;
  predictBtn.innerHTML       = '<span class="btn-icon">🔍</span> Predict Performance';
}

function showApiError(message) {
  hideLoading();
  apiError.hidden   = false;
  apiErrorMsg.textContent = message;
}

function showFormError(errors) {
  formError.hidden      = false;
  formError.textContent = errors.join("  •  ");
}

function clearFormError() {
  formError.hidden      = true;
  formError.textContent = "";
}

/* ════════════════════════════════════════════════════════════════
   RENDER RESULTS
════════════════════════════════════════════════════════════════ */

/**
 * Animate the SVG score ring to reflect 0–100 score.
 * stroke-dashoffset is reduced from RING_CIRCUMFERENCE toward 0
 * as the score increases from 0 to 100.
 */
function renderScoreRing(score) {
  scoreNumber.textContent = score.toFixed(1);
  const pct    = Math.min(Math.max(score, 0), 100) / 100;
  const offset = RING_CIRCUMFERENCE * (1 - pct);
  ringFill.style.strokeDashoffset = offset;
}

/**
 * Apply colour to ring based on the predicted category.
 */
function setRingColor(category) {
  const meta = CATEGORY_META[category] || CATEGORY_META["Average"];
  ringFill.style.stroke = meta.ringColor;
}

/**
 * Render the performance category badge.
 */
function renderCategory(category) {
  // Remove previous badge class
  categoryBadge.className = "category-badge";

  const meta = CATEGORY_META[category];
  if (meta) {
    categoryBadge.classList.add(meta.badgeClass);
    categoryDesc.textContent = meta.desc;
  }
  categoryBadge.textContent = category;
}

/**
 * Render classification confidence bars.
 * confidence: { "Average": 12.5, "Excellent": 0.0, ... }  (values in %)
 */
function renderConfidence(confidence, activeCategory) {
  confidenceBars.innerHTML = "";

  // Display in a logical order
  const ORDER = ["Excellent", "Good", "Average", "Needs Improvement"];

  ORDER.forEach(cat => {
    const pct  = confidence[cat] !== undefined ? confidence[cat] : 0;
    const meta = CATEGORY_META[cat];
    const isActive = cat === activeCategory;

    const row = document.createElement("div");
    row.className = "conf-row";

    // Label
    const label = document.createElement("div");
    label.className = "conf-label";
    label.textContent = cat;
    if (isActive) label.style.fontWeight = "700";

    // Bar track + fill
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = `bar-fill ${meta ? meta.barClass : "bar-good"}`;
    fill.style.width = "0%";   // starts at 0, animated below
    track.appendChild(fill);

    // Percentage label
    const pctLabel = document.createElement("div");
    pctLabel.className = "bar-pct";
    pctLabel.textContent = `${pct.toFixed(1)}%`;

    row.append(label, track, pctLabel);
    confidenceBars.appendChild(row);

    // Animate after brief delay so CSS transition fires
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
    });
  });
}

/**
 * Render feature importance bars.
 * importance: { "study_hours_per_week": 52.05, ... }  (values in %)
 * Already sorted descending by the backend.
 */
function renderImportance(importance) {
  importanceBars.innerHTML = "";

  // Max value for relative bar width scaling
  const maxVal = Math.max(...Object.values(importance));

  Object.entries(importance).forEach(([feature, pct]) => {
    const relWidth = maxVal > 0 ? (pct / maxVal) * 100 : 0;
    const label    = FEATURE_LABELS[feature] || feature;

    const row = document.createElement("div");
    row.className = "imp-row";

    const labelEl = document.createElement("div");
    labelEl.className = "imp-label";
    labelEl.textContent = label;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "imp-fill";
    fill.style.width = "0%";
    track.appendChild(fill);

    const pctLabel = document.createElement("div");
    pctLabel.className = "imp-pct";
    pctLabel.textContent = `${pct.toFixed(1)}%`;

    row.append(labelEl, track, pctLabel);
    importanceBars.appendChild(row);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => { fill.style.width = `${relWidth}%`; });
    });
  });
}

/**
 * Render personalised recommendations as list items.
 */
function renderRecommendations(recs) {
  recommendList.innerHTML = "";
  recs.forEach(text => {
    const li = document.createElement("li");
    li.textContent = text;
    recommendList.appendChild(li);
  });
}

/**
 * Full result render — called once API response is received.
 */
function renderResults(data) {
  hideLoading();

  // Score ring
  renderScoreRing(data.predicted_score);
  setRingColor(data.performance_category);

  // Category badge
  renderCategory(data.performance_category);

  // Confidence
  renderConfidence(data.classification_confidence, data.performance_category);

  // Feature importance
  renderImportance(data.feature_importance);

  // Recommendations
  renderRecommendations(data.personalized_recommendations);

  // Reveal results
  resultContent.hidden = false;

  // Scroll so the Personalised Recommendations card is visible after prediction
  var predictPanel = document.getElementById("predict");
  if (predictPanel && recommendList) {
    var recsCard = recommendList.closest(".glass-card") || recommendList;
    predictPanel.scrollTo({ top: recsCard.offsetTop - 20, behavior: "smooth" });
  }
}

/* ════════════════════════════════════════════════════════════════
   API CALL
════════════════════════════════════════════════════════════════ */

async function callPredictAPI(payload) {
  showLoading();

  try {
    const response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    const json = await response.json();

    if (!response.ok) {
      // Backend returned a structured error (422, 400, 500, 503, …)
      const msg = json.error || `Server returned HTTP ${response.status}.`;
      showApiError(msg);
      return;
    }

    // Success — render the results
    renderResults(json);

  } catch (err) {
    // Network error — Flask server probably not running
    if (err instanceof TypeError && err.message.includes("fetch")) {
      showApiError(
        "Cannot reach the EduPredict AI server at " + API_URL + ". " +
        "Please start the Flask backend with: python app.py"
      );
    } else {
      showApiError("An unexpected error occurred: " + err.message);
    }
  }
}

/* ════════════════════════════════════════════════════════════════
   RESET
════════════════════════════════════════════════════════════════ */

function resetAll() {
  form.reset();
  clearFormError();
  resultsSection.hidden = true;
  resultContent.hidden  = true;
  apiError.hidden       = true;
  loadingState.hidden   = true;

  // Clear invalid states
  FIELD_RULES.forEach(rule => {
    document.getElementById(rule.id).classList.remove("invalid");
  });

  // Reset ring
  scoreNumber.textContent         = "--";
  ringFill.style.strokeDashoffset = RING_CIRCUMFERENCE;
  ringFill.style.stroke           = "var(--clr-accent)";
}

/* ════════════════════════════════════════════════════════════════
   EVENT LISTENERS
════════════════════════════════════════════════════════════════ */

// Form submission
form.addEventListener("submit", function (e) {
  e.preventDefault();
  clearFormError();

  const { valid, data, errors } = validateForm();

  if (!valid) {
    showFormError(errors);
    return;
  }

  // data keys exactly match Flask FEATURE_COLS:
  // attendance_pct, study_hours_per_week, assignment_avg, internal_marks,
  // prev_exam_score, assignment_completion, participation_level, num_backlogs
  callPredictAPI(data);
});

// Reset button
resetBtn.addEventListener("click", resetAll);

// Clear individual field's invalid state on user input
FIELD_RULES.forEach(rule => {
  const el = document.getElementById(rule.id);
  el.addEventListener("input", () => el.classList.remove("invalid"));
  el.addEventListener("change", () => el.classList.remove("invalid"));
});

/* ════════════════════════════════════════════════════════════════
   SPA NAVIGATION ROUTER
   Manages section switching, active nav state, mobile menu,
   scroll-reveal, and navbar shadow.
   All ML/API/form logic above (lines 1–461) is completely untouched.
════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  /* ── Page IDs ──────────────────────────────────────────────── */
  var PAGES       = ["home", "predict", "how-it-works", "about"];
  var currentPage = null; // null so first call always runs

  /* ── DOM refs ──────────────────────────────────────────────── */
  var navbar    = document.getElementById("navbar");
  var navToggle = document.getElementById("nav-toggle");
  var navLinks  = document.getElementById("nav-links");

  /* ─────────────────────────────────────────────────────────────
     navigateTo(pageId)
     Shows the target section; hides all others; updates nav state.
  ───────────────────────────────────────────────────────────── */
  function navigateTo(pageId) {
    if (!pageId || PAGES.indexOf(pageId) === -1) { pageId = "home"; }
    if (pageId === currentPage) { return; } // already showing this page

    currentPage = pageId;

    // Show target section, hide the rest
    PAGES.forEach(function (id) {
      var sec = document.getElementById(id);
      if (!sec) { return; }

      if (id === pageId) {
        // Make visible first, then reset scroll and animation
        sec.classList.add("active");
        sec.scrollTop = 0;
        sec.classList.remove("page-enter");
        void sec.offsetWidth; // force reflow so animation restarts
        sec.classList.add("page-enter");
        // Reveal child elements after a brief RAF so the browser has
        // painted the section (avoids IntersectionObserver root issues
        // with display:none → display:block transitions)
        triggerReveal(sec);
      } else {
        sec.classList.remove("active", "page-enter");
      }
    });

    // Highlight the matching nav button(s)
    updateNavHighlight(pageId);

    // Close mobile menu if open
    closeMenu();
  }

  /* ─────────────────────────────────────────────────────────────
     updateNavHighlight(pageId)
  ───────────────────────────────────────────────────────────── */
  function updateNavHighlight(pageId) {
    document.querySelectorAll("[data-nav]").forEach(function (el) {
      el.classList.toggle("active", el.getAttribute("data-nav") === pageId);
    });
  }

  /* ─────────────────────────────────────────────────────────────
     triggerReveal(sectionEl)
     Makes all .reveal children visible.
     Uses double-RAF so the browser has finished painting the newly
     displayed section before we add .visible (avoids the case where
     IntersectionObserver fires before layout is recalculated).
  ───────────────────────────────────────────────────────────── */
  function triggerReveal(sectionEl) {
    var els = sectionEl.querySelectorAll(".reveal");
    if (!els.length) { return; }
    // Reset first so re-entering a section re-animates
    els.forEach(function (el) { el.classList.remove("visible"); });
    // Double RAF: first RAF = end of current paint, second = next frame
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        els.forEach(function (el) { el.classList.add("visible"); });
      });
    });
  }

  /* ─────────────────────────────────────────────────────────────
     Mobile menu helpers
  ───────────────────────────────────────────────────────────── */
  function closeMenu() {
    if (navLinks)  { navLinks.classList.remove("open"); }
    if (navToggle) { navToggle.setAttribute("aria-expanded", "false"); }
  }

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      var open = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  /* ─────────────────────────────────────────────────────────────
     Navbar scroll shadow — listen on each page section's own scroll
  ───────────────────────────────────────────────────────────── */
  PAGES.forEach(function (id) {
    var sec = document.getElementById(id);
    if (sec && navbar) {
      sec.addEventListener("scroll", function () {
        navbar.classList.toggle("scrolled", sec.scrollTop > 10);
      }, { passive: true });
    }
  });

  /* ─────────────────────────────────────────────────────────────
     Wire up ALL [data-nav] elements (nav buttons, brand, hero CTA)
  ───────────────────────────────────────────────────────────── */
  document.querySelectorAll("[data-nav]").forEach(function (el) {
    el.addEventListener("click", function (e) {
      e.preventDefault();
      navigateTo(el.getAttribute("data-nav"));
    });
  });

  /* ─────────────────────────────────────────────────────────────
     Initialise — show Home on load
  ───────────────────────────────────────────────────────────── */
  navigateTo("home");

}());
