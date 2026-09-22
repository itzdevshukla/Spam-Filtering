/**
 * SPIDERSHIELD — Client-side Interactive Logic & Cyber Telemetry
 * Architect: Dev Shukla
 * Features:
 *  1. Smooth scroll to #scanner on "Launch Spidey Scan" (NO page reload)
 *  2. Real-time char & word counters
 *  3. Preset simulation target injection
 *  4. Asynchronous AJAX prediction with scanning animation HUD (800ms - 1200ms)
 *  5. Dynamic Dual Probability Bar Visualization
 *  6. Mobile navigation auto-collapse & scroll tracking
 *  7. Ctrl+Enter keyboard submission
 */

document.addEventListener("DOMContentLoaded", () => {
    // --------------------------------------------------------------------------
    // 1. CRITICAL BUG FIX: "LAUNCH SPIDEY SCAN" CTA BUTTON
    // --------------------------------------------------------------------------
    const launchScanBtn = document.getElementById("launchScanBtn");
    const scannerSection = document.getElementById("scanner");
    const textarea = document.getElementById("message_text");

    if (launchScanBtn) {
        launchScanBtn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();

            if (scannerSection) {
                scannerSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

                if (textarea) {
                    setTimeout(() => {
                        textarea.focus();
                    }, 500);
                }
            }
        });
    }

    // --------------------------------------------------------------------------
    // 2. NAVBAR SCROLL GLASS EFFECT & MOBILE MENU AUTO-CLOSE
    // --------------------------------------------------------------------------
    const mainNavbar = document.getElementById("mainNavbar");
    const navCollapse = document.getElementById("spidershieldNav");
    const navLinks = document.querySelectorAll(".nav-scroll-link");

    window.addEventListener("scroll", () => {
        if (mainNavbar) {
            if (window.scrollY > 25) {
                mainNavbar.classList.add("scrolled");
            } else {
                mainNavbar.classList.remove("scrolled");
            }
        }
    }, { passive: true });

    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            const href = link.getAttribute("href");
            if (href && href.startsWith("#")) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: "smooth", block: "start" });
                }

                // Auto collapse mobile menu if open
                if (navCollapse && navCollapse.classList.contains("show")) {
                    const bsCollapse = bootstrap.Collapse.getInstance(navCollapse) || new bootstrap.Collapse(navCollapse);
                    bsCollapse.hide();
                }
            }
        });
    });

    // --------------------------------------------------------------------------
    // 3. TEXTAREA COUNTERS & CLEAR BUTTON
    // --------------------------------------------------------------------------
    const charCountEl = document.getElementById("charCount");
    const wordCountEl = document.getElementById("wordCount");
    const validationErrorEl = document.getElementById("inputValidationError");
    const clearBtn = document.getElementById("clearPayloadBtn");

    const updateCounters = () => {
        if (!textarea) return;
        const text = textarea.value;
        if (charCountEl) charCountEl.textContent = text.length;
        if (wordCountEl) {
            const words = text.trim().split(/\s+/).filter(Boolean);
            wordCountEl.textContent = words.length;
        }
        if (text.trim().length > 0 && validationErrorEl) {
            validationErrorEl.classList.add("d-none");
        }
    };

    if (textarea) {
        textarea.addEventListener("input", updateCounters);
        updateCounters();

        // Keyboard Shortcut: Ctrl + Enter / Cmd + Enter
        textarea.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();
                const analyzeBtn = document.getElementById("analyzeMessageBtn");
                if (analyzeBtn) analyzeBtn.click();
            }
        });
    }

    if (clearBtn && textarea) {
        clearBtn.addEventListener("click", () => {
            textarea.value = "";
            updateCounters();
            if (validationErrorEl) validationErrorEl.classList.add("d-none");
            
            const resultContainer = document.getElementById("resultContainer");
            if (resultContainer) resultContainer.classList.add("d-none");
            
            const scanningHud = document.getElementById("scanningHud");
            if (scanningHud) scanningHud.classList.add("d-none");
            
            textarea.focus();
        });
    }

    // --------------------------------------------------------------------------
    // 4. RADAR SIMULATION TARGET PRESETS
    // --------------------------------------------------------------------------
    const presetButtons = document.querySelectorAll(".preset-btn");
    presetButtons.forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.preventDefault();
            const sampleText = btn.getAttribute("data-preset");
            if (textarea && sampleText) {
                textarea.value = sampleText;
                updateCounters();
                if (validationErrorEl) validationErrorEl.classList.add("d-none");

                // Highlight textarea briefly
                const glowContainer = document.querySelector(".textarea-glow-container");
                if (glowContainer) {
                    glowContainer.style.background = "linear-gradient(135deg, var(--spidey-red), var(--cyber-blue))";
                    setTimeout(() => {
                        glowContainer.style.background = "";
                    }, 600);
                }
                textarea.focus();
            }
        });
    });

    // --------------------------------------------------------------------------
    // 5. ASYNCHRONOUS SCANNER & TEMPORARY SCANNING HUD ANIMATION
    // --------------------------------------------------------------------------
    const analyzeBtn = document.getElementById("analyzeMessageBtn");
    const scanningHud = document.getElementById("scanningHud");
    const scanningStepText = document.getElementById("scanningStepText");
    const scanningProgressBar = document.getElementById("scanningProgressBar");
    const scanningPctText = document.getElementById("scanningPctText");
    const resultContainer = document.getElementById("resultContainer");
    const analyzeForm = document.getElementById("analyzeForm");

    if (analyzeBtn && textarea) {
        analyzeBtn.addEventListener("click", async (e) => {
            e.preventDefault();

            const text = textarea.value.trim();
            if (!text) {
                if (validationErrorEl) {
                    validationErrorEl.classList.remove("d-none");
                }
                textarea.focus();
                return;
            }

            if (validationErrorEl) validationErrorEl.classList.add("d-none");

            // Hide previous results
            if (resultContainer) {
                resultContainer.classList.add("d-none");
            }

            // Show Scanning HUD
            if (scanningHud) {
                scanningHud.classList.remove("d-none");
                scanningHud.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }

            // Animate steps and progress bar
            let progress = 15;
            if (scanningProgressBar) scanningProgressBar.style.width = `${progress}%`;
            if (scanningPctText) scanningPctText.textContent = `${progress}%`;
            if (scanningStepText) scanningStepText.textContent = "Scanning neural patterns...";

            const stepTimer1 = setTimeout(() => {
                progress = 55;
                if (scanningProgressBar) scanningProgressBar.style.width = `${progress}%`;
                if (scanningPctText) scanningPctText.textContent = `${progress}%`;
                if (scanningStepText) scanningStepText.textContent = "Checking message characteristics & n-grams...";
            }, 300);

            const stepTimer2 = setTimeout(() => {
                progress = 85;
                if (scanningProgressBar) scanningProgressBar.style.width = `${progress}%`;
                if (scanningPctText) scanningPctText.textContent = `${progress}%`;
                if (scanningStepText) scanningStepText.textContent = "Evaluating threat probability & log-odds...";
            }, 600);

            const startTime = Date.now();

            try {
                // Call Authoritative Django REST API endpoint
                const response = await fetch("/api/predict/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ text: text }),
                });

                if (!response.ok) {
                    throw new Error(`Server returned HTTP ${response.status}`);
                }

                const data = await response.json();

                // Ensure minimum animation duration of ~850ms for smooth user experience
                const elapsed = Date.now() - startTime;
                const remainingDelay = Math.max(0, 850 - elapsed);

                setTimeout(() => {
                    clearTimeout(stepTimer1);
                    clearTimeout(stepTimer2);

                    if (scanningProgressBar) scanningProgressBar.style.width = "100%";
                    if (scanningPctText) scanningPctText.textContent = "100%";
                    if (scanningStepText) scanningStepText.textContent = "Verdict compiled.";

                    setTimeout(() => {
                        if (scanningHud) scanningHud.classList.add("d-none");
                        renderDynamicResult(data);
                    }, 250);
                }, remainingDelay);

            } catch (err) {
                console.warn("[SpiderShield] AJAX classification fallback to standard form POST:", err);
                clearTimeout(stepTimer1);
                clearTimeout(stepTimer2);
                if (scanningHud) scanningHud.classList.add("d-none");
                // Resilient fallback to traditional Django POST submission
                if (analyzeForm) {
                    analyzeForm.submit();
                }
            }
        });
    }

    // --------------------------------------------------------------------------
    // 6. RENDER DYNAMIC RESULT CARD
    // --------------------------------------------------------------------------
    function renderDynamicResult(data) {
        if (!resultContainer) return;

        const isSpam = data.is_spam;
        const spamPct = (data.spam_percentage !== undefined) ? data.spam_percentage : (data.spam_probability * 100).toFixed(1);
        const hamPct = (data.ham_percentage !== undefined) ? data.ham_percentage : ((1 - data.spam_probability) * 100).toFixed(1);
        const riskLevel = data.risk_level || (isSpam ? "CRITICAL" : "SAFE");
        const latency = data.inference_time_ms ? Number(data.inference_time_ms).toFixed(2) : "2.40";
        const threshold = data.optimal_threshold ? Number(data.optimal_threshold).toFixed(2) : "0.35";

        let spamDriversHtml = "";
        if (data.spam_drivers && data.spam_drivers.length > 0) {
            spamDriversHtml = `
                <div class="mb-3">
                    <div class="text-danger small font-mono mb-1 fw-bold">
                        Vectors Pushing Toward SPAM:
                    </div>
                    <div class="d-flex flex-wrap gap-1">
                        ${data.spam_drivers.map(d => `
                            <span class="signal-pill pill-spam" title="Weight: ${Number(d.weight || 0).toFixed(3)}, Value: ${Number(d.value || 0).toFixed(2)}">
                                ${escapeHtml(d.feature)} <strong>+${Number(d.contribution || 0).toFixed(2)}</strong>
                            </span>
                        `).join("")}
                    </div>
                </div>
            `;
        } else if (data.top_signals && isSpam) {
            const spamTokens = data.top_signals.filter(s => s.contribution > 0);
            if (spamTokens.length > 0) {
                spamDriversHtml = `
                    <div class="mb-3">
                        <div class="text-danger small font-mono mb-1 fw-bold">
                            Vectors Pushing Toward SPAM:
                        </div>
                        <div class="d-flex flex-wrap gap-1">
                            ${spamTokens.map(s => `
                                <span class="signal-pill pill-spam">
                                    ${escapeHtml(s.feature)} <strong>+${Number(s.contribution).toFixed(2)}</strong>
                                </span>
                            `).join("")}
                        </div>
                    </div>
                `;
            }
        }

        let hamDriversHtml = "";
        if (data.ham_drivers && data.ham_drivers.length > 0) {
            hamDriversHtml = `
                <div>
                    <div class="text-info small font-mono mb-1 fw-bold">
                        Vectors Pushing Toward HAM (Safe):
                    </div>
                    <div class="d-flex flex-wrap gap-1">
                        ${data.ham_drivers.map(d => `
                            <span class="signal-pill pill-ham" title="Weight: ${Number(d.weight || 0).toFixed(3)}, Value: ${Number(d.value || 0).toFixed(2)}">
                                ${escapeHtml(d.feature)} <strong>${Number(d.contribution || 0).toFixed(2)}</strong>
                            </span>
                        `).join("")}
                    </div>
                </div>
            `;
        } else if (data.top_signals && !isSpam) {
            const hamTokens = data.top_signals.filter(s => s.contribution < 0);
            if (hamTokens.length > 0) {
                hamDriversHtml = `
                    <div>
                        <div class="text-info small font-mono mb-1 fw-bold">
                            Vectors Pushing Toward HAM (Safe):
                        </div>
                        <div class="d-flex flex-wrap gap-1">
                            ${hamTokens.map(s => `
                                <span class="signal-pill pill-ham">
                                    ${escapeHtml(s.feature)} <strong>${Number(s.contribution).toFixed(2)}</strong>
                                </span>
                            `).join("")}
                        </div>
                    </div>
                `;
            }
        }

        const cardHtml = `
            <div class="spider-result-card ${isSpam ? "result-spam" : "result-ham"} p-4 p-md-5 rounded-4">
                <!-- Status Header -->
                <div class="d-flex justify-content-between align-items-start flex-wrap gap-3 mb-4">
                    <div class="d-flex align-items-center gap-3">
                        <div class="result-status-icon-wrap">
                            ${isSpam ? '<span class="result-icon-threat">🚨</span>' : '<span class="result-icon-safe">✓</span>'}
                        </div>
                        <div>
                            <div class="result-label-badge ${isSpam ? "badge-threat" : "badge-safe"} mb-1">
                                ${isSpam ? "THREAT DETECTED" : "SYSTEM CLEAR"}
                            </div>
                            <h3 class="result-status-heading mb-0">
                                ${isSpam ? "SPAM DETECTED" : "MESSAGE SAFE"}
                            </h3>
                            <p class="result-status-subtext mb-0">
                                ${isSpam ? "This message appears suspicious and matches phishing/scam heuristics." : "No significant spam pattern found. Payload appears genuine."}
                            </p>
                        </div>
                    </div>
                    <div class="result-risk-pill ${isSpam ? "pill-threat" : "pill-safe"}">
                        ${escapeHtml(riskLevel)} RISK
                    </div>
                </div>

                <!-- Dual Probability Visualization -->
                <div class="probability-vis-box p-3 p-md-4 rounded-3 mb-4">
                    <h5 class="text-white font-mono small fw-bold mb-3 d-flex justify-content-between align-items-center">
                        <span>PROBABILITY DISTRIBUTION</span>
                        <span class="text-muted" style="font-size: 0.8rem;">DECISION CUTOFF: ${threshold}</span>
                    </h5>

                    <!-- SPAM Bar -->
                    <div class="mb-3">
                        <div class="d-flex justify-content-between text-white small font-mono mb-1">
                            <span class="text-danger fw-bold"><i class="bi bi-shield-x me-1"></i> SPAM PROBABILITY</span>
                            <span class="text-danger fw-bold">${spamPct}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill bar-fill-spam" id="dynamicSpamFill" style="width: 0%;"></div>
                        </div>
                    </div>

                    <!-- HAM Bar -->
                    <div>
                        <div class="d-flex justify-content-between text-white small font-mono mb-1">
                            <span class="text-info fw-bold"><i class="bi bi-shield-check me-1"></i> HAM (SAFE) PROBABILITY</span>
                            <span class="text-info fw-bold">${hamPct}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill bar-fill-ham" id="dynamicHamFill" style="width: 0%;"></div>
                        </div>
                    </div>
                </div>

                <!-- KPIs & Telemetry Grid -->
                <div class="row g-2 mb-4">
                    <div class="col-6 col-md-4">
                        <div class="kpi-mini-card">
                            <div class="kpi-mini-val ${isSpam ? "text-danger" : "text-info"}">
                                ${spamPct}%
                            </div>
                            <div class="kpi-mini-label">Spam Threat</div>
                        </div>
                    </div>
                    <div class="col-6 col-md-4">
                        <div class="kpi-mini-card">
                            <div class="kpi-mini-val text-warning">
                                ${latency} ms
                            </div>
                            <div class="kpi-mini-label">Inference Latency</div>
                        </div>
                    </div>
                    <div class="col-12 col-md-4">
                        <div class="kpi-mini-card">
                            <div class="kpi-mini-val text-white">
                                ${threshold}
                            </div>
                            <div class="kpi-mini-label">Boundary Threshold</div>
                        </div>
                    </div>
                </div>

                <!-- Signal Decomposition -->
                <div class="signal-decomposition-box pt-3 border-top border-secondary-subtle">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span class="text-white small fw-bold font-mono">
                            <i class="bi bi-cpu-fill text-danger me-1"></i> SPIDEY-SENSE TOKEN DRIVERS (LOG-ODDS)
                        </span>
                        <span class="text-info font-mono small">w_i &times; x_i</span>
                    </div>
                    ${spamDriversHtml}
                    ${hamDriversHtml}
                </div>
            </div>
        `;

        resultContainer.innerHTML = cardHtml;
        resultContainer.classList.remove("d-none");

        // Animate the probability bars smoothly after injection
        requestAnimationFrame(() => {
            setTimeout(() => {
                const spamFill = document.getElementById("dynamicSpamFill");
                const hamFill = document.getElementById("dynamicHamFill");
                if (spamFill) spamFill.style.width = `${spamPct}%`;
                if (hamFill) hamFill.style.width = `${hamPct}%`;
            }, 50);
        });

        // Smooth scroll result into viewport
        resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
