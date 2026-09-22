/**
 * SpamShield Frontend Interactive Scripts
 * Featuring Marvel Superhero Theme Switcher:
 * - Spider-Man (Dev Shukla)
 * - Captain America (Harsh)
 * - Iron Man (Abhiraj)
 */

const THEME_CONFIG = {
    spiderman: {
        name: "Spider-Man",
        badge: "🕷️ Spidey-Sense Active • 98.8% Accuracy",
        heroTitle: "Spidey-Sense Spam Detection",
        member: "Dev Shukla (Lead ML Engineer)",
        tagline: "With Great ML Accuracy Comes Great Spam Responsibility",
        icon: "🕷️",
        heroColor: "#e11d48",
    },
    captainamerica: {
        name: "Captain America",
        badge: "🛡️ Vibranium Protocol Active • 98.8% Accuracy",
        heroTitle: "Vibranium Shield Defense",
        member: "Harsh (NLP & Signal Processing Lead)",
        tagline: "I Can Filter Spam All Day",
        icon: "🛡️",
        heroColor: "#3b82f6",
    },
    ironman: {
        name: "Iron Man",
        badge: "⚡ Arc Reactor Online • 98.8% Accuracy",
        heroTitle: "JARVIS AI Neural Protocol",
        member: "Abhiraj (Backend & API Architect)",
        tagline: "Sometimes You Gotta Filter Before You Walk",
        icon: "⚡",
        heroColor: "#d97706",
    }
};

function applyMarvelTheme(themeKey) {
    if (!THEME_CONFIG[themeKey]) themeKey = "spiderman";
    
    document.documentElement.setAttribute("data-theme", themeKey);
    localStorage.setItem("spamshield_theme", themeKey);

    const config = THEME_CONFIG[themeKey];

    // Update Theme Buttons Active State
    document.querySelectorAll(".theme-btn").forEach(btn => {
        if (btn.getAttribute("data-set-theme") === themeKey) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    // Update Dynamic Navbar & Hero Elements
    const heroLiveBadge = document.getElementById("heroLiveBadgeText");
    if (heroLiveBadge) heroLiveBadge.textContent = config.badge;

    const heroTitleEl = document.getElementById("heroTitleDisplay");
    if (heroTitleEl) heroTitleEl.textContent = config.heroTitle;

    const heroMemberEl = document.getElementById("heroMemberDisplay");
    if (heroMemberEl) heroMemberEl.textContent = config.member;

    const heroTaglineEl = document.getElementById("heroTaglineDisplay");
    if (heroTaglineEl) heroTaglineEl.textContent = `"${config.tagline}"`;

    const heroIconEl = document.getElementById("heroIconDisplay");
    if (heroIconEl) heroIconEl.textContent = config.icon;
}

document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Theme from localStorage or Default to Spider-Man (Dev)
    const savedTheme = localStorage.getItem("spamshield_theme") || "spiderman";
    applyMarvelTheme(savedTheme);

    // 2. Attach Event Listeners to Theme Switcher Buttons
    document.querySelectorAll(".theme-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const themeKey = btn.getAttribute("data-set-theme");
            applyMarvelTheme(themeKey);
        });
    });

    // 3. Text Area Counter
    const textarea = document.getElementById("message_text");
    const charCounter = document.getElementById("charCount");
    const wordCounter = document.getElementById("wordCount");

    if (textarea && charCounter) {
        const updateCounters = () => {
            const text = textarea.value;
            charCounter.textContent = text.length;
            if (wordCounter) {
                const words = text.trim().split(/\s+/).filter(Boolean);
                wordCounter.textContent = words.length;
            }
        };

        textarea.addEventListener("input", updateCounters);
        updateCounters();
    }

    // 4. Preset Click Handlers
    const presetButtons = document.querySelectorAll(".preset-btn");
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const sampleText = btn.getAttribute("data-preset");
            if (textarea && sampleText) {
                textarea.value = sampleText;
                textarea.dispatchEvent(new Event("input"));
                textarea.focus();
                textarea.classList.add("border-primary");
                setTimeout(() => textarea.classList.remove("border-primary"), 500);
            }
        });
    });

    // 5. Copy to Clipboard Utility
    const copyButtons = document.querySelectorAll(".btn-copy");
    copyButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-target");
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                navigator.clipboard.writeText(targetEl.textContent.trim()).then(() => {
                    const originalHtml = btn.innerHTML;
                    btn.innerHTML = '<i class="bi bi-check2"></i> Copied!';
                    setTimeout(() => {
                        btn.innerHTML = originalHtml;
                    }, 2000);
                });
            }
        });
    });

    // 6. Batch Table Filter
    const filterInput = document.getElementById("batchTableFilter");
    if (filterInput) {
        filterInput.addEventListener("input", () => {
            const filter = filterInput.value.toLowerCase();
            const rows = document.querySelectorAll(".batch-result-row");
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(filter) ? "" : "none";
            });
        });
    }
});
