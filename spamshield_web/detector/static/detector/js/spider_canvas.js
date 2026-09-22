/**
 * Interactive Cyber-Web Background Canvas
 * Simulates dynamic spider-web filaments connecting floating tech nodes
 */

(function () {
    const canvas = document.createElement("canvas");
    canvas.id = "spiderWebCanvas";
    canvas.style.position = "fixed";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none";
    canvas.style.zIndex = "-1";
    canvas.style.opacity = "0.75";
    document.body.prepend(canvas);

    const ctx = canvas.getContext("2d");
    let width, height;
    let particles = [];
    const particleCount = 55;
    const maxDistance = 140;

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    window.addEventListener("resize", resize);
    resize();

    class WebParticle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.7;
            this.vy = (Math.random() - 0.5) * 0.7;
            this.radius = Math.random() * 2 + 1;
            // Alternating Spider Red (#ff003c) and Web Cyan (#00d2ff) and Gold (#ffb703)
            const colors = [
                "rgba(255, 0, 60, 0.8)",
                "rgba(0, 210, 255, 0.8)",
                "rgba(255, 183, 3, 0.7)"
            ];
            this.color = colors[Math.floor(Math.random() * colors.length)];
            this.pulse = Math.random() * Math.PI;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;
            this.pulse += 0.03;

            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;
        }

        draw() {
            const currentRadius = this.radius + Math.sin(this.pulse) * 0.8;
            ctx.beginPath();
            ctx.arc(this.x, this.y, Math.max(0.5, currentRadius), 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.shadowColor = this.color;
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    for (let i = 0; i < particleCount; i++) {
        particles.push(new WebParticle());
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        // Draw connecting spider-web threads
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < maxDistance) {
                    const alpha = (1 - dist / maxDistance) * 0.35;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);

                    // Web filament gradient
                    const gradient = ctx.createLinearGradient(
                        particles[i].x, particles[i].y,
                        particles[j].x, particles[j].y
                    );
                    gradient.addColorStop(0, `rgba(255, 0, 60, ${alpha})`);
                    gradient.addColorStop(0.5, `rgba(0, 102, 255, ${alpha * 0.8})`);
                    gradient.addColorStop(1, `rgba(0, 210, 255, ${alpha})`);

                    ctx.strokeStyle = gradient;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }

        // Draw web nodes
        particles.forEach(p => {
            p.update();
            p.draw();
        });

        requestAnimationFrame(animate);
    }

    animate();
})();
