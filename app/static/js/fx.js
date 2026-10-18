/* Жалпы эффекттер: айналдырғанда пайда болу, шыны панельдегі прожектор, батырма толқыны */
(function () {
    'use strict';
    document.documentElement.classList.remove('no-js');
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Айналдырғанда пайда болу: элемент экранға кірген сәтте .is-in қосылады.
    // IntersectionObserver орнына тікелей тексеріс — кез келген браузерде және жасырын блок ашылғанда да сенімді.
    let pending = [];
    function check() {
        const h = window.innerHeight || document.documentElement.clientHeight;
        pending = pending.filter((el) => {
            if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') return true; // әзірге жасырын
            const r = el.getBoundingClientRect();
            if (r.top < h - 40 && r.bottom > 0) { el.classList.add('is-in'); return false; }
            return true;
        });
    }
    let raf = 0;
    const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; check(); }); };
    window.fxReveal = function (root) {
        (root || document).querySelectorAll('.reveal:not(.is-in)').forEach((el) => {
            if (reduce) el.classList.add('is-in'); else if (!pending.includes(el)) pending.push(el);
        });
        check();
        setTimeout(check, 60);
    };
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    window.fxReveal();

    // Курсор прожекторы (шыны панельдерде)
    document.addEventListener('pointermove', (e) => {
        const g = e.target.closest && e.target.closest('.glass');
        if (!g) return;
        const r = g.getBoundingClientRect();
        g.style.setProperty('--mx', (e.clientX - r.left) + 'px');
        g.style.setProperty('--my', (e.clientY - r.top) + 'px');
    }, { passive: true });

    // Батырма толқыны
    document.addEventListener('pointerdown', (e) => {
        const b = e.target.closest && e.target.closest('.btn');
        if (!b || b.disabled || reduce) return;
        const r = b.getBoundingClientRect(), s = Math.max(r.width, r.height);
        const w = document.createElement('span');
        w.className = 'ripple';
        w.style.cssText = 'width:' + s + 'px;height:' + s + 'px;left:' + (e.clientX - r.left - s / 2) + 'px;top:' + (e.clientY - r.top - s / 2) + 'px';
        b.appendChild(w);
        setTimeout(() => w.remove(), 650);
    });

    // Сан біртіндеп өседі
    window.fxCount = function (el, to, ms) {
        if (reduce || !to) { el.textContent = to; return; }
        const t0 = performance.now(); ms = ms || 1100;
        (function step(t) {
            const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
            el.textContent = Math.round(to * e);
            if (k < 1) requestAnimationFrame(step);
        })(t0);
    };
})();
