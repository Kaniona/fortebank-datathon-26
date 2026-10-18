/* ForteVision · нәтиже беті: прогресс сақинасы, кезеңдер, вердикт, уақыт шкаласы, оқиғалар */
(function () {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const jobId = $('jobIdDisplay').textContent.trim();
    let apiKey = '';
    try { apiKey = localStorage.getItem('fortebank_api_key') || ''; } catch (e) {}

    const POLL_MS = 1500, MAX_POLLS = 240;          // 6 минут
    const RING_LEN = 2 * Math.PI * 96;              // шеңбер ұзындығы
    let polls = 0, timer = null, shown = 0;

    // Сервердің кезең атауы → кезеңдер тізіміндегі индекс
    const STAGE_INDEX = [
        [/жүкт|кезек/i, 0], [/адам/i, 1], [/көрініс/i, 2], [/зат/i, 3], [/дәлел|дайын/i, 4],
    ];

    const ICON_ALERT = '<svg viewBox="0 0 28 28" fill="none"><path d="M14 4 25 23H3L14 4Z" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/><path d="M14 11v6" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><circle cx="14" cy="20.2" r="1.3" fill="currentColor"/></svg>';
    const ICON_OK = '<svg viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="11" stroke="currentColor" stroke-width="2.2"/><path d="m9 14.5 3.5 3.5L19.5 10" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    const ICON_LAW = '<svg viewBox="0 0 16 16" fill="none"><path d="M8 2v12M3 14h10M4 5h8M4 5 2 9.5a2 2 0 0 0 4 0L4 5Zm8 0-2 4.5a2 2 0 0 0 4 0L12 5Z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>';
    const ICON_WARN = '<svg viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6.3" stroke="currentColor" stroke-width="1.3"/><path d="M8 4.8v3.6" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/><circle cx="8" cy="11" r=".8" fill="currentColor"/></svg>';

    // ── Көмекшілер ──────────────────────────────────────────────
    function esc(v) { const d = document.createElement('div'); d.textContent = v == null ? '' : String(v); return d.innerHTML; }
    function tc(sec) {
        sec = Math.max(0, Number(sec) || 0);
        const m = Math.floor(sec / 60), s = sec - m * 60;
        return String(m).padStart(2, '0') + ':' + s.toFixed(1).padStart(4, '0');
    }
    function human(sec) {
        sec = Number(sec) || 0;
        const m = Math.floor(sec / 60), s = Math.round(sec - m * 60);
        return m ? m + ' мин ' + s + ' с' : (Math.round(sec * 10) / 10) + ' с';
    }

    // ── Өңделу күйі ─────────────────────────────────────────────
    $('ringValue').style.strokeDasharray = RING_LEN;
    $('ringValue').style.strokeDashoffset = RING_LEN;
    function setProgress(pct, stage) {
        pct = Math.max(2, Math.min(100, Number(pct) || 0));
        shown = Math.max(shown, pct);                     // сақина артқа жүрмейді
        $('ringValue').style.strokeDashoffset = RING_LEN * (1 - shown / 100);
        $('procPct').textContent = Math.round(shown) + '%';
        let idx = 0;
        for (const [re, i] of STAGE_INDEX) if (stage && re.test(stage)) { idx = i; }
        document.querySelectorAll('.step').forEach((li) => {
            const n = Number(li.dataset.step);
            li.classList.toggle('is-done', n < idx);
            li.classList.toggle('is-active', n === idx);
            li.querySelector('.step__dot').textContent = n < idx ? '✓' : String(n + 1);
        });
    }
    function showError(title, text) {
        const box = $('resultStatus');
        box.classList.add('is-error');
        $('procTitle').textContent = title;
        $('procText').textContent = text;
        $('steps').hidden = true;
    }

    async function poll() {
        if (++polls > MAX_POLLS) { stop(); return showError('Күту уақыты бітті', 'Нәтиже 6 минутта дайын болмады. Бетті жаңартыңыз немесе бейнені қайта жүктеңіз.'); }
        try {
            const r = await fetch('/api/v1/jobs/' + encodeURIComponent(jobId), { headers: { 'X-API-Key': apiKey } });
            if (r.status === 401) {
                const k = prompt('API-кілтті енгізіңіз:');
                if (k) { apiKey = k.trim().replace(/^API_KEY=/, ''); try { localStorage.setItem('fortebank_api_key', apiKey); } catch (e) {} }
                return;
            }
            if (r.status === 404) { stop(); return showError('Тапсырма табылмады', 'Сілтеме қате немесе тапсырма өшірілген. Бейнені қайта жүктеңіз.'); }
            if (!r.ok) return;
            const d = await r.json();
            if (d.status === 'done') { stop(); finish(d); }
            else if (d.status === 'failed') { stop(); showError('Бейнені өңдеу мүмкін болмады', d.error || 'Басқа файлмен қайталап көріңіз.'); }
            else setProgress(d.progress, d.stage);
        } catch (e) { /* желі уақытша үзілсе — келесі сұрауда қайталаймыз */ }
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; } }
    poll(); timer = setInterval(poll, POLL_MS);

    // Нәтиже келгенде: сақинаны 100%-ға толтырып, сосын нәтижені көрсету
    function finish(d) {
        setProgress(100, 'дайын');
        document.querySelectorAll('.step').forEach((li) => { li.classList.add('is-done'); li.classList.remove('is-active'); li.querySelector('.step__dot').textContent = '✓'; });
        setTimeout(() => render(d), 650);
    }

    // ── Нәтиже ──────────────────────────────────────────────────
    function render(d) {
        const list = d.violations || [];
        const vi = d.video_info || {};
        $('resultStatus').hidden = true;
        $('resultContent').hidden = false;

        const card = $('verdictCard');
        if (d.verdict === 'violation') {
            card.classList.add('verdict--violation');
            $('verdictIcon').innerHTML = ICON_ALERT;
            $('verdictTitle').textContent = 'Бұзушылық белгілері анықталды';
            $('verdictDescription').textContent = 'Әр оқиғаны төмендегі кадрлар мен норма бойынша тексеріңіз.';
        } else {
            card.classList.add('verdict--clear');
            $('verdictIcon').innerHTML = ICON_OK;
            $('verdictTitle').textContent = 'Бұзушылық табылмады';
            $('verdictDescription').textContent = 'Тексерілетін белгілердің ешқайсысы бейнеде көрінбеді.';
            $('noViolationsMsg').hidden = false;
        }
        window.fxCount ? window.fxCount($('verdictCount'), list.length) : ($('verdictCount').textContent = list.length);

        if (vi.resolution || vi.duration_sec) {
            $('videoMeta').hidden = false;
            $('metaResolution').textContent = vi.resolution || '—';
            $('metaFps').textContent = vi.fps ? Number(vi.fps).toFixed(1) : '—';
            $('metaDuration').textContent = vi.duration_sec ? human(vi.duration_sec) : '—';
            $('metaElapsed').textContent = vi.processing_sec ? human(vi.processing_sec) : '—';
        }
        if (vi.duration_sec) renderTimeline(list, Number(vi.duration_sec));
        list.forEach((v, i) => $('violationsList').appendChild(eventCard(v, i)));
        if (window.fxReveal) window.fxReveal($('resultContent'));
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function renderTimeline(list, dur) {
        const wrap = $('timeline'), track = $('timelineTrack');
        wrap.hidden = false;
        if (!list.length) wrap.classList.add('timeline--clear');
        $('timelineLegend').textContent = '00:00 — ' + tc(dur) + (list.length ? ' · ' + list.length + ' белгі' : ' · таза');
        const pad = 12;
        const pos = (t) => 'calc(' + pad + 'px + (100% - ' + 2 * pad + 'px) * ' + (Math.min(Math.max(t, 0), dur) / dur).toFixed(4) + ')';
        const step = dur > 90 ? 20 : dur > 40 ? 10 : 5;
        for (let t = 0; t <= dur + 0.01; t += step) {
            const tick = document.createElement('span');
            tick.className = 'timeline__tick'; tick.style.left = pos(t); tick.textContent = Math.round(t) + 'с';
            track.appendChild(tick);
        }
        list.forEach((v, i) => {
            const m = document.createElement('button');
            m.type = 'button'; m.className = 'marker';
            m.style.left = pos(v.timecode_sec);
            m.style.animationDelay = (0.9 + i * 0.15) + 's';
            m.setAttribute('aria-label', (i + 1) + '-оқиға: ' + v.type + ', ' + tc(v.timecode_sec));
            m.innerHTML = '<span class="marker__label">' + esc(tc(v.timecode_sec) + ' · ' + v.type) + '</span><span class="marker__pin"></span>';
            m.addEventListener('click', () => {
                const el = $('event-' + i);
                if (!el) return;
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                el.classList.remove('is-flash'); void el.offsetWidth; el.classList.add('is-flash');
            });
            track.appendChild(m);
        });
    }

    function eventCard(v, i) {
        const el = document.createElement('article');
        el.className = 'glass event reveal';
        el.style.setProperty('--d', (0.1 + i * 0.08) + 's');
        el.id = 'event-' + i;
        const frames = v.evidence_frames || [];
        const first = frames[0];

        let shot = '<div class="shot">';
        if (first) shot += '<img data-role="main" src="data:image/jpeg;base64,' + first.image_base64 + '" alt="Дәлел кадры, ' + esc(tc(first.timestamp_sec)) + '">';
        shot += '<div class="scanline" aria-hidden="true"></div><div class="shot__corners" aria-hidden="true"><span></span><span></span><span></span><span></span></div>';
        shot += '<div class="hud hud--top"><span class="hud__rec"><i aria-hidden="true"></i>CAM 01</span><span data-role="osd-t">' + (first ? tc(first.timestamp_sec) : '') + '</span></div>';
        shot += '<div class="hud hud--bottom"><span class="hud__stat">ДӘЛЕЛ ' + (frames.length ? '1/' + frames.length : '') + '</span><span>FORTEVISION AI</span></div></div>';

        let thumbs = '';
        if (frames.length > 1) {
            thumbs = '<div class="thumbs" aria-label="Дәлел кадрлары">';
            frames.forEach((f, k) => {
                thumbs += '<button type="button" class="thumb' + (k === 0 ? ' is-active' : '') + '" data-k="' + k + '" aria-label="Кадр ' + esc(tc(f.timestamp_sec)) + '">' +
                          '<img src="data:image/jpeg;base64,' + f.image_base64 + '" alt=""><span>' + esc(tc(f.timestamp_sec)) + '</span></button>';
            });
            thumbs += '</div>';
        }

        const rule = v.rule;
        const norm = rule ? '<div class="norm"><p class="label">' + ICON_LAW + 'Норма</p>' +
            '<p class="norm__doc">' + esc(rule.document) + ' <span class="norm__clause">' + esc(rule.clause) + '</span></p>' +
            (rule.quote ? '<blockquote class="norm__quote">«' + esc(rule.quote) + '»</blockquote>' : '') +
            (rule.note ? '<p class="norm__note">' + esc(rule.note) + '</p>' : '') +
            (rule.url ? '<a class="norm__link" href="' + encodeURI(rule.url) + '" target="_blank" rel="noopener noreferrer">' + esc(rule.url) + ' ↗</a>' : '') +
            '</div>' : '';
        const limits = v.limitations ? '<div class="limits"><p class="label">' + ICON_WARN + 'Не ескеру керек</p><p>' + esc(v.limitations) + '</p></div>' : '';
        const confPct = typeof v.confidence === 'number' ? Math.round(v.confidence * 100) : null;
        const conf = confPct !== null ? '<span class="chip conf">сенімділік ' + confPct + '%<span class="conf__bar"><i data-w="' + confPct + '"></i></span></span>' : '';

        el.innerHTML =
            '<div>' + shot + thumbs + '</div>' +
            '<div>' +
                '<div class="event__meta"><span class="chip chip--time">' + esc(tc(v.timecode_sec)) + '</span><span class="chip chip--n">оқиға ' + (i + 1) + '</span>' + conf + '</div>' +
                '<h3 class="event__title">' + esc(v.type) + '</h3>' +
                '<p class="event__desc">' + esc(v.description) + '</p>' +
                norm + limits +
            '</div>';

        // Сенімділік жолағы карточка көрінгенде толады
        setTimeout(() => el.querySelectorAll('.conf__bar i').forEach((b) => { b.style.width = b.dataset.w + '%'; }), 400);

        el.querySelectorAll('.thumb').forEach((b) => b.addEventListener('click', () => {
            const k = Number(b.dataset.k), f = frames[k];
            if (!f) return;
            const img = el.querySelector('[data-role="main"]');
            if (img) {
                img.style.opacity = '0';
                setTimeout(() => { img.src = 'data:image/jpeg;base64,' + f.image_base64; img.alt = 'Дәлел кадры, ' + tc(f.timestamp_sec); img.style.opacity = '1'; }, 180);
            }
            const t = el.querySelector('[data-role="osd-t"]'); if (t) t.textContent = tc(f.timestamp_sec);
            const st = el.querySelector('.shot .hud__stat'); if (st) st.textContent = 'ДӘЛЕЛ ' + (k + 1) + '/' + frames.length;
            el.querySelectorAll('.thumb').forEach((x) => x.classList.toggle('is-active', x === b));
        }));
        return el;
    }
})();
