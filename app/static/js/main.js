/* ForteBank · басты бет: бейне таңдау, жүктеу, тексеруді бастау */
(function () {
    'use strict';

    const MAX_MB = 200;
    const ALLOWED = ['mov', 'mp4', 'avi', 'mkv'];

    const $ = (id) => document.getElementById(id);
    const zone = $('uploadZone'), fileInput = $('fileInput'), selectBtn = $('selectBtn');
    const info = $('uploadInfo'), fileName = $('fileName'), fileSize = $('fileSize');
    const keyInput = $('apiKeyInput'), analyzeBtn = $('analyzeBtn');
    const statusArea = $('statusArea'), statusText = $('statusText'), statusPct = $('statusPct'), bar = $('progressBar');
    const errorBox = $('errorMessage');
    const osdClock = $('osdClock'), osdState = $('osdState'), osdStat = $('osdStat');

    let selectedFile = null;

    try { keyInput.value = localStorage.getItem('fortebank_api_key') || ''; } catch (e) {}

    // ── Монитордағы сағат ───────────────────────────────────────
    function tick() {
        const d = new Date();
        osdClock.textContent = [d.getHours(), d.getMinutes(), d.getSeconds()]
            .map((n) => String(n).padStart(2, '0')).join(':');
    }
    tick(); setInterval(tick, 1000);

    // ── Көмекші функциялар ──────────────────────────────────────
    const formatSize = (b) => b < 1048576 ? (b / 1024).toFixed(0) + ' КБ' : (b / 1048576).toFixed(1) + ' МБ';
    const cleanKey = (v) => v.trim().replace(/^API_KEY=/, '').replace(/^["']|["']$/g, '');
    function showError(msg) { errorBox.textContent = msg; errorBox.hidden = false; }
    function hideError() { errorBox.hidden = true; }
    function refreshButton() { analyzeBtn.disabled = !(selectedFile && cleanKey(keyInput.value)); }
    function setProgress(pct, text) {
        statusArea.hidden = false;
        bar.style.width = pct + '%';
        statusPct.textContent = Math.round(pct) + '%';
        if (text) statusText.textContent = text;
    }

    // ── Файл таңдау ─────────────────────────────────────────────
    function setFile(file) {
        if (!file) return;
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (!ALLOWED.includes(ext)) { showError('Бұл пішім қабылданбайды. Рұқсат етілгені: ' + ALLOWED.join(', ') + '.'); return; }
        if (file.size > MAX_MB * 1048576) { showError('Файл ' + MAX_MB + ' МБ-тан үлкен. Қысқарақ жазбаны таңдаңыз.'); return; }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatSize(file.size) + ' · дайын';
        info.classList.add('is-visible');
        osdState.textContent = 'ФАЙЛ ДАЙЫН';
        osdStat.textContent = 'AI · талдауға дайын';
        hideError();
        refreshButton();
        if (!cleanKey(keyInput.value)) keyInput.focus();
    }

    selectBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
    zone.addEventListener('click', () => fileInput.click());
    zone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
    fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

    ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add('is-dragover'); }));
    ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove('is-dragover'); }));
    zone.addEventListener('drop', (e) => setFile(e.dataTransfer.files[0]));

    keyInput.addEventListener('input', refreshButton);
    keyInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !analyzeBtn.disabled) analyzeBtn.click(); });

    // ── Жүктеу ─────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', function () {
        if (!selectedFile) return;
        const apiKey = cleanKey(keyInput.value);
        if (!apiKey) { showError('API-кілтті енгізіңіз.'); keyInput.focus(); return; }
        try { localStorage.setItem('fortebank_api_key', apiKey); } catch (e) {}

        hideError();
        analyzeBtn.disabled = true;
        osdState.textContent = 'ЖҮКТЕЛУДЕ';
        setProgress(0, 'Файл жүктелуде…');

        // XHR — жүктеу пайызын көрсету үшін
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/v1/analyze');
        xhr.setRequestHeader('X-API-Key', apiKey);
        xhr.upload.onprogress = (e) => { if (e.lengthComputable) setProgress(e.loaded / e.total * 100, 'Файл жүктелуде…'); };
        xhr.onerror = () => fail('Серверге қосылу мүмкін болмады. Интернетті тексеріп, қайталаңыз.');
        xhr.onload = () => {
            let data = {};
            try { data = JSON.parse(xhr.responseText); } catch (e) {}
            if (xhr.status === 401) return fail('API-кілт сәйкес келмеді. Ұйымдастырушы берген кілтті бос орынсыз, толық қойыңыз.');
            if (xhr.status === 429) return fail('Қате кілт тым көп рет енгізілді. Бір минуттан кейін қайталаңыз.');
            if (xhr.status === 413) return fail('Файл тым үлкен. ' + MAX_MB + ' МБ-тан аспайтын жазбаны таңдаңыз.');
            if (xhr.status < 200 || xhr.status >= 300 || !data.job_id) {
                return fail((data && data.detail) || ('Жүктеу сәтсіз аяқталды (HTTP ' + xhr.status + ').'));
            }
            setProgress(100, 'Жүктелді — талдауға жіберілді');
            osdState.textContent = 'ТАЛДАУҒА ЖІБЕРІЛДІ';
            window.location.href = '/results/' + encodeURIComponent(data.job_id);
        };
        const form = new FormData();
        form.append('file', selectedFile);
        xhr.send(form);
    });

    function fail(msg) {
        statusArea.hidden = true;
        osdState.textContent = 'ҚАТЕ';
        showError(msg);
        refreshButton();
    }
})();
