const App = (() => {
    let bootstrap = {};
    const bootstrapEl = document.getElementById('bootstrap-data');
    if (bootstrapEl) {
        try {
            bootstrap = JSON.parse(bootstrapEl.textContent);
        } catch (err) {
            console.error('Failed to parse bootstrap data', err);
        }
    }

    const voices = Array.isArray(bootstrap.voices) ? bootstrap.voices : [];
    const languages = Array.isArray(bootstrap.languages) ? bootstrap.languages : [];
    const modelInfo = bootstrap.model_info || {};
    const iconUrls = bootstrap.icon_urls || {};

    const elements = {};
    const state = {
        selectedLanguage: languages.length > 0 && languages[0].code ? languages[0].code : 'a',
        selectedGender: 'all',
        selectedVoice: null,
        isGenerating: false,
    };

    const batch = {
        rows: [],
        counter: 0,
        isRunning: false,
    };

    const formatters = {
        grade: (grade) => grade ? `Grade ${grade}` : 'Ungraded',
        genderIcon: (gender) => {
            if (gender === 'male') return iconUrls.male || iconUrls.female;
            if (gender === 'female') return iconUrls.female || iconUrls.male;
            return iconUrls.other || iconUrls.female || iconUrls.male;
        },
    };

    function cacheElements() {
        elements.navButtons = document.querySelectorAll('[data-nav]');
        elements.views = {
            single: document.getElementById('view-single'),
            batch: document.getElementById('view-batch'),
            voices: document.getElementById('view-voices'),
        };
        elements.statusPill = document.getElementById('modelStatusPill');
        elements.statusText = elements.statusPill ? elements.statusPill.querySelector('.status-pill__text') : null;
        elements.languageSelect = document.getElementById('languageSelect');
        elements.genderFilter = document.getElementById('genderFilter');
        elements.voiceList = document.getElementById('voiceList');
        elements.voiceCardTemplate = document.getElementById('voiceCardTemplate');
        elements.voiceDetails = {
            name: document.getElementById('detailVoiceName'),
            id: document.getElementById('detailVoiceId'),
            lang: document.getElementById('detailVoiceLang'),
            gender: document.getElementById('detailVoiceGender'),
        };
        elements.summary = {
            voices: document.getElementById('summaryVoices'),
            languages: document.getElementById('summaryLanguages'),
            gender: document.getElementById('summaryGender'),
        };
        elements.speedSlider = document.getElementById('speedSlider');
        elements.speedValue = document.getElementById('speedValue');
        elements.scriptInput = document.getElementById('singleText');
        elements.charCount = document.getElementById('charCount');
        elements.generateBtn = document.getElementById('generateBtn');
        elements.outputPlaceholder = document.getElementById('outputPlaceholder');
        elements.outputArea = document.getElementById('outputArea');
        elements.audioPlayer = document.getElementById('singleAudio');
        elements.outputMeta = document.getElementById('outputMeta');
        elements.waveform = document.getElementById('waveformPlaceholder');
        elements.batch = {
            body: document.getElementById('batchTableBody'),
            addRowBtn: document.getElementById('addRowBtn'),
            loadSampleBtn: document.getElementById('loadSampleBtn'),
            runBtn: document.getElementById('runBatchBtn'),
            summary: document.getElementById('batchSummary'),
            stream: document.getElementById('jobStream'),
        };
        elements.jobCardTemplate = document.getElementById('jobCardTemplate');
        elements.voiceLibraryList = document.getElementById('voiceLibraryList');
        elements.voiceSearch = document.getElementById('voiceSearch');
    }

    function bindNavigation() {
        elements.navButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const view = btn.dataset.nav;
                elements.navButtons.forEach((b) => b.classList.toggle('active', b === btn));
                Object.keys(elements.views).forEach((key) => {
                    const el = elements.views[key];
                    if (el) {
                        el.classList.toggle('view--active', key === view);
                    }
                });
                if (view === 'voices') renderVoiceLibrary();
            });
        });
    }

    function populateLanguageSelect() {
        if (!elements.languageSelect) return;
        elements.languageSelect.innerHTML = '';
        languages.forEach((lang) => {
            const option = document.createElement('option');
            option.value = lang.code;
            option.textContent = `${lang.label} (${lang.code})`;
            elements.languageSelect.appendChild(option);
        });
        elements.languageSelect.value = state.selectedLanguage;
    }

    function bindFilters() {
        if (elements.languageSelect) {
            elements.languageSelect.addEventListener('change', () => {
                state.selectedLanguage = elements.languageSelect.value;
                const defaultVoice = voices.find((v) => v.language_code === state.selectedLanguage);
                if (defaultVoice) state.selectedVoice = defaultVoice;
                renderVoiceList();
            });
        }

        if (elements.genderFilter) {
            elements.genderFilter.querySelectorAll('button').forEach((btn) => {
                btn.addEventListener('click', () => {
                    state.selectedGender = btn.dataset.gender;
                    elements.genderFilter.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === btn));
                    renderVoiceList();
                });
            });
        }
    }

    function createVoiceCard(voice, isActive = false) {
        if (!elements.voiceCardTemplate || !elements.voiceCardTemplate.content) {
            return document.createElement('div');
        }
        const tpl = elements.voiceCardTemplate.content.cloneNode(true);
        const card = tpl.querySelector('[data-voice]');
        const idEl = tpl.querySelector('[data-voice-id]');
        const nameEl = tpl.querySelector('[data-voice-name]');
        const genderEl = tpl.querySelector('[data-voice-gender]');
        const gradeEl = tpl.querySelector('[data-voice-grade]');
        const selectBtn = tpl.querySelector('.voice-card__select');

        card.dataset.voice = voice.id;
        if (isActive) card.classList.add('active');
        idEl.textContent = voice.id;
        nameEl.textContent = voice.display_name || voice.id;
        genderEl.src = formatters.genderIcon(voice.gender);
        genderEl.alt = voice.gender || 'gender';
        gradeEl.textContent = voice.overall_grade || '—';
        selectBtn.addEventListener('click', () => setSelectedVoice(voice.id));
        card.addEventListener('click', (event) => {
            if (event.target === selectBtn) return;
            setSelectedVoice(voice.id);
        });
        return card;
    }

    function renderVoiceList() {
        if (!elements.voiceList) return;
        elements.voiceList.innerHTML = '';
        const filtered = voices.filter((voice) => {
            if (voice.language_code !== state.selectedLanguage) return false;
            if (state.selectedGender !== 'all' && voice.gender !== state.selectedGender) return false;
            return true;
        });

        if (!state.selectedVoice || state.selectedVoice.language_code !== state.selectedLanguage) {
            state.selectedVoice = filtered[0] || voices[0];
        }

        filtered.forEach((voice) => {
            const isActive = state.selectedVoice && state.selectedVoice.id === voice.id;
            const card = createVoiceCard(voice, isActive);
            elements.voiceList.appendChild(card);
        });

        updateVoiceDetails();
        updateSummary();
    }

    function setSelectedVoice(voiceId) {
        const voice = voices.find((v) => v.id === voiceId);
        if (!voice) return;
        state.selectedVoice = voice;
        state.selectedLanguage = voice.language_code;
        if (elements.languageSelect) elements.languageSelect.value = state.selectedLanguage;
        renderVoiceList();
    }

    function updateVoiceDetails() {
        if (!state.selectedVoice || !elements.voiceDetails) return;
        const voice = state.selectedVoice;
        if (elements.voiceDetails.name) {
            elements.voiceDetails.name.textContent = voice.display_name || voice.id;
        }
        if (elements.voiceDetails.id) {
            elements.voiceDetails.id.textContent = voice.id;
        }
        if (elements.voiceDetails.lang) {
            elements.voiceDetails.lang.textContent = voice.language_label;
        }
        if (elements.voiceDetails.gender) {
            elements.voiceDetails.gender.src = formatters.genderIcon(voice.gender);
            elements.voiceDetails.gender.alt = voice.gender || 'gender';
        }
    }

    function updateSummary() {
        if (!elements.summary) return;
        const maleCount = voices.filter((voice) => voice.gender === 'male').length;
        const femaleCount = voices.filter((voice) => voice.gender === 'female').length;
        const otherCount = voices.length - maleCount - femaleCount;
        const genderPieces = [];
        if (femaleCount) genderPieces.push(`${femaleCount}F`);
        if (maleCount) genderPieces.push(`${maleCount}M`);
        if (otherCount) genderPieces.push(`${otherCount}U`);

        if (elements.summary.voices) {
            elements.summary.voices.textContent = voices.length;
        }
        if (elements.summary.languages) {
            elements.summary.languages.textContent = languages.length;
        }
        if (elements.summary.gender) {
            elements.summary.gender.textContent = genderPieces.length ? genderPieces.join(' / ') : '—';
        }
    }

    function bindSingleControls() {
        if (elements.speedSlider && elements.speedValue) {
            elements.speedSlider.addEventListener('input', () => {
                elements.speedValue.textContent = `${Number(elements.speedSlider.value).toFixed(1)}×`;
            });
        }

        if (elements.scriptInput && elements.charCount) {
            elements.scriptInput.addEventListener('input', () => {
                const length = elements.scriptInput.value.length;
                elements.charCount.textContent = `${length} ${length === 1 ? 'character' : 'characters'}`;
            });
        }

        if (elements.generateBtn) {
            elements.generateBtn.addEventListener('click', generateSingle);
        }
    }

    async function generateSingle() {
        if (state.isGenerating) return;
        const text = elements.scriptInput ? elements.scriptInput.value.trim() : '';
        if (!text) {
            alert('Please provide some text to synthesize.');
            return;
        }
        if (!state.selectedVoice) {
            alert('Select a voice before generating speech.');
            return;
        }

        setGenerating(true);
        try {
            const payload = {
                text,
                voice: state.selectedVoice.id,
                language: state.selectedVoice.language_code,
                speed: Number(elements.speedSlider ? elements.speedSlider.value : 1),
            };
            const response = await fetch('/generate_speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || data.message || 'Generation failed');
            }
            showOutput(data, payload);
        } catch (error) {
            console.error(error);
            alert(`Generation failed: ${error.message}`);
        } finally {
            setGenerating(false);
        }
    }

    function setGenerating(flag) {
        state.isGenerating = flag;
        if (elements.generateBtn) {
            elements.generateBtn.disabled = flag;
            elements.generateBtn.textContent = flag ? 'Generating…' : 'Generate Speech';
        }
    }

    function showOutput(data, payload) {
        if (elements.outputPlaceholder) {
            elements.outputPlaceholder.classList.add('hidden');
        }
        if (elements.outputArea) {
            elements.outputArea.classList.remove('hidden');
        }
        if (elements.audioPlayer) {
            elements.audioPlayer.src = `/download_audio/${data.audio_path}`;
            elements.audioPlayer.load();
        }
        if (elements.outputMeta) {
            const metaParts = [];
            if (payload.voice) metaParts.push(`Voice ${payload.voice}`);
            metaParts.push(`Speed ${Number(payload.speed).toFixed(1)}×`);
            if (data.audio_size) metaParts.push(`${(data.audio_size / 1024).toFixed(1)} KB`);
            elements.outputMeta.textContent = metaParts.join(' · ');
        }
    }

    async function refreshStatus() {
        if (!elements.statusPill || !elements.statusText) return;
        try {
            const response = await fetch('/model_status');
            const data = await response.json();
            elements.statusPill.classList.remove('status-pill--warning', 'status-pill--error');
            const deviceLabel = data && data.device ? data.device : 'unknown';
            if (data.loaded) {
                elements.statusText.textContent = `Aurora Voice Studio · ${deviceLabel}`;
            } else if (data.model_exists) {
                elements.statusPill.classList.add('status-pill--warning');
                elements.statusText.textContent = 'Model found, awaiting initialization';
            } else {
                elements.statusPill.classList.add('status-pill--error');
                elements.statusText.textContent = 'Model unavailable · demo mode';
            }
        } catch (error) {
            elements.statusPill.classList.add('status-pill--error');
            elements.statusText.textContent = `Status check failed: ${error.message}`;
        }
    }

    function getVoicesByLanguage(lang) {
        return voices.filter((voice) => voice.language_code === lang);
    }

    function addBatchRow(prefill = {}) {
        const fallbackLang = languages.length > 0 && languages[0].code ? languages[0].code : 'a';
        const language = prefill.language || state.selectedLanguage || fallbackLang;
        const voicesForLang = getVoicesByLanguage(language);
        const voice = prefill.voice || (voicesForLang.length > 0 ? voicesForLang[0].id : (voices.length > 0 ? voices[0].id : 'af_heart'));
        batch.counter += 1;
        batch.rows.push({
            id: batch.counter,
            text: prefill.text || '',
            language,
            voice,
            speed: Number(prefill.speed || 1).toFixed(1),
            status: prefill.status || 'Pending',
            audio_path: prefill.audio_path || null,
            audio_size: prefill.audio_size || null,
        });
        renderBatchTable();
    }

    function bindBatchControls() {
        if (elements.batch.addRowBtn) {
            elements.batch.addRowBtn.addEventListener('click', () => addBatchRow());
        }
        if (elements.batch.loadSampleBtn) {
            elements.batch.loadSampleBtn.addEventListener('click', loadSampleData);
        }
        if (elements.batch.runBtn) {
            elements.batch.runBtn.addEventListener('click', runBatchGeneration);
        }
    }

    function renderBatchTable() {
        const tbody = elements.batch.body;
        if (!tbody) return;
        tbody.innerHTML = '';
        batch.rows.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.dataset.rowId = String(row.id);

            const textCell = document.createElement('td');
            const textarea = document.createElement('textarea');
            textarea.className = 'batch-textarea';
            textarea.value = row.text;
            textarea.placeholder = `Row ${index + 1} text…`;
            textarea.addEventListener('input', () => {
                row.text = textarea.value;
            });
            textCell.appendChild(textarea);

            const langCell = document.createElement('td');
            const langSelect = document.createElement('select');
            langSelect.className = 'batch-select';
            languages.forEach((lang) => {
                const opt = document.createElement('option');
                opt.value = lang.code;
                opt.textContent = `${lang.label} (${lang.code})`;
                langSelect.appendChild(opt);
            });
            langSelect.value = row.language;
            langSelect.addEventListener('change', () => {
                row.language = langSelect.value;
                const voicesForLang = getVoicesByLanguage(row.language);
                row.voice = voicesForLang.length > 0 ? voicesForLang[0].id : row.voice;
                renderBatchTable();
            });
            langCell.appendChild(langSelect);

            const voiceCell = document.createElement('td');
            const voiceSelect = document.createElement('select');
            voiceSelect.className = 'batch-select';
            getVoicesByLanguage(row.language).forEach((voice) => {
                const opt = document.createElement('option');
                opt.value = voice.id;
                opt.textContent = voice.id;
                voiceSelect.appendChild(opt);
            });
            voiceSelect.value = row.voice;
            voiceSelect.addEventListener('change', () => {
                row.voice = voiceSelect.value;
            });
            voiceCell.appendChild(voiceSelect);

            const speedCell = document.createElement('td');
            const speedInput = document.createElement('input');
            speedInput.type = 'number';
            speedInput.min = '0.5';
            speedInput.max = '2';
            speedInput.step = '0.1';
            speedInput.value = row.speed;
            speedInput.className = 'batch-speed';
            speedInput.addEventListener('change', () => {
                const val = parseFloat(speedInput.value);
                if (!Number.isFinite(val)) {
                    speedInput.value = '1.0';
                    row.speed = '1.0';
                } else {
                    row.speed = Math.min(2, Math.max(0.5, val)).toFixed(1);
                    speedInput.value = row.speed;
                }
            });
            speedCell.appendChild(speedInput);

            const statusCell = document.createElement('td');
            statusCell.textContent = row.status || 'Pending';

            const outputCell = document.createElement('td');
            if (row.audio_path) {
                const link = document.createElement('a');
                link.href = `/download_audio/${row.audio_path}`;
                link.textContent = 'Download';
                link.target = '_blank';
                outputCell.appendChild(link);
            } else {
                outputCell.textContent = '—';
            }

            tr.appendChild(textCell);
            tr.appendChild(langCell);
            tr.appendChild(voiceCell);
            tr.appendChild(speedCell);
            tr.appendChild(statusCell);
            tr.appendChild(outputCell);
            tbody.appendChild(tr);
        });
        renderBatchSummary();
    }

    function renderBatchSummary() {
        const summary = elements.batch.summary;
        if (!summary) return;
        const total = batch.rows.length;
        const queued = batch.rows.filter((row) => row.status === 'Pending').length;
        const complete = batch.rows.filter((row) => row.status === 'Success').length;
        const failed = batch.rows.filter((row) => row.status === 'Failed').length;
        summary.innerHTML = `
            <div class="summary-line"><span>Total items</span><strong>${total}</strong></div>
            <div class="summary-line"><span>Pending</span><strong>${queued}</strong></div>
            <div class="summary-line"><span>Success</span><strong>${complete}</strong></div>
            <div class="summary-line"><span>Failed</span><strong>${failed}</strong></div>
        `;
    }

    function loadSampleData() {
        batch.rows = [];
        batch.counter = 0;
        addBatchRow({ text: 'Welcome to the Kokoro TTS showcase, powered by af_heart.', language: 'a', voice: 'af_heart', speed: 1.0 });
        addBatchRow({ text: 'Bonjour et merci de votre attention.', language: 'f', voice: 'ff_siwis', speed: 0.9 });
        addBatchRow({ text: 'こんにちは、本日のハイライトです。', language: 'j', voice: 'jf_tebukuro', speed: 1.1 });
    }

    async function runBatchGeneration() {
        if (batch.isRunning || !batch.rows.length) return;
        const invalid = batch.rows.some((row) => !row.text.trim());
        if (invalid) {
            alert('Please fill in text for all rows before running the batch.');
            return;
        }

        batch.isRunning = true;
        if (elements.batch.runBtn) {
            elements.batch.runBtn.disabled = true;
            elements.batch.runBtn.textContent = 'Processing…';
        }

        batch.rows.forEach((row) => {
            row.status = 'Processing…';
        });
        renderBatchTable();

        try {
            const payload = {
                items: batch.rows.map((row) => ({
                    text: row.text,
                    language: row.language,
                    voice: row.voice,
                    speed: parseFloat(row.speed),
                })),
            };

            const response = await fetch('/batch_generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Batch generation failed');
            }
            ingestBatchResults(data.results || []);
        } catch (error) {
            console.error(error);
            alert(`Batch generation failed: ${error.message}`);
        } finally {
            batch.isRunning = false;
            if (elements.batch.runBtn) {
                elements.batch.runBtn.disabled = false;
                elements.batch.runBtn.textContent = 'Generate Batch';
            }
        }
    }

    function ingestBatchResults(results) {
        if (!Array.isArray(results)) return;
        results.forEach((result) => {
            const row = batch.rows[result.index];
            if (!row) return;
            if (result.success) {
                row.status = 'Success';
                row.audio_path = result.audio_path;
                row.audio_size = result.audio_size;
            } else {
                row.status = 'Failed';
            }
        });
        renderBatchTable();
        renderJobStream(results);
    }

    function renderJobStream(results) {
        const stream = elements.batch.stream;
        if (!stream || !Array.isArray(results)) return;
        stream.innerHTML = '';
        results.forEach((result) => {
            if (!elements.jobCardTemplate || !elements.jobCardTemplate.content) {
                return;
            }
            const tpl = elements.jobCardTemplate.content.cloneNode(true);
            const card = tpl.querySelector('.job-card');
            const title = tpl.querySelector('[data-job-title]');
            const status = tpl.querySelector('[data-job-status]');
            const progress = tpl.querySelector('[data-job-progress]');
            const actions = tpl.querySelector('[data-job-actions]');

            const row = batch.rows[result.index];
            const voiceLabel = row && row.voice ? row.voice : 'voice';
            title.textContent = `#${result.index + 1} · ${voiceLabel}`;
            status.textContent = result.success ? 'Completed' : `Failed: ${result.error || 'Unknown error'}`;
            progress.innerHTML = `<span style="width:${result.success ? '100' : '0'}%"></span>`;

            if (result.success && row && row.audio_path) {
                const playBtn = document.createElement('button');
                playBtn.textContent = 'Play';
                playBtn.addEventListener('click', () => {
                    const audio = new Audio(`/download_audio/${row.audio_path}`);
                    audio.play();
                });
                const downloadBtn = document.createElement('button');
                downloadBtn.textContent = 'Download';
                downloadBtn.addEventListener('click', () => {
                    window.open(`/download_audio/${row.audio_path}`, '_blank');
                });
                actions.appendChild(playBtn);
                actions.appendChild(downloadBtn);
            }
            stream.appendChild(card);
        });
    }

    function renderVoiceLibrary() {
        if (!elements.voiceLibraryList) return;
        const query = elements.voiceSearch && elements.voiceSearch.value ? elements.voiceSearch.value.toLowerCase() : '';
        elements.voiceLibraryList.innerHTML = '';
        voices
            .filter((voice) => !query || voice.id.toLowerCase().includes(query) || (voice.language_label || '').toLowerCase().includes(query))
            .forEach((voice) => {
                const card = createVoiceCard(voice, state.selectedVoice && state.selectedVoice.id === voice.id);
                elements.voiceLibraryList.appendChild(card);
            });
    }

    function bindVoiceSearch() {
        if (elements.voiceSearch) {
            elements.voiceSearch.addEventListener('input', renderVoiceLibrary);
        }
    }

    function initBatchModule() {
        if (!batch.rows.length) addBatchRow();
        renderBatchSummary();
    }

    function init() {
        cacheElements();
        bindNavigation();
        populateLanguageSelect();
        bindFilters();
        renderVoiceList();
        bindSingleControls();
        refreshStatus();
        setInterval(refreshStatus, 60000);
        bindBatchControls();
        initBatchModule();
        bindVoiceSearch();
        renderVoiceLibrary();
        updateSummary();
    }

    return { init };
})();

document.addEventListener('DOMContentLoaded', App.init);
