
const tg = window.Telegram?.WebApp;
if (tg) {
  tg.ready();
  // Sync viewport height BEFORE expanding — Telegram's WebView
  // may report an incorrect height if expand() fires before the
  // CSS custom properties (--app-height / --vvh) are set.
  document.body.classList.add('telegram');
  requestAnimationFrame(() => {
    tg.expand();
  });
}

// ---------------------------------------------------------------------------
// Web mode detection: true when running outside Telegram WebApp
// ---------------------------------------------------------------------------
let _isWebMode = false;
let _webAuthConfig = null; // { enabled, bot_username, bot_id }
// Web session is managed via HttpOnly cookie — no JS-accessible token storage.
// _webHasSession tracks whether the user has an active cookie session (set after
// successful login, cleared on logout / 401).
let _webHasSession = false;
let _widgetCheckInterval = null; // cleared on navigation away from login page

let _shellRendered = false;  // флаг: основной shell уже в DOM

// ob-choice-card кликает через data-атрибуты — не через inline onclick.
// Это убирает мусорную перерисовку при каждом клике по карточкам онбординга.
document.addEventListener('click', function(e) {
  const card = e.target.closest('[data-ob-handler]');
  if (!card) return;
  const handler = card.getAttribute('data-ob-handler');
  const value = card.getAttribute('data-ob-value');
  if (handler && typeof window[handler] === 'function') {
    window[handler](value);
  }
});

// Delegated input handler for onboarding custom fields (CSP-safe: no inline oninput).
// Listens for 'input' events on elements with data-ob-input attribute.
document.addEventListener('input', function(e) {
  const el = e.target.closest('[data-ob-input]');
  if (!el) return;
  const handler = el.getAttribute('data-ob-input');
  if (handler && typeof window[handler] === 'function') {
    window[handler](el.value);
  }
});

// Event delegation for the dashboard hero action button (data-hero-action).
// Only a fixed allowlist of action keys is accepted to prevent JS injection.
const _HERO_ACTION_MAP = {
  channels:  () => switchTab('channels'),
  profile:   () => openChannelProfile(),
  plan:      () => switchTab('plan'),
  posts:     () => switchTab('posts'),
  autopost:  () => switchTab('autopost'),
  settings:  () => openSettingsModal(),
  analytics: () => openAnalyticsDetails(),
};
document.addEventListener('click', function(e) {
  const btn = e.target.closest('[data-hero-action]');
  if (!btn) return;
  const key = btn.getAttribute('data-hero-action');
  const fn = _HERO_ACTION_MAP[key];
  if (fn) fn();
});

// Event delegation for the delete-channel button (data-delete-channel).
// Channel name is stored in a data attribute (HTML-escaped) to avoid
// injecting user-controlled data into an onclick attribute.
document.addEventListener('click', function(e) {
  const btn = e.target.closest('[data-delete-channel]');
  if (!btn) return;
  const id = parseInt(btn.getAttribute('data-delete-channel'), 10);
  const name = btn.getAttribute('data-channel-name') || 'канал';
  if (id) deleteChannel(id, name);
});

// ---------------------------------------------------------------------------
// Global event delegation: data-action attribute → calls named function
// Supported data attributes:
//   data-action="functionName"       → calls window.functionName()
//   data-action-arg="value"          → passes as first argument
//   data-action-arg2="value"         → passes as second argument
// This eliminates the need for inline onclick= handlers.
// ---------------------------------------------------------------------------
const _ACTION_ALLOWLIST = Object.freeze(new Set([
  'closeModal', 'switchTab', 'showTariffsModal', 'buyTariff',
  'openDraftEditor', 'openChannelProfile', 'openSettingsModal',
  'openAnalyticsDetails', 'openPlanGenerator', 'openChatPicker',
  'generateDraft', 'generatePlan', 'createPlanItem', 'savePlanItem',
  'createSchedule', 'saveChannel', 'createDraft', 'saveDraft',
  'publishDraft', 'previewDraft', 'previewEditorDraft', 'resetEditorDraft',
  'generatePostInEditor', 'addHashtagsToEditor', 'rewriteEditorText',
  'redoEditorChange', 'syncEditorPreview', 'clearEditorMedia',
  'addEditorButtonRow', 'clearEditorButtons', 'removeEditorButtonRow',
  'askAIAssistant', 'onboardingPrev', 'onboardingNext', 'completeOnboarding',
  'runCompetitorSpy', 'runNewsSniperNow', 'refreshInbox', 'backToChat',
  'deleteSchedule', 'deleteInboxMedia', 'createDraftFromInbox',
  'activateChannel', 'toggleAutopost', 'updateAutopostMode', 'toggleAutopostNews',
  '_webStartLogin', '_webStartDemo', '_showWebLanding',
  'saveSettings', 'openScheduleModal', 'saveAutopostInterval',
  // Added for full inline onclick migration:
  'openChannelModal', 'deleteDraft', 'openPlanEditor', 'deletePlanItem',
  'openCompetitorSpyModal', 'showPaywallModal', 'openNewsSniperModal',
  'generatePostFromPlan', 'openAIAssistant', 'webLogout',
  // Source chips UI
  'addNewsSource', 'removeNewsSource',
  // Welcome screen
  'welcomeSaveChannel',
]));
document.addEventListener('click', function(e) {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.getAttribute('data-action');
  if (!_ACTION_ALLOWLIST.has(action)) return;
  const fn = window[action];
  if (typeof fn !== 'function') return;
  e.preventDefault();
  const arg = btn.getAttribute('data-action-arg');
  const arg2 = btn.getAttribute('data-action-arg2');
  if (arg2 !== null && arg2 !== undefined) fn(arg, arg2);
  else if (arg !== null && arg !== undefined) fn(arg);
  else fn();
  // Support data-dismiss-modal for compound actions (e.g. switchTab + closeModal)
  if (btn.hasAttribute('data-dismiss-modal')) {
    if (typeof window.closeModal === 'function') window.closeModal();
  }
});

// Delegation: confirm dialog buttons (dynamic resolvers)
document.addEventListener('click', function(e) {
  const btn = e.target.closest('[data-confirm-resolve]');
  if (!btn) return;
  const id = btn.getAttribute('data-confirm-resolve');
  const val = btn.getAttribute('data-confirm-value') === 'true';
  if (window.__confirmResolvers && window.__confirmResolvers[id]) {
    window.__confirmResolvers[id](val);
  }
});

// Delegation: media upload trigger (file input click)
document.addEventListener('click', function(e) {
  const btn = e.target.closest('[data-trigger-upload]');
  if (!btn) return;
  const target = document.getElementById(btn.getAttribute('data-trigger-upload'));
  if (target) target.click();
});

// Delegation: promise-based dialog buttons (onboarding media prompt, etc.)
document.addEventListener('click', function(e) {
  const btn = e.target.closest('[data-resolve-id]');
  if (!btn) return;
  const resolveId = btn.getAttribute('data-resolve-id');
  const action = btn.getAttribute('data-resolve-action'); // 'ok' or 'skip'
  const fn = window[`${resolveId}_${action}`];
  if (typeof fn === 'function') fn();
});

// Delegation: onchange handlers for form controls (data-change-action)
// Supports checkbox (passes checked state) and select (passes value).
const _CHANGE_ACTION_ALLOWLIST = Object.freeze(new Set([
  'toggleAutopost', 'updateAutopostMode', 'toggleAutopostNews',
]));
document.addEventListener('change', function(e) {
  const el = e.target.closest('[data-change-action]');
  if (!el) return;
  const action = el.getAttribute('data-change-action');
  if (!_CHANGE_ACTION_ALLOWLIST.has(action)) return;
  const fn = window[action];
  if (typeof fn !== 'function') return;
  // For checkboxes pass checked boolean, for selects pass value string
  if (el.type === 'checkbox') fn(el.checked);
  else fn(el.value);
});

// Delegation: Enter key in source add input triggers addNewsSource
document.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && e.target.id === 'news-source-input') {
    e.preventDefault();
    if (typeof addNewsSource === 'function') addNewsSource();
  }
});

const state = {
  user: null,
  data: null,
  activeTab: 'dashboard',
  loading: false,
  modalCleanup: null,
  refreshPromises: new Map(),
  awaitMediaHandled: false,
  editorOpen: false,
  onboarding: {
    active: false,
    step: 0,
    answers: {
      topic: '',
      audience: '',
      style: '',
      mode: '',
      formats: [],
      frequency: '',
      constraints: [],
      channelId: null,
      customTopic: '',
    }
  },
  autosaveTimer: null,
  autosaveInFlight: false,
  lastAutosavedHash: '',
  editorHistory: { undo: [], redo: [], lastCapturedHash: '' },
  pendingDeletedDraftIds: new Set(),
  pendingDeletedPlanIds: new Set(),
};

const UI_STATE_KEY = 'neurosmm_miniapp_ui_v1';
const AWAIT_MEDIA_KEY = 'neurosmm_miniapp_await_media_v1';
const AI_ASSISTANT_SESSION_KEY = 'neurosmm_miniapp_ai_session_v1';


function getViewportHeightPx() {
  const vv = window.visualViewport;
  const height = vv && Number(vv.height) > 0 ? Number(vv.height) : window.innerHeight;
  return Math.max(320, Math.round(height || 0));
}

function syncViewportHeight() {
  const height = getViewportHeightPx();
  document.documentElement.style.setProperty('--vvh', `${height}px`);
  document.documentElement.style.setProperty('--app-height', `${height}px`);
}

function bindViewportObservers() {
  syncViewportHeight();
  const vv = window.visualViewport;
  const onChange = () => {
    syncViewportHeight();
    const chat = document.getElementById('ai-assistant-chat');
    if (chat) chat.scrollTop = chat.scrollHeight;
  };
  window.addEventListener('resize', onChange, { passive: true });
  window.addEventListener('orientationchange', onChange, { passive: true });
  if (vv) {
    vv.addEventListener('resize', onChange, { passive: true });
    vv.addEventListener('scroll', onChange, { passive: true });
  }
}

// Set viewport height CSS variables early so layout doesn't flash.
syncViewportHeight();

function safeJsonParse(raw, fallback = null) {
  try { return raw ? JSON.parse(raw) : fallback; } catch { return fallback; }
}

function parseNewsSource(draft) {
  if (!draft || String(draft.draft_source || '') !== 'news_sniper') return null;
  return safeJsonParse(draft.news_source_json, null);
}

function saveUiState() {
  try {
    const payload = {
      activeTab: state.activeTab || 'dashboard',
      editorOpen: !!state.editorOpen,
      editor: state.editorOpen ? getCurrentEditorSnapshot() : null,
      ts: Date.now(),
    };
    localStorage.setItem(UI_STATE_KEY, JSON.stringify(payload));
  } catch {}
}

function loadUiState() {
  try {
    const saved = safeJsonParse(localStorage.getItem(UI_STATE_KEY), null);
    // Migrate legacy 'more' tab to 'autopost'
    if (saved && saved.activeTab === 'more') saved.activeTab = 'autopost';
    return saved;
  } catch { return null; }
}

function clearEditorUiState() {
  try { localStorage.removeItem(UI_STATE_KEY); } catch {}
  state.lastAutosavedHash = '';
  state.autosaveInFlight = false;
  if (state.autosaveTimer) {
    clearTimeout(state.autosaveTimer);
    state.autosaveTimer = null;
  }
}

function editorSnapshotHash(snapshot = {}) {
  try {
    return JSON.stringify({
      text: snapshot.text || '',
      prompt: snapshot.prompt || '',
      media_ref: snapshot.media_ref || '',
      media_type: snapshot.media_type || 'none',
      media_meta_json: snapshot.media_meta_json || '',
      buttons_json: snapshot.buttons_json || '[]',
      pin_post: snapshot.pin_post ? 1 : 0,
      comments_enabled: snapshot.comments_enabled ? 1 : 0,
      ad_mark: snapshot.ad_mark ? 1 : 0,
    });
  } catch { return ''; }
}

function initEditorHistory(snapshot = null) {
  const snap = snapshot || getCurrentEditorSnapshot() || {};
  const hash = editorSnapshotHash(snap);
  state.editorHistory = { undo: hash ? [snap] : [], redo: [], lastCapturedHash: hash };
}

function maybeCaptureEditorHistory() {
  const snap = getCurrentEditorSnapshot();
  if (!snap) return;
  const hash = editorSnapshotHash(snap);
  if (!hash || hash === state.editorHistory.lastCapturedHash) return;
  state.editorHistory.undo.push(snap);
  if (state.editorHistory.undo.length > 30) state.editorHistory.undo.shift();
  state.editorHistory.redo = [];
  state.editorHistory.lastCapturedHash = hash;
}

function stepEditorHistory(direction = 'undo') {
  const current = getCurrentEditorSnapshot();
  if (!current) return;
  if (direction === 'undo') {
    const stack = state.editorHistory.undo || [];
    if (stack.length <= 1) return toast('Больше нечего отменять');
    const currentSnap = stack.pop();
    state.editorHistory.redo.push(currentSnap);
    const prev = stack[stack.length - 1];
    applyEditorSnapshot(prev || {});
    state.editorHistory.lastCapturedHash = editorSnapshotHash(prev || {});
    scheduleEditorAutosave(true);
    return;
  }
  const next = (state.editorHistory.redo || []).pop();
  if (!next) return toast('Больше нечего вернуть');
  state.editorHistory.undo.push(next);
  applyEditorSnapshot(next);
  state.editorHistory.lastCapturedHash = editorSnapshotHash(next || {});
  scheduleEditorAutosave(true);
}

function scheduleEditorAutosave(immediate = false) {
  const snap = getCurrentEditorSnapshot();
  if (!snap) return;
  saveUiState();
  if (state.autosaveTimer) clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(() => { autosaveCurrentDraft(); }, immediate ? 80 : 700);
}

async function autosaveCurrentDraft() {
  const snap = getCurrentEditorSnapshot();
  if (!snap || state.autosaveInFlight) return;
  const hash = editorSnapshotHash(snap);
  if (!hash || hash === state.lastAutosavedHash) return;
  const draftId = Number(snap.draftId || 0);
  const marker = document.getElementById('editor-autosave-status');
  if (!draftId) {
    state.lastAutosavedHash = hash;
    if (marker) marker.textContent = 'Локально сохранено';
    return;
  }
  state.autosaveInFlight = true;
  if (marker) marker.textContent = 'Сохраняю…';
  const savedHash = hash;
  try {
    const body = readDraftForm();
    await api(`/api/drafts/${draftId}`, { method: 'PATCH', body: JSON.stringify(body) });
    state.lastAutosavedHash = savedHash;
    if (marker) marker.textContent = 'Сохранено';
  } catch (e) {
    if (marker) marker.textContent = 'Ошибка сохранения';
    console.error('autosave failed', e);
  } finally {
    state.autosaveInFlight = false;
    const latestSnap = getCurrentEditorSnapshot();
    const latestHash = latestSnap ? editorSnapshotHash(latestSnap) : '';
    if (latestHash && latestHash !== state.lastAutosavedHash) {
      scheduleEditorAutosave(false);
    }
  }
}


function clearAwaitMediaState() {
  try { localStorage.removeItem(AWAIT_MEDIA_KEY); } catch {}
  state.awaitMediaHandled = false;
}

function setAwaitMediaState(payload) {
  try { localStorage.setItem(AWAIT_MEDIA_KEY, JSON.stringify({ ...(payload || {}), ts: Date.now() })); } catch {}
  state.awaitMediaHandled = false;
}

function getAwaitMediaState() {
  try { return safeJsonParse(localStorage.getItem(AWAIT_MEDIA_KEY), null); } catch { return null; }
}

function getCurrentEditorSnapshot() {
  const textEl = document.getElementById('dr-text');
  const promptEl = document.getElementById('dr-prompt');
  const channelEl = document.getElementById('dr-channel');
  const topicEl = document.getElementById('dr-topic');
  const mediaRefEl = document.getElementById('dr-media-ref');
  const mediaTypeEl = document.getElementById('dr-media-type');
  if (!textEl && !promptEl && !mediaRefEl) return null;
  return {
    draftId: Number(document.getElementById('dr-id')?.value || 0) || 0,
    text: textEl?.value || '',
    prompt: promptEl?.value || '',
    channel_target: channelEl?.value || '',
    topic: topicEl?.value || '',
    media_ref: mediaRefEl?.value || '',
    media_type: mediaTypeEl?.value || 'none',
    media_meta_json: document.getElementById('dr-media-meta')?.value || '',
    buttons_json: document.getElementById('dr-buttons')?.value || '[]',
    pin_post: document.getElementById('dr-pin')?.checked ? 1 : 0,
    comments_enabled: document.getElementById('dr-comments')?.checked ? 1 : 0,
    ad_mark: document.getElementById('dr-ad')?.checked ? 1 : 0,
  };
}

function applyEditorSnapshot(snapshot = {}) {
  const map = {
    'dr-text': snapshot.text || '',
    'dr-prompt': snapshot.prompt || '',
    'dr-channel': snapshot.channel_target || '',
    'dr-topic': snapshot.topic || '',
    'dr-media-ref': snapshot.media_ref || '',
    'dr-media-type': snapshot.media_type || 'none',
    'dr-media-meta': snapshot.media_meta_json || '',
    'dr-buttons': snapshot.buttons_json || '[]',
  };
  Object.entries(map).forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.value = value;
  });
  const pin = document.getElementById('dr-pin'); if (pin) pin.checked = Number(snapshot.pin_post || 0) === 1;
  const comments = document.getElementById('dr-comments'); if (comments) comments.checked = Number(snapshot.comments_enabled || 0) !== 0;
  const ad = document.getElementById('dr-ad'); if (ad) ad.checked = Number(snapshot.ad_mark || 0) === 1;
  refreshEditorButtonsList();
  refreshEditorMediaPreview();
  syncEditorPreview();
  initEditorHistory(getCurrentEditorSnapshot());
  state.lastAutosavedHash = editorSnapshotHash(getCurrentEditorSnapshot() || {});
  saveUiState();
}

function startAwaitMediaForEditor() {
  const snapshot = getCurrentEditorSnapshot();
  setAwaitMediaState({
    active: true,
    draftId: Number(snapshot?.draftId || 0) || 0,
    lastKnownMediaId: Number(getMediaInbox()?.[0]?.id || 0) || 0,
    snapshot,
  });
  saveUiState();
}

async function consumeAwaitedMediaIfAny() {
  const wait = getAwaitMediaState();
  if (!wait?.active || state.awaitMediaHandled) return false;
  const pending = state.data?.pending_media || null;
  const latest = state.data?.latest_media || getMediaInbox()?.[0] || null;
  const candidate = pending || latest;
  if (!candidate || !candidate.id) return false;
  const candidateId = Number(candidate.id || 0);
  const lastKnownMediaId = Number(wait.lastKnownMediaId || 0);
  if (!pending && candidateId <= lastKnownMediaId) return false;

  state.awaitMediaHandled = true;
  try {
    showBusy('Прикрепляю видео из чата…');
    if (Number(wait.draftId || 0) > 0) {
      await api(`/api/media/inbox/${candidateId}/attach`, {
        method: 'POST',
        body: JSON.stringify({ draft_id: Number(wait.draftId) })
      });
      await refreshSections(['core','drafts','media_inbox'], { silent: true });
      clearAwaitMediaState();
      openDraftEditor(Number(wait.draftId));
      toast('Видео прикреплено к текущему черновику');
      return true;
    }

    const created = await api(`/api/media/inbox/${candidateId}/draft`, { method: 'POST', body: JSON.stringify({}) });
    const newDraftId = Number(created?.draft_id || created?.draft?.id || 0);
    if (!newDraftId) throw new Error('Не удалось создать черновик из видео');
    const snapshot = wait.snapshot || {};
    const patch = {
      text: snapshot.text || created?.draft?.text || '',
      prompt: snapshot.prompt || '',
      topic: snapshot.topic || '',
      channel_target: snapshot.channel_target || state.data?.settings?.channel_target || '',
      buttons_json: snapshot.buttons_json || '[]',
      pin_post: Number(snapshot.pin_post || 0),
      comments_enabled: Number(snapshot.comments_enabled || 0) ? 1 : 0,
      ad_mark: Number(snapshot.ad_mark || 0),
    };
    await api(`/api/drafts/${newDraftId}`, { method: 'PATCH', body: JSON.stringify(patch) });
    await refreshSections(['core','drafts','media_inbox'], { silent: true });
    clearAwaitMediaState();
    switchTab('posts');
    openDraftEditor(newDraftId);
    toast('Видео автоматически добавлено в новый черновик');
    return true;
  } catch (e) {
    state.awaitMediaHandled = false;
    toast(e.message || 'Не удалось прикрепить видео');
    return false;
  } finally { hideBusy(); }
}

function parseLaunchParams(raw) {
  const out = new URLSearchParams();
  const text = String(raw || '').replace(/^#/, '');
  if (!text) return out;
  for (const [k, v] of new URLSearchParams(text).entries()) out.append(k, v);
  return out;
}

function extractTelegramInitData() {
  const direct = tg?.initData || window.Telegram?.WebApp?.initData || '';
  if (direct) {
    try { sessionStorage.setItem('tg_init_data', direct); } catch {}
    return direct;
  }
  const fromHash = parseLaunchParams(window.location.hash).get('tgWebAppData') || '';
  if (fromHash) {
    try { sessionStorage.setItem('tg_init_data', fromHash); } catch {}
    return fromHash;
  }
  const fromSearch = new URLSearchParams(window.location.search).get('tgWebAppData') || '';
  if (fromSearch) {
    try { sessionStorage.setItem('tg_init_data', fromSearch); } catch {}
    return fromSearch;
  }
  try { return sessionStorage.getItem('tg_init_data') || ''; } catch { return ''; }
}

function escapeHtml(str = '') {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function nl2br(str = '') {
  return escapeHtml(str).replace(/\n/g, '<br>');
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  // Auth transport: Telegram initData or web session cookie
  const initData = extractTelegramInitData();
  if (initData) {
    headers['X-Telegram-Init-Data'] = initData;
    headers['Authorization'] = `tma ${initData}`;
  } else if (_isWebMode && _webHasSession) {
    // Cookie is sent automatically by fetch with credentials: 'same-origin'
  } else {
    throw new Error('Требуется авторизация');
  }

  const timeoutMs = Number(options.timeoutMs || 25000);
  const controller = options.signal ? null : new AbortController();
  const timer = controller ? setTimeout(() => controller.abort('timeout'), timeoutMs) : null;

  let res;
  try {
    res = await fetch(path, { ...options, headers, credentials: 'same-origin', signal: options.signal || controller.signal });
  } catch (e) {
    if (timer) clearTimeout(timer);
    const msg = String(e?.name || e || '');
    if (msg.includes('AbortError') || String(e).includes('timeout')) {
      throw new Error('Сервер отвечает слишком долго. Попробуй ещё раз.');
    }
    throw new Error('Ошибка сети или соединения с Mini App');
  }
  if (timer) clearTimeout(timer);

  const contentType = res.headers.get('content-type') || '';
  let payload = null;
  try {
    payload = contentType.includes('application/json') ? await res.json() : await res.text();
  } catch {
    payload = null;
  }
  if (!res.ok) {
    const detail = (payload && (payload.detail || payload.error || payload.message)) || `Ошибка ${res.status}`;
    if (res.status === 402) {
      const info = typeof detail === 'object' ? detail : { message: String(detail) };
      showPaywallModal(info.message || String(detail), info.code || 'limit_reached');
      const err = new Error(info.message || String(detail));
      err.status = 402;
      err.code = info.code || 'limit_reached';
      throw err;
    }
    if (res.status === 403) {
      const info = typeof detail === 'object' ? detail : { message: String(detail) };
      if (info.code === 'draft_limit_reached') {
        showPaywallModal(info.message || String(detail), 'draft_limit_reached');
        const err = new Error(info.message || String(detail));
        err.status = 403;
        err.code = 'draft_limit_reached';
        throw err;
      }
      if (info.code === 'channel_limit_reached') {
        showPaywallModal(info.message || String(detail), 'channel_limit_reached');
        const err = new Error(info.message || String(detail));
        err.status = 403;
        err.code = 'channel_limit_reached';
        throw err;
      }
    }
    const err = new Error(typeof detail === 'object' ? (detail.message || JSON.stringify(detail)) : detail);
    err.status = res.status;
    throw err;
  }
  return payload;
}

function toast(message) {
  const wrap = document.querySelector('.toast-wrap') || document.body.appendChild(Object.assign(document.createElement('div'), { className: 'toast-wrap' }));
  wrap.innerHTML = '';
  const node = Object.assign(document.createElement('div'), { className: 'toast', textContent: message });
  wrap.appendChild(node);
  setTimeout(() => node.remove(), 2600);
}

function toastError(e) {
  if (e && e.status === 402) return; // paywall modal already shown
  toast((e && e.message) || 'Произошла ошибка');
}

function setInlineStatus(targetId, message = '', tone = 'neutral') {
  const el = document.getElementById(targetId);
  if (!el) return;
  el.className = `inline-status ${tone}`;
  el.textContent = message || '';
}

function showUndoToast(label, onUndo, ttl = 4200) {
  const wrap = document.querySelector('.toast-wrap') || document.body.appendChild(Object.assign(document.createElement('div'), { className: 'toast-wrap' }));
  const node = document.createElement('div');
  node.className = 'toast toast-undo';
  const timer = setTimeout(() => node.remove(), ttl);
  node.innerHTML = `<div>${escapeHtml(label)}</div><button class="btn small ghost">Отменить</button>`;
  node.querySelector('button')?.addEventListener('click', () => { clearTimeout(timer); node.remove(); try { onUndo(); } catch {} });
  wrap.appendChild(node);
}

function confirmAction(message) {
  return window.confirm(message);
}

function confirmActionModal(message, confirmText = 'Удалить') {
  return new Promise((resolve) => {
    const id = `confirm-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    window.__confirmResolvers = window.__confirmResolvers || {};
    window.__confirmResolvers[id] = (result) => {
      try { resolve(Boolean(result)); } finally {
        delete window.__confirmResolvers[id];
        const root = document.getElementById('confirm-root');
        if (root) { root.classList.remove('open'); root.innerHTML = ''; }
      }
    };
    let root = document.getElementById('confirm-root');
    if (!root) {
      root = document.createElement('div');
      root.id = 'confirm-root';
      root.className = 'modal-backdrop confirm-backdrop';
      document.body.appendChild(root);
    }
    root.innerHTML = `
      <div class="modal confirm-modal">
        <div class="modal-head">
          <div class="section-title">Подтверждение</div>
        </div>
        <div class="stack modal-body-stack">
          <div class="confirm-copy">
            <div class="confirm-icon">✦</div>
            <div class="confirm-text">
              <div class="confirm-title">Подтверди действие</div>
              <div class="note">${escapeHtml(message)}</div>
            </div>
          </div>
        </div>
        <div class="item-actions sticky-actions">
          <button class="btn danger" data-confirm-resolve="${id}" data-confirm-value="true">${escapeHtml(confirmText)}</button>
          <button class="btn ghost" data-confirm-resolve="${id}" data-confirm-value="false">Отмена</button>
        </div>
      </div>
    `;
    root.classList.add('open');
    root.onclick = (e) => { if (e.target === root) window.__confirmResolvers[id]?.(false); };
  });
}

async function animateCardRemoval(card) {
  if (!card) return;
  const h = Math.max(card.offsetHeight || 0, 72);
  card.style.maxHeight = `${h}px`;
  card.style.overflow = 'hidden';
  card.style.willChange = 'opacity, transform, max-height';
  await new Promise(resolve => requestAnimationFrame(resolve));
  card.classList.add('draft-removing');
  await new Promise(resolve => setTimeout(resolve, 180));
  card.classList.add('card-collapsing');
  card.style.maxHeight = '0px';
  await new Promise(resolve => setTimeout(resolve, 360));
}

function showBusy(message = 'Загрузка…') {
  document.body.classList.add('busy');
  let root = document.getElementById('busy-overlay');
  if (!root) {
    root = document.createElement('div');
    root.id = 'busy-overlay';
    root.className = 'busy-overlay';
    document.body.appendChild(root);
  }
  root.innerHTML = `
    <div class="busy-overlay-card">
      <div class="spinner"></div>
      <div class="busy-overlay-text">${escapeHtml(message)}</div>
    </div>
  `;
  root.classList.add('open');
}

function hideBusy() {
  document.body.classList.remove('busy');
  document.getElementById('busy-overlay')?.classList.remove('open');
}

function closeModal() {
  if (typeof state.modalCleanup === 'function') {
    try { state.modalCleanup(); } catch {}
  }
  state.modalCleanup = null;
  state.editorOpen = false;
  document.body.classList.remove('modal-open');
  document.getElementById('modal-root')?.classList.remove('open');
  saveUiState();
}

function modal(title, bodyHtml, actionsHtml = '') {
  let root = document.getElementById('modal-root');
  if (!root) {
    root = document.createElement('div');
    root.id = 'modal-root';
    root.className = 'modal-backdrop';
    document.body.appendChild(root);
  }
  root.innerHTML = `
    <div class="modal">
      <div class="modal-head">
        <div>
          <div class="section-title">${escapeHtml(title)}</div>
        </div>
        <button class="btn small ghost" id="modal-close">Закрыть</button>
      </div>
      <div class="stack modal-body-stack">${bodyHtml}</div>
      ${actionsHtml ? `<div id="editor-inline-status" class="inline-status"></div><div class="item-actions sticky-actions">${actionsHtml}</div>` : ''}
    </div>
  `;
  root.classList.add('open');
  document.body.classList.add('modal-open');
  syncViewportHeight();
  document.getElementById('modal-close').onclick = closeModal;
  root.onclick = (e) => { if (e.target === root) closeModal(); };
}

function showPaywallModal(message, code = 'limit_reached') {
  let title, icon;
  if (code === 'upgrade_required') {
    title = 'Функция доступна в PRO';
    icon = '💎';
  } else if (code === 'draft_limit_reached') {
    title = 'Лимит черновиков';
    icon = '📋';
  } else if (code === 'channel_limit_reached') {
    title = 'Лимит каналов';
    icon = '📡';
  } else if (code === 'spy_locked') {
    title = '🕵️‍♂️ Шпион конкурентов — Max';
    icon = '🔐';
  } else if (code === 'sniper_locked') {
    title = '⚡ News Sniper — Pro+';
    icon = '🔐';
  } else {
    title = 'Лимит исчерпан';
    icon = '🔒';
  }
  const bodyHtml = `
    <div class="paywall-modal-body">
      <div class="paywall-icon">${icon}</div>
      <div class="paywall-title">${escapeHtml(title)}</div>
      <div class="paywall-message">${escapeHtml(message || '')}</div>
    </div>
  `;
  const actions = code === 'draft_limit_reached'
    ? `<button class="btn primary" data-action="switchTab" data-action-arg="posts" data-dismiss-modal="true">Открыть черновики</button><button class="btn ghost" data-action="showTariffsModal">Тарифы</button>`
    : `<button class="btn primary" data-action="showTariffsModal">Узнать тарифы</button><button class="btn ghost" data-action="closeModal">Закрыть</button>`;
  modal(title, bodyHtml, actions);
}

function showTariffsModal() {
  closeModal();
  const sub = state.data?.subscription || {};
  const tier = sub.subscription_tier || 'free';
  const tariffs = state.data?.tariffs || {};
  const limits = state.data?.limits || {};
  const fLimits = limits.feature_limits_free || tariffs.feature_limits_free || {};
  const fUsage = limits.feature_usage || {};
  const genLimit = Number(fLimits.generate ?? limits.generations_limit_free ?? 5);
  const genUsed = Number(fUsage.generate ?? sub.generations_used ?? 0);
  const proRub = Number(tariffs.pro_rub ?? 490);
  const maxRub = Number(tariffs.max_rub ?? 990);
  const draftLimits = tariffs.draft_limits || { free: 5, pro: 15, max: 50 };
  const channelLimits = tariffs.channel_limits || { free: 1, pro: 3, max: 10 };
  const _u = (feat) => Number(fUsage[feat] ?? 0);
  const _l = (feat, fb) => Number(fLimits[feat] ?? fb);
  const bodyHtml = `
    <div class="tariffs-list">
      <div class="tariff-card ${tier === 'free' ? 'tariff-active' : ''}">
        <div class="tariff-head"><span class="tariff-badge free">Free</span><span class="tariff-price">Бесплатно</span></div>
        <ul class="tariff-features">
          <li>✍️ ${genLimit} генераций/мес (${genUsed}/${genLimit})</li>
          <li>🔄 ${_l('rewrite',5)} рерайтов/мес (${_u('rewrite')}/${_l('rewrite',5)})</li>
          <li>#️⃣ ${_l('hashtags',5)} хэштегов/мес (${_u('hashtags')}/${_l('hashtags',5)})</li>
          <li>🖼 ${_l('assets',3)} ассетов/мес (${_u('assets')}/${_l('assets',3)})</li>
          <li>📋 ${_l('plan_generate',2)} генер. плана/мес (${_u('plan_generate')}/${_l('plan_generate',2)})</li>
          <li>🤖 ${_l('assistant',3)} запросов ассистенту/мес (${_u('assistant')}/${_l('assistant',3)})</li>
          <li>📊 ${_l('ai_insights',1)} AI-аналитика/мес (${_u('ai_insights')}/${_l('ai_insights',1)})</li>
          <li>🎙️ ${_l('voice',1)} Voice-to-Post/мес (${_u('voice')}/${_l('voice',1)})</li>
          <li>📰 ${_l('news_generate',1)} генер. новости/мес (${_u('news_generate')}/${_l('news_generate',1)})</li>
          <li>До ${draftLimits.free ?? 5} черновиков</li>
          <li>${channelLimits.free ?? 1} канал</li>
          <li>Ручная публикация</li>
          <li class="disabled">Автопостинг по расписанию</li>
          <li class="disabled">News Sniper (авто)</li>
        </ul>
      </div>
      <div class="tariff-card ${tier === 'pro' ? 'tariff-active' : ''}">
        <div class="tariff-head"><span class="tariff-badge pro">Pro</span><span class="tariff-price">${proRub} ₽/мес</span></div>
        <ul class="tariff-features">
          <li>♾️ Безлимитные генерации</li>
          <li>♾️ Рерайт, хэштеги, ассеты, план</li>
          <li>♾️ Ассистент и AI-аналитика</li>
          <li>♾️ Voice-to-Post без ограничений</li>
          <li>♾️ Генерация новостей без ограничений</li>
          <li>До ${draftLimits.pro ?? 15} черновиков</li>
          <li>До ${channelLimits.pro ?? 3} каналов</li>
          <li>Автопостинг по расписанию</li>
          <li>News Sniper (авто)</li>
          <li class="disabled">Шпион конкурентов</li>
        </ul>
        ${tier !== 'pro' && tier !== 'max' ? `<button class="btn primary tariff-buy-btn" data-action="buyTariff" data-action-arg="pro">Купить за ${proRub} ₽</button>` : ''}
      </div>
      <div class="tariff-card ${tier === 'max' ? 'tariff-active' : ''}">
        <div class="tariff-head"><span class="tariff-badge max">Max / Agency</span><span class="tariff-price">${maxRub} ₽/мес</span></div>
        <ul class="tariff-features">
          <li>♾️ Безлимитные генерации</li>
          <li>♾️ Рерайт, хэштеги, ассеты, план</li>
          <li>♾️ Ассистент и AI-аналитика</li>
          <li>♾️ Voice-to-Post без ограничений</li>
          <li>⚡ News Sniper (авто)</li>
          <li>До ${draftLimits.max ?? 50} черновиков</li>
          <li>До ${channelLimits.max ?? 10} каналов</li>
          <li>Автопостинг по расписанию</li>
          <li>🕵️‍♂️ Шпион конкурентов</li>
          <li>Мультиканальность</li>
          <li>Приоритетная поддержка</li>
        </ul>
        ${tier !== 'max' ? `<button class="btn primary tariff-buy-btn" data-action="buyTariff" data-action-arg="max">Купить за ${maxRub} ₽</button>` : ''}
      </div>
    </div>
  `;
  modal('💎 Тарифы', bodyHtml, `<button class="btn ghost" data-action="closeModal">Закрыть</button>`);
}

async function buyTariff(tier) {
  try {
    showBusy('Создаём платёж…');
    const result = await api('/api/payments/create', {
      method: 'POST',
      body: JSON.stringify({ tier }),
    });
    hideBusy();
    const url = result?.confirmation_url || result?.payment_url || '';
    if (url) {
      if (tg?.openLink) {
        tg.openLink(url);
      } else {
        window.open(url, '_blank', 'noopener');
      }
    } else {
      toast('Не удалось получить ссылку на оплату');
    }
  } catch (e) {
    hideBusy();
    toast(e?.message || 'Ошибка при создании платежа');
  }
}

function formatDateTime(dt) {
  const raw = String(dt || '');
  if (!raw) return '—';
  const normalized = raw.replace(' ', 'T');
  const d = new Date(normalized);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
  });
}

function buildAuthenticatedMediaUrl(url = '') {
  const raw = String(url || '').trim();
  if (!raw) return '';
  const initData = extractTelegramInitData();
  if (!initData) return raw;
  const separator = raw.includes('?') ? '&' : '?';
  return `${raw}${separator}tgWebAppData=${encodeURIComponent(initData)}`;
}

function normalizeMediaRef(ref = '') {
  const raw = String(ref || '').trim();
  if (!raw) return '';
  if (raw.startsWith('tgfile:')) {
    const parts = raw.split(':');
    const kind = parts[1] || 'photo';
    const fileId = parts.slice(2).join(':');
    return buildAuthenticatedMediaUrl(`/api/media/telegram?kind=${encodeURIComponent(kind)}&file_id=${encodeURIComponent(fileId)}`);
  }
  if (raw.startsWith('/uploads/')) return buildAuthenticatedMediaUrl(raw);
  if (raw.includes('/uploads/')) return buildAuthenticatedMediaUrl('/uploads/' + raw.split('/uploads/').pop());
  if (raw.includes('/generated_images/')) return buildAuthenticatedMediaUrl('/generated-images/' + raw.split('/generated_images/').pop());
  if (raw.includes('/generated-images/')) return buildAuthenticatedMediaUrl('/generated-images/' + raw.split('/generated-images/').pop());
  if (raw.startsWith('/api/media/telegram')) return buildAuthenticatedMediaUrl(raw);
  return raw;
}

function guessMediaType(ref = '') {
  const value = String(ref || '').toLowerCase();
  if (!value) return 'none';
  if (value.startsWith('tgfile:video:')) return 'video';
  if (value.startsWith('tgfile:photo:')) return 'photo';
  if (/\.(mp4|mov|webm|mkv)$/i.test(value)) return 'video';
  if (/\.(jpg|jpeg|png|gif|webp)$/i.test(value)) return 'photo';
  if (value.startsWith('/uploads/') || value.startsWith('/generated-images/') || value.startsWith('/api/media/telegram')) return 'photo';
  return 'photo';
}

function renderMediaNode(ref = '', explicitType = '') {
  const mediaRef = normalizeMediaRef(ref);
  const type = explicitType || guessMediaType(mediaRef);
  if (!mediaRef || type === 'none') return '<div class="editor-media-empty">Медиа не выбрано</div>';
  if (type === 'video') return `<video controls playsinline preload="metadata" src="${escapeHtml(mediaRef)}"></video>`;
  return `<img src="${escapeHtml(mediaRef)}" alt="preview" loading="lazy" onerror="handleMediaLoadError(this)">`;
}

function handleMediaLoadError(img) {
  if (img.dataset.retried) {
    img.closest('.editor-media-preview,.preview-media')?.classList.add('is-broken');
    img.outerHTML = '<div class="editor-media-empty">Не удалось загрузить медиа</div>';
  } else {
    img.dataset.retried = '1';
    // On first retry, rebuild the auth URL in case init data was not available
    // at the time of original rendering. base is always the URL without query params.
    const base = img.src.split('?')[0];
    const hadAuth = img.src.includes('tgWebAppData=');
    const initData = !hadAuth ? extractTelegramInitData() : null;
    if (initData && base.includes('/api/')) {
      img.src = `${base}?tgWebAppData=${encodeURIComponent(initData)}&t=${Date.now()}`;
    } else {
      img.src = base + '?t=' + Date.now();
    }
  }
}

function renderButtonsPreview(buttonsJson = '[]') {
  try {
    const rows = JSON.parse(buttonsJson || '[]');
    const valid = Array.isArray(rows) ? rows.filter(x => x && x.text && x.url) : [];
    if (!valid.length) return '';
    return `<div class="preview-buttons">${valid.map(btn => `<a class="preview-btn" href="${escapeHtml(btn.url)}" target="_blank" rel="noreferrer">${escapeHtml(btn.text)}</a>`).join('')}</div>`;
  } catch {
    return '';
  }
}

function renderLivePreviewCard(draft) {
  const mediaRef = normalizeMediaRef(draft?.media_ref || '');
  const mediaType = draft?.media_type && draft?.media_type !== 'none' ? draft.media_type : guessMediaType(mediaRef);
  const mediaMetaJson = draft?.media_meta_json || '';
  const media = mediaRef ? `<div class="preview-media">${renderMediaNode(mediaRef, mediaType)}</div>` : '';
  return `
    <div class="tg-preview-card">
      <div class="tg-preview-header">
        <div class="tg-preview-avatar">N</div>
        <div>
          <div class="tg-preview-title">${escapeHtml(resolveChannelLabel(draft?.channel_target) || activeChannel()?.title || 'Предпросмотр поста')}</div>
          <div class="tg-preview-meta">сейчас</div>
        </div>
      </div>
      ${media}
      <div class="tg-preview-text">${nl2br((draft?.text || draft?.prompt || 'Без текста').trim())}</div>
      ${renderButtonsPreview(draft?.buttons_json || '[]')}
      <div class="chips">
        ${draft?.pin_post ? '<div class="chip">Закрепление</div>' : ''}
        ${draft?.comments_enabled !== 0 ? '<div class="chip">Комментарии</div>' : ''}
        ${draft?.ad_mark ? '<div class="chip">Реклама</div>' : ''}
      </div>
    </div>
  `;
}

function activeChannel() {
  return state.data?.active_channel || null;
}

function visibleDrafts() {
  const hidden = state.pendingDeletedDraftIds || new Set();
  return (state.data?.drafts || []).filter(d => !hidden.has(Number(d?.id || 0)));
}

function visiblePlanItems() {
  const hidden = state.pendingDeletedPlanIds || new Set();
  return (state.data?.plan || []).filter(p => !hidden.has(Number(p?.id || 0)));
}

function effectiveDraftsCurrent() {
  const serverCount = Number(state.data?.drafts_current ?? state.data?.limits?.drafts_current ?? 0);
  const pending = Number(state.pendingDeletedDraftIds?.size || 0);
  const visible = visibleDrafts().length;
  if (serverCount > 0 || pending > 0) return Math.max(0, serverCount - pending);
  return visible;
}

function buildRhythmSparkline(points = []) {
  const prepared = (points || []).map((item, index) => ({
    key: String(item?.key || index),
    label: String(item?.label || ''),
    hint: String(item?.hint || ''),
    value: Math.max(0, Number(item?.value || 0)),
    index,
  }));
  if (!prepared.length) return '';

  const max = Math.max(1, ...prepared.map(item => item.value));
  const width = 320;
  const height = 168;
  const padLeft = 20;
  const padRight = 20;
  const padTop = 34;
  const padBottom = 42;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;
  const step = prepared.length > 1 ? innerW / (prepared.length - 1) : innerW / 2;

  const coords = prepared.map((item, index) => {
    const x = padLeft + (prepared.length > 1 ? step * index : innerW / 2);
    const y = padTop + innerH - ((item.value / max) * innerH);
    return {
      ...item,
      x: Number(x.toFixed(2)),
      y: Number(y.toFixed(2)),
    };
  });

  const linePoints = coords.map(point => `${point.x},${point.y}`).join(' ');
  const areaPath = coords.length
    ? `M ${coords[0].x} ${padTop + innerH} L ${coords.map(point => `${point.x} ${point.y}`).join(' L ')} L ${coords[coords.length - 1].x} ${padTop + innerH} Z`
    : '';
  const guideLines = [0.25, 0.5, 0.75].map(fr => {
    const y = padTop + innerH - innerH * fr;
    return `<line class="sparkline-grid-line" x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}"></line>`;
  }).join('');

  const dots = coords.map(point => `
    <g class="sparkline-point-group">
      <circle class="sparkline-point-glow" cx="${point.x}" cy="${point.y}" r="9"></circle>
      <circle class="sparkline-point" cx="${point.x}" cy="${point.y}" r="5.5"></circle>
      <text class="sparkline-value-label" x="${point.x}" y="${Math.max(18, point.y - 14)}" text-anchor="middle">${point.value}</text>
      <text class="sparkline-axis-label" x="${point.x}" y="${height - 12}" text-anchor="middle">${escapeHtml(point.label)}</text>
    </g>
  `).join('');

  const legend = prepared.map(item => `
    <div class="sparkline-legend-item">
      <span class="sparkline-legend-dot"></span>
      <div class="sparkline-legend-copy">
        <b>${escapeHtml(item.label)}</b>
        <span>${escapeHtml(item.hint || '')}</span>
      </div>
    </div>
  `).join('');

  return `
    <div class="sparkline-card">
      <div class="sparkline-head">
        <div class="sparkline-title">Активность</div>
        <div class="sparkline-sub">Цифры у точек — реальные значения</div>
      </div>
      <div class="sparkline-shell">
        <svg class="sparkline-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
          <defs>
            <linearGradient id="sparklineAreaGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#7fd7ff" stop-opacity="0.24"></stop>
              <stop offset="100%" stop-color="#7fd7ff" stop-opacity="0"></stop>
            </linearGradient>
            <linearGradient id="sparklineStrokeGradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#7fd7ff"></stop>
              <stop offset="100%" stop-color="#73dbc8"></stop>
            </linearGradient>
          </defs>
          ${guideLines}
          <path class="sparkline-area" d="${areaPath}"></path>
          <polyline class="sparkline-line" points="${linePoints}"></polyline>
          ${dots}
        </svg>
      </div>
      <div class="sparkline-legend">${legend}</div>
    </div>`;
}

function renderScoreRing(value = 0, sizeClass = 'score-ring-sm') {
  const score = Math.max(0, Math.min(100, Math.round(Number(value || 0))));
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = Number((circumference * (1 - score / 100)).toFixed(3));
  return `
    <div class="score-ring ${sizeClass}" aria-label="Готовность ${score}%" data-score="${score}" style="--ring-offset:${offset};--circumference:${circumference.toFixed(3)};">
      <svg viewBox="0 0 120 120" role="presentation" aria-hidden="true">
        <defs>
          <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="var(--accent, #00d4aa)"/>
            <stop offset="100%" stop-color="var(--accent-2, #4d7cff)"/>
          </linearGradient>
        </defs>
        <circle class="score-ring-track" cx="60" cy="60" r="${radius}"></circle>
        <circle class="score-ring-value" cx="60" cy="60" r="${radius}" stroke-dasharray="${circumference.toFixed(3)}"></circle>
      </svg>
      <div class="score-ring-label">${score}%</div>
    </div>`;
}

function buildAnalyticsChartRows(signals = []) {
  return (signals || []).map(sig => `
    <div class="analytics-chart-row">
      <div class="analytics-chart-label">${escapeHtml(sig.label || '')}</div>
      <div class="analytics-chart-track"><span style="width:${Math.max(0, Math.min(100, Number(sig.value || 0)))}%"></span></div>
      <div class="analytics-chart-value">${Math.max(0, Math.min(100, Number(sig.value || 0)))}%</div>
    </div>`).join('');
}

function buildActivityMiniBars(items = []) {
  const prepared = (items || []).map(x => ({ ...x, value: Number(x?.value || 0) }));
  const max = Math.max(1, ...prepared.map(x => x.value));
  return prepared.map(item => {
    const height = Math.max(18, Math.round((item.value / max) * 72));
    return `
      <div class="mini-bar-item">
        <div class="mini-bar-column"><span style="height:${height}px"></span></div>
        <div class="mini-bar-value">${item.value}</div>
        <div class="mini-bar-label">${escapeHtml(item.label || '')}</div>
      </div>`;
  }).join('');
}


function ensureStateData() {
  if (!state.data) state.data = {};
  if (!state.data.settings) state.data.settings = {};
}

function applyBootstrapCore(core = {}) {
  ensureStateData();
  if (core.user) state.user = core.user;
  if (core.telegram_user_id) state.data.telegram_user_id = core.telegram_user_id;
  if (core.active_channel !== undefined) state.data.active_channel = core.active_channel;
  if (core.stats !== undefined) state.data.stats = core.stats;
  if (core.analytics !== undefined) state.data.analytics = core.analytics;
  if (core.drafts_current !== undefined) state.data.drafts_current = core.drafts_current;
  if (core.settings) state.data.settings = { ...(state.data.settings || {}), ...(core.settings || {}) };
  updateOnboardingFromSettings();
  if (core.pending_media !== undefined) state.data.pending_media = core.pending_media;
  if (core.latest_media !== undefined) state.data.latest_media = core.latest_media;
  if (core.has_pending_media !== undefined) state.data.has_pending_media = !!core.has_pending_media;
  if (core.bot_username !== undefined) state.data.bot_username = core.bot_username;
  if (core.limits !== undefined) state.data.limits = core.limits;
  if (core.subscription !== undefined) state.data.subscription = core.subscription;
  if (core.tariffs !== undefined) state.data.tariffs = core.tariffs;
}

async function refreshSections(sections = [], { silent = true } = {}) {
  const unique = [...new Set((sections || []).filter(Boolean))];
  if (!unique.length) return;
  ensureStateData();
  const tasks = unique.map(async (section) => {
    if (state.refreshPromises.has(section)) {
      await state.refreshPromises.get(section);
      return;
    }
    const promise = (async () => {
      if (section === 'core') {
        const core = await api('/api/bootstrap/core');
        applyBootstrapCore(core);
        return;
      }
      if (section === 'channels') {
        const payload = await api('/api/channels');
        state.data.channels = payload.channels || [];
        if (payload.active_channel !== undefined) state.data.active_channel = payload.active_channel;
        return;
      }
      if (section === 'drafts') {
        const payload = await api('/api/drafts');
        state.data.drafts = Array.isArray(payload) ? payload : (payload.drafts || []);
        if (!Array.isArray(payload) && payload.drafts_current !== undefined) state.data.drafts_current = payload.drafts_current;
        else state.data.drafts_current = state.data.drafts.length;
        return;
      }
      if (section === 'plan') {
        const payload = await api('/api/plan');
        state.data.plan = Array.isArray(payload) ? payload : (payload.plan || []);
        return;
      }
      if (section === 'schedules') {
        const payload = await api('/api/schedules');
        state.data.schedules = Array.isArray(payload) ? payload : (payload.schedules || []);
        return;
      }
      if (section === 'media_inbox') {
        const payload = await api('/api/media/inbox');
        state.data.media_inbox = Array.isArray(payload) ? payload : (payload.media_inbox || []);
        return;
      }
      if (section === 'settings') {
        const payload = await api('/api/settings');
        state.data.settings = { ...(state.data.settings || {}), ...(payload.settings || payload || {}) };
        return;
      }
      // 'stats' is served by /api/bootstrap/core — alias to a lightweight stats refresh
      if (section === 'stats') {
        const core = await api('/api/bootstrap/core');
        if (core.stats !== undefined) state.data.stats = core.stats;
        if (core.active_channel !== undefined) state.data.active_channel = core.active_channel;
        return;
      }
    })();
    state.refreshPromises.set(section, promise);
    try {
      await promise;
    } finally {
      state.refreshPromises.delete(section);
    }
  });
  if (!silent) {
    state.loading = true;
    render();
  }
  try {
    await Promise.all(tasks);
  } finally {
    if (!silent) {
      state.loading = false;
      render();
    }
  }
}

// ---------------------------------------------------------------------------
// Web mode: landing page, login, demo
// ---------------------------------------------------------------------------

function _clearWidgetCheck() {
  if (_widgetCheckInterval) { clearInterval(_widgetCheckInterval); _widgetCheckInterval = null; }
}

function _showWebLanding() {
  _clearWidgetCheck();
  const root = document.getElementById('app');
  if (!root) return;
  _shellRendered = false;
  root.innerHTML = _renderLandingPage();
  _webFetchAuthConfig();
}

function _renderLandingPage() {
  return `
    <div class="web-landing">
      <div class="web-landing-inner">
        <nav class="web-nav">
          <div class="web-nav-brand">N<span>SMM</span></div>
          <button class="btn primary web-nav-cta" data-action="_webStartLogin">Войти</button>
        </nav>

        <section class="web-hero web-hero-split">
          <div class="web-hero-glow"></div>
          <div class="web-hero-glow-2"></div>
          <div class="web-hero-left">
            <div class="web-hero-badge">✦ AI-powered · Telegram</div>
            <h1 class="web-hero-title">Neuro<span class="accent">SMM</span></h1>
            <p class="web-hero-subtitle">Полностью автономный ИИ-менеджер для вашего Telegram-канала. Контент, публикации, аналитика — всё на автопилоте.</p>
            <div class="web-hero-cta web-hero-cta-row">
              <button class="btn primary web-cta-main" data-action="_webStartLogin">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.2 4.4L2.4 10.8c-.6.2-.6 1.1 0 1.3l4.5 1.5 1.7 5.3c.1.4.6.6 1 .4l2.5-1.8 4.3 3.2c.5.3 1.1 0 1.2-.5L21.8 5.4c.2-.6-.3-1.2-.6-1z"/></svg>
                Войти через Telegram
              </button>
              <button class="btn ghost web-cta-secondary" data-action="_webStartDemo">Демо</button>
            </div>
          </div>
          <div class="web-hero-right">
            <div class="web-hero-visual">
              <div class="web-hero-card-stack">
                <div class="web-hero-mini-card web-hero-mini-card-1">
                  <span class="web-hero-mini-icon">✦</span>
                  <div><b>Пост готов</b><span>ИИ создал черновик по теме канала</span></div>
                </div>
                <div class="web-hero-mini-card web-hero-mini-card-2">
                  <span class="web-hero-mini-icon">📊</span>
                  <div><b>+32 подписчика</b><span>Рост за последнюю неделю</span></div>
                </div>
                <div class="web-hero-mini-card web-hero-mini-card-3">
                  <span class="web-hero-mini-icon">⚡</span>
                  <div><b>Новость найдена</b><span>Автопубликация через 2 часа</span></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="web-trust-bar">
          <div class="web-trust-item">
            <span class="web-trust-num">24/7</span>
            <span class="web-trust-label">автопилот</span>
          </div>
          <div class="web-trust-divider"></div>
          <div class="web-trust-item">
            <span class="web-trust-num">ИИ</span>
            <span class="web-trust-label">контент</span>
          </div>
          <div class="web-trust-divider"></div>
          <div class="web-trust-item">
            <span class="web-trust-num">0₽</span>
            <span class="web-trust-label">старт</span>
          </div>
        </section>

        <section class="web-section web-features-section">
          <h2 class="web-section-title">Всё для роста канала</h2>
          <p class="web-section-desc">Один инструмент вместо десяти сервисов</p>
          <div class="web-features-grid">
            <div class="web-feat-card web-feat-card-accent">
              <div class="web-feat-card-icon"><span>✦</span></div>
              <div class="web-feat-card-title">Генерация</div>
              <div class="web-feat-card-desc">ИИ создаёт уникальные посты и подбирает изображения под стиль вашего канала</div>
            </div>
            <div class="web-feat-card">
              <div class="web-feat-card-icon"><span>📅</span></div>
              <div class="web-feat-card-title">Автопостинг</div>
              <div class="web-feat-card-desc">Умное расписание и автоматическая публикация в лучшее время</div>
            </div>
            <div class="web-feat-card">
              <div class="web-feat-card-icon"><span>📊</span></div>
              <div class="web-feat-card-title">Аналитика</div>
              <div class="web-feat-card-desc">Детальная статистика роста, вовлечённости и лучшее время для постов</div>
            </div>
            <div class="web-feat-card">
              <div class="web-feat-card-icon"><span>🔍</span></div>
              <div class="web-feat-card-title">Мониторинг</div>
              <div class="web-feat-card-desc">Отслеживание новостей и контента конкурентов в реальном времени</div>
            </div>
          </div>
        </section>

        <section class="web-section web-how-section">
          <h2 class="web-section-title">Как это работает</h2>
          <div class="web-how-steps">
            <div class="web-how-step">
              <div class="web-how-num">1</div>
              <div class="web-how-step-title">Подключи канал</div>
              <div class="web-how-step-desc">Добавь бота в свой Telegram-канал</div>
            </div>
            <div class="web-how-connector"></div>
            <div class="web-how-step">
              <div class="web-how-num">2</div>
              <div class="web-how-step-title">Настрой стиль</div>
              <div class="web-how-step-desc">Выбери тон, темы и расписание публикаций</div>
            </div>
            <div class="web-how-connector"></div>
            <div class="web-how-step">
              <div class="web-how-num">3</div>
              <div class="web-how-step-title">ИИ ведёт канал</div>
              <div class="web-how-step-desc">Нейросеть создаёт и публикует контент автоматически</div>
            </div>
          </div>
        </section>

        <section class="web-bottom-cta">
          <div class="web-bottom-cta-card">
            <h3 class="web-bottom-cta-title">Готовы автоматизировать канал?</h3>
            <p class="web-bottom-cta-desc">Начните бесплатно — без карты и обязательств</p>
            <div class="web-bottom-cta-actions">
              <button class="btn primary web-cta-main" data-action="_webStartLogin">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.2 4.4L2.4 10.8c-.6.2-.6 1.1 0 1.3l4.5 1.5 1.7 5.3c.1.4.6.6 1 .4l2.5-1.8 4.3 3.2c.5.3 1.1 0 1.2-.5L21.8 5.4c.2-.6-.3-1.2-.6-1z"/></svg>
                Войти через Telegram
              </button>
              <a class="btn outline web-cta-tg-link" id="web-open-tg-link" href="#" target="_blank" rel="noopener">
                Открыть в Telegram
              </a>
            </div>
          </div>
        </section>

        <footer class="web-footer">NeuroSMM &copy; ${new Date().getFullYear()}</footer>
      </div>
    </div>
  `;
}

async function _webFetchAuthConfig() {
  if (_webAuthConfig) return _webAuthConfig;
  try {
    const res = await fetch('/api/web-auth/config');
    if (res.ok) {
      _webAuthConfig = await res.json();
      const link = document.getElementById('web-open-tg-link');
      if (link && _webAuthConfig.bot_username) {
        link.href = 'https://t.me/' + encodeURIComponent(_webAuthConfig.bot_username);
      }
    }
  } catch { /* ignore */ }
  return _webAuthConfig || { enabled: false, bot_username: '', bot_id: '' };
}

async function _webStartLogin() {
  const cfg = await _webFetchAuthConfig();
  if (!cfg.enabled || !cfg.bot_id) {
    toast('Веб-авторизация не настроена. Используйте Telegram Mini App.');
    return;
  }
  _showWebLoginPage(cfg);
}

function _showWebLoginPage(cfg) {
  const root = document.getElementById('app');
  if (!root) return;
  _shellRendered = false;

  root.innerHTML = `
    <div class="web-login-page">
      <div class="web-login-inner">
        <button class="btn ghost web-back-btn" data-action="_showWebLanding">← Назад</button>
        <div class="web-login-brand">
          <div class="web-login-logo">N<span>SMM</span></div>
        </div>
        <h2 class="web-login-title">Вход в NeuroSMM</h2>
        <p class="web-login-desc">Авторизуйтесь через Telegram, чтобы управлять каналами.</p>
        <div id="web-tg-login-wrap" class="web-tg-login-wrap">
          <div class="web-widget-loading" id="web-widget-loading">
            <div class="web-widget-spinner"></div>
            <span>Загрузка виджета…</span>
          </div>
        </div>
        <p class="web-login-hint" id="web-login-hint" style="display:none">Нажмите кнопку для авторизации через Telegram.</p>
        <div id="web-widget-fallback" class="web-widget-fallback" style="display:none">
          <p class="web-fallback-text">Виджет Telegram не загрузился.</p>
          <a id="web-fallback-link" class="btn primary web-cta-main" href="#" target="_blank" rel="noopener">Войти через Telegram напрямую</a>
          <p class="web-fallback-hint">Или попробуйте открыть эту страницу в обычном браузере</p>
        </div>
        <div class="web-login-footer">
          <span>Безопасный вход через официальный Telegram Login</span>
        </div>
      </div>
    </div>
  `;

  _injectTelegramWidget(cfg);
}

function _injectTelegramWidget(cfg) {
  const wrap = document.getElementById('web-tg-login-wrap');
  if (!wrap) return;

  // ----- Diagnostic collector: logged to console on success/failure -----
  const diag = [];
  const logDiag = (level) => console[level]('[TG Widget]', diag.join(' | '));
  diag.push('bot_username=' + (cfg.bot_username || '(empty)'));
  diag.push('bot_id=' + (cfg.bot_id || '(empty)'));
  diag.push('protocol=' + location.protocol);
  diag.push('host=' + location.host);

  // Sanitise bot_username: strip leading '@' if present
  const botUser = String(cfg.bot_username || '').replace(/^@/, '');
  if (!botUser) {
    diag.push('ERROR: bot_username is empty — widget cannot render');
    logDiag('warn');
    _showWidgetFallback(cfg, 'Бот не настроен для веб-авторизации (bot_username пуст).');
    return;
  }

  if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
    diag.push('WARNING: not HTTPS — widget may not work');
  }

  // Listen for CSP violations that might block the widget
  const cspViolations = [];
  const cspHandler = (e) => {
    const info = 'CSP-block: ' + (e.blockedURI || '?') + ' directive=' + (e.violatedDirective || '?');
    cspViolations.push(info);
    diag.push(info);
  };
  document.addEventListener('securitypolicyviolation', cspHandler);

  // Build the widget <script> element.
  // Use data-auth-url (redirect mode) instead of data-onauth (JS callback mode).
  // Redirect mode does NOT require 'unsafe-eval' in CSP — the widget doesn't
  // need to call eval()/new Function() to invoke a callback string.
  const callbackUrl = location.origin + '/api/web-auth/telegram-callback';
  const script = document.createElement('script');
  script.async = true;
  script.src = 'https://telegram.org/js/telegram-widget.js?22';
  script.setAttribute('data-telegram-login', botUser);
  script.setAttribute('data-size', 'large');
  script.setAttribute('data-radius', '14');
  script.setAttribute('data-auth-url', callbackUrl);
  script.setAttribute('data-request-access', 'write');

  script.onerror = () => {
    diag.push('ERROR: telegram-widget.js failed to load (network or CSP block)');
    if (cspViolations.length) diag.push('CSP violations: ' + cspViolations.join('; '));
    logDiag('error');
    _clearWidgetCheck();
    document.removeEventListener('securitypolicyviolation', cspHandler);
    _showWidgetFallback(cfg, 'Скрипт Telegram виджета не загрузился. Проверьте подключение к интернету.');
  };

  script.onload = () => {
    diag.push('script loaded OK');
    // Check if Telegram SDK objects are present
    diag.push('Telegram.Login=' + (typeof window.Telegram?.Login?.auth));
  };

  wrap.appendChild(script);

  // Poll for the iframe created by telegram-widget.js
  _clearWidgetCheck();
  let attempts = 0;
  _widgetCheckInterval = setInterval(() => {
    attempts++;
    const iframe = wrap.querySelector('iframe');
    if (iframe) {
      _clearWidgetCheck();
      document.removeEventListener('securitypolicyviolation', cspHandler);
      diag.push('iframe appeared after ' + (attempts * 500) + 'ms');
      logDiag('info');
      const loading = document.getElementById('web-widget-loading');
      if (loading) loading.style.display = 'none';
      const hint = document.getElementById('web-login-hint');
      if (hint) hint.style.display = '';
      return;
    }
    // After 6 seconds, try popup fallback; after 10 seconds, give up
    if (attempts === 12) {
      diag.push('SLOW: iframe not found after 6s — showing popup fallback');
      logDiag('warn');
      _showPopupLoginButton(cfg, wrap);
    }
    if (attempts >= 20) {
      _clearWidgetCheck();
      document.removeEventListener('securitypolicyviolation', cspHandler);
      diag.push('TIMEOUT: iframe never appeared after 10s');
      const scriptInDom = wrap.querySelector('script[data-telegram-login]');
      diag.push('script_in_dom=' + !!scriptInDom);
      if (cspViolations.length) diag.push('CSP violations: ' + cspViolations.join('; '));

      // Classify the failure reason for better diagnostics
      let reason = 'Виджет Telegram не загрузился.';
      let errorType = 'unknown';

      if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
        reason += ' Возможная причина: страница не на HTTPS.';
        errorType = 'no_https';
      }
      if (cspViolations.length) {
        reason += ' Обнаружены блокировки CSP — проверьте консоль браузера.';
        errorType = 'csp_block';
      }
      if (!scriptInDom) {
        reason += ' Скрипт виджета не добавлен в DOM.';
        errorType = 'script_missing';
      } else if (!document.querySelector('iframe[src*="oauth.telegram.org"]')) {
        reason += ' Telegram не создал iframe — проверьте, что домен ' + location.host + ' привязан к боту @' + botUser + ' в BotFather (команда /setdomain).';
        errorType = 'no_iframe';
      }

      diag.push('error_type=' + errorType);

      // Try network check: is telegram.org reachable?
      // The ?22 matches the widget script version used above for cache consistency.
      try {
        fetch('https://telegram.org/js/telegram-widget.js?22', {mode: 'no-cors', cache: 'no-store'})
          .then(() => diag.push('telegram.org fetch=ok'))
          .catch(() => {
            diag.push('telegram.org fetch=FAILED (VPN/firewall may block telegram.org)');
            reason += ' telegram.org может быть заблокирован вашим VPN или сетью.';
          })
          .finally(() => logDiag('warn'));
      } catch (e) {
        diag.push('telegram.org fetch=exception');
      }

      logDiag('warn');
      _showWidgetFallback(cfg, reason);
    }
  }, 500);
}

function _showWidgetFallback(cfg, reason) {
  const loading = document.getElementById('web-widget-loading');
  if (loading) loading.style.display = 'none';
  const fallback = document.getElementById('web-widget-fallback');
  if (fallback) {
    fallback.style.display = '';
    // Update fallback text with specific reason
    const textEl = fallback.querySelector('.web-fallback-text');
    if (textEl && reason) textEl.textContent = reason;
    const link = document.getElementById('web-fallback-link');
    if (link && cfg.bot_username) {
      link.href = 'https://t.me/' + encodeURIComponent(cfg.bot_username.replace(/^@/, ''));
    }
  }
  // Always offer popup fallback alongside the fallback block
  const wrap = document.getElementById('web-tg-login-wrap');
  if (wrap) _showPopupLoginButton(cfg, wrap);
}

/**
 * Show a styled "Log in with Telegram" popup button using the Telegram.Login.auth()
 * API exposed by telegram-widget.js.  This bypasses the iframe completely and opens
 * a Telegram auth popup directly.  Works even when the iframe can't render (e.g.
 * domain mismatch, CSP eval issues, etc.).
 */
function _showPopupLoginButton(cfg, container) {
  if (container.querySelector('.web-popup-login-btn')) return; // already added
  const botId = cfg.bot_id || '';
  if (!botId) return;

  const btn = document.createElement('button');
  btn.className = 'btn primary web-popup-login-btn';
  btn.textContent = '🔑 Войти через Telegram (popup)';
  btn.setAttribute('type', 'button');
  container.appendChild(btn);

  // Hide the spinner if still visible
  const loading = document.getElementById('web-widget-loading');
  if (loading) loading.style.display = 'none';

  btn.addEventListener('click', () => {
    // Method 1: Use Telegram.Login.auth() API (exposed by telegram-widget.js)
    if (typeof window.Telegram?.Login?.auth === 'function') {
      console.info('[TG Widget] Using Telegram.Login.auth() popup');
      window.Telegram.Login.auth(
        { bot_id: botId, request_access: 'write' },
        (data) => {
          if (data) window._onTelegramLoginAuth(data);
        }
      );
      return;
    }
    // Method 2: Open Telegram OAuth URL directly in a popup
    const authUrl = 'https://oauth.telegram.org/auth?bot_id=' +
      encodeURIComponent(botId) +
      '&origin=' + encodeURIComponent(location.origin) +
      '&request_access=write&embed=0&return_to=' + encodeURIComponent(location.href);
    console.info('[TG Widget] Opening direct OAuth popup:', authUrl);
    const popup = window.open(authUrl, 'telegram_auth', 'width=550,height=470,left=100,top=100');
    if (!popup) {
      toast('Разрешите всплывающие окна для авторизации через Telegram');
    }
  });
}

// Global callback for Telegram Login Widget
window._onTelegramLoginAuth = async function(user) {
  if (!user || !user.id) {
    toast('Ошибка авторизации Telegram');
    return;
  }
  try {
    showBusy('Авторизация…');
    const res = await fetch('/api/web-auth/telegram-login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(user),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Ошибка авторизации');
    }
    _webHasSession = true;
    hideBusy();
    loadBootstrap();
  } catch (e) {
    hideBusy();
    toast(e.message || 'Ошибка авторизации');
  }
};

function _webStartDemo() {
  _clearWidgetCheck();
  const root = document.getElementById('app');
  if (!root) return;
  _shellRendered = false;

  root.innerHTML = `
    <div class="web-demo-page">
      <div class="web-demo-inner">
        <button class="btn ghost web-back-btn" data-action="_showWebLanding">← Назад</button>
        <div class="web-demo-header">
          <h2 class="web-demo-title">Превью NeuroSMM</h2>
          <p class="web-demo-desc">Так выглядит управление вашим каналом</p>
        </div>
        <div class="web-demo-preview">
          <div class="web-demo-mockup">
            <div class="web-demo-topbar">
              <span class="web-demo-app-name">NeuroSMM</span>
              <span class="web-demo-badge-pro">Pro</span>
              <span class="web-demo-channel">@my_channel</span>
            </div>
            <div class="web-demo-content">
              <div class="web-demo-card web-demo-card-highlight">
                <div class="web-demo-card-title">📊 Ваш канал</div>
                <div class="web-demo-stats">
                  <div class="web-demo-stat"><strong>1 247</strong><span>подписчиков</span></div>
                  <div class="web-demo-stat"><strong>+32</strong><span>за неделю</span></div>
                  <div class="web-demo-stat"><strong>4.2%</strong><span>ER</span></div>
                </div>
              </div>
              <div class="web-demo-card">
                <div class="web-demo-card-title">✦ Готовый пост</div>
                <div class="web-demo-card-text">ИИ сгенерировал пост о трендах вашей ниши. Автопубликация через 2 часа.</div>
                <div class="web-demo-card-actions">
                  <span class="web-demo-btn-mock">Редактировать</span>
                  <span class="web-demo-btn-mock accent">Опубликовать</span>
                </div>
              </div>
              <div class="web-demo-card">
                <div class="web-demo-card-title">📰 Новость по теме</div>
                <div class="web-demo-card-text">Найдена свежая публикация — сохранена в черновики для обработки.</div>
              </div>
              <div class="web-demo-card">
                <div class="web-demo-card-title">📅 План на неделю</div>
                <div class="web-demo-plan-slots">
                  <div class="web-demo-plan-slot"><span>Пн</span><span>10:00</span></div>
                  <div class="web-demo-plan-slot"><span>Ср</span><span>12:00</span></div>
                  <div class="web-demo-plan-slot"><span>Пт</span><span>18:00</span></div>
                </div>
              </div>
            </div>
            <div class="web-demo-nav">
              <div class="web-demo-nav-item active">⌂<span>Главная</span></div>
              <div class="web-demo-nav-item">✎<span>Посты</span></div>
              <div class="web-demo-nav-item">▣<span>План</span></div>
              <div class="web-demo-nav-item">◎<span>Каналы</span></div>
              <div class="web-demo-nav-item">⏱<span>Автопост</span></div>
            </div>
          </div>
        </div>
        <div class="web-demo-label">Демо-данные · не сохраняются</div>
        <div class="web-demo-actions">
          <button class="btn primary web-cta-main" data-action="_webStartLogin">Войти для полного доступа</button>
          <button class="btn ghost web-cta-secondary" data-action="_showWebLanding">Вернуться</button>
        </div>
      </div>
    </div>
  `;
}

async function webLogout() {
  try {
    await fetch('/api/web-auth/logout', { method: 'POST', credentials: 'same-origin' });
  } catch { /* best-effort */ }
  _webHasSession = false;
  _isWebMode = true;
  state.data = null;
  state.user = null;
  _shellRendered = false;
  _showWebLanding();
}

function _restoreEditorIfNeeded() {
  const saved = loadUiState();
  if (!state.onboarding.active && saved?.editorOpen && saved?.editor) {
    const restoreId = Number(saved.editor.draftId || 0);
    if (restoreId && (state.data?.drafts || []).some(d => Number(d.id) === restoreId)) {
      openDraftEditor(restoreId);
    } else if (saved.editor && (saved.editor.text || saved.editor.prompt || saved.editor.media_ref)) {
      openDraftEditor(null);
      applyEditorSnapshot(saved.editor);
    }
  }
}

async function loadBootstrap() {
  // Detect web mode: no Telegram initData available
  const hasTgAuth = !!extractTelegramInitData();
  _isWebMode = !hasTgAuth;

  if (_isWebMode) {
    // In web mode, we cannot inspect the HttpOnly cookie from JS.
    // If we don't know we have a session yet, attempt a lightweight probe.
    // If no cookie is present the API call will fail with 401 and we show landing.
    if (!_webHasSession) {
      try {
        const probe = await fetch('/api/bootstrap/core', { credentials: 'same-origin' });
        if (probe.status === 401) {
          _showWebLanding();
          return;
        }
        // Cookie is valid — mark session active and use the response
        _webHasSession = true;
        const core = await probe.json();
        state.loading = true;
        render();
        state.data = {};
        state.user = core.user || null;
        applyBootstrapCore(core);
        const saved = loadUiState();
        if (saved?.activeTab) state.activeTab = saved.activeTab;
        await refreshSections(['channels', 'drafts', 'plan', 'schedules', 'media_inbox'], { silent: true });
        state.loading = false;
        render();
        _restoreEditorIfNeeded();
        await consumeAwaitedMediaIfAny();
        return;
      } catch {
        _showWebLanding();
        return;
      }
    }
  }

  state.loading = true;
  render();
  try {
    const saved = loadUiState();
    if (saved?.activeTab) state.activeTab = saved.activeTab;
    const core = await api('/api/bootstrap/core');
    state.data = {};
    state.user = core.user || null;
    applyBootstrapCore(core);
    await refreshSections(['channels', 'drafts', 'plan', 'schedules', 'media_inbox'], { silent: true });
  } catch (e) {
    if (_isWebMode && (e.status === 401 || e.message === 'Требуется авторизация')) {
      // Cookie expired or invalid — clear session and show landing page
      _webHasSession = false;
      state.loading = false;
      _showWebLanding();
      return;
    }
    if ((e.message || '').includes('лимит черновиков')) { showDraftLimitModal(); } else { toast(e.message); }
    console.error('bootstrap failed', e);
  } finally {
    state.loading = false;
    render();
    _restoreEditorIfNeeded();
    await consumeAwaitedMediaIfAny();
  }
}

function navBtn(id, title, icon) {
  return `<button class="nav-btn ${state.activeTab === id ? 'active' : ''}" data-action="switchTab" data-action-arg="${id}"><div>${icon}</div><div>${title}</div></button>`;
}

function switchTab(id) {
  // Backwards compat: redirect legacy 'more' tab to autopost
  if (id === 'more') id = 'autopost';
  if (state.activeTab === id) return;
  state.activeTab = id;
  saveUiState();

  // Если shell уже в DOM — обновляем только body и nav без полной перерисовки
  const surface = document.querySelector('.main-surface');
  const nav = document.querySelector('.bottom-nav');
  if (surface && _shellRendered) {
    surface.innerHTML = `<div class="tab-enter">${renderBody()}</div>`;
    if (nav) {
      nav.innerHTML = _buildNavButtons();
    }
    return;
  }
  render();
}

function _buildNavButtons() {
  /* Clean inline SVG icons — uniform 22×22 stroke-based set */
  const icons = {
    dashboard: '<svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    posts:     '<svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 113 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>',
    plan:      '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>',
    channels:  '<svg viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    autopost:  '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
  };
  return navBtn('dashboard', 'Главная', icons.dashboard) +
    navBtn('posts', 'Посты', icons.posts) +
    navBtn('plan', 'План', icons.plan) +
    navBtn('channels', 'Каналы', icons.channels) +
    navBtn('autopost', 'Автопост', icons.autopost);
}


function activeChannelTitle() {
  const active = activeChannel();
  if (active?.title) return active.title;
  const raw = active?.channel_target || state.data?.settings?.channel_target || '';
  // Numeric Telegram chat IDs are 6+ digits (e.g. -1001234567890); never show them as labels
  if (/^-?\d{6,}$/.test(String(raw))) return 'Канал без названия';
  return raw || 'Канал не выбран';
}

function resolveChannelLabel(channelTarget = '') {
  const raw = String(channelTarget || '').trim();
  if (!raw) return activeChannelTitle();
  const found = (state.data?.channels || []).find(ch => String(ch.channel_target || '').trim() === raw || String(ch.id) === raw || String(ch.channel_profile_id || '') === raw || String(ch.title || '').trim() === raw);
  if (found?.title) return found.title;
  // Never show any raw numeric ID as a label (Telegram chat IDs and internal DB IDs)
  if (/^-?\d+$/.test(raw)) return activeChannel()?.title || 'Канал без названия';
  return raw;
}

function getMediaInbox() {
  if (Array.isArray(state.data?.media_inbox)) return state.data.media_inbox;
  if (Array.isArray(state.data?.mediaInbox)) return state.data.mediaInbox;
  return [];
}


function currentDraftsLimit() {
  const limits = state.data?.limits || {};
  const drafts = visibleDrafts();
  const current = Number(limits.drafts_current ?? state.data?.drafts_current ?? drafts.length ?? 0);
  const max = Math.max(1, Number(limits.drafts_max ?? 15));
  return { current, max, reached: current >= max };
}

function draftLimitTone(percent) {
  if (percent >= 92) return 'danger';
  if (percent >= 75) return 'warn';
  return 'normal';
}

function showDraftLimitModal() {
  const { current, max } = currentDraftsLimit();
  modal('Лимит черновиков', `
    <div class="note">Достигнут лимит черновиков <b>${current} / ${max}</b>. Удали лишние черновики или опубликуй часть, чтобы освободить место.</div>
  `, `<button class="btn primary" data-action="closeModal">Понятно</button>`);
}

function formatFileSize(bytes) {
  const n = Number(bytes || 0);
  if (!n) return '';
  if (n >= 1024*1024*1024) return `${(n/(1024*1024*1024)).toFixed(1)} ГБ`;
  return `${(n/(1024*1024)).toFixed(1)} МБ`;
}

function mediaInboxCard(item) {
  const title = item.title || 'Видео из чата';
  const meta = [
    item.kind === 'video' ? 'Видео' : 'Файл',
    item.size_mb ? `${item.size_mb} МБ` : '',
    item.duration_sec ? `${item.duration_sec} с` : '',
    formatDateTime(item.created_at || item.ts || '')
  ].filter(Boolean).join(' · ');
  return `
    <div class="media-card clean-media-card compact-media-row media-inbox-card-fixed media-inbox-card-commercial">
      <div class="media-thumb media-thumb-fixed">${renderMediaNode(item.file_url || item.file_path || item.preview_url || '', item.kind || 'video')}</div>
      <div class="media-card-body media-inbox-body-fixed">
        <div class="media-card-title">${escapeHtml(title)}</div>
        <div class="media-card-meta">${escapeHtml(meta || 'Файл из чата')}</div>
        <div class="media-inbox-actions-fixed media-inbox-actions-commercial">
          <button class="btn small primary" data-action="createDraftFromInbox" data-action-arg="${Number(item.id)}">Выбрать</button>
          <button class="btn small ghost" data-action="deleteInboxMedia" data-action-arg="${Number(item.id)}">Удалить</button>
        </div>
      </div>
    </div>
  `;
}

function bigVideoPanel(compact = false) {
  const inbox = getMediaInbox();
  const latest = inbox[0] || state.data?.latest_media || null;
  const count = inbox.length;
  const meta = count ? `${count} ${count === 1 ? 'файл' : 'файла'} из чата` : 'Пока пусто';
  return `
    <div class="card video-flow-card video-flow-card-clean"><div class="card-inner compact-section-card stack">
      <div class="section-title">Видео из чата</div>
      <div class="section-desc">Открой окно со списком файлов из чата, инструкцией и быстрым выбором видео.</div>
      <div class="video-cta-row">
        <button class="btn primary btn-wide-mobile" data-action="openChatPicker">Открыть медиатеку</button>
      </div>
      <div class="video-inline-meta">${escapeHtml(meta)}${latest ? ` · ${escapeHtml((latest.title || 'Видео').slice(0,60))}` : ''}</div>
    </div></div>
  `;
}


function computeFallbackAnalytics(settings, activeChannel, stats, drafts, plan, mediaInbox, schedules) {
  const topic = String(settings.topic || '').trim();
  const audience = String(settings.channel_audience || '').trim();
  const style = String(settings.channel_style || settings.channel_style_preset || '').trim();
  const mode = String(settings.channel_mode || settings.posting_mode || '').trim();
  const frequency = String(settings.channel_frequency || '').trim();
  const formats = safeJsonParse(settings.channel_formats, []);
  const constraints = safeJsonParse(settings.content_constraints, []);
  const newsSources = safeJsonParse(settings.news_sources_json || settings.news_sources, []);
  const onboardingDone = Boolean(Number(settings.onboarding_completed || 0) === 1 || (topic && audience && style && (frequency || mode) && formats.length >= 2));
  const planCount = Array.isArray(plan) ? plan.filter(x => Number(x?.posted || 0) !== 1).length : 0;
  const draftsCount = Array.isArray(drafts) ? drafts.filter(x => String(x?.status || 'draft').toLowerCase() === 'draft').length : 0;
  const mediaCount = Array.isArray(mediaInbox) ? mediaInbox.length : 0;
  const scheduleCount = Array.isArray(schedules) ? schedules.filter(x => Number(x?.enabled ?? 1) !== 0).length : 0;
  const totalPosts = Number(stats.total_posts || 0);
  const postedLast7d = Number(stats.posted_last_7d || 0);
  const refs = [...(drafts || []), ...(mediaInbox || [])].map(x => String(x?.media_ref || x?.file_id || '')).filter(Boolean);
  const dupRate = refs.length > 1 ? Math.round((1 - (new Set(refs).size / refs.length)) * 100) : 0;
  let score = 0;
  score += topic ? 14 : 0;
  score += audience ? 10 : 0;
  score += style ? 10 : 0;
  score += mode ? 6 : 0;
  score += frequency ? 8 : 0;
  score += Math.min(16, formats.length * 4);
  score += Math.min(8, constraints.length * 2);
  score += activeChannel ? 10 : 0;
  score += onboardingDone ? 8 : 0;
  score += Math.min(18, planCount * 3);
  score += Math.min(15, draftsCount * 5);
  score += Math.min(10, mediaCount * 3);
  score += Math.min(10, scheduleCount * 5);
  score += Math.min(8, postedLast7d * 2);
  score += Math.min(6, Math.floor(totalPosts / 4));
  if (mode === 'news' || mode === 'both') score += Math.min(10, newsSources.length * 3);
  score -= Math.min(22, Math.round(dupRate / 2));
  if (!activeChannel) score = Math.min(score, 42);
  if (!onboardingDone) score = Math.min(score, 58);
  if (planCount < 2) score = Math.min(score, 68);
  if (draftsCount < 2) score = Math.min(score, 76);
  if (mediaCount < 1) score = Math.min(score, 82);
  if (scheduleCount < 1 && (mode === 'autopilot' || mode === 'review')) score = Math.min(score, 78);
  if ((mode === 'news' || mode === 'both') && !newsSources.length) score = Math.min(score, 72);
  score = Math.max(0, Math.min(100, Math.round(score)));
  const signals = [
    { key:'profile', label:'Профиль канала', value: Math.max(0, Math.min(100, (topic?25:0)+(audience?20:0)+(style?20:0)+(mode?15:0)+(frequency?10:0)+Math.min(10, formats.length*2))), hint:'Тема, аудитория, стиль, режим и частота', action:'Дозаполни onboarding до конца' },
    { key:'content_reserve', label:'Запас контента', value: Math.max(0, Math.min(100, Math.min(55, planCount*11)+Math.min(45, draftsCount*18))), hint:`${planCount} в плане · ${draftsCount} черновиков`, action:'Создай резерв плана и черновиков' },
    { key:'media', label:'Медиарезерв', value: Math.max(0, Math.min(100, Math.min(85, mediaCount*22) - Math.min(20, dupRate))), hint:`${mediaCount} файлов · повторы ${dupRate}%`, action:'Добавь больше разных медиа' },
    { key:'autopost', label:'Автопостинг', value: Math.max(0, Math.min(100, (activeChannel?25:0)+(frequency?15:0)+Math.min(30, scheduleCount*25)+Math.min(30, postedLast7d*6))), hint:`${scheduleCount} слотов · ${postedLast7d} публикаций за 7 дней`, action:'Настрой ритм публикаций' },
  ];
  const weakest = signals.slice().sort((a,b)=>a.value-b.value)[0];
  return {
    score,
    summary: { onboarding_completed: onboardingDone ? 1 : 0, plan_count: planCount, drafts_count: draftsCount, media_count: mediaCount, schedule_count: scheduleCount },
    signals,
    next_step: weakest?.action || 'Усиль профиль и запас контента',
    weakest_key: weakest?.key || 'profile',
    recommendations: signals.filter(x => x.value < 70).map(x => x.action),
    rubrics: formats,
  };
}


function buildSmartAnalytics() {
  const analytics = state.data?.analytics || {};
  const settings = state.data?.settings || {};
  const drafts = visibleDrafts();
  const plan = visiblePlanItems();
  const media = Array.isArray(state.data?.media_inbox) ? state.data.media_inbox : getMediaInbox();
  const schedules = Array.isArray(state.data?.schedules) ? state.data.schedules : [];
  const activeChannel = state.data?.active_channel || null;
  const backendSignals = Array.isArray(analytics.signals) ? analytics.signals : [];
  const liveSummary = {
    plan_count: plan.filter(x => Number(x?.posted || 0) === 0).length,
    drafts_count: drafts.filter(x => String(x?.status || 'draft').toLowerCase() === 'draft').length,
    media_count: media.length,
    schedule_count: schedules.filter(x => Number(x?.enabled ?? 1) !== 0).length,
    onboarding_completed: (Number(analytics?.summary?.onboarding_completed || 0) === 1 || String(settings.onboarding_completed || '0') === '1' || Boolean(String(settings.topic || '').trim() && String(settings.channel_audience || '').trim() && String(settings.channel_style || settings.channel_style_preset || '').trim() && (safeJsonParse(settings.channel_formats, []).length >= 2))) ? 1 : 0,
    active_channel: activeChannel ? 1 : 0,
  };
  const fallback = computeFallbackAnalytics(settings, activeChannel, state.data?.stats || {}, drafts, plan, media, schedules);
  const useBackend = backendSignals.length > 0;
  const score = useBackend ? Number(analytics.score || analytics.readiness || 0) : Number(fallback.score || 0);
  const signals = useBackend ? backendSignals : fallback.signals;
  const weakest = useBackend ? (analytics.weakest_area || signals.slice().sort((a,b)=>Number(a.value||0)-Number(b.value||0))[0]) : signals.slice().sort((a,b)=>Number(a.value||0)-Number(b.value||0))[0];
  return {
    score: Math.max(0, Math.min(100, score || 0)),
    next_step: String((useBackend ? analytics.next_step : fallback.next_step) || weakest?.action || 'Усиль профиль и запас контента'),
    weakest_key: String((useBackend ? analytics.weakest_key : fallback.weakest_key) || weakest?.key || ''),
    signals,
    recommendations: useBackend ? (Array.isArray(analytics.recommendations) ? analytics.recommendations : []) : fallback.recommendations,
    rubrics: Array.isArray(analytics.rubrics) && analytics.rubrics.length ? analytics.rubrics : (fallback.rubrics || []),
    summary: { ...(analytics.summary || {}), ...liveSummary }
  };
}

function getDashboardImprovementAction(settings, analytics, activeChannel, planCount, draftsCount) {
  const mode = String(settings.channel_mode || '').trim();
  const topic = String(settings.topic || '').trim();
  const audience = String(settings.channel_audience || '').trim();
  const style = String(settings.channel_style || '').trim();
  const sources = safeJsonParse(settings.news_sources_json || settings.news_sources || '[]', []);

  if (!activeChannel) {
    return { label: 'Подключить канал', hint: 'Без активного канала автопилот не сможет публиковать.', actionKey: 'channels', cta: 'Открыть каналы' };
  }
  if (!topic || !audience || !style || Number((analytics.summary||{}).onboarding_completed || (String(settings.onboarding_completed || '0') === '1' ? 1 : 0)) !== 1) {
    return { label: 'Дозаполнить профиль канала', hint: 'Сейчас каналу не хватает базовой стратегии: темы, аудитории, стиля и правил.', actionKey: 'profile', cta: 'Открыть onboarding' };
  }
  if (planCount < 5) {
    return { label: 'Усилить контент-план', hint: 'Боту нужен запас тем и идей, иначе он быстро упрётся в пустой план.', actionKey: 'plan', cta: 'Открыть план' };
  }
  if (draftsCount < 2) {
    return { label: 'Создать резерв черновиков', hint: 'Пара готовых черновиков делает постинг устойчивее и спасает от простоев.', actionKey: 'posts', cta: 'Открыть посты' };
  }
  if ((mode === 'news' || mode === 'both') && !sources.length) {
    return { label: 'Добавить новостные источники', hint: 'Новостной режим без источников быстро скатится в слабую и общую генерацию.', actionKey: 'profile', cta: 'Открыть onboarding' };
  }

  const weakMap = {
    profile: { label: 'Усилить профиль канала', hint: 'Нужно уточнить стратегию канала в onboarding.', actionKey: 'profile', cta: 'Открыть onboarding' },
    content_reserve: { label: 'Пополнить запас контента', hint: 'Добавь новые идеи и черновики, чтобы канал не простаивал.', actionKey: 'plan', cta: 'Открыть план' },
    autopost: { label: 'Настроить автопостинг', hint: 'Добавь частоту и расписание, чтобы публикации шли стабильно.', actionKey: 'autopost', cta: 'Открыть автопост' },
    media: { label: 'Добавить медиарезерв', hint: 'Заранее подготовь изображения и видео для ближайших публикаций.', actionKey: 'posts', cta: 'Открыть посты' },
    variety: { label: 'Добавить новые рубрики', hint: 'План стал слишком однотипным — нужно расширить подачу и углы.', actionKey: 'profile', cta: 'Открыть onboarding' },
    news: { label: 'Уточнить актуальный режим', hint: 'Для новостного канала нужно больше опоры на источники и профиль.', actionKey: 'profile', cta: 'Открыть onboarding' },
  };
  return weakMap[String(analytics.weakest_key || '')] || { label: 'Открыть аналитику', hint: 'Проверь, что сейчас сильнее всего просаживает устойчивость канала.', actionKey: 'analytics', cta: 'Открыть аналитику' };
}

function openAnalyticsDetails() {
  const a = buildSmartAnalytics();
  const stats = state.data?.stats || {};
  const summary = a.summary || {};
  const activityItems = [
    { key: 'posts_total', label: 'Посты', value: Number(stats.total_posts || 0), hint: 'Всего опубликовано' },
    { key: 'posts_week', label: '7 дней', value: Number(stats.posted_last_7d || 0), hint: 'Публикации за неделю' },
    { key: 'plan', label: 'План', value: Number(summary.plan_count || 0), hint: 'Идей в очереди' },
    { key: 'drafts', label: 'Черновики', value: Number(summary.drafts_count || 0), hint: 'Готовых заготовок' },
  ];
  const sparkline = buildRhythmSparkline(activityItems);
  const body = `
    <div class="stack">
      <div class="analytics-modal-head analytics-modal-head-v2">
        <div>
          <div class="section-title">Подробная аналитика</div>
          <div class="section-desc">Ключевые сигналы по каналу.</div>
        </div>
        <div class="analytics-modal-ring-wrap">
          ${renderScoreRing(a.score, 'score-ring-lg')}
          <div class="dashboard-score-label">Готовность</div>
        </div>
      </div>
      <div class="card analytics-modal-chart-card"><div class="card-inner stack compact-dashboard-card">
        ${sparkline}
        <div class="section-head-inline">
          <div>
            <div class="section-title mini-title">Сигналы</div>
            <div class="section-desc">Что уже собрано, а что ещё проседает.</div>
          </div>
        </div>
        <div class="analytics-modal-visual-grid analytics-modal-visual-grid-single">
          <div class="analytics-chart-list">
            ${buildAnalyticsChartRows(a.signals)}
          </div>
        </div>
      </div></div>
      <div class="analytics-grid analytics-grid-modal">
        ${a.signals.map(sig => `
          <div class="analytics-item analytics-item-modal analytics-item-soft">
            <div class="analytics-item-top"><b>${escapeHtml(sig.label)}</b><span>${sig.value}%</span></div>
            <div class="progress-track"><span style="width:${sig.value}%"></span></div>
            <div class="analytics-item-hint">${escapeHtml(sig.hint)}</div>
            <div class="analytics-item-action">Что сделать: ${escapeHtml(sig.action)}</div>
          </div>
        `).join('')}
      </div>
      <div class="card"><div class="card-inner stack compact-section-card">
        <div class="section-title small-title">Что улучшить</div>
        <div class="analytics-recommend-list">${a.recommendations.map(x => `<div class="analytics-recommend-item">${escapeHtml(x)}</div>`).join('')}</div>
      </div></div>
      ${Array.isArray(a.rubrics) && a.rubrics.length ? `<div class="analytics-rubrics">${a.rubrics.map(x => `<span class="meta-pill">${escapeHtml(x)}</span>`).join('')}</div>` : ''}
    </div>`;
  modal('Умная аналитика', body, `<button class="btn primary" data-action="switchTab" data-action-arg="autopost" data-dismiss-modal="true">Открыть автопост</button><button class="btn ghost" data-action="closeModal">Закрыть</button>`);
}

function analyticsBlock() {
  const a = buildSmartAnalytics();
  const stats = state.data?.stats || {};
  const summary = a.summary || {};
  const activityItems = [
    { key: 'posts_total', label: 'Посты', value: Number(stats.total_posts || 0), hint: 'Всего опубликовано' },
    { key: 'posts_week', label: '7 дней', value: Number(stats.posted_last_7d || 0), hint: 'Публикации за неделю' },
    { key: 'plan', label: 'План', value: Number(summary.plan_count || 0), hint: 'Идей в очереди' },
    { key: 'media', label: 'Медиа', value: Number(summary.media_count || 0), hint: 'Файлов в резерве' },
  ];
  const sparkline = buildRhythmSparkline(activityItems);
  return `
    <div class="card analytics-card analytics-card-clickable analytics-card-charted" data-action="openAnalyticsDetails"><div class="card-inner stack compact-dashboard-card">
      <div class="section-head-inline analytics-head-inline">
        <div>
          <div class="section-title small-title">Умная аналитика</div>
          <div class="section-desc">Показывает, где канал уже собран, а где ещё проседает.</div>
        </div>
        <div class="analytics-badge-inline">${a.next_step ? escapeHtml(a.next_step) : 'Открыть детали'}</div>
      </div>
      ${sparkline}
      <div class="analytics-visual-split analytics-visual-split-single">
        <div class="analytics-chart-list compact-chart-list">
          ${buildAnalyticsChartRows(a.signals)}
        </div>
      </div>
      <div class="analytics-inline-link">Открыть детали</div>
    </div></div>
  `;
}

function dashboardView() {
  const stats = state.data?.stats || {};
  const drafts = visibleDrafts();
  const plan = visiblePlanItems().slice().sort((a,b)=> String(a.dt||'').localeCompare(String(b.dt||'')));
  const settings = state.data?.settings || {};
  const smartAnalytics = buildSmartAnalytics();
  const activeChannelData = state.data?.active_channel || null;
  const nextPlan = plan.find(x => !x.posted) || null;
  const latestDraft = drafts[0] || null;
  const postsCount = Number(stats.total_posts || 0);
  const postedLast7d = Number(stats.posted_last_7d || 0);
  const planCount = Number(smartAnalytics.summary.plan_count || plan.filter(x => !x.posted).length || 0);
  const draftsCount = Number(smartAnalytics.summary.drafts_count || drafts.length || 0);
  const mediaCount = Number(smartAnalytics.summary.media_count || 0);
  const readinessScore = Math.max(0, Math.min(100, Number(smartAnalytics.score || 0)));
  const autopilotReady = Boolean(
    Number(smartAnalytics.summary.onboarding_completed || 0) === 1 &&
    activeChannelData &&
    String(settings.topic || '').trim() &&
    readinessScore >= 55
  );
  const channelLabel = activeChannelData
    ? resolveChannelLabel(activeChannelData.title || activeChannelData.channel_target || '')
    : 'Канал не выбран';
  const topicLabel = String(settings.topic || '').trim() || 'Тема не указана';
  const nextAction = getDashboardImprovementAction(settings, smartAnalytics, activeChannelData, planCount, draftsCount);

  return `
    <div class="home-stage stack home-stage-reset home-dashboard-v6">
      <div class="card dashboard-top-card glass-card"><div class="card-inner dashboard-top-inner">
        <div class="dashboard-top-copy">
          <div class="dashboard-top-eyebrow">Активный канал</div>
          <div class="dashboard-channel-name">${escapeHtml(channelLabel)}</div>
          <div class="dashboard-topic-line">${escapeHtml(topicLabel)}</div>
          <div class="dashboard-meta-pills dashboard-meta-pills-top">
            <span class="meta-pill ${autopilotReady ? 'is-good' : ''}">${autopilotReady ? '● Автопилот готов' : '○ Автопилот не готов'}</span>
            ${mediaCount ? `<span class="meta-pill">Медиа: ${mediaCount}</span>` : ''}
          </div>
          <div class="dashboard-top-glow"></div>
        </div>
        <button class="dashboard-score-panel" data-action="openAnalyticsDetails" title="Открыть аналитику">
          ${renderScoreRing(readinessScore, 'score-ring-lg')}
          <div class="dashboard-score-label">Готовность</div>
        </button>
      </div></div>

      <div class="dashboard-quick-row quick-actions-row-top quick-actions-row-main">
        <button class="quick-chip quick-chip-compact active" data-action="openDraftEditor"><span>✎</span><b>Новый пост</b></button>
        <button class="quick-chip quick-chip-compact" data-action="switchTab" data-action-arg="plan"><span>▣</span><b>План</b></button>
        <button class="quick-chip quick-chip-compact" data-action="switchTab" data-action-arg="channels"><span>◎</span><b>Каналы</b></button>
        <button class="quick-chip quick-chip-compact" data-action="openChannelProfile"><span>⚙</span><b>Профиль</b></button>
      </div>

      <div class="card next-action-hero-card glass-card"><div class="card-inner next-action-hero-inner">
        <div class="next-action-hero-copy">
          <div class="next-action-hero-label">${escapeHtml(nextAction.label)}</div>
          <div class="next-action-hero-hint">${escapeHtml(nextAction.hint)}</div>
          <div class="hero-micro-pills">
            <span class="hero-micro-pill">${draftsCount} черновиков</span>
            <span class="hero-micro-pill">${planCount} идей</span>
            <span class="hero-micro-pill">${postedLast7d} за 7 дней</span>
          </div>
        </div>
        <button class="btn primary next-action-hero-btn" data-hero-action="${escapeHtml(nextAction.actionKey)}">${escapeHtml(nextAction.cta)}</button>
      </div></div>

      <div class="dashboard-stat-row">
        <button class="dashboard-stat-pill" data-action="switchTab" data-action-arg="posts"><span>${postsCount}</span><small>опубликовано</small></button>
        <button class="dashboard-stat-pill ${draftsCount < 2 ? 'low' : ''}" data-action="switchTab" data-action-arg="posts"><span>${draftsCount}</span><small>черновиков</small></button>
        <button class="dashboard-stat-pill ${planCount < 3 ? 'low' : ''}" data-action="switchTab" data-action-arg="plan"><span>${planCount}</span><small>в плане</small></button>
        <button class="dashboard-stat-pill" data-action="switchTab" data-action-arg="channels"><span>${postedLast7d}</span><small>за 7 дней</small></button>
      </div>

      <div class="dashboard-secondary-grid">
        <div class="card interactive-card glass-card" data-action="${nextPlan ? 'switchTab' : 'openPlanGenerator'}" ${nextPlan ? 'data-action-arg="plan"' : ''}><div class="card-inner compact-dashboard-card stack">
          <div class="dashboard-card-head">
            <div class="section-title mini-title">Ближайшая идея</div>
            <span class="card-arrow">→</span>
          </div>
          ${nextPlan
            ? `<div class="meta-brief-title clamp-3">${escapeHtml((nextPlan.prompt || nextPlan.topic || 'Идея публикации').slice(0, 140))}</div>
               <div class="section-desc">${escapeHtml(formatDateTime(nextPlan.dt))}</div>
               <div class="card-inline-action">Открыть план</div>`
            : `<div class="section-desc empty-hint">Добавь несколько идей в план — бот наберёт запас и перестанет останавливаться.</div>
               <div class="card-inline-action">Сгенерировать план</div>`
          }
        </div></div>
        <div class="card interactive-card glass-card" data-action="openDraftEditor" ${latestDraft ? `data-action-arg="${latestDraft.id}"` : ''}><div class="card-inner compact-dashboard-card stack">
          <div class="dashboard-card-head">
            <div class="section-title mini-title">Последний черновик</div>
            <span class="card-arrow">→</span>
          </div>
          ${latestDraft
            ? `<div class="meta-brief-title clamp-3">${escapeHtml(((latestDraft.text || latestDraft.prompt || 'Черновик')).slice(0, 140))}</div>
               <div class="section-desc">Готов к доработке и публикации.</div>
               <div class="card-inline-action">Открыть редактор</div>`
            : `<div class="section-desc empty-hint">Черновиков пока нет — открой редактор и создай первый пост.</div>
               <div class="card-inline-action">Создать черновик</div>`
          }
        </div></div>
      </div>

      ${analyticsBlock()}
    </div>
  `;
}

function channelsView() {
  const channels = state.data?.channels || [];
  const activeId = state.data?.active_channel?.id;
  const tier = state.data?.subscription?.subscription_tier || 'free';
  const maxChannels = state.data?.limits?.channels_max || (tier === 'max' ? 10 : tier === 'pro' ? 3 : 1);
  // Next tier locked slot preview counts (upsell motivation)
  const nextTierMax = tier === 'free' ? 3 : tier === 'pro' ? 10 : 0;
  const emptySlots = Math.max(0, maxChannels - channels.length);
  const lockedSlots = Math.max(0, nextTierMax - maxChannels);

  const filledHtml = channels.map(ch => `
    <div class="item item-strong channel-item-pro channel-item-single-shell ${activeId === ch.id ? 'channel-item-active' : ''}">
      <div class="item-row">
        <div>
          <div class="item-title">${escapeHtml(ch.title || 'Без названия')}</div>
          <div class="item-sub">${escapeHtml(ch.topic || 'Тема не указана')}</div>
        </div>
        ${activeId === ch.id ? '<div class="badge badge-accent">Активный</div>' : '<div class="badge">Канал</div>'}
      </div>
      <div class="item-actions">
        ${activeId === ch.id
          ? `<button class="btn small ghost" data-action="openChannelProfile">Профиль канала</button>`
          : `<button class="btn small primary" data-action="activateChannel" data-action-arg="${ch.id}">Сделать активным</button>`}
        <button class="btn small danger" data-delete-channel="${ch.id}" data-channel-name="${escapeHtml(ch.title || 'канал')}">Удалить</button>
      </div>
    </div>`).join('');

  const emptyHtml = Array.from({ length: emptySlots }, () => `
    <div class="channel-slot channel-slot-empty" data-action="openChannelModal">
      <div class="channel-slot-icon">＋</div>
      <div class="channel-slot-text">
        <b>Добавить канал</b>
        <span>Слот доступен на вашем тарифе</span>
      </div>
    </div>`).join('');

  const lockedHtml = Array.from({ length: lockedSlots }, () => `
    <div class="channel-slot channel-slot-locked" data-action="showTariffsModal">
      <div class="channel-slot-icon">🔒</div>
      <div class="channel-slot-text">
        <b>Заблокированный слот</b>
        <span>Доступен на тарифе ${tier === 'free' ? 'PRO' : 'MAX'} — нажмите для апгрейда</span>
      </div>
    </div>`).join('');

  // Linked channels / mirror publishing hint (show when 2+ channels exist)
  const linkedHint = channels.length >= 2 ? `
    <div class="channel-linked-hint card">
      <div class="card-inner">
        <div class="channel-linked-header">
          <span class="channel-linked-icon">🔗</span>
          <div>
            <strong>Связанная публикация</strong>
            <div class="channel-linked-desc">Публикуйте один пост сразу в несколько каналов. При публикации из редактора вы можете выбрать дополнительные каналы для одновременной отправки.</div>
          </div>
        </div>
      </div>
    </div>
  ` : '';

  return `
    <div class="stack page-stack-tight">
      <div class="section-head">
        <div>
          <div class="section-title">Каналы</div>
          <div class="section-desc">Слоты каналов по вашему тарифу. Добавляйте и управляйте каналами.</div>
        </div>
        ${channels.length < maxChannels
          ? '<button class="btn primary" data-action="openChannelModal">Добавить канал</button>'
          : '<button class="btn ghost" data-action="showTariffsModal">Улучшить тариф</button>'}
      </div>
      <div class="channel-slots-grid">
        ${filledHtml}
        ${emptyHtml}
        ${lockedHtml}
      </div>
      ${linkedHint}
    </div>
  `;
}

function draftCard(d) {
  const hasChannel = !!String(d.channel_target || '').trim();
  const isFailed = String(d.status || '').toLowerCase() === 'failed';
  const text = (d.text || d.prompt || 'Без текста').trim();
  const mediaLabel = d.media_type && d.media_type !== 'none' ? (d.media_type === 'video' ? 'видео' : 'фото') : 'без медиа';
  const subtitleBits = [resolveChannelLabel(d.channel_target), mediaLabel];
  const source = String(d.draft_source || '');
  const sourceIcon = source === 'voice' ? '🎙️' : source === 'news_sniper' ? '⚡' : '';
  const newsSrc = parseNewsSource(d);
  const sourceBadge = source === 'voice'
    ? '<div class="badge badge-voice" title="Создан из голосового сообщения">🎙️ Голос</div>'
    : source === 'news_sniper'
      ? '<div class="badge badge-sniper" title="Срочная новость от ИИ">⚡ Срочно</div>'
      : '';
  const newsSourceLine = newsSrc
    ? `<div class="item-sub news-source-line" title="${escapeHtml(newsSrc.url || '')}">📰 ${escapeHtml(newsSrc.domain || '')}${newsSrc.published_at ? ' · ' + escapeHtml(newsSrc.published_at.slice(0, 10)) : ''}</div>`
    : '';
  const failedBadge = '<div class="badge badge-danger" title="Ошибка публикации">⚠ Ошибка</div>';
  const statusBadge = isFailed ? failedBadge : (hasChannel ? '<div class="badge badge-accent">Готов</div>' : '<div class="badge">Нужен канал</div>');
  const publishBtn = hasChannel
    ? `<button class="btn small secondary" data-action="publishDraft" data-action-arg="${d.id}">${isFailed ? 'Повторить' : 'Опубликовать'}</button>`
    : '';
  return `
    <div class="item draft-card draft-card-tone${isFailed ? ' draft-card-failed' : ''}${source === 'news_sniper' ? ' draft-card-sniper' : source === 'voice' ? ' draft-card-voice' : ''}" data-draft-id="${d.id}">
      <div class="item-row">
        <div class="stack-tight">
          <div class="item-title">${sourceIcon ? sourceIcon + ' ' : ''}${escapeHtml(text.slice(0, 120) || 'Без текста')}</div>
          <div class="item-sub">${escapeHtml(subtitleBits.join(' · '))}</div>
          ${newsSourceLine}
        </div>
        <div class="badge-group">${sourceBadge}${statusBadge}</div>
      </div>
      <div class="item-actions item-actions-compact draft-actions-grid">
        <button class="btn small" data-action="openDraftEditor" data-action-arg="${d.id}">Редактировать</button>
        ${publishBtn}
        <button class="btn small danger" data-action="deleteDraft" data-action-arg="${d.id}">Удалить</button>
      </div>
    </div>
  `;
}

function postsView() {
  const drafts = visibleDrafts();
  const limits = state.data?.limits || {};
  const current = effectiveDraftsCurrent();
  const max = Math.max(1, Number(limits.drafts_max ?? 15));
  const percent = Math.min(100, Math.round((current / max) * 100));
  const tone = draftLimitTone(percent);
  const nextPlanItem = visiblePlanItems()
    .filter(x => !x.posted)
    .sort((a, b) => String(a.dt || '').localeCompare(String(b.dt || '')))[0] || null;
  const nextPlanPrompt = nextPlanItem ? (nextPlanItem.prompt || nextPlanItem.topic || nextPlanItem.idea || '') : '';
  const hasSniperDrafts = drafts.some(d => d.draft_source === 'news_sniper');
  const hasSpyDrafts = drafts.some(d => d.draft_source === 'spy');
  const tier = state.data?.subscription?.subscription_tier || 'free';
  const sniperBanner = hasSniperDrafts
    ? `<div class="card sniper-banner">
        <div class="card-inner">
          <span class="sniper-banner-icon">⚡</span>
          <div class="sniper-banner-text"><b>Срочные инфоповоды</b><br>ИИ нашёл актуальные новости по вашей теме и подготовил посты ниже.</div>
        </div>
      </div>`
    : '';
  const spyBanner = hasSpyDrafts
    ? `<div class="card sniper-banner spy-banner">
        <div class="card-inner">
          <span class="sniper-banner-icon">🕵️‍♂️</span>
          <div class="sniper-banner-text"><b>Идеи от Шпиона</b><br>Ниже — черновики, сгенерированные на основе анализа конкурентов.</div>
        </div>
      </div>`
    : '';
  const spyButton = tier === 'max'
    ? `<button class="btn ghost posts-action-btn spy-btn" data-action="openCompetitorSpyModal" title="Анализ конкурентов">🕵️‍♂️ Шпион</button>`
    : `<button class="btn ghost posts-action-btn spy-btn locked-feature-btn" data-action="showPaywallModal" data-action-arg="Шпион конкурентов доступен на тарифе Max. Анализируй каналы конкурентов и создавай уникальный контент на их основе." data-action-arg2="spy_locked"" title="Только для Max">🕵️‍♂️ Шпион <span class="feature-lock-badge">Max</span></button>`;
  const sniperButton = tier === 'max'
    ? `<button class="btn ghost posts-action-btn" data-action="openNewsSniperModal" title="News Sniper — запустить сейчас">⚡ News Sniper</button>`
    : '';
  return `
    <div class="posts-layout posts-layout-pro stack page-stack-tight">
      <div class="card posts-hero posts-hero-clean"><div class="card-inner stack compact-section-card">
        <div class="section-title">Посты</div>
        <div class="section-desc">Главный сценарий: открой редактор, сгенерируй пост через ИИ внутри него и сразу доработай перед публикацией.</div>
        <div class="posts-primary-actions posts-actions-grid">
          <button class="btn primary posts-main-btn posts-btn-full" data-action="openDraftEditor">Открыть редактор</button>
          <div class="posts-secondary-row">
            ${spyButton}
            ${sniperButton || `<button class="btn ghost posts-action-btn posts-action-equal" data-action="${nextPlanItem ? 'generatePostFromPlan' : 'openPlanGenerator'}" ${nextPlanItem ? `data-action-arg="${escapeHtml(String(nextPlanItem.id))}" title="${escapeHtml(nextPlanPrompt)}"` : ''}>${nextPlanItem ? '▣ Из плана' : '▣ План'}</button>`}
          </div>
          ${sniperButton && nextPlanItem ? `<button class="btn ghost posts-action-btn posts-btn-full" data-action="generatePostFromPlan" data-action-arg="${escapeHtml(String(nextPlanItem.id))}" title="${escapeHtml(nextPlanPrompt)}">▣ Из плана</button>` : ''}
        </div>
        ${nextPlanItem ? `<div class="section-desc plan-hint-desc">Следующая идея из плана: <b>${escapeHtml(nextPlanPrompt.slice(0, 60))}${nextPlanPrompt.length > 60 ? '…' : ''}</b></div>` : ''}
      </div></div>

      ${bigVideoPanel(false)}

      <div class="draft-limit-card draft-limit-card-accent draft-limit-${tone}">
        <div class="draft-limit-head">
          <div class="draft-limit-title">Черновики</div>
          <div class="draft-limit-meta">${current} / ${max}</div>
        </div>
        <div class="draft-progress-bar"><span class="draft-progress-fill ${tone}" style="width:${percent}%"></span></div>
      </div>

      <div class="section-head section-head-tight section-head-soft">
        <div>
          <div class="section-title">Мои черновики</div>
          <div class="section-desc">Все рабочие посты в одном месте.</div>
        </div>
      </div>

      ${sniperBanner}
      ${spyBanner}

      <div class="list drafts-list stack">${drafts.length ? drafts.map(draftCard).join('') : `<div class="empty-state-card">${nextPlanItem ? `<div class="empty-state-icon">▣</div><div class="empty-state-title">Черновиков пока нет</div><div class="empty-state-text">Следующая идея из плана: <b>${escapeHtml(nextPlanPrompt.slice(0, 80))}</b></div><button class="btn primary" data-action="generatePostFromPlan" data-action-arg="${escapeHtml(String(nextPlanItem.id))}">Создать пост из плана</button>` : '<div class="empty">Пока пусто. Открой редактор и создай первый пост.</div>'}</div>`}</div>
    </div>
  `;
}

function planCard(item) {
  return `
    <div class="item item-strong" data-plan-id="${Number(item.id || 0)}">
      <div class="item-row">
        <div>
          <div class="item-title">${escapeHtml(item.prompt || item.topic || 'Идея публикации')}</div>
          <div class="item-sub">${formatDateTime(item.dt)}</div>
        </div>
        ${item.posted ? '<div class="badge">Готово</div>' : '<div class="badge badge-accent">Запланировано</div>'}
      </div>
      <div class="item-actions">
        <button class="btn small" data-action="openPlanEditor" data-action-arg="${item.id}">Редактировать</button>
        <button class="btn small danger" data-action="deletePlanItem" data-action-arg="${item.id}">Удалить</button>
      </div>
    </div>
  `;
}

function planView() {
  const items = visiblePlanItems().slice().sort((a, b) => (a.dt > b.dt ? 1 : -1));
  const upcoming = items.filter(x => !x.posted);
  const done = items.filter(x => x.posted);
  return `
    <div class="stack page-stack-tight">
      <div class="section-head plan-page-head">
        <div>
          <div class="section-title">Контент-план</div>
          <div class="section-desc">Запас идей и тем, из которых бот готовит посты.</div>
        </div>
        <div class="item-actions plan-head-actions">
          <button class="btn ghost" data-action="openPlanGenerator">Сгенерировать</button>
          <button class="btn primary" data-action="openPlanEditor">Добавить</button>
        </div>
      </div>
      ${upcoming.length
        ? `<div class="list plan-list">${upcoming.map(planCard).join('')}</div>`
        : `<div class="empty-state-card">
             <div class="empty-state-icon">▣</div>
             <div class="empty-state-title">План пустой</div>
             <div class="empty-state-text">Без запаса тем автопилот остановится. Добавь 5–10 идей или сгенерируй план одной кнопкой.</div>
             <button class="btn primary" data-action="openPlanGenerator">Сгенерировать план</button>
           </div>`
      }
      ${done.length ? `
        <div class="section-head section-head-tight"><div class="section-title mini-title">Уже опубликовано</div></div>
        <div class="list plan-list plan-list-done">${done.map(planCard).join('')}</div>
      ` : ''}
    </div>
  `;
}

function _computeAutopostStatus(postsEnabled, newsEnabled, postingMode, scheduleRows, channel) {
  const now = new Date();
  const hour = now.getHours();
  const hhmm = now.toTimeString().slice(0, 5);

  if (!postsEnabled) {
    return { html: '<div class="autopost-realtime-status off">⏸ Автопостинг выключен. Публикация по расписанию не производится.</div>' };
  }

  if (!channel) {
    return { html: '<div class="autopost-realtime-status warn">⚠ Канал не выбран. Автопост не может работать без привязанного канала.</div>' };
  }

  const lines = [];

  // Night pause check (23:00 - 08:00)
  if (hour >= 23 || hour < 8) {
    lines.push('🌙 Сейчас ночная пауза (23:00–08:00). Посты не публикуются.');
  }

  // Find next schedule slot
  const enabledSlots = (scheduleRows || []).filter(r => r.enabled !== 0);
  if (enabledSlots.length === 0) {
    lines.push('📭 Нет активных слотов расписания. Добавьте хотя бы один слот.');
  } else {
    // Find the nearest upcoming slot today or tomorrow
    const todaySlots = enabledSlots.map(r => r.time_hhmm).sort();
    const upcoming = todaySlots.find(t => t > hhmm);
    if (upcoming) {
      lines.push(`⏰ Следующий слот: <strong>${upcoming}</strong>`);
    } else if (todaySlots.length > 0) {
      lines.push(`⏰ Сегодня слоты отработали. Следующий: <strong>${todaySlots[0]}</strong> завтра`);
    }
    lines.push(`📊 Всего слотов: ${enabledSlots.length} | Мин. пауза: 45 мин`);
  }

  // News status
  if ((postingMode === 'both' || postingMode === 'news') && newsEnabled) {
    lines.push('📰 Авто-новости активны — проверяются каждые 30 мин');
  }

  const html = lines.length > 0
    ? `<div class="autopost-realtime-status active">${lines.join('<br>')}</div>`
    : '';
  return { html };
}

function autopostView() {
  const s = state.data?.settings || {};
  const postsEnabled = s.posts_enabled === '1';
  const newsEnabled = s.news_enabled === '1';
  const postingMode = s.posting_mode || 'both';
  const interval = s.news_interval_hours || '6';
  const rows = state.data?.schedules || [];
  const channel = activeChannel();
  const channels = state.data?.channels || [];
  const planCount = (state.data?.plan || []).length;
  const draftsCount = (state.data?.drafts || []).length;

  // Compute real-time autopost status
  const statusDetails = _computeAutopostStatus(postsEnabled, newsEnabled, postingMode, rows, channel);

  // Build active sources list
  const sources = [];
  if (postingMode === 'both' || postingMode === 'posts') {
    sources.push({icon: '✦', name: 'ИИ-генерация', desc: 'Автоматические посты по теме канала', active: true});
  }
  if ((postingMode === 'both' || postingMode === 'news') && newsEnabled) {
    sources.push({icon: '📰', name: 'Авто-новости', desc: `Каждые ${interval}ч, ночью — пауза`, active: true});
  }
  if (s.source_auto_draft !== '0') {
    sources.push({icon: '📋', name: 'Черновики из источников', desc: 'Новости и конкуренты → черновики', active: true});
  }
  sources.push({icon: '▣', name: 'Контент-план', desc: `${planCount} идей в очереди`, active: planCount > 0});
  sources.push({icon: '✎', name: 'Готовые черновики', desc: `${draftsCount} черновиков`, active: draftsCount > 0});

  const modeLabel = {both: 'Посты и новости', news: 'Только новости', posts: 'Только посты'}[postingMode] || postingMode;

  // Channel switcher for autopost
  const channelSwitcherHtml = channels.length > 1 ? `
    <div class="autopost-channel-switcher">
      <div class="autopost-channel-label">Канал для автопоста</div>
      <div class="autopost-channel-chips">
        ${channels.map(ch => `
          <button class="autopost-channel-chip ${channel && channel.id === ch.id ? 'active' : ''}"
                  data-action="activateChannel" data-action-arg="${ch.id}">
            ${escapeHtml(resolveChannelLabel(ch.title || ch.channel_target || ''))}
          </button>
        `).join('')}
      </div>
    </div>
  ` : '';

  return `
    <div class="stack page-stack-tight autopost-page">
      <div class="section-head">
        <div>
          <div class="section-title">Автопост</div>
          <div class="section-desc">Единая система публикации — ИИ, план, новости и черновики работают вместе.</div>
        </div>
      </div>

      <!-- Master toggle + status -->
      <div class="autopost-status-card card">
        <div class="card-inner">
          <div class="autopost-status-row">
            <div class="autopost-status-info">
              <div class="autopost-status-indicator ${postsEnabled ? 'on' : 'off'}">
                <span class="autopost-status-dot"></span>
                <span>${postsEnabled ? 'Автопост включён' : 'Автопост выключен'}</span>
              </div>
              <div class="autopost-status-meta">
                ${channel ? `<span>Канал: <strong>${escapeHtml(resolveChannelLabel(channel.title || channel.channel_target || ''))}</strong></span>` : '<span class="text-warn" role="status" aria-label="Требуется выбрать канал">Канал не выбран</span>'}
                <span>Режим: <strong>${modeLabel}</strong></span>
              </div>
              ${statusDetails.html}
            </div>
            <label class="switch modern-switch autopost-master-toggle">
              <input type="checkbox" id="autopost-toggle" ${postsEnabled ? 'checked' : ''} data-change-action="toggleAutopost"/>
              <span class="switch-ui"></span>
            </label>
          </div>
        </div>
      </div>

      ${channelSwitcherHtml}

      <!-- Content pipeline -->
      <div class="autopost-pipeline">
        <div class="autopost-pipeline-title">Источники контента</div>
        <div class="autopost-pipeline-grid">
          ${sources.map(src => `
            <div class="autopost-pipeline-item ${src.active ? 'active' : 'inactive'}">
              <span class="autopost-pipeline-icon">${src.icon}</span>
              <div class="autopost-pipeline-info">
                <strong>${src.name}</strong>
                <span>${src.desc}</span>
              </div>
              <span class="autopost-pipeline-status">${src.active ? '●' : '○'}</span>
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Publishing settings -->
      <div class="autopost-section-group">
        <div class="autopost-section-label">Режим и параметры</div>
        <div class="autopost-section-desc">Управление типом и ритмом публикаций</div>
        <div class="card autopost-settings-card">
          <div class="card-inner stack">
            <div class="field">
              <div class="label">Режим публикации</div>
              <select class="select" id="ap-posting-mode" data-change-action="updateAutopostMode">
                ${[['both','Посты и новости'],['news','Только новости'],['posts','Только посты']]
                  .map(([v,l]) => `<option value="${v}" ${postingMode === v ? 'selected' : ''}>${l}</option>`).join('')}
              </select>
            </div>
            <div class="autopost-pacing">
              <span class="autopost-pacing-icon">⏱</span>
              <span>Мин. пауза между постами: <strong>45 мин</strong></span>
            </div>
            <div class="autopost-news-settings" style="${(postingMode === 'both' || postingMode === 'news') ? '' : 'display:none'}">
              <label class="switch modern-switch">
                <input type="checkbox" id="ap-news-enabled" ${newsEnabled ? 'checked' : ''} data-change-action="toggleAutopostNews"/>
                <span class="switch-ui"></span><span>Авто-новости</span>
              </label>
              <div class="field">
                <div class="label">Интервал новостей, ч</div>
                <input class="input" id="ap-news-interval" value="${escapeHtml(interval)}" type="number" min="1" max="48"/>
              </div>
              <button class="btn primary autopost-save-btn" data-action="saveAutopostInterval">Сохранить интервал</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Schedule slots -->
      <div class="autopost-section-group">
        <div class="autopost-section-label-row">
          <div class="autopost-section-label">Расписание</div>
          <button class="btn primary small" data-action="openScheduleModal">+ Слот</button>
        </div>
        <div class="autopost-section-desc">${rows.length ? `${rows.length} слотов настроено` : 'Нет активных слотов'}</div>
        <div class="autopost-schedule-section">
          <div class="list">
            ${rows.length ? rows.map(row => `
              <div class="autopost-slot-card">
                <div class="autopost-slot-main">
                  <div class="autopost-slot-time">${escapeHtml(row.time_hhmm)}</div>
                  <div class="autopost-slot-days">${escapeHtml(row.days_label || row.days || '*')}</div>
                </div>
                <button class="btn small danger" data-action="deleteSchedule" data-action-arg="${row.id}">Удалить</button>
              </div>`).join('') : '<div class="autopost-empty-state">Добавьте слоты — бот будет публиковать в указанное время.</div>'}
          </div>
        </div>
      </div>

      <!-- Detailed settings -->
      <div class="autopost-section-group">
        <div class="autopost-section-label">Расширенные настройки</div>
        <div class="autopost-section-desc">Тонкая настройка канала и параметров</div>
        <button class="autopost-full-settings-btn" data-action="openSettingsModal">
          <span>⚙ Все настройки канала</span>
          <span class="autopost-full-settings-arrow">›</span>
        </button>
      </div>
    </div>
  `;
}

async function toggleAutopost(enabled) {
  try {
    await api('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify({ posts_enabled: enabled ? '1' : '0' })
    });
    toast(enabled ? 'Автопост включён' : 'Автопост выключен');
    await refreshSections(['settings','core'], { silent: false });
  } catch (e) {
    toast(e.message);
  }
}

async function toggleAutopostNews(enabled) {
  try {
    await api('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify({ news_enabled: enabled ? '1' : '0' })
    });
    await refreshSections(['settings','core'], { silent: true });
  } catch (e) {
    toast(e.message);
  }
}

async function updateAutopostMode(mode) {
  try {
    await api('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify({ posting_mode: mode })
    });
    await refreshSections(['settings','core'], { silent: true });
    // Re-render to show/hide news settings
    const surface = document.querySelector('.main-surface');
    if (surface && _shellRendered) {
      surface.innerHTML = `<div class="tab-enter">${renderBody()}</div>`;
    }
  } catch (e) {
    toast(e.message);
  }
}

async function saveAutopostInterval() {
  const val = document.getElementById('ap-news-interval')?.value;
  if (!val) return;
  const num = Number(val);
  if (!Number.isFinite(num) || num < 1 || num > 48) {
    toast('Интервал должен быть от 1 до 48 ч');
    return;
  }
  try {
    await api('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify({ news_interval_hours: val })
    });
    toast('Интервал сохранён');
    await refreshSections(['settings'], { silent: true });
  } catch (e) {
    toast(e.message);
  }
}

function openSettingsModal() {
  const s = state.data?.settings || {};
  const logoutBtn = _isWebMode ? `
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--glass-border)">
      <button class="btn ghost danger" style="width:100%" data-action="webLogout" data-dismiss-modal>Выйти из аккаунта</button>
    </div>
  ` : '';
  const body = `
    <div class="settings-modal-body stack">
      <div class="settings-sections">
        <div class="settings-section">
          <div class="settings-section-header">
            <div class="settings-section-icon">⚙</div>
            <div><div class="settings-section-title">Основные</div><div class="settings-section-desc">Тема канала и режим работы</div></div>
          </div>
          <div class="stack">
            <div class="field">
              <div class="label">Тема канала</div>
              <div class="field-hint">Основная тематика, вокруг которой строится весь контент</div>
              <textarea class="textarea" id="set-topic">${escapeHtml(s.topic || '')}</textarea>
            </div>
            <label class="switch modern-switch">
              <input type="checkbox" id="set-posts-enabled" ${s.posts_enabled === '1' ? 'checked' : ''}/>
              <span class="switch-ui"></span><span>Автопостинг</span>
            </label>
            <div class="field">
              <div class="label">Режим публикации</div>
              <select class="select" id="set-posting-mode">${[
                ['both','Посты и новости'],
                ['news','Только новости'],
                ['posts','Только посты']
              ].map(([value,label]) => `<option value="${value}" ${s.posting_mode === value ? 'selected' : ''}>${label}</option>`).join('')}</select>
            </div>
          </div>
        </div>

        <div class="settings-section">
          <div class="settings-section-header">
            <div class="settings-section-icon">✦</div>
            <div><div class="settings-section-title">Генерация ИИ</div><div class="settings-section-desc">Стиль, рубрики, сценарии и ограничения</div></div>
          </div>
          <div class="stack">
            <div class="field">
              <div class="label">Стиль канала</div>
              <textarea class="textarea" id="set-channel-style" placeholder="Например: живой, уверенный, без пафоса">${escapeHtml(s.channel_style || '')}</textarea>
            </div>
            <div class="field">
              <div class="label">Рубрики и их частота</div>
              <textarea class="textarea" id="set-rubrics" placeholder="Например: Разбор ошибок — раз в неделю">${escapeHtml(s.rubrics_schedule || s.content_rubrics || '')}</textarea>
            </div>
            <div class="field">
              <div class="label">Сценарии постов</div>
              <textarea class="textarea" id="set-scenarios" placeholder="Например: экспертный пост, мягкий прогрев">${escapeHtml(s.post_scenarios || '')}</textarea>
            </div>
            <div class="field">
              <div class="label">Исключения (чёрный список)</div>
              <textarea class="textarea" id="set-exclusions" placeholder="Например: политика, реклама конкурентов">${escapeHtml(s.content_exclusions || '')}</textarea>
            </div>
          </div>
        </div>

        <div class="settings-section">
          <div class="settings-section-header">
            <div class="settings-section-icon">📰</div>
            <div><div class="settings-section-title">Новости</div><div class="settings-section-desc">Автоматический подбор новостей</div></div>
          </div>
          <div class="stack">
            <label class="switch modern-switch">
              <input type="checkbox" id="set-news-enabled" ${s.news_enabled === '1' ? 'checked' : ''}/>
              <span class="switch-ui"></span><span>Авто-новости</span>
            </label>
            <label class="switch modern-switch">
              <input type="checkbox" id="set-news-strict" ${(s.news_strict_mode === '1' || s.news_strict_mode === undefined) ? 'checked' : ''}/>
              <span class="switch-ui"></span><span>Новости строго по теме</span>
            </label>
            <div class="field">
              <div class="label">Интервал новостей, часы</div>
              <input class="input" id="set-news-interval" value="${escapeHtml(s.news_interval_hours || '6')}" />
            </div>
            <div class="field">
              <div class="label">Источники новостей</div>
              <div class="field-hint">Домены или каналы, откуда подтягиваются новости. Нажмите «+» чтобы добавить.</div>
              <div class="source-chips-container" id="news-sources-chips">
                ${_renderSourceChips(s.news_sources || '')}
              </div>
              <div class="source-add-row">
                <input class="input source-add-input" id="news-source-input" placeholder="Например, vc.ru или @channel" />
                <button class="btn small primary source-add-btn" data-action="addNewsSource" type="button">+</button>
              </div>
              <input type="hidden" id="set-news-sources" value="${escapeHtml(s.news_sources || '')}" />
            </div>
          </div>
        </div>

        <div class="settings-section">
          <div class="settings-section-header">
            <div class="settings-section-icon">🔗</div>
            <div><div class="settings-section-title">Координация источников</div><div class="settings-section-desc">Как бот обрабатывает контент из автоматических источников</div></div>
          </div>
          <div class="stack">
            <label class="switch modern-switch">
              <input type="checkbox" id="set-source-auto-draft" ${(s.source_auto_draft === '1' || s.source_auto_draft === undefined) ? 'checked' : ''}/>
              <span class="switch-ui"></span><span>Сохранять в черновики</span>
            </label>
            <div class="field-hint">Рекомендуется. Новости и Шпион конкурентов сохраняют контент как черновики.</div>
          </div>
        </div>
      </div>
      ${logoutBtn}
    </div>
  `;
  modal('Настройки', body, `<button class="btn primary" data-action="saveSettings" data-dismiss-modal="true">Сохранить</button><button class="btn ghost" data-action="closeModal">Отмена</button>`);
}

function renderBody() {
  switch (state.activeTab) {
    case 'channels': return channelsView();
    case 'posts': return postsView();
    case 'plan': return planView();
    case 'autopost': return autopostView();
    case 'more': return autopostView(); // backwards compat
    default: return dashboardView();
  }
}

/* ============================================================
   WELCOME SCREEN (first run — no channels yet)
   ============================================================ */
function _needsWelcome() {
  if (!state.data) return false;
  const channels = state.data.channels || [];
  const obDone = Number(state.data.settings?.onboarding_completed || 0) === 1;
  // Show welcome screen when user has no channels and onboarding hasn't been completed
  const needs = channels.length === 0 && !obDone;
  // Clean stale browser state on fresh start to prevent ghost old data
  if (needs) {
    try { localStorage.removeItem(UI_STATE_KEY); } catch {}
    try { localStorage.removeItem(AWAIT_MEDIA_KEY); } catch {}
    try { sessionStorage.removeItem(AI_ASSISTANT_SESSION_KEY); } catch {}
  }
  return needs;
}

function _renderWelcomeScreen() {
  const welcomeStep = state._welcomeStep || 'landing';

  if (welcomeStep === 'landing') {
    return `
      <div class="shell welcome-screen">
        <div class="welcome-wrap">
          <div class="welcome-hero welcome-hero-landing">
            <div class="welcome-logo">N<span>SMM</span></div>
            <h1 class="welcome-title">NeuroSMM</h1>
            <p class="welcome-tagline">ИИ-менеджер вашего Telegram-канала</p>
            <div class="welcome-features">
              <div class="welcome-feature"><span class="welcome-feature-icon">✍️</span><span>Автоматическая генерация постов</span></div>
              <div class="welcome-feature"><span class="welcome-feature-icon">📊</span><span>Аналитика и статистика</span></div>
              <div class="welcome-feature"><span class="welcome-feature-icon">🗓️</span><span>Контент-план и расписание</span></div>
              <div class="welcome-feature"><span class="welcome-feature-icon">🖼️</span><span>Подбор изображений</span></div>
            </div>
            <button class="btn primary welcome-hero-btn" data-action="welcomeGoToAddChannel">
              Начать настройку
              <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5l7 7-7 7"/></svg>
            </button>
          </div>
        </div>
      </div>
    `;
  }

  // Step 2: Add channel form
  return `
    <div class="shell welcome-screen">
      <div class="welcome-wrap">
        <div class="welcome-hero">
          <div class="welcome-logo welcome-logo-sm">N<span>SMM</span></div>
          <h1 class="welcome-title">Привяжите ваш канал</h1>
          <p class="welcome-subtitle">Укажите название и идентификатор Telegram-канала для начала работы</p>
        </div>

        <div class="welcome-card">
          <div class="welcome-form">
            <div class="field">
              <div class="label">Название канала <span class="field-required">*</span></div>
              <div class="field-hint">Как вы хотите видеть канал в интерфейсе</div>
              <input class="input" id="welcome-ch-title" placeholder="Например, Чиповые новости" required aria-required="true">
            </div>
            <div class="field">
              <div class="label">Канал / chat_id / @username <span class="field-required">*</span></div>
              <div class="field-hint">@username канала или числовой ID</div>
              <input class="input" id="welcome-ch-target" placeholder="@my_channel" required aria-required="true">
            </div>
          </div>

          <button class="btn primary welcome-submit-btn" data-action="welcomeSaveChannel">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5l7 7-7 7"/></svg>
            Привязать канал и продолжить
          </button>
        </div>

        <div class="welcome-steps-preview">
          <div class="welcome-step-pill">
            <span class="welcome-step-num active">1</span>
            <span class="welcome-step-label active">Канал</span>
          </div>
          <div class="welcome-step-connector"></div>
          <div class="welcome-step-pill">
            <span class="welcome-step-num">2</span>
            <span class="welcome-step-label">Профиль</span>
          </div>
          <div class="welcome-step-connector"></div>
          <div class="welcome-step-pill">
            <span class="welcome-step-num">3</span>
            <span class="welcome-step-label">Автопилот</span>
          </div>
        </div>
      </div>
    </div>
  `;
}

function welcomeGoToAddChannel() {
  state._welcomeStep = 'addChannel';
  render();
}

async function welcomeSaveChannel() {
  const titleVal = (document.getElementById('welcome-ch-title')?.value || '').trim();
  const targetVal = (document.getElementById('welcome-ch-target')?.value || '').trim();
  if (!titleVal) { toast('Укажите название канала'); return; }
  if (!targetVal) { toast('Укажите идентификатор канала'); return; }
  try {
    showBusy('Привязываю канал…');
    await api('/api/channels', {
      method: 'POST',
      body: JSON.stringify({
        title: titleVal,
        channel_target: targetVal,
        make_active: true,
      })
    });
    toast('Канал привязан!');
    await refreshSections(['core', 'channels', 'settings'], { silent: true });
    // Transition directly to onboarding
    openChannelProfile();
  } catch (e) {
    toast(e.message || 'Не удалось привязать канал');
  } finally {
    hideBusy();
  }
}

function _subscriptionBadge() {
  const sub = state.data?.subscription || {};
  const tier = sub.subscription_tier || 'free';
  const label = { free: 'Free', pro: 'Pro', max: 'Max' }[tier] || tier;
  return `<button class="subscription-badge tier-${tier}" data-action="showTariffsModal" title="Тариф: ${label}">${label}</button>`;
}

function _buildAppShell() {
  return `
    <div class="shell app-shell app-shell-v3">
      <div class="topbar mini-header-wrap">
        <div class="mini-header">
          <div class="mini-header-title">NeuroSMM</div>
          <div class="mini-header-right">
            ${_subscriptionBadge()}
            <div class="top-status-pill mini-channel-pill"><span>Активный канал</span><strong>${escapeHtml(activeChannelTitle())}</strong></div>
            <button class="header-settings-btn" data-action="openSettingsModal" title="Настройки" aria-label="Настройки">⚙</button>
          </div>
        </div>
      </div>
      <div class="main-surface"><div class="tab-enter">${renderBody()}</div></div>
      <button class="ai-fab" data-action="openAIAssistant" title="ИИ-помощник" aria-label="ИИ-помощник">
        <span>✦</span>
      </button>
      <div class="bottom-nav floating-nav nav-v3">
        ${_buildNavButtons()}
      </div>
    </div>
  `;
}

function render() {
  const root = document.getElementById('app');
  if (!root) return;

  if (!state.data && state.loading) {
    _shellRendered = false;
    root.innerHTML = `<div class="shell loading"><div class="loading-card loading-card-wave"><div class="spinner"></div><div class="section-title">Загружаю панель…</div><div class="section-desc">Подготавливаю канал, черновики и план.</div></div></div>`;
    return;
  }

  // Welcome screen: shown on first run when user has no channels
  if (_needsWelcome() && !state.onboarding.active) {
    _shellRendered = false;
    root.innerHTML = _renderWelcomeScreen();
    return;
  }

  if (state.onboarding.active) {
    const existingOb = root.querySelector('.onboarding-shell');
    if (existingOb) {
      _patchOnboardingShell(existingOb);
      return;
    }
    _shellRendered = false;
    root.innerHTML = renderOnboardingShell();
    return;
  }

  if (_shellRendered && root.querySelector('.app-shell-v3')) {
    const surface = root.querySelector('.main-surface');
    const topStatus = root.querySelector('.mini-channel-pill strong');
    if (surface) {
      surface.innerHTML = `<div class="tab-enter">${renderBody()}</div>`;
      if (topStatus) topStatus.textContent = activeChannelTitle();
      return;
    }
  }

  root.innerHTML = _buildAppShell();
  _shellRendered = true;
}


async function activateChannel(profileId) {
  try {
    showBusy('Переключаю канал…');
    await api('/api/channels/activate', { method: 'POST', body: JSON.stringify({ profile_id: profileId }) });
    toast('Активный канал обновлён');
    // Refresh core + channels + settings — all three change on channel switch
    await refreshSections(['core','channels','settings'], { silent: false });
    switchTab('channels');
  } catch (e) {
    toast(e.message);
  } finally { hideBusy(); }
}


function openChannelModal() {
  modal('Новый канал', `
    <div class="field"><div class="label">Название канала <span class="field-required">*</span></div><div class="field-hint">Человекочитаемое название для интерфейса</div><input class="input" id="ch-title" placeholder="Например, Чиповые новости" required aria-required="true"></div>
    <div class="field"><div class="label">Канал / chat_id / @username <span class="field-required">*</span></div><input class="input" id="ch-target" placeholder="@my_channel" required aria-required="true"></div>
    <div class="field"><div class="label">Тема канала</div><textarea class="textarea" id="ch-topic" placeholder="О чём канал"></textarea></div>
    <label class="switch"><input type="checkbox" id="ch-active" checked> <span>Сделать активным сразу</span></label>
  `, `<button class="btn primary" data-action="saveChannel">Сохранить</button>`);
}

async function saveChannel() {
  try {
    const titleVal = (document.getElementById('ch-title').value || '').trim();
    const targetVal = (document.getElementById('ch-target').value || '').trim();
    if (!titleVal) { toast('Укажите название канала'); return; }
    if (!targetVal) { toast('Укажите идентификатор канала'); return; }
    showBusy('Проверяю доступ к каналу…');
    const makeActive = document.getElementById('ch-active').checked;
    await api('/api/channels', {
      method: 'POST',
      body: JSON.stringify({
        title: titleVal,
        channel_target: targetVal,
        topic: document.getElementById('ch-topic').value,
        make_active: makeActive,
      })
    });
    closeModal();
    toast('Канал сохранён');
    await refreshSections(['core','channels','settings'], { silent: false });
    // If the new channel was made active, launch onboarding for it
    // so the user configures topic/style/audience specifically for this channel
    if (makeActive) {
      openChannelProfile();
    } else {
      switchTab('channels');
    }
  } catch (e) {
    toast(e.message);
  } finally {
    hideBusy();
  }
}

async function deleteChannel(id, title) {
  if (!confirmAction(`Удалить канал "${title}"?`)) return;
  try {
    await api(`/api/channels/${id}`, { method: 'DELETE' });
    toast('Канал удалён');
    await refreshSections(['core','channels'], { silent: false });
    switchTab('channels');
  } catch (e) {
    toast(e.message);
  }
}


function draftEditorHtml(draft = null) {
  const currentChannel = draft?.channel_target || activeChannel()?.channel_target || state.data?.settings?.channel_target || '';
  const currentTopic = draft?.topic || state.data?.settings?.topic || '';
  const mediaRef = normalizeMediaRef(draft?.media_ref || '');
  const mediaType = draft?.media_type && draft?.media_type !== 'none' ? draft.media_type : guessMediaType(mediaRef);
  const mediaMetaJson = draft?.media_meta_json || '';
  const autoHashtagsEnabled = String(state.data?.settings?.auto_hashtags || '0') === '1';
  const editorNewsSrc = parseNewsSource(draft);
  const newsSourceCard = editorNewsSrc ? `
    <div class="card news-source-card"><div class="card-inner stack compact-section-card">
      <div class="section-title media-title-small">📰 Источник новости</div>
      <div class="news-source-details">
        <div class="item-sub"><b>Заголовок:</b> ${escapeHtml(editorNewsSrc.original_headline || '')}</div>
        <div class="item-sub"><b>Домен:</b> ${escapeHtml(editorNewsSrc.domain || '')}</div>
        ${editorNewsSrc.published_at ? '<div class="item-sub"><b>Дата:</b> ' + escapeHtml(editorNewsSrc.published_at.slice(0, 10)) + '</div>' : ''}
        ${editorNewsSrc.url ? '<div class="item-sub"><a href="' + escapeHtml(editorNewsSrc.url) + '" target="_blank" rel="noopener">Открыть оригинал ↗</a></div>' : ''}
      </div>
    </div></div>` : '';

  return `
    <div class="editor-shell redesigned-editor">
      <input type="hidden" id="dr-id" value="${draft ? Number(draft.id) : 0}">
      <input type="hidden" id="dr-channel" value="${escapeHtml(currentChannel)}">
      <input type="hidden" id="dr-topic" value="${escapeHtml(currentTopic)}">
      <div class="editor-grid editor-grid-pro editor-grid-balanced">
        ${newsSourceCard}
        <div class="card ai-priority-card"><div class="card-inner stack ai-priority-inner compact-section-card">
          <div class="section-title media-title-small">ИИ-помощник</div>
          <div class="section-desc">Попроси ИИ собрать живой пост для канала. Он заполнит редактор и отдельно предложит заголовок, CTA и короткую версию.</div>
          <div class="field">
            <div class="label">Задача для ИИ</div>
            <input class="input" id="dr-prompt" value="${escapeHtml(draft?.prompt || '')}" placeholder="Например: сильный пост с пользой и без воды">
          </div>
          <div class="hero-actions hero-actions-tight editor-ai-actions two-equal-actions">
            <button class="btn primary" type="button" data-action="generatePostInEditor" data-action-arg="${draft ? Number(draft.id) : 0}">Написать пост</button>
            <button class="btn ghost" type="button" data-action="addHashtagsToEditor">Хэштеги</button>
          </div>
          <div class="editor-ai-mini-actions">
            <button class="btn small ghost" type="button" data-action="rewriteEditorText" data-action-arg="improve">Улучшить</button>
            <button class="btn small ghost" type="button" data-action="rewriteEditorText" data-action-arg="shorter">Короче</button>
          </div>
          <div class="editor-history-row">
            <button class="btn small ghost" type="button" data-action="redoEditorChange">Вернуть</button>
            <span class="editor-autosave-status" id="editor-autosave-status">Автосохранение включено</span>
          </div>
          <div class="section-desc">${autoHashtagsEnabled ? 'Авто-хэштеги включены в настройках.' : 'Авто-хэштеги выключены. Можно добавить их кнопкой вручную.'}</div>
          <div class="editor-ai-assets" id="editor-ai-assets"></div>
        </div></div>

        <div class="card editor-main-card"><div class="card-inner stack compact-section-card">
          <div class="field">
            <div class="label">Текст поста</div>
            <textarea class="textarea editor-textarea contrast" id="dr-text" placeholder="После генерации здесь появится готовый текст поста. Можешь дополнить его вручную.">${escapeHtml(draft?.text || '')}</textarea>
          </div>

          <div class="editor-actions-row editor-actions-row-pro two-equal-actions">
            <button class="btn ghost" type="button" data-action="resetEditorDraft">Очистить</button>
            <button class="btn ghost" type="button" data-action="syncEditorPreview">Обновить превью</button>
          </div>
        </div></div>

        <div class="editor-side">
          <div class="card media-choice-card"><div class="card-inner stack compact-section-card">
            <div class="section-title media-title-small">Медиа</div>
            <div class="section-desc">Загрузи фото или выбери видео из чата. После выбора редактор сразу покажет, что медиа уже прикреплено.</div>

            <div class="media-actions-grid media-actions-grid-two">
              <button class="media-action-btn upload" type="button" data-trigger-upload="dr-file">
                <span class="media-action-icon">＋</span>
                <span><b>Загрузить</b><small>Фото или короткое видео</small></span>
              </button>
              <button class="media-action-btn chat" type="button" data-action="openChatPicker">
                <span class="media-action-icon">▶</span>
                <span><b>Из чата</b><small>Выбрать видео</small></span>
              </button>
            </div>

            <input class="input hidden-file-input" id="dr-file" type="file" accept="image/*,video/*">
            <input type="hidden" id="dr-media-ref" value="${escapeHtml(mediaRef)}">
            <input type="hidden" id="dr-media-type" value="${escapeHtml(mediaType || 'none')}">
            <input type="hidden" id="dr-media-meta" value="${escapeHtml(mediaMetaJson || '')}">

            <div class="editor-media-card media-preview-box">
              <div class="editor-media-preview" id="dr-media-preview">${renderMediaNode(mediaRef, mediaType)}</div>
              <div class="upload-status" id="dr-upload-status">${mediaRef ? 'Медиа выбрано' : 'Медиа не выбрано'}</div>
              <div class="section-desc" id="dr-media-attribution"></div>
              <div class="item-actions media-type-row media-type-row-compact">
                <button class="btn small ghost" type="button" data-action="clearEditorMedia">Убрать</button>
                <button class="btn small ghost" type="button" data-set-media="photo">Фото</button>
                <button class="btn small ghost" type="button" data-set-media="video">Видео</button>
                <button class="btn small ghost" type="button" data-set-media="none">Без медиа</button>
              </div>
            </div>
          </div></div>

          <details class="card ai-card" open>
            <summary class="card-inner summary-card"><div><div class="section-title media-title-small">Дополнительно</div><div class="section-desc">Кнопки, параметры публикации и тонкая настройка.</div></div></summary>
            <div class="card-inner stack compact-section-card">
              <div class="buttons-builder-grid">
                <div class="field"><div class="label">Текст кнопки</div><input class="input" id="dr-btn-text" placeholder="Например: Записаться"></div>
                <div class="field"><div class="label">Ссылка кнопки</div><input class="input" id="dr-btn-url" placeholder="https://..."></div>
              </div>
              <div class="item-actions compact-actions-row compact-single-action two-builder-actions">
                <button class="btn small ghost" type="button" data-action="addEditorButtonRow">Добавить кнопку</button>
                <button class="btn small ghost" type="button" data-action="clearEditorButtons">Очистить кнопки</button>
              </div>
              <div id="editor-buttons-list" class="editor-buttons-list"></div>
              <input class="input" id="dr-buttons" value="${escapeHtml(draft?.buttons_json || '[]')}" type="hidden">
              <div class="editor-switch-stack">
                <label class="switch modern-switch full-width-switch">
                  <input type="checkbox" id="dr-pin" ${draft?.pin_post ? 'checked' : ''}>
                  <span class="switch-ui"></span><span>Закрепить</span>
                </label>
                <label class="switch modern-switch full-width-switch">
                  <input type="checkbox" id="dr-comments" ${draft?.comments_enabled !== 0 ? 'checked' : ''}>
                  <span class="switch-ui"></span><span>Комментарии</span>
                </label>
                <label class="switch modern-switch full-width-switch">
                  <input type="checkbox" id="dr-ad" ${draft?.ad_mark ? 'checked' : ''}>
                  <span class="switch-ui"></span><span>Маркировка рекламы</span>
                </label>
              </div>
            </div>
          </details>
        </div>
      </div>
    </div>
  `;
}


function readEditorButtons() {
  try {
    const raw = document.getElementById('dr-buttons')?.value || '[]';
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(x => x && x.text && x.url) : [];
  } catch {
    return [];
  }
}

function writeEditorButtons(rows) {
  const input = document.getElementById('dr-buttons');
  if (input) input.value = JSON.stringify(rows || []);
  refreshEditorButtonsList();
  syncEditorPreview();
}

function refreshEditorButtonsList() {
  const root = document.getElementById('editor-buttons-list');
  if (!root) return;
  const rows = readEditorButtons();
  if (!rows.length) {
    root.innerHTML = '<div class="editor-buttons-empty">Кнопок пока нет</div>';
    return;
  }
  root.innerHTML = rows.map((btn, idx) => `
    <div class="editor-button-row">
      <div class="editor-button-copy">
        <strong>${escapeHtml(btn.text || '')}</strong>
        <span>${escapeHtml(btn.url || '')}</span>
      </div>
      <button class="btn small danger" type="button" data-action="removeEditorButtonRow" data-action-arg="${idx}">Убрать</button>
    </div>
  `).join('');
}

function addEditorButtonRow() {
  const textEl = document.getElementById('dr-btn-text');
  const urlEl = document.getElementById('dr-btn-url');
  const text = textEl?.value?.trim() || '';
  const url = urlEl?.value?.trim() || '';
  if (!text || !url) {
    toast('Заполни текст и ссылку кнопки');
    return;
  }
  const rows = readEditorButtons();
  rows.push({ text, url });
  writeEditorButtons(rows);
  if (textEl) textEl.value = '';
  if (urlEl) urlEl.value = '';
}

function removeEditorButtonRow(index) {
  const rows = readEditorButtons();
  rows.splice(Number(index), 1);
  writeEditorButtons(rows);
}

function clearEditorButtons() {
  writeEditorButtons([]);
}

function bindDraftEditorEvents() {
  const onInput = (e) => {
    if (['dr-text','dr-prompt','dr-channel','dr-buttons','dr-media-ref'].includes(e.target?.id)) {
      if (e.target.id === 'dr-media-ref') {
        const type = document.getElementById('dr-media-type');
        if (type && type.value === 'none' && e.target.value.trim()) {
          type.value = guessMediaType(e.target.value.trim());
        }
        refreshEditorMediaPreview();
      }
      if (['dr-text','dr-prompt'].includes(e.target.id)) maybeCaptureEditorHistory();
      syncEditorPreview();
      scheduleEditorAutosave(false);
    }
  };
  const onChange = async (e) => {
    if (e.target?.id === 'dr-file') {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        toast('Загружаю файл…');
        const payload = new FormData();
        payload.append('file', file);
        const uploaded = await api('/api/upload', { method: 'POST', body: payload });
        const ref = document.getElementById('dr-media-ref');
        if (ref) ref.value = uploaded.media_ref || uploaded.url || '';
        setEditorMediaType(uploaded.media_type || ((file.type || '').startsWith('video/') ? 'video' : 'photo'));
        toast('Файл загружен');
      } catch (err) {
        toast(err.message || 'Не удалось загрузить файл');
      }
      return;
    }
    if (['dr-media-ref','dr-media-type','dr-pin','dr-comments','dr-ad'].includes(e.target?.id)) {
      refreshEditorMediaPreview();
      syncEditorPreview();
      scheduleEditorAutosave(false);
    }
  };
  const onClick = (e) => {
    const mediaBtn = e.target.closest('[data-set-media]');
    if (mediaBtn) {
      setEditorMediaType(mediaBtn.dataset.setMedia);
      scheduleEditorAutosave(false);
      return;
    }
  };

  document.addEventListener('input', onInput);
  document.addEventListener('change', onChange);
  document.addEventListener('click', onClick);
  state.modalCleanup = () => {
    document.removeEventListener('input', onInput);
    document.removeEventListener('change', onChange);
    document.removeEventListener('click', onClick);
  };
}


function buildAIAssistantFallback(question = '') {
  const q = String(question || '').trim().toLowerCase();
  const analytics = buildSmartAnalytics();
  const summary = analytics.summary || {};
  const channel = activeChannel();
  const draftsCount = Number(summary.drafts_count || 0);
  const planCount = Number(summary.plan_count || 0);
  const mediaCount = Number(summary.media_count || 0);
  const scheduleCount = Number(summary.schedule_count || 0);
  const score = Number(analytics.score || 0);
  const weakest = (analytics.signals || []).slice().sort((a, b) => Number(a.value || 0) - Number(b.value || 0))[0] || null;

  if (!q) return `Готовность канала ${score}%. В запасе ${draftsCount} черновиков, ${planCount} идей и ${mediaCount} медиа.`;
  if (q.includes('автопилот') || q.includes('автопост')) {
    if (!channel) return 'Сначала подключи и активируй канал. Без активного канала автопилот не запустится.';
    if (scheduleCount < 1) return 'Сейчас нет слотов публикации. Добавь хотя бы один слот в настройках.';
    return `Автопилот сейчас на уровне ${score}%. Слабое место: ${weakest?.label || 'аналитика'}. ${weakest?.action || ''}`.trim();
  }
  if (q.includes('чернов')) return draftsCount > 0 ? `Сейчас ${draftsCount} черновиков. Для устойчивости держи хотя бы 2–4.` : 'Черновиков нет. Создай хотя бы 2 заготовки.';
  if (q.includes('план') || q.includes('иде')) return planCount > 0 ? `В плане ${planCount} идей. Нормальный запас — 5–10.` : 'План пустой. Сгенерируй хотя бы 5 идей.';
  if (q.includes('медиа')) return mediaCount > 0 ? `В медиарезерве ${mediaCount} файлов. Держи запас под ближайшие публикации.` : 'Медиарезерв пустой. Добавь несколько изображений или видео.';
  if (q.includes('канал')) return channel ? `Сейчас активен канал «${resolveChannelLabel(channel.title || channel.channel_target || '')}».` : 'Активный канал не выбран.';
  return `Канал сейчас готов на ${score}%. ${analytics.next_step || 'Открой аналитику и усили самое слабое место.'}`;
}


function loadAIAssistantSession() {
  try {
    const raw = sessionStorage.getItem(AI_ASSISTANT_SESSION_KEY);
    const parsed = safeJsonParse(raw, []);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveAIAssistantSession(messages = []) {
  try { sessionStorage.setItem(AI_ASSISTANT_SESSION_KEY, JSON.stringify(messages || [])); } catch {}
}

function pushAIAssistantSession(role, text = '', analytics = null) {
  const items = loadAIAssistantSession();
  items.push({
    role: role === 'user' ? 'user' : 'ai',
    text: String(text || ''),
    analytics: analytics || null,
    ts: Date.now(),
  });
  while (items.length > 30) items.shift();
  saveAIAssistantSession(items);
}

function renderStoredAIAssistantMessages() {
  const items = loadAIAssistantSession();
  if (!items.length) {
    const channel = activeChannel();
    const analytics = buildSmartAnalytics();
    const score = Number(analytics.score || 0);
    const greeting = channel
      ? `<p>Привет! Канал «${escapeHtml(resolveChannelLabel(channel.title || channel.channel_target || ''))}» готов на ${score}%.</p><p>Спроси что угодно: как улучшить посты, настроить автопилот, что делать дальше.</p>`
      : `<p>Привет! Я помогу настроить канал и автопостинг.</p><p>Спроси, с чего начать, или задай конкретный вопрос.</p>`;
    return `
      <div class="ai-assistant-day">ИИ-помощник</div>
      <div class="ai-assistant-msg ai">
        <div class="ai-assistant-bubble">
          ${greeting}
        </div>
      </div>`;
  }
  return `
    <div class="ai-assistant-day">Текущая сессия</div>
    ${items.map(item => {
      const cls = item.role === 'user' ? 'user' : 'ai';
      const body = item.role === 'user'
        ? `<p>${escapeHtml(item.text || '')}</p>`
        : formatAssistantText(item.text || '');
      const actions = item.role === 'ai' ? buildAssistantActionsHtml(item.text || '', item.analytics || null) : '';
      return `<div class="ai-assistant-msg ${cls}"><div class="ai-assistant-bubble">${body}${actions}</div></div>`;
    }).join('')}`;
}


function renderAIAssistantMessages() {
  return `
    <div class="ai-assistant-chat" id="ai-assistant-chat">
      ${renderStoredAIAssistantMessages()}
    </div>`;
}

function formatAssistantText(text = '') {
  const normalized = String(text || '').replace(/\r/g, '').trim();
  if (!normalized) return '<p>Ответ пустой.</p>';
  const paragraphs = normalized.split(/\n{2,}/).map(block => block.trim()).filter(Boolean);
  if (!paragraphs.length) return `<p>${escapeHtml(normalized)}</p>`;
  return paragraphs.map(block => {
    const lines = block.split('\n').map(x => x.trim()).filter(Boolean);
    if (!lines.length) return '';
    const bulletLike = lines.every(line => /^[-•*]|^\d+[\.)]/.test(line));
    if (bulletLike) {
      return `<ul>${lines.map(line => `<li>${escapeHtml(line.replace(/^[-•*]\s*/, '').replace(/^\d+[\.)]\s*/, ''))}</li>`).join('')}</ul>`;
    }
    return lines.map(line => `<p>${escapeHtml(line)}</p>`).join('');
  }).join('');
}

function getAssistantActions(text = '', analytics = null) {
  const a = analytics || buildSmartAnalytics();
  const summary = a.summary || {};
  const lower = String(text || '').toLowerCase();
  const actions = [];
  const pushAction = (label, action, arg, dismissModal) => {
    if (!label || !action) return;
    if (actions.some(item => item.label === label)) return;
    actions.push({ label, action, arg: arg || null, dismissModal: !!dismissModal });
  };

  // Detect intent from AI response text
  const mentionsDrafts = /чернов|пост|редакт|написа/i.test(lower);
  const mentionsPlan = /план|иде[ияй]|генер/i.test(lower);
  const mentionsAutopost = /автопост|автопилот|расписан|слот|публикац/i.test(lower);
  const mentionsChannels = /канал|подключ/i.test(lower);
  const mentionsProfile = /профил|стиль|аудитор|ниш|тон|onboarding/i.test(lower);
  const mentionsAnalytics = /аналитик|статистик|готовност/i.test(lower);
  const mentionsMedia = /медиа|фото|изображ|видео/i.test(lower);

  // Only show actions when the response is clearly action-oriented
  const isActionOriented = /сделать|открой|перейд|добав|создай|настро|включ|выключ|запуст|что делать/i.test(lower);
  const hasSpecificAdvice = /вкладк|раздел|кнопк/i.test(lower);

  if (!isActionOriented && !hasSpecificAdvice) {
    // Pure informational response — no buttons needed
    return [];
  }

  // Context-specific actions based on what the AI mentions
  if (mentionsDrafts) pushAction('Открыть посты', 'switchTab', 'posts', true);
  if (mentionsPlan) pushAction('Открыть план', 'switchTab', 'plan', true);
  if (mentionsAutopost) pushAction('Открыть автопост', 'switchTab', 'autopost', true);
  if (mentionsChannels) pushAction('Открыть каналы', 'switchTab', 'channels', true);
  if (mentionsProfile) pushAction('Профиль канала', 'openChannelProfile', null, true);
  if (mentionsAnalytics) pushAction('Аналитика', 'openAnalyticsDetails', null, true);

  // If no specific section was mentioned, suggest based on weakest signal
  if (!actions.length) {
    const weakest = (a.signals || []).slice().sort((x, y) => Number(x.value || 0) - Number(y.value || 0))[0] || null;
    if (weakest?.key === 'content_reserve' || Number(summary.plan_count || 0) < 5) pushAction('Открыть план', 'switchTab', 'plan', true);
    else if (Number(summary.drafts_count || 0) < 2) pushAction('Открыть посты', 'switchTab', 'posts', true);
    else if (weakest?.key === 'autopost') pushAction('Открыть автопост', 'switchTab', 'autopost', true);
    else pushAction('Аналитика', 'openAnalyticsDetails', null, true);
  }

  return actions.slice(0, 2);
}

function buildAssistantActionsHtml(text = '', analytics = null) {
  const actions = getAssistantActions(text, analytics);
  if (!actions.length) return '';
  return `<div class="ai-assistant-actions">${actions.map(item => {
    let attrs = `data-action="${item.action}"`;
    if (item.arg) attrs += ` data-action-arg="${escapeHtml(item.arg)}"`;
    if (item.dismissModal) attrs += ` data-dismiss-modal`;
    return `<button class="ai-action-btn" type="button" ${attrs}>${escapeHtml(item.label)}</button>`;
  }).join('')}</div>`;
}

function appendAIAssistantMessage(role, text, extraClass = '', analytics = null) {
  const chat = document.getElementById('ai-assistant-chat');
  if (!chat) return null;
  const cls = role === 'user' ? 'user' : 'ai';
  const node = document.createElement('div');
  node.className = `ai-assistant-msg ${cls} ${extraClass}`.trim();
  const isThinking = extraClass.includes('is-thinking');
  const body = isThinking
    ? `<span class="typing-dot-1"></span><span class="typing-dot-2"></span><span class="typing-dot-3"></span>`
    : (role === 'user' ? `<p>${escapeHtml(text || '')}</p>` : formatAssistantText(text || ''));
  const actions = role === 'ai' && !isThinking ? buildAssistantActionsHtml(text || '', analytics) : '';
  node.innerHTML = `<div class="ai-assistant-bubble">${body}${actions}</div>`;
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
  return node;
}

function openAIAssistant(prefill = '') {
  const body = `
    <div class="ai-assistant-shell">
      <div class="ai-assistant-chat-wrap">${renderAIAssistantMessages()}</div>
      <div class="ai-assistant-composer">
        <div class="ai-assistant-input-row">
          <input class="input ai-assistant-input" id="ai-assistant-input" placeholder="Спроси: что улучшить, как настроить..." value="${escapeHtml(prefill)}">
          <button class="btn primary ai-assistant-send" type="button" data-action="askAIAssistant">Отправить</button>
        </div>
      </div>
    </div>`;
  modal('ИИ-помощник', body);
  const root = document.getElementById('modal-root');
  if (root) root.classList.add('ai-assistant-backdrop');
  const modalNode = document.querySelector('#modal-root .modal');
  if (modalNode) modalNode.classList.add('ai-assistant-modal');
  const bodyNode = document.querySelector('#modal-root .modal-body-stack');
  if (bodyNode) bodyNode.classList.add('ai-assistant-modal-body');
  syncViewportHeight();
  requestAnimationFrame(() => {
    const chat = document.getElementById('ai-assistant-chat');
    if (chat) chat.scrollTop = chat.scrollHeight;
  });
  setTimeout(() => {
    const input = document.getElementById('ai-assistant-input');
    if (input) {
      input.focus();
      input.onkeydown = (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          askAIAssistant();
        }
      };
    }
  }, 60);
}

function buildAIAssistantFallback(question = '') {
  const q = String(question || '').trim().toLowerCase();
  const analytics = buildSmartAnalytics();
  const summary = analytics.summary || {};
  const channel = activeChannel();
  const draftsCount = Number(summary.drafts_count || 0);
  const planCount = Number(summary.plan_count || 0);
  const mediaCount = Number(summary.media_count || 0);
  const scheduleCount = Number(summary.schedule_count || 0);
  const score = Number(analytics.score || 0);
  const weakest = (analytics.signals || []).slice().sort((a, b) => Number(a.value || 0) - Number(b.value || 0))[0] || null;

  if (!q) return `Готовность канала ${score}%. В запасе ${draftsCount} черновиков, ${planCount} идей и ${mediaCount} медиа.`;
  if (q.includes('автопилот') || q.includes('автопост')) {
    if (!channel) return 'Сначала подключи и активируй канал. Без активного канала автопилот не запустится.';
    if (scheduleCount < 1) return 'Сейчас нет слотов публикации. Добавь хотя бы один слот в настройках.';
    return `Автопилот сейчас на уровне ${score}%. Слабое место: ${weakest?.label || 'аналитика'}. ${weakest?.action || ''}`.trim();
  }
  if (q.includes('чернов')) return draftsCount > 0 ? `Сейчас ${draftsCount} черновиков. Для устойчивости держи хотя бы 2–4.` : 'Черновиков нет. Создай хотя бы 2 заготовки.';
  if (q.includes('план') || q.includes('иде')) return planCount > 0 ? `В плане ${planCount} идей. Нормальный запас — 5–10.` : 'План пустой. Сгенерируй хотя бы 5 идей.';
  if (q.includes('медиа')) return mediaCount > 0 ? `В медиарезерве ${mediaCount} файлов. Держи запас под ближайшие публикации.` : 'Медиарезерв пустой. Добавь несколько изображений или видео.';
  if (q.includes('канал')) return channel ? `Сейчас активен канал «${resolveChannelLabel(channel.title || channel.channel_target || '')}».` : 'Активный канал не выбран.';

  const parts = [];
  parts.push(`Канал сейчас готов на ${score}%.`);
  if (weakest) {
    parts.push(`Слабее всего сейчас «${weakest.label}» — ${weakest.value}%.`);
    if (weakest.action) parts.push(`Что сделать: ${weakest.action}.`);
  }
  parts.push(`В запасе ${draftsCount} черновиков, ${planCount} идей и ${mediaCount} медиа.`);
  return parts.join(' ');
}

async function askAIAssistant(forcedQuestion = '') {
  const input = document.getElementById('ai-assistant-input');
  const question = String(forcedQuestion || input?.value || '').trim();
  const chat = document.getElementById('ai-assistant-chat');
  if (!chat) {
    openAIAssistant(question);
    return;
  }
  if (!question) return;

  appendAIAssistantMessage('user', question);
  pushAIAssistantSession('user', question);
  if (input) input.value = '';

  const thinking = appendAIAssistantMessage('ai', 'Смотрю данные по каналу…', 'is-thinking');
  try {
    const payload = await api('/api/assistant/chat', {
      method: 'POST',
      body: JSON.stringify({ question, session_history: loadAIAssistantSession().slice(-5) })
    });
    const responseText = String(payload?.text || '').trim() || buildAIAssistantFallback(question);
    if (thinking) thinking.remove();
    appendAIAssistantMessage('ai', responseText, '', payload?.analytics || null);
    pushAIAssistantSession('ai', responseText, payload?.analytics || null);
  } catch (e) {
    if (thinking) thinking.remove();
    const fallbackText = buildAIAssistantFallback(question);
    appendAIAssistantMessage('ai', fallbackText);
    pushAIAssistantSession('ai', fallbackText, null);
    toast(e.message || 'Не удалось получить ответ помощника');
  }
}

function openDraftEditor(draftId = null) {
  const draft = draftId ? (state.data?.drafts || []).find(d => Number(d.id) === Number(draftId)) : null;
  const saved = loadUiState();
  state.editorOpen = true;
  modal(
    draft ? 'Редактор черновика' : 'Новый пост',
    draftEditorHtml(draft),
    draft
      ? `<div class="editor-footer-pair"><button class="btn ghost" data-action="previewDraft" data-action-arg="${draft.id}">Предпросмотр</button><button class="btn secondary" data-action="publishDraft" data-action-arg="${draft.id}">Опубликовать</button></div><button class="btn primary full" data-action="saveDraft" data-action-arg="${draft.id}">Сохранить</button>`
      : `<div class="editor-footer-pair"><button class="btn ghost" data-action="previewEditorDraft">Предпросмотр</button><button class="btn ghost" data-action="resetEditorDraft">Очистить</button></div><button class="btn primary full" data-action="createDraft">Сохранить черновик</button>`
  );
  bindDraftEditorEvents();
  refreshEditorButtonsList();
  refreshEditorMediaPreview();
  syncEditorPreview();
  initEditorHistory(getCurrentEditorSnapshot());
  state.lastAutosavedHash = editorSnapshotHash(getCurrentEditorSnapshot() || {});
  if (saved?.editor && draft) {
    const sameDraft = Number(saved.editor.draftId || 0) === Number(draft?.id || 0);
    if (sameDraft) {
      applyEditorSnapshot(saved.editor);
    }
  }
  saveUiState();
}


async function addHashtagsToEditor() {
  const textEl = document.getElementById('dr-text');
  const promptEl = document.getElementById('dr-prompt');
  const text = textEl?.value?.trim() || '';
  const prompt = promptEl?.value?.trim() || '';
  if (!text && !prompt) {
    toast('Сначала добавь текст или задачу для ИИ');
    return;
  }
  try {
    showBusy('Добавляю хэштеги…');
    const payload = await api('/api/ai/add-hashtags', {
      method: 'POST',
      body: JSON.stringify({
        text,
        prompt,
        topic: document.getElementById('dr-topic')?.value || '',
      })
    });
    const nextText = payload?.text || payload?.result || payload?.draft?.text || payload?.content || '';
    if (!nextText) throw new Error('Сервис не вернул обновлённый текст');
    if (textEl) textEl.value = nextText;
    syncEditorPreview();
    toast('Хэштеги добавлены');
  } catch (e) {
    toast(e.message || 'Не удалось добавить хэштеги');
  } finally {
    hideBusy();
  }
}

async function runEditorAIGeneration() {
  const prompt = document.getElementById('dr-prompt')?.value?.trim() || '';
  const rawMediaRef = document.getElementById('dr-media-ref')?.value?.trim() || '';
  const rawMediaType = document.getElementById('dr-media-type')?.value || 'none';
  const draftId = Number(document.getElementById('dr-id')?.value || 0) || 0;
  if (!prompt) { toast('Опиши, какой пост нужен ИИ'); return; }

  const isUnsplashAutoMedia = /^https:\/\/images\.unsplash\.com\//i.test(rawMediaRef);
  const mediaRef = (!draftId && isUnsplashAutoMedia) ? '' : rawMediaRef;
  const mediaType = (!draftId && isUnsplashAutoMedia) ? 'none' : rawMediaType;
  const mediaMetaJson = (!draftId && isUnsplashAutoMedia) ? '' : (document.getElementById('dr-media-meta')?.value || '');

  if (!draftId && isUnsplashAutoMedia) {
    const mediaEl = document.getElementById('dr-media-ref');
    const typeEl = document.getElementById('dr-media-type');
    const metaEl = document.getElementById('dr-media-meta');
    if (mediaEl) mediaEl.value = '';
    if (typeEl) typeEl.value = 'none';
    if (metaEl) metaEl.value = '';
  }

  try {
    showBusy('Генерирую пост…');
    const payload = await api('/api/ai/generate-post', {
      method: 'POST',
      body: JSON.stringify({ prompt, force_image: true, draft_id: draftId, current_text: document.getElementById('dr-text')?.value || '', media_ref: mediaRef, media_type: mediaType, media_meta_json: mediaMetaJson, buttons_json: document.getElementById('dr-buttons')?.value || '[]', pin_post: document.getElementById('dr-pin')?.checked ? 1 : 0, comments_enabled: document.getElementById('dr-comments')?.checked ? 1 : 0, ad_mark: document.getElementById('dr-ad')?.checked ? 1 : 0 }),
      timeoutMs: 120000
    });
    const textEl = document.getElementById('dr-text');
    if (textEl) textEl.value = payload?.draft?.text || payload?.text || '';
    renderEditorAssetsPanel(payload || {});
    const btnTextInput = document.getElementById('dr-btn-text');
    if (btnTextInput && (payload?.button_text || '')) btnTextInput.value = payload.button_text || '';
    const mediaEl = document.getElementById('dr-media-ref');
    if (mediaEl && (payload?.media_ref || payload?.draft?.media_ref)) mediaEl.value = payload.media_ref || payload.draft.media_ref || '';
    const mediaMetaEl = document.getElementById('dr-media-meta');
    if (mediaMetaEl) mediaMetaEl.value = payload?.media_meta_json || payload?.draft?.media_meta_json || '';
    const typeEl = document.getElementById('dr-media-type');
    if (typeEl && (payload?.media_type || payload?.draft?.media_type)) typeEl.value = payload.media_type || payload.draft.media_type || 'photo';
    refreshEditorMediaPreview();
    maybeCaptureEditorHistory();
    syncEditorPreview();
    scheduleEditorAutosave(true);
    const toastMsg = payload?.warning || payload?.image_warning || 'Готово';
    toast(toastMsg);
    if (payload?.warning) {
      const marker = document.getElementById('editor-autosave-status');
      if (marker) marker.textContent = `⚠ ${payload.warning}`;
    }
  } catch (e) {
    if (e.status === 402) return;
    // Handle structured generation_failed error with retry UI
    if (e.status === 422 && e.message && (e.message.includes('generation_failed') || e.message.includes('качественный пост'))) {
      // Extract human-readable reason if available from the error message
      let reason = '';
      try {
        // The server returns detail as JSON: {code, message, reason}
        // FastAPI serializes it; our fetch wrapper puts detail text in e.message
        const m = e.message || '';
        // Try to find a parenthesized reason like "(Текст не прошёл проверку качества)"
        const match = m.match(/\(([^)]+)\)/);
        if (match) reason = match[1];
      } catch (_) {}
      _showGenerationFailedModal(reason);
      return;
    }
    toast(e.message || 'Не удалось сгенерировать пост');
  } finally {
    hideBusy();
  }
}

/** Show a user-friendly modal when generation quality gate fails with a retry button. */
function _showGenerationFailedModal(reason) {
  const reasonText = reason
    ? `Причина: ${reason}.`
    : 'ИИ не смог создать достаточно качественный текст по вашему запросу.';
  const bodyHtml = `
    <div class="paywall-modal-body">
      <div class="paywall-icon">✍️</div>
      <div class="paywall-title">Не удалось сгенерировать пост</div>
      <div class="paywall-message">${reasonText} Попробуйте ещё раз — каждая попытка использует новый угол подачи. Или измените запрос для лучшего результата.</div>
    </div>
  `;
  const actionsHtml = `<button class="btn primary" id="gen-retry-btn">Попробовать ещё раз</button><button class="btn ghost" data-action="closeModal">Закрыть</button>`;
  modal('Генерация не удалась', bodyHtml, actionsHtml);
  setTimeout(() => {
    const retryBtn = document.getElementById('gen-retry-btn');
    if (retryBtn) retryBtn.onclick = () => { closeModal(); generatePostInEditor(); };
  }, 50);
}


async function generatePostInEditor() {
  try {
    await runEditorAIGeneration();
  } catch (e) {
    if (e.status === 402) return;
    toast(e?.message || 'Не удалось сгенерировать пост');
  } finally {
    hideBusy();
  }
}


async function generatePostFromPlan(planItemId) {
  const item = visiblePlanItems().find(x => Number(x.id) === Number(planItemId));
  if (!item) { toast('Идея из плана не найдена'); return; }
  const prompt = String(item.prompt || item.topic || item.idea || '').trim();
  if (!prompt) { toast('У этой идеи нет текста'); return; }
  openDraftEditor();
  await new Promise(r => setTimeout(r, 120));
  const promptEl = document.getElementById('dr-prompt');
  if (promptEl) promptEl.value = prompt;
  scheduleEditorAutosave(false);
}

function renderEditorAssetsPanel(bundle = {}) {
  const root = document.getElementById('editor-ai-assets');
  if (!root) return;
  const title = String(bundle.title || '').trim();
  const cta = String(bundle.cta || '').trim();
  const short = String(bundle.short || '').trim();
  const buttonText = String(bundle.button_text || '').trim();
  if (!title && !cta && !short && !buttonText) {
    root.innerHTML = '';
    return;
  }
  root.innerHTML = `
    <div class="ai-assets-grid">
      <div class="ai-asset-chip"><b>Заголовок</b><span>${escapeHtml(title || '—')}</span></div>
      <div class="ai-asset-chip"><b>Призыв</b><span>${escapeHtml(cta || '—')}</span></div>
      <div class="ai-asset-chip wide"><b>Короткая версия</b><span>${escapeHtml(short || '—')}</span></div>
      <div class="ai-asset-chip"><b>Текст кнопки</b><span>${escapeHtml(buttonText || '—')}</span></div>
    </div>`;
}

async function rewriteEditorText(mode = 'improve') {
  const textEl = document.getElementById('dr-text');
  const promptEl = document.getElementById('dr-prompt');
  const text = textEl?.value?.trim() || '';
  if (!text) { toast('Сначала нужен текст поста'); return; }
  try {
    const labels = { improve: 'Улучшаю текст…', shorter: 'Сокращаю…', selling: 'Усиливаю подачу…' };
    showBusy(labels[mode] || 'Редактирую текст…');
    const payload = await api('/api/ai/rewrite', {
      method: 'POST',
      body: JSON.stringify({
        text,
        prompt: promptEl?.value || '',
        topic: document.getElementById('dr-topic')?.value || '',
        mode,
      })
    });
    if (textEl) textEl.value = payload?.text || text;
    maybeCaptureEditorHistory();
    syncEditorPreview();
    scheduleEditorAutosave(true);
    toast('Текст обновлён');
  } catch (e) {
    toast(e.message || 'Не удалось обновить текст');
  } finally { hideBusy(); }
}

async function fillEditorAssets() {
  const text = document.getElementById('dr-text')?.value?.trim() || '';
  const prompt = document.getElementById('dr-prompt')?.value?.trim() || '';
  if (!text && !prompt) { toast('Сначала нужна тема или текст'); return; }
  try {
    showBusy('Собираю заголовок, CTA и короткую версию…');
    const payload = await api('/api/ai/assets', {
      method: 'POST',
      body: JSON.stringify({ text, prompt, topic: document.getElementById('dr-topic')?.value || '' })
    });
    renderEditorAssetsPanel(payload || {});
    scheduleEditorAutosave(true);
    toast(payload?.image_warning || 'Готово');
  } catch (e) {
    toast(e.message || 'Не удалось собрать дополнительные элементы');
  } finally { hideBusy(); }
}

function toggleAdvanced() {
  const box = document.getElementById('advanced-box');
  if (box) box.classList.toggle('open', document.getElementById('adv-toggle').checked);
}

function setEditorMediaType(type) {
  const typeInput = document.getElementById('dr-media-type');
  if (typeInput) typeInput.value = type;
  document.querySelectorAll('[data-set-media]').forEach(btn => btn.classList.toggle('active', btn.dataset.setMedia === type));
  if (type === 'none') {
    const ref = document.getElementById('dr-media-ref');
    if (ref) ref.value = '';
    const meta = document.getElementById('dr-media-meta');
    if (meta) meta.value = '';
  }
  refreshEditorMediaPreview();
  syncEditorPreview();
}

function refreshEditorMediaPreview() {
  const mediaRef = normalizeMediaRef(document.getElementById('dr-media-ref')?.value?.trim() || '');
  const mediaType = document.getElementById('dr-media-type')?.value || guessMediaType(mediaRef);
  const box = document.getElementById('dr-media-preview');
  const status = document.getElementById('dr-upload-status');
  const attr = document.getElementById('dr-media-attribution');
  if (status) status.textContent = mediaRef ? 'Медиа выбрано' : 'Медиа не выбрано';
  if (!box) return;
  box.classList.remove('is-broken');
  box.innerHTML = renderMediaNode(mediaRef, mediaType);
  let meta = null;
  try { meta = JSON.parse(document.getElementById('dr-media-meta')?.value || 'null'); } catch {}
  if (attr) {
    if (meta?.attribution_html) attr.textContent = meta.attribution_html;
    else if (meta?.attribution_text) attr.textContent = meta.attribution_text;
    else attr.textContent = '';
  }
}

function clearEditorMedia() {
  const ref = document.getElementById('dr-media-ref');
  if (ref) ref.value = '';
  const meta = document.getElementById('dr-media-meta');
  if (meta) meta.value = '';
  setEditorMediaType('none');
}

function resetEditorDraft() {
  ['dr-text','dr-prompt','dr-media-ref','dr-buttons'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = id === 'dr-buttons' ? '[]' : '';
  });
  const idEl = document.getElementById('dr-id');
  if (idEl) idEl.value = '0';
  const type = document.getElementById('dr-media-type');
  if (type) type.value = 'none';
  ['dr-pin','dr-ad'].forEach(id => { const el = document.getElementById(id); if (el) el.checked = false; });
  const comments = document.getElementById('dr-comments'); if (comments) comments.checked = true;
  const upload = document.getElementById('dr-file'); if (upload) upload.value = '';
  clearAwaitMediaState();
  clearEditorUiState();
  refreshEditorButtonsList();
  refreshEditorMediaPreview();
  syncEditorPreview();
  initEditorHistory(getCurrentEditorSnapshot());
  saveUiState();
  toast('Редактор очищен');
}

function undoEditorChange() { stepEditorHistory('undo'); }
function redoEditorChange() { stepEditorHistory('redo'); }

function collectEditorPreviewDraft() {
  return {
    text: document.getElementById('dr-text')?.value || '',
    prompt: document.getElementById('dr-prompt')?.value || '',
    channel_target: document.getElementById('dr-channel')?.value || activeChannel()?.title || '',
    media_ref: normalizeMediaRef(document.getElementById('dr-media-ref')?.value || ''),
    media_type: document.getElementById('dr-media-type')?.value || 'none',
    media_meta_json: document.getElementById('dr-media-meta')?.value || '',
    buttons_json: document.getElementById('dr-buttons')?.value || '[]',
    pin_post: document.getElementById('dr-pin')?.checked ? 1 : 0,
    comments_enabled: document.getElementById('dr-comments')?.checked ? 1 : 0,
    ad_mark: document.getElementById('dr-ad')?.checked ? 1 : 0,
  };
}

function syncEditorPreview() {
  const box = document.getElementById('editor-live-preview');
  if (!box) return;
  box.innerHTML = renderLivePreviewCard(collectEditorPreviewDraft());
}

function readDraftForm() {
  return {
    text: document.getElementById('dr-text').value,
    prompt: document.getElementById('dr-prompt')?.value || '',
    channel_target: document.getElementById('dr-channel').value,
    topic: document.getElementById('dr-topic')?.value || '',
    media_type: document.getElementById('dr-media-type')?.value || (document.getElementById('dr-media-ref').value ? 'photo' : 'none'),
    media_ref: normalizeMediaRef(document.getElementById('dr-media-ref').value),
    media_meta_json: document.getElementById('dr-media-meta')?.value || '',
    buttons_json: document.getElementById('dr-buttons')?.value || '[]',
    pin_post: document.getElementById('dr-pin')?.checked ? 1 : 0,
    comments_enabled: document.getElementById('dr-comments')?.checked ? 1 : 0,
    ad_mark: document.getElementById('dr-ad')?.checked ? 1 : 0,
  };
}

async function createDraft() {
  try {
    const body = readDraftForm();
    if (!body.text.trim() && !body.prompt.trim()) return toast('Добавь текст поста или промпт');
    showBusy('Сохраняю черновик…');
    const payload = await api('/api/drafts', { method: 'POST', body: JSON.stringify(body) });
    state.lastAutosavedHash = editorSnapshotHash(getCurrentEditorSnapshot() || {});
    clearAwaitMediaState();
    clearEditorUiState();
    closeModal();
    toast('Черновик создан');
    await refreshSections(['core','drafts','media_inbox'], { silent: false });
    switchTab('posts');
  } catch (e) {
    if ((e.message || '').includes('лимит черновиков')) showDraftLimitModal();
    else toast(e.message);
  } finally {
    hideBusy();
  }
}

async function saveDraft(id) {
  if (state.autosaveTimer) { clearTimeout(state.autosaveTimer); state.autosaveTimer = null; }
  try {
    const body = readDraftForm();
    showBusy('Сохраняю черновик…');
    await api(`/api/drafts/${id}`, { method: 'PATCH', body: JSON.stringify(body) });
    state.lastAutosavedHash = editorSnapshotHash(getCurrentEditorSnapshot() || {});
    clearAwaitMediaState();
    clearEditorUiState();
    closeModal();
    setInlineStatus('editor-inline-status', 'Черновик обновлён', 'success'); toast('Черновик обновлён');
    await refreshSections(['core','drafts','media_inbox'], { silent: false });
    switchTab('posts');
  } catch (e) {
    toast(e.message);
  } finally {
    hideBusy();
  }
}

function removeDraftFromState(id) {
  const draftId = Number(id);
  if (!Array.isArray(state.data?.drafts)) return null;
  const idx = state.data.drafts.findIndex(x => Number(x?.id) === draftId);
  if (idx < 0) return null;
  const [removed] = state.data.drafts.splice(idx, 1);
  if (typeof state.data.drafts_current === 'number') state.data.drafts_current = Math.max(0, state.data.drafts_current - 1);
  return removed;
}

function restoreDraftToState(draft) {
  if (!draft) return;
  state.data = state.data || {};
  state.data.drafts = Array.isArray(state.data.drafts) ? state.data.drafts : [];
  state.data.drafts.unshift(draft);
  state.data.drafts_current = Number(state.data.drafts_current || 0) + 1;
  render();
}

function removePlanFromState(id) {
  const itemId = Number(id);
  if (!Array.isArray(state.data?.plan)) return null;
  const idx = state.data.plan.findIndex(x => Number(x?.id) === itemId);
  if (idx < 0) return null;
  const [removed] = state.data.plan.splice(idx, 1);
  return removed;
}

function restorePlanToState(item) {
  if (!item) return;
  state.data = state.data || {};
  state.data.plan = Array.isArray(state.data.plan) ? state.data.plan : [];
  state.data.plan.unshift(item);
  render();
}

async function deleteDraft(id) {
  if (!(await confirmActionModal('Удалить черновик?', 'Удалить черновик'))) return;
  const draftId = Number(id);
  if (state.pendingDeletedDraftIds.has(draftId)) return;
  removeDraftFromState(draftId);
  state.pendingDeletedDraftIds.add(draftId);
  render();
  try {
    await api(`/api/drafts/${draftId}`, { method: 'DELETE' });
    state.pendingDeletedDraftIds.delete(draftId);
    toast('Черновик удалён');
    await refreshSections(['core','drafts'], { silent: true });
    render();
  } catch (e) {
    state.pendingDeletedDraftIds.delete(draftId);
    toast(e.message || 'Не удалось удалить черновик');
    await refreshSections(['core','drafts'], { silent: true });
    render();
  }
}

function previewEditorDraft() {
  modal('Предпросмотр', renderLivePreviewCard(collectEditorPreviewDraft()), `<button class="btn primary" data-action="closeModal">Закрыть</button>`);
}

function previewDraft(id) {
  const draft = (state.data?.drafts || []).find(d => Number(d.id) === Number(id));
  if (!draft) return toast('Черновик не найден');
  modal('Предпросмотр', renderLivePreviewCard(draft), `<button class="btn primary" data-action="closeModal">Закрыть</button>`);
}

async function publishDraft(id) {
  try {
    const draft = (state.data?.drafts || []).find(d => Number(d.id) === Number(id));
    if (!draft) return toast('Черновик не найден');
    if (!draft.channel_target || !String(draft.channel_target).trim()) {
      toast('Сначала выбери канал в редакторе');
      return openDraftEditor(id);
    }
    const channelLabel = resolveChannelLabel(draft.channel_target || '');
    const preview = String((draft.text || draft.prompt || '').trim()).slice(0, 180);

    // Check for mirror-eligible channels (other channels owned by user)
    const channels = state.data?.channels || [];
    const otherChannels = channels.filter(ch =>
      String(ch.channel_target || '') !== String(draft.channel_target || '') && ch.channel_target
    );

    let mirrorTargets = [];
    let confirmMsg = `Опубликовать пост в канал «${channelLabel}»?${preview ? '\n\n' + preview : ''}`;

    if (otherChannels.length > 0) {
      confirmMsg += '\n\nОпубликовать также в другие каналы?';
    }

    const confirmed = await confirmActionModal(confirmMsg, 'Опубликовать');
    if (!confirmed) return;

    // If there are other channels, show mirror selection
    if (otherChannels.length > 0) {
      mirrorTargets = await _selectMirrorChannels(otherChannels, draft.channel_target);
    }

    showBusy('Проверяю черновик и отправляю в Telegram…');
    const payload = await api('/api/drafts/publish', {
      method: 'POST',
      body: JSON.stringify({ draft_id: id, mirror_targets: mirrorTargets }),
      timeoutMs: 120000,
    });
    clearAwaitMediaState();
    closeModal();
    const publishedTo = resolveChannelLabel(payload?.channel_target || draft.channel_target || '');
    let msg = publishedTo ? `Пост опубликован в «${publishedTo}»` : 'Пост опубликован';
    if (payload?.mirror_results?.length) {
      const mirrorOk = payload.mirror_results.filter(r => r.ok).length;
      const mirrorFail = payload.mirror_results.filter(r => !r.ok).length;
      if (mirrorOk > 0) msg += ` + ${mirrorOk} зеркальных каналов`;
      if (mirrorFail > 0) msg += ` (${mirrorFail} ошибок)`;
    }
    toast(msg);
  } catch (e) {
    toast(e.message || 'Ошибка публикации в Telegram');
  } finally {
    hideBusy();
    await refreshSections(['drafts','core','media_inbox','stats'], { silent: false });
  }
}

async function _selectMirrorChannels(otherChannels, primaryTarget) {
  return new Promise(resolve => {
    // Use a unique ID to avoid conflicts with simultaneous modals
    const resolveId = '_mirror_' + Date.now();
    const body = `
      <div class="mirror-channel-select">
        <p class="mirror-desc">Выберите каналы для зеркальной публикации:</p>
        <div class="mirror-channel-list">
          ${otherChannels.map(ch => `
            <label class="switch modern-switch mirror-channel-item">
              <input type="checkbox" class="mirror-ch-checkbox" value="${escapeHtml(ch.channel_target)}" />
              <span class="switch-ui"></span>
              <span>${escapeHtml(resolveChannelLabel(ch.title || ch.channel_target || ''))}</span>
            </label>
          `).join('')}
        </div>
      </div>
    `;
    function cleanup() {
      delete window[resolveId + '_ok'];
      delete window[resolveId + '_skip'];
    }
    window[resolveId + '_ok'] = () => {
      const checks = document.querySelectorAll('.mirror-ch-checkbox:checked');
      const targets = Array.from(checks).map(c => c.value);
      cleanup();
      closeModal();
      resolve(targets);
    };
    window[resolveId + '_skip'] = () => {
      cleanup();
      closeModal();
      resolve([]);
    };
    const actions = `
      <button class="btn primary" data-resolve-id="${resolveId}" data-resolve-action="ok">Продолжить</button>
      <button class="btn ghost" data-resolve-id="${resolveId}" data-resolve-action="skip">Пропустить</button>
    `;
    modal('Зеркальная публикация', body, actions);
  });
}


function backToChat() {
  saveUiState();
  if (tg?.close) {
    tg.close();
    return;
  }
  history.back();
}

async function refreshInbox() {
  await refreshSections(['media_inbox'], { silent: false });
  toast('Список обновлён');
  if (state.activeTab !== 'posts') switchTab('posts');
}

function openChatPicker() {
  const inbox = getMediaInbox();
  startAwaitMediaForEditor();
  const body = `
    <div class="note">1. Нажми кнопку ниже и отправь видео боту в чат.<br>2. Вернись в Mini App.<br>3. Новое видео попробует прикрепиться к текущему черновику автоматически.</div>
    <div class="item-actions">
      <button class="btn primary" data-action="backToChat" data-dismiss-modal>Открыть чат</button>
      <button class="btn ghost" data-action="refreshInbox" data-dismiss-modal="true">Обновить список</button>
    </div>
    ${inbox.length ? `<div class="media-grid compact" style="margin-top:8px">${inbox.map(mediaInboxCard).join('')}</div>` : '<div class="empty" style="margin-top:8px">Видео из чата пока нет.</div>'}
  `;
  modal('Медиатека из чата', body);
}

async function createDraftFromInbox(itemId) {
  try {
    showBusy('Открываю видео в редакторе…');
    const payload = await api(`/api/media/inbox/${itemId}/draft`, { method: 'POST', body: JSON.stringify({}) });
    clearAwaitMediaState();
    closeModal();
    toast('Видео добавлено в новый черновик');
    await refreshSections(['core','drafts','media_inbox'], { silent: false });
    switchTab('posts');
    const draftId = Number(payload?.draft_id || payload?.id || payload?.draft?.id || 0);
    if (draftId) openDraftEditor(draftId);
  } catch (e) {
    if ((e.message || '').includes('лимит черновиков')) showDraftLimitModal();
    else toast(e.message || 'Не удалось выбрать видео');
  } finally {
    hideBusy();
  }
}

function openGenerateDraftModal() {
  modal('Сгенерировать черновик', `
    <div class="field">
      <div class="label">Что нужно сгенерировать</div>
      <textarea class="textarea" id="gd-prompt" placeholder="Например: пост про тренды в полупроводниках для Telegram-канала"></textarea>
    </div>
    <div class="note">Готовый черновик появится в списке сразу после генерации.</div>
  `, `<button class="btn primary" data-action="generateDraft">Создать черновик</button>`);
}

async function generateDraft() {
  try {
    showBusy('Генерирую черновик…');
    await api('/api/drafts/generate', { method: 'POST', body: JSON.stringify({ prompt: document.getElementById('gd-prompt').value }), timeoutMs: 120000 });
    closeModal();
    toast('ИИ-черновик создан');
    await refreshSections(['core','drafts','media_inbox'], { silent: false });
    switchTab('posts');
  } catch (e) {
    if ((e.message || '').includes('лимит черновиков')) showDraftLimitModal();
    else toast(e.message);
  } finally {
    hideBusy();
  }
}

async function deleteInboxMedia(itemId) {
  if (!confirmAction('Удалить это видео из мини-приложения?')) return;
  try {
    await api(`/api/media/inbox/${itemId}`, { method: 'DELETE' });
    await refreshSections(['core','media_inbox'], { silent: false });
    toast('Видео удалено');
  } catch (e) {
    toast(e.message || 'Не удалось удалить видео');
  }
}

function openPlanGenerator() {
  modal('Генерация контент-плана', `
    <div class="grid-2">
      <div class="field"><div class="label">Стартовая дата</div><input class="input" id="pg-date" type="date"></div>
      <div class="field"><div class="label">Время первого поста</div><input class="input" id="pg-time" type="time" value="10:00"></div>
    </div>
    <div class="grid-2">
      <div class="field"><div class="label">Сколько дней</div><select class="select" id="pg-days">${[7,14,30,60].map(v => `<option value="${v}">${v}</option>`).join('')}</select></div>
      <div class="field"><div class="label">Постов в день</div><select class="select" id="pg-ppd">${[1,2,3].map(v => `<option value="${v}">${v}</option>`).join('')}</select></div>
    </div>
    <div class="field"><div class="label">Тема</div><textarea class="textarea" id="pg-topic">${escapeHtml(state.data?.settings?.topic || '')}</textarea></div>
    <div class="note">План теперь опирается на onboarding: аудиторию, стиль, режим, форматы и ограничения. Здесь лучше менять только тему, если нужен отдельный цикл публикаций.</div>
    <label class="switch"><input type="checkbox" id="pg-clear" checked> <span>Очистить старый непубликованный план</span></label>
  `, `<button class="btn primary" data-action="generatePlan">Создать план</button>`);
}

async function generatePlan() {
  try {
    showBusy('Собираю контент-план…', 'wave');
    await api('/api/plan/generate', {
      method: 'POST',
      body: JSON.stringify({
        start_date: document.getElementById('pg-date').value,
        post_time: document.getElementById('pg-time').value,
        days: Number(document.getElementById('pg-days').value),
        posts_per_day: Number(document.getElementById('pg-ppd').value),
        topic: document.getElementById('pg-topic').value,
        clear_existing: document.getElementById('pg-clear').checked,
      }),
      timeoutMs: 120000,
    });
    closeModal();
    toast('Контент-план создан');
    await refreshSections(['core','plan'], { silent: false });
    switchTab('plan');
  } catch (e) {
    toast(e.message || 'Не удалось создать контент-план');
  } finally {
    hideBusy();
  }
}

function openPlanEditor(itemId = null) {
  const item = itemId ? (state.data?.plan || []).find(p => Number(p.id) === Number(itemId)) : null;
  modal(item ? 'Редактировать элемент плана' : 'Новый элемент плана', `
    <div class="field"><div class="label">Дата и время</div><input class="input" id="pl-dt" type="datetime-local" value="${escapeHtml(item?.dt ? String(item.dt).replace(' ', 'T').slice(0,16) : '')}"></div>
    <div class="field"><div class="label">Текст / идея публикации</div><textarea class="textarea" id="pl-prompt">${escapeHtml(item?.prompt || item?.topic || '')}</textarea></div>
    <div class="note">Достаточно заполнить поле идеи. Бот сам сгенерирует полноценный пост к нужной дате.</div>
  `, item ? `<button class="btn primary" data-action="savePlanItem" data-action-arg="${item.id}">Сохранить</button>` : `<button class="btn primary" data-action="createPlanItem">Создать</button>`);
}

async function createPlanItem() {
  try {
    await api('/api/plan', {
      method: 'POST',
      body: JSON.stringify({
        dt: document.getElementById('pl-dt').value.replace('T', ' '),
        prompt: document.getElementById('pl-prompt').value,
        topic: '',
      })
    });
    closeModal();
    toast('Элемент добавлен');
    await refreshSections(['core','plan'], { silent: false });
    switchTab('plan');
  } catch (e) {
    toast(e.message);
  }
}

async function savePlanItem(id) {
  try {
    await api(`/api/plan/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({
        dt: document.getElementById('pl-dt').value.replace('T', ' '),
        prompt: document.getElementById('pl-prompt').value,
      })
    });
    closeModal();
    toast('Элемент обновлён');
    await refreshSections(['core','plan'], { silent: false });
    switchTab('plan');
  } catch (e) {
    toast(e.message);
  }
}

async function deletePlanItem(id) {
  if (!(await confirmActionModal('Удалить элемент контент-плана?', 'Удалить элемент'))) return;
  const itemId = Number(id);
  const removed = removePlanFromState(itemId);
  state.pendingDeletedPlanIds.add(itemId);
  const card = document.querySelector(`[data-plan-id="${itemId}"]`);
  if (card) {
    card.style.pointerEvents = 'none';
    await animateCardRemoval(card);
  }
  render();
  let undone = false;
  showUndoToast('Элемент плана удалён', () => {
    undone = true;
    state.pendingDeletedPlanIds.delete(itemId);
    restorePlanToState(removed);
    toast('Удаление отменено');
  });
  await new Promise(resolve => setTimeout(resolve, 4100));
  if (undone) return;
  try {
    await api(`/api/plan/${itemId}`, { method: 'DELETE' });
    state.pendingDeletedPlanIds.delete(itemId);
    toast('Элемент плана удалён окончательно');
    await refreshSections(['core','plan'], { silent: false });
  } catch (e) {
    state.pendingDeletedPlanIds.delete(itemId);
    restorePlanToState(removed);
    toast(e.message || 'Не удалось удалить элемент плана');
    await refreshSections(['core','plan'], { silent: false });
  }
}

function openScheduleModal() {
  modal('Новый слот расписания', `
    <div class="field"><div class="label">Время</div><input class="input" id="sc-time" type="time"></div>
    <div class="field"><div class="label">Дни</div><input class="input" id="sc-days" value="*" placeholder="* или mon,tue,fri"></div>
    <div class="note">Укажи <b>*</b> для каждого дня или список через запятую: mon,tue,wed...</div>
  `, `<button class="btn primary" data-action="createSchedule">Сохранить</button>`);
}

async function createSchedule() {
  try {
    await api('/api/schedules', {
      method: 'POST',
      body: JSON.stringify({
        time_hhmm: document.getElementById('sc-time').value,
        days: document.getElementById('sc-days').value,
      })
    });
    closeModal();
    toast('Слот добавлен');
    await refreshSections(['core','schedules'], { silent: false });
    switchTab('autopost');
  } catch (e) {
    if (e.status !== 402) toast(e.message);
  }
}

async function deleteSchedule(id) {
  if (!confirmAction('Удалить слот?')) return;
  try {
    await api(`/api/schedules/${id}`, { method: 'DELETE' });
    toast('Слот удалён');
    await refreshSections(['schedules','core'], { silent: false });
  } catch (e) {
    toast(e.message);
  }
}



const ONBOARDING_TOPICS = ['Технологии','Здоровье','Бизнес','Новости','Игры','Образование','Личный бренд','Другое'];
const ONBOARDING_AUTHOR_ROLES = [
  { id:'expert', title:'Эксперт / специалист', desc:'Делишься знаниями и опытом в своей нише' },
  { id:'business_owner', title:'Бизнес / услуги', desc:'Продвигаешь бизнес, продукт или услугу' },
  { id:'media', title:'Новостной / медиа', desc:'Публикуешь новости, обзоры, аналитику' },
  { id:'blogger', title:'Блогер / личный бренд', desc:'Личные мысли, наблюдения, история' },
  { id:'brand', title:'Бренд / команда', desc:'Ведёшь канал от имени компании или команды' },
  { id:'educator', title:'Образовательный', desc:'Обучаешь, объясняешь, делаешь разборы' },
];
const ONBOARDING_AUDIENCES = ['Широкая аудитория','Новички','Специалисты','Клиенты','Локальное сообщество'];
const ONBOARDING_STYLES = ['Экспертный','Простой и дружелюбный','Живой и дерзкий','Спокойный и полезный','Новостной','Вовлекающий'];
const ONBOARDING_MODES = [
  { id:'autopilot', title:'Автопилот', desc:'Сам готовит и публикует контент' },
  { id:'review', title:'С подтверждением', desc:'Готовит кандидатов, ты одобряешь' },
  { id:'drafts_only', title:'Только идеи и черновики', desc:'Без автопубликации' }
];
const ONBOARDING_FORMATS = ['Новости','Разборы','Полезные советы','FAQ','Подборки','Мифы и ошибки','Кейсы','Вовлекающие посты'];
const ONBOARDING_FREQUENCIES = [
  { id:'daily_1', title:'1 пост в день' },
  { id:'daily_2', title:'2 поста в день' },
  { id:'weekly_3_5', title:'3–5 постов в неделю' },
  { id:'flexible', title:'Гибко' }
];
const ONBOARDING_CONSTRAINTS = ['Без политики','Без кликбейта','Без длинных текстов','Без жести','Без повторов','Только проверенные источники'];
const ONBOARDING_TOTAL_STEPS = 7;

function openChannelProfile() {
  updateOnboardingFromSettings(true);
  const s = onboardingSettings();
  state.onboarding.active = true;
  state.onboarding.step = resolveOnboardingStepFromAnswers(state.onboarding.answers || {}, s);
  render();
}

function resolveOnboardingStepFromAnswers(answers = {}, settings = {}) {
  const a = answers || {};
  const topic = ((a.topic === 'Другое' ? a.customTopic : a.topic) || settings.topic || '').trim();
  if (!topic) return 1;
  if (!a.authorRole) return 2;
  if (!a.audience || !a.style) return 3;
  if (!a.mode) return 4;
  if (!Array.isArray(a.formats) || a.formats.length === 0) return 5;
  if (!a.frequency || !a.channelId) return 6;
  return 0;
}

function onboardingSettings() { return state.data?.settings || {}; }

function updateOnboardingFromSettings(force = false) {
  const s = onboardingSettings();
  const prev = force ? {} : (state.onboarding.answers || {});
  const topicFromSettings = String(s.topic || '').trim();
  const chosenTopic = prev.topic || (ONBOARDING_TOPICS.includes(topicFromSettings) ? topicFromSettings : (topicFromSettings ? 'Другое' : ''));
  const customTopic = !ONBOARDING_TOPICS.includes(topicFromSettings) ? topicFromSettings : String(prev.customTopic || '');
  // Resolve author role: if saved role is a preset ID use it, otherwise it's a custom description
  const savedRole = String(prev.authorRole || s.author_role_type || '').trim();
  const savedRoleDesc = String(prev.customAuthorRole || s.author_role_description || '').trim();
  const authorRole = savedRoleDesc || savedRole;
  state.onboarding.answers = {
    topic: chosenTopic,
    customTopic,
    authorRole,
    customAuthorRole: ONBOARDING_AUTHOR_ROLES.some(r => r.id === authorRole) ? '' : authorRole,
    audience: String(prev.audience || s.channel_audience || ''),
    style: String(prev.style || s.channel_style_preset || ''),
    mode: String(prev.mode || s.channel_mode || ''),
    frequency: String(prev.frequency || s.channel_frequency || ''),
    formats: Array.isArray(prev.formats) && prev.formats.length ? prev.formats : (safeJsonParse(s.channel_formats, []) || []),
    constraints: Array.isArray(prev.constraints) && prev.constraints.length ? prev.constraints : (safeJsonParse(s.content_constraints, []) || []),
    channelId: Number(prev.channelId || activeChannel()?.id || 0) || null,
  };
  state.onboarding.step = Math.max(0, Math.min(ONBOARDING_TOTAL_STEPS - 1, Number(state.onboarding.step || 0)));
}


function onboardingCanContinue(step) {
  const a = state.onboarding.answers || {};
  if (step === 0) return true;
  if (step === 1) return !!((a.topic === 'Другое' ? a.customTopic : a.topic) || '').trim();
  if (step === 2) return !!a.authorRole;
  if (step === 3) return !!a.audience && !!a.style;
  if (step === 4) return !!a.mode;
  if (step === 5) return Array.isArray(a.formats) && a.formats.length > 0;
  if (step === 6) return !!a.frequency;
  return true;
}

function renderChoiceCards(items, selected, handler, multi = false) {
  return `<div class="onboarding-options-grid">${
    items.map(item => {
      const value = typeof item === 'string' ? item : item.id;
      const title = typeof item === 'string' ? item : item.title;
      const desc = typeof item === 'string' ? '' : (item.desc || '');
      const active = multi ? (Array.isArray(selected) && selected.includes(value)) : (selected === value);
      return `<button type="button" class="ob-choice-card ${active ? 'selected' : ''}" data-ob-handler="${escapeHtml(handler)}" data-ob-value="${escapeHtml(String(value))}"><div class="ob-choice-title">${escapeHtml(title)}</div>${desc ? `<div class="ob-choice-desc">${escapeHtml(desc)}</div>` : ''}${active ? `<span class="ob-choice-check">✓</span>` : ''}</button>`;
    }).join('')
  }</div>`;
}


function renderChannelChoices() {
  const channels = state.data?.channels || [];
  if (!channels.length) {
    return `<div class="onboarding-empty-note"><div class="note-title">Сначала добавь канал</div><div class="note-text">Финальный шаг нельзя завершить без хотя бы одного подключённого канала.</div></div>`;
  }
  return `<div class="onboarding-options-grid">${
    channels.map(ch => `<button type="button" class="ob-choice-card ${Number(state.onboarding.answers.channelId || 0) === Number(ch.id) ? 'selected' : ''}" data-ob-handler="onboardingSelectChannel" data-ob-value="${Number(ch.id)}"><div class="ob-choice-title">${escapeHtml(resolveChannelLabel(ch.title || ch.channel_target || ''))}</div><div class="ob-choice-desc">${escapeHtml(/^-?\d{6,}$/.test(String(ch.channel_target || '')) ? '' : (ch.channel_target || ''))}</div>${Number(state.onboarding.answers.channelId || 0) === Number(ch.id) ? `<span class="ob-choice-check">✓</span>` : ''}</button>`).join('')
  }</div>`;
}

function renderDualChoiceColumns(leftTitle, leftHtml, rightTitle, rightHtml) {
  return `
    <div class="onboarding-dual-grid">
      <div class="onboarding-dual-col">
        <div class="onboarding-dual-title">${escapeHtml(leftTitle)}</div>
        ${leftHtml}
      </div>
      <div class="onboarding-dual-col">
        <div class="onboarding-dual-title">${escapeHtml(rightTitle)}</div>
        ${rightHtml}
      </div>
    </div>
  `;
}

function renderOnboardingStep() {
  const s = Number(state.onboarding.step || 0);
  const a = state.onboarding.answers || {};
  if (s === 0) {
    return `<div class="ob-hero"><div class="ob-kicker">NeuroSMM</div><h1 class="ob-title">Настроим канал один раз — дальше бот возьмёт рутину на себя</h1><p class="ob-subtitle">Всего ${ONBOARDING_TOTAL_STEPS} коротких шагов: тема, роль автора, аудитория и стиль, режим, форматы, ритм и канал.</p></div>`;
  }
  if (s === 1) {
    return `<div class="ob-step-header"><div class="section-title">О чём канал</div><div class="ob-step-text">Выбери основную тему. Если ниша своя — задай её вручную.</div></div>${renderChoiceCards(ONBOARDING_TOPICS, a.topic, 'onboardingSetTopic')}${a.topic === 'Другое' ? `<div class="field"><div class="label">Своя тема</div><input class="input" value="${escapeHtml(a.customTopic || '')}" data-ob-input="onboardingSetCustomTopic" placeholder="Например: ремонт айфонов"></div>` : ''}`;
  }
  if (s === 2) {
    const customRoleActive = a.authorRole && !ONBOARDING_AUTHOR_ROLES.some(r => r.id === a.authorRole);
    return `<div class="ob-step-header"><div class="section-title">Кто ведёт канал</div><div class="ob-step-text">Выбери подходящую роль или опиши свою в одно-два слова. Это определяет тон и стиль текстов.</div></div>${renderChoiceCards(ONBOARDING_AUTHOR_ROLES, customRoleActive ? '' : a.authorRole, 'onboardingSetAuthorRole')}<div class="field" style="margin-top:12px"><div class="label">Или опиши свою роль</div><input class="input" value="${escapeHtml(customRoleActive ? a.authorRole : (a.customAuthorRole || ''))}" data-ob-input="onboardingSetCustomAuthorRole" placeholder="Например: врач, мастер маникюра, продюсер"></div>`;
  }
  if (s === 3) {
    const bothSelected = a.audience && a.style;
    const alertClass = bothSelected ? 'ob-must-select-both done' : 'ob-must-select-both';
    const alertText = bothSelected
      ? '✓ Аудитория и стиль выбраны — можно идти дальше'
      : '⚠ Нужно выбрать ОБА пункта: аудиторию И стиль';
    return `<div class="ob-step-header"><div class="section-title">Для кого и как писать</div><div class="ob-step-text">Выбери аудиторию и стиль. Без этого бот не поймёт, как именно подавать контент.</div></div>
      <div class="${alertClass}">${alertText}</div>
      <div class="ob-picked-row">
        <div class="ob-picked-pill ${a.audience ? 'filled' : ''}">${a.audience ? `Аудитория: ${escapeHtml(a.audience)}` : '① Выбери аудиторию ↓'}</div>
        <div class="ob-picked-pill ${a.style ? 'filled' : ''}">${a.style ? `Стиль: ${escapeHtml(a.style)}` : '② Выбери стиль ↓'}</div>
      </div>
      ${renderDualChoiceColumns('Аудитория', `<div id="ob-audience-block">${renderChoiceCards(ONBOARDING_AUDIENCES, a.audience, 'onboardingSetAudience')}</div>`, 'Стиль', `<div id="ob-style-block">${renderChoiceCards(ONBOARDING_STYLES, a.style, 'onboardingSetStyle')}</div>`)}
      <div class="ob-inline-hint">Сначала аудитория, потом стиль. После выбора обоих пунктов кнопка «Дальше» станет активной.</div>`;
  }
  if (s === 4) {
    return `<div class="ob-step-header"><div class="section-title">Как бот должен работать</div><div class="ob-step-text">Выбери уровень автоматизации — от полного автопилота до аккуратной подготовки черновиков.</div></div>${renderChoiceCards(ONBOARDING_MODES, a.mode, 'onboardingSetMode')}`;
  }
  if (s === 5) {
    return `<div class="ob-step-header"><div class="section-title">Какие форматы публиковать</div><div class="ob-step-text">Выбери 3–5 форматов, чтобы контент был не однотипным.</div></div>${renderChoiceCards(ONBOARDING_FORMATS, a.formats || [], 'onboardingToggleFormat', true)}`;
  }
  if (s === 6) {
    return `<div class="ob-step-header"><div class="section-title">Ритм, рамки и канал</div><div class="ob-step-text">Финальный шаг: частота, ограничения и активный канал для публикаций.</div></div>${renderDualChoiceColumns('Частота', renderChoiceCards(ONBOARDING_FREQUENCIES, a.frequency, 'onboardingSetFrequency'), 'Ограничения', renderChoiceCards(ONBOARDING_CONSTRAINTS, a.constraints || [], 'onboardingToggleConstraint', true))}<div class="onboarding-channel-block"><div class="onboarding-dual-title">Куда публиковать</div>${renderChannelChoices()}</div><div class="ob-finish-note">Выбери частоту и активный канал. Только после этого можно завершить запуск автопилота.</div>`;
  }
  return '';
}

/**
 * Частичная замена онбординг-shell без разрушения DOM.
 * Сохраняет фокус в input-полях и не вызывает полный перерендер.
 */
function _patchOnboardingShell(shell) {
  if (!shell) return;
  const total = ONBOARDING_TOTAL_STEPS;
  const cur = Math.max(0, Math.min(total - 1, Number(state.onboarding.step || 0)));
  const pct = Math.round(((cur + 1) / total) * 100);
  const last = cur === total - 1;

  const bar = shell.querySelector('.onboarding-progress-bar span');
  if (bar) bar.style.width = `${pct}%`;
  const meta = shell.querySelector('.onboarding-progress-meta span');
  if (meta) meta.textContent = `Шаг ${cur + 1} из ${total}`;

  const card = shell.querySelector('.onboarding-card');
  if (card) {
    card.innerHTML = `<div class="ob-step-enter">${renderOnboardingStep()}</div>`;
  }

  const actions = shell.querySelector('.onboarding-actions');
  if (actions) {
    const canCont = onboardingCanContinue(cur);
    actions.innerHTML =
      `<button type="button" class="btn ghost${cur === 0 ? ' disabled' : ''}" ${cur === 0 ? 'disabled' : ''} data-action="onboardingPrev">Назад</button>` +
      (last
        ? `<button type="button" class="btn primary" ${canCont ? '' : 'disabled'} data-action="completeOnboarding">Запустить канал</button>`
        : `<button type="button" class="btn primary" ${canCont ? '' : 'disabled'} data-action="onboardingNext">Дальше</button>`);
  }

  const needHint = (cur === 3 || cur === 6) && !onboardingCanContinue(cur);
  const hintText = cur === 6
    ? 'Чтобы завершить настройку, выбери частоту публикаций.'
    : 'Чтобы перейти дальше, выбери и аудиторию, и стиль.';
  let hintEl = shell.querySelector('.ob-bottom-hint');
  if (needHint) {
    if (hintEl) {
      hintEl.textContent = hintText;
    } else {
      const wrap = shell.querySelector('.onboarding-wrap');
      if (wrap) {
        const newHint = document.createElement('div');
        newHint.className = 'ob-bottom-hint';
        newHint.textContent = hintText;
        const actionsEl = wrap.querySelector('.onboarding-actions');
        if (actionsEl) wrap.insertBefore(newHint, actionsEl);
        else wrap.appendChild(newHint);
      }
    }
  } else if (hintEl) {
    hintEl.remove();
  }
}

function _patchOnboardingOrRender() {
  if (state.onboarding.active) {
    const shell = document.querySelector('.onboarding-shell');
    if (shell) {
      _patchOnboardingShell(shell);
      return;
    }
  }
  render();
}

function renderOnboardingShell() {
  const total = ONBOARDING_TOTAL_STEPS;
  const cur = Math.max(0, Math.min(total - 1, Number(state.onboarding.step || 0)));
  const pct = Math.round(((cur + 1) / total) * 100);
  const last = cur === total - 1;
  const needHint = (cur === 3 || cur === 6) && !onboardingCanContinue(cur);
  return `<div class="shell onboarding-shell"><div class="onboarding-wrap"><div class="onboarding-progress"><div class="onboarding-progress-bar"><span style="width:${pct}%"></span></div><div class="onboarding-progress-meta"><span>Шаг ${cur + 1} из ${total}</span></div></div><div class="onboarding-card">${renderOnboardingStep()}</div>${needHint ? `<div class="ob-bottom-hint">${cur === 6 ? 'Чтобы завершить настройку, выбери частоту публикаций.' : 'Чтобы перейти дальше, выбери и аудиторию, и стиль.'}</div>` : ''}<div class="onboarding-actions"><button type="button" class="btn ghost ${cur === 0 ? 'disabled' : ''}" ${cur === 0 ? 'disabled' : ''} data-action="onboardingPrev">Назад</button>${last ? `<button type="button" class="btn primary" ${onboardingCanContinue(cur) ? '' : 'disabled'} data-action="completeOnboarding">Запустить канал</button>` : `<button type="button" class="btn primary" ${onboardingCanContinue(cur) ? '' : 'disabled'} data-action="onboardingNext">Дальше</button>`}</div></div></div>`;
}

function onboardingSetTopic(v) {
  state.onboarding.answers.topic = v;
  if (v !== 'Другое') state.onboarding.answers.customTopic = '';
  _patchOnboardingOrRender();
}
function onboardingSetCustomTopic(v) {
  state.onboarding.answers.customTopic = v || '';
  const actions = document.querySelector('.onboarding-actions');
  if (actions) {
    const cur = Number(state.onboarding.step || 0);
    const canCont = onboardingCanContinue(cur);
    const nextBtn = actions.querySelector('button:last-child');
    if (nextBtn && !nextBtn.getAttribute('onclick')?.includes('completeOnboarding')) {
      nextBtn.disabled = !canCont;
    }
  }
}
function onboardingSetAuthorRole(v) {
  state.onboarding.answers.authorRole = v;
  state.onboarding.answers.customAuthorRole = '';
  _patchOnboardingOrRender();
}
function onboardingSetCustomAuthorRole(v) {
  const trimmed = (v || '').trim();
  state.onboarding.answers.customAuthorRole = trimmed;
  if (trimmed) {
    state.onboarding.answers.authorRole = trimmed;
  } else {
    // If custom field cleared and no preset was previously selected, reset
    state.onboarding.answers.authorRole = '';
  }
  // Update button state without full re-render to preserve cursor
  const actions = document.querySelector('.onboarding-actions');
  if (actions) {
    const cur = Number(state.onboarding.step || 0);
    const canCont = onboardingCanContinue(cur);
    const nextBtn = actions.querySelector('button:last-child');
    if (nextBtn) nextBtn.disabled = !canCont;
  }
}
function onboardingSetAudience(v) {
  state.onboarding.answers.audience = v;
  _patchOnboardingOrRender();
  setTimeout(() => document.getElementById('ob-style-block')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 30);
}
function onboardingSetStyle(v) {
  state.onboarding.answers.style = v;
  _patchOnboardingOrRender();
  setTimeout(() => document.querySelector('.onboarding-actions')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 30);
}
function onboardingSetMode(v) { state.onboarding.answers.mode = v; _patchOnboardingOrRender(); }
function onboardingSetFrequency(v) { state.onboarding.answers.frequency = v; _patchOnboardingOrRender(); }
function onboardingSelectChannel(v) { state.onboarding.answers.channelId = Number(v) || null; _patchOnboardingOrRender(); }

function onboardingToggleFormat(v) {
  const arr = Array.isArray(state.onboarding.answers.formats) ? [...state.onboarding.answers.formats] : [];
  const i = arr.indexOf(v);
  if (i >= 0) arr.splice(i, 1); else arr.push(v);
  state.onboarding.answers.formats = arr;
  _patchOnboardingOrRender();
}
function onboardingToggleConstraint(v) {
  const arr = Array.isArray(state.onboarding.answers.constraints) ? [...state.onboarding.answers.constraints] : [];
  const i = arr.indexOf(v);
  if (i >= 0) arr.splice(i, 1); else arr.push(v);
  state.onboarding.answers.constraints = arr;
  _patchOnboardingOrRender();
}
function onboardingNext() {
  const s = Number(state.onboarding.step || 0);
  if (!onboardingCanContinue(s)) return;
  state.onboarding.step = Math.min(ONBOARDING_TOTAL_STEPS - 1, s + 1);
  _patchOnboardingOrRender();
}
function onboardingPrev() {
  state.onboarding.step = Math.max(0, Number(state.onboarding.step || 0) - 1);
  _patchOnboardingOrRender();
}

function buildOnboardingStylePreset(style, audience) {
  const styleMap = {
    'Экспертный': 'экспертный, авторитетный, структурированный',
    'Простой и дружелюбный': 'простой, дружелюбный, без сложных терминов',
    'Живой и дерзкий': 'живой, дерзкий, с юмором, нестандартный',
    'Спокойный и полезный': 'спокойный, полезный, информативный',
    'Новостной': 'нейтральный, лаконичный, новостной',
    'Вовлекающий': 'вовлекающий, диалоговый, с вопросами к аудитории',
  };
  const audienceMap = {
    'Широкая аудитория': 'доступно для всех',
    'Новички': 'понятно для новичков',
    'Специалисты': 'на уровне профессионалов',
    'Клиенты': 'ориентировано на потенциальных клиентов',
    'Локальное сообщество': 'для локального сообщества',
  };
  const stylePart = styleMap[style] || style || 'профессиональный';
  const audPart = audienceMap[audience] || audience || '';
  return audPart ? `${stylePart}, ${audPart}` : stylePart;
}

function buildOnboardingRubrics(formats) {
  if (!formats || !formats.length) return '';
  return formats.map(f => `${f} — регулярно`).join('; ');
}

function buildOnboardingScenarios(mode, frequency, constraints, audience) {
  const freqMap = {
    'daily_1': '1 пост в день',
    'daily_2': '2 поста в день',
    'weekly_3_5': '3–5 постов в неделю',
    'flexible': 'гибкий график',
  };
  const modeMap = {
    'autopilot': 'автопубликация без подтверждения',
    'review': 'с ручным подтверждением каждого поста',
    'drafts_only': 'только черновики, без автопубликации',
  };
  const parts = [
    `Режим: ${modeMap[mode] || mode}`,
    `Частота: ${freqMap[frequency] || frequency}`,
  ];
  if (constraints && constraints.length) {
    parts.push(`Ограничения: ${constraints.join(', ')}`);
  }
  return parts.join('. ');
}

async function completeOnboarding() {
  const a = state.onboarding.answers || {};
  const topic = ((a.topic === 'Другое' ? a.customTopic : a.topic) || '').trim();
  if (!topic || !a.authorRole || !a.audience || !a.style || !a.mode || !(a.formats || []).length || !a.frequency) {
    toast('Нужно заполнить основные шаги настройки');
    return;
  }

  // Map onboarding mode to backend posting_mode and posts_enabled
  const modeToPostingMode = {
    'autopilot': 'both',
    'review': 'posts',
    'drafts_only': 'manual',
  };
  const postingMode = modeToPostingMode[a.mode] || 'manual';
  const postsEnabled = a.mode === 'drafts_only' ? '0' : '1';

  const patchPayload = {
    topic,
    author_role_type: ONBOARDING_AUTHOR_ROLES.some(r => r.id === a.authorRole) ? a.authorRole : 'expert',
    author_role_description: ONBOARDING_AUTHOR_ROLES.some(r => r.id === a.authorRole) ? '' : a.authorRole,
    channel_style: buildOnboardingStylePreset(a.style, a.audience),
    content_rubrics: buildOnboardingRubrics(a.formats || []),
    post_scenarios: buildOnboardingScenarios(a.mode, a.frequency, a.constraints || [], a.audience),
    channel_audience: a.audience,
    channel_style_preset: a.style,
    channel_mode: a.mode,
    channel_formats: JSON.stringify(a.formats || []),
    channel_frequency: a.frequency,
    content_constraints: JSON.stringify(a.constraints || []),
    onboarding_completed: '1',
    news_enabled: (a.formats || []).includes('Новости') ? '1' : '0',
    posting_mode: postingMode,
    posts_enabled: postsEnabled,
  };

  try {
    showBusy('Сохраняю профиль канала…');
    await api('/api/settings', { method: 'PATCH', body: JSON.stringify(patchPayload) });
    if (a.channelId) {
      await api('/api/channels/activate', { method: 'POST', body: JSON.stringify({ profile_id: Number(a.channelId) }) });
    }
    await refreshSections(['core','settings','channels'], { silent: true });
    updateOnboardingFromSettings();
    state.onboarding.active = false;
    state.activeTab = 'dashboard';
    render();
    toast('Профиль канала сохранён');
  } catch (e) {
    toast(e.message || 'Не удалось сохранить настройку');
  } finally {
    hideBusy();
  }
}

/* ---- Source chips helpers ---- */
function _parseSourcesList(raw) {
  return String(raw || '').split(/[\n,;]+/).map(s => s.trim()).filter(Boolean);
}

function _renderSourceChips(raw) {
  const items = _parseSourcesList(raw);
  if (!items.length) return '<div class="source-chips-empty">Источники не добавлены</div>';
  return items.map((src, i) => `
    <span class="source-chip">
      <span class="source-chip-text">${escapeHtml(src)}</span>
      <button class="source-chip-remove" data-action="removeNewsSource" data-action-arg="${i}" type="button" title="Удалить">&times;</button>
    </span>
  `).join('');
}

function _syncSourceChips() {
  const hidden = document.getElementById('set-news-sources');
  const container = document.getElementById('news-sources-chips');
  if (hidden && container) {
    container.innerHTML = _renderSourceChips(hidden.value);
  }
}

function addNewsSource() {
  const input = document.getElementById('news-source-input');
  const hidden = document.getElementById('set-news-sources');
  if (!input || !hidden) return;
  const val = input.value.trim();
  if (!val) return;
  const items = _parseSourcesList(hidden.value);
  const valLower = val.toLowerCase();
  if (items.some(s => s.toLowerCase() === valLower)) {
    toast('Этот источник уже добавлен');
    input.value = '';
    return;
  }
  items.push(val);
  hidden.value = items.join(', ');
  input.value = '';
  _syncSourceChips();
}

function removeNewsSource(indexStr) {
  const hidden = document.getElementById('set-news-sources');
  if (!hidden) return;
  const items = _parseSourcesList(hidden.value);
  const idx = parseInt(indexStr, 10);
  if (idx >= 0 && idx < items.length) {
    items.splice(idx, 1);
    hidden.value = items.join(', ');
    _syncSourceChips();
  }
}

async function saveSettings() {
  try {
    /* Collect news_sources from hidden input (chip-based UI) with textarea fallback */
    const newsSourcesEl = document.getElementById('set-news-sources');
    const newsSourcesVal = newsSourcesEl ? newsSourcesEl.value : '';
    await api('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify({
        posts_enabled: document.getElementById('set-posts-enabled').checked ? '1' : '0',
        news_enabled: document.getElementById('set-news-enabled').checked ? '1' : '0',
        posting_mode: document.getElementById('set-posting-mode').value,
        news_interval_hours: document.getElementById('set-news-interval').value,
        news_sources: newsSourcesVal,
        topic: document.getElementById('set-topic').value,
        channel_style: document.getElementById('set-channel-style')?.value || '',
        rubrics_schedule: document.getElementById('set-rubrics')?.value || '',
        content_rubrics: document.getElementById('set-rubrics')?.value || '',
        post_scenarios: document.getElementById('set-scenarios')?.value || '',
        content_exclusions: document.getElementById('set-exclusions')?.value || '',
        news_strict_mode: document.getElementById('set-news-strict')?.checked ? '1' : '0',
        source_auto_draft: document.getElementById('set-source-auto-draft')?.checked ? '1' : '0',
      })
    });
    toast('Настройки сохранены');
    await refreshSections(['settings','core'], { silent: false });
  } catch (e) {
    toast(e.message);
  }
}

function openCompetitorSpyModal() {
  const bodyHtml = `
    <div class="spy-modal-body stack">
      <div class="spy-modal-icon">🕵️‍♂️</div>
      <div class="spy-modal-title">Шпион конкурентов</div>
      <div class="spy-modal-desc">Введите ссылку на публичный Telegram-канал конкурента. ИИ проанализирует последние посты и создаст 3 уникальных черновика в вашем стиле.</div>
      <div class="field">
        <div class="label">Ссылка на канал</div>
        <input class="input" id="spy-channel-input" placeholder="t.me/durov или просто durov" autocomplete="off" />
      </div>
      <div id="spy-status" class="inline-status" style="min-height:18px"></div>
    </div>
  `;
  modal('🕵️‍♂️ Шпион конкурентов', bodyHtml,
    `<button class="btn primary" data-action="runCompetitorSpy">Анализировать</button><button class="btn ghost" data-action="closeModal">Отмена</button>`
  );
}

async function runCompetitorSpy() {
  const input = document.getElementById('spy-channel-input');
  const statusEl = document.getElementById('spy-status');
  const channelLink = (input?.value || '').trim();
  if (!channelLink) {
    if (statusEl) { statusEl.textContent = 'Введите ссылку на канал'; statusEl.className = 'inline-status danger'; }
    return;
  }
  if (statusEl) { statusEl.textContent = 'Анализирую канал…'; statusEl.className = 'inline-status neutral'; }
  const btn = document.querySelector('.modal .btn.primary');
  if (btn) btn.disabled = true;
  try {
    const result = await api('/api/competitor/spy', {
      method: 'POST',
      body: JSON.stringify({ channel_link: channelLink }),
    });
    closeModal();
    toast(`🕵️‍♂️ Создано ${result.count || 0} черновика на основе анализа конкурента`);
    await refreshSections(['drafts', 'core'], { silent: false });
    render();
  } catch (e) {
    if (e.status === 402 || e.status === 403) return; // paywall shown already
    if (statusEl) { statusEl.textContent = e.message || 'Ошибка'; statusEl.className = 'inline-status danger'; }
    if (btn) btn.disabled = false;
  }
}

function openNewsSniperModal() {
  // Reset any leftover state
  if (state._sniperTimer) { clearInterval(state._sniperTimer); state._sniperTimer = null; }
  if (state._sniperAbort) { try { state._sniperAbort.abort(); } catch {} state._sniperAbort = null; }
  state._sniperRunning = false;

  const bodyHtml = `
    <div class="sniper-modal-body stack">
      <div class="sniper-header">
        <div class="sniper-header-icon">⚡</div>
        <div class="sniper-header-copy">
          <div class="sniper-header-title">News Sniper</div>
          <div class="sniper-header-subtitle">Автоматический поиск актуальных новостей</div>
        </div>
      </div>
      <div class="sniper-description">
        ИИ найдёт самые свежие и горячие новости по теме вашего канала и подготовит черновик поста — за один клик.
      </div>
      <div id="sniper-state-wrap"></div>
    </div>
  `;
  modal('⚡ News Sniper', bodyHtml,
    `<button class="btn primary" id="sniper-run-btn" data-action="runNewsSniperNow">Найти новость и создать черновик</button>` +
    `<button class="btn ghost" data-action="closeModal">Отмена</button>`
  );
  state.modalCleanup = function() {
    if (state._sniperTimer) { clearInterval(state._sniperTimer); state._sniperTimer = null; }
    if (state._sniperAbort) { try { state._sniperAbort.abort(); } catch {} state._sniperAbort = null; }
    state._sniperRunning = false;
  };
}

function _sniperSetState(stateType, message, extra) {
  const wrap = document.getElementById('sniper-state-wrap');
  if (!wrap) return;
  if (stateType === 'idle') {
    wrap.innerHTML = '';
    return;
  }
  if (stateType === 'loading') {
    wrap.innerHTML = `
      <div class="sniper-state-card is-loading">
        <div class="sniper-spinner"></div>
        <div class="sniper-state-text">${escapeHtml(message)}</div>
        <div class="sniper-elapsed" id="sniper-elapsed">0 сек</div>
      </div>
    `;
    return;
  }
  if (stateType === 'success') {
    wrap.innerHTML = `
      <div class="sniper-state-card is-success">
        <div class="sniper-state-icon">✓</div>
        <div class="sniper-state-text">${escapeHtml(message)}</div>
      </div>
    `;
    return;
  }
  if (stateType === 'error') {
    wrap.innerHTML = `
      <div class="sniper-state-card is-error">
        <div class="sniper-state-icon">✕</div>
        <div class="sniper-state-text">${escapeHtml(message)}</div>
      </div>
    `;
    return;
  }
}

async function runNewsSniperNow() {
  if (state._sniperRunning) return;
  state._sniperRunning = true;

  const btn = document.getElementById('sniper-run-btn');
  if (btn) btn.disabled = true;

  // Show loading state with elapsed timer
  _sniperSetState('loading', 'Ищу актуальные новости…');
  let elapsed = 0;
  state._sniperTimer = setInterval(() => {
    elapsed++;
    const el = document.getElementById('sniper-elapsed');
    if (el) el.textContent = `${elapsed} сек`;
  }, 1000);

  // Abort controller with 60s timeout
  const TIMEOUT_MS = 60000;
  const controller = new AbortController();
  state._sniperAbort = controller;
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const result = await api('/api/news/sniper/run', {
      method: 'POST',
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    if (state._sniperTimer) { clearInterval(state._sniperTimer); state._sniperTimer = null; }

    // Show brief success state, then close
    _sniperSetState('success', 'Черновик с новостью создан!');
    if (btn) btn.disabled = false;
    state._sniperRunning = false;

    setTimeout(() => {
      closeModal();
      toast('⚡ Черновик с новостью создан!');
      refreshSections(['drafts', 'core'], { silent: false }).then(() => render());
    }, 800);

  } catch (e) {
    clearTimeout(timeoutId);
    if (state._sniperTimer) { clearInterval(state._sniperTimer); state._sniperTimer = null; }
    state._sniperRunning = false;
    state._sniperAbort = null;

    // Paywall/forbidden errors are handled by api() — just clean up
    if (e.status === 402 || e.status === 403) {
      if (btn) btn.disabled = false;
      _sniperSetState('idle', '');
      return;
    }

    // Determine user-friendly error message
    let msg;
    const isAbort = e.name === 'AbortError' || String(e?.message || '').includes('слишком долго');
    const raw = String(e?.message || e || '');
    if (isAbort) {
      msg = 'Сервер не ответил вовремя. Попробуйте ещё раз.';
    } else if (raw.includes('Актуальных новостей')) {
      msg = raw;
    } else if (raw.includes('Ошибка сети') || raw.includes('Failed to fetch')) {
      msg = 'Ошибка сети. Проверьте соединение и попробуйте снова.';
    } else {
      msg = raw || 'Произошла ошибка. Попробуйте ещё раз.';
    }

    _sniperSetState('error', msg);
    if (btn) btn.disabled = false;
  }
}

window.switchTab = switchTab;
window.openDraftEditor = openDraftEditor;
window.openGenerateDraftModal = openGenerateDraftModal;
window.openPlanGenerator = openPlanGenerator;
window.openPlanEditor = openPlanEditor;
window.openChannelProfile = openChannelProfile;
window.openScheduleModal = openScheduleModal;
window.openChannelModal = openChannelModal;
window.openAnalyticsDetails = openAnalyticsDetails;
window.createDraft = createDraft;
window.saveDraft = saveDraft;
window.generateDraft = generateDraft;
window.publishDraft = publishDraft;
window.backToChat = backToChat;
window.refreshInbox = refreshInbox;
window.openChatPicker = openChatPicker;
window.generatePostInEditor = generatePostInEditor;
window.generatePostFromPlan = generatePostFromPlan;
window.createDraftFromInbox = createDraftFromInbox;
window.deleteInboxMedia = deleteInboxMedia;
window.deleteDraft = deleteDraft;
window.previewDraft = previewDraft;
window.activateChannel = activateChannel;
window.saveChannel = saveChannel;
window.deleteChannel = deleteChannel;
window.generatePlan = generatePlan;
window.createPlanItem = createPlanItem;
window.savePlanItem = savePlanItem;
window.deletePlanItem = deletePlanItem;
window.createSchedule = createSchedule;
window.deleteSchedule = deleteSchedule;
window.saveSettings = saveSettings;
window.openSettingsModal = openSettingsModal;
window.toggleAutopost = toggleAutopost;
window.toggleAutopostNews = toggleAutopostNews;
window.updateAutopostMode = updateAutopostMode;
window.toggleAdvanced = toggleAdvanced;
window.setEditorMediaType = setEditorMediaType;
window.refreshEditorMediaPreview = refreshEditorMediaPreview;
window.clearEditorMedia = clearEditorMedia;
window.resetEditorDraft = resetEditorDraft;
window.openCompetitorSpyModal = openCompetitorSpyModal;
window.runCompetitorSpy = runCompetitorSpy;
window.openNewsSniperModal = openNewsSniperModal;
window.runNewsSniperNow = runNewsSniperNow;
window.closeModal = closeModal;
window.showPaywallModal = showPaywallModal;
window.showTariffsModal = showTariffsModal;
window.buyTariff = buyTariff;
window.addHashtagsToEditor = addHashtagsToEditor;
window.rewriteEditorText = rewriteEditorText;
window.fillEditorAssets = fillEditorAssets;
window.previewEditorDraft = previewEditorDraft;
window.addEditorButtonRow = addEditorButtonRow;
window.removeEditorButtonRow = removeEditorButtonRow;
window.clearEditorButtons = clearEditorButtons;
window.undoEditorChange = undoEditorChange;
window.redoEditorChange = redoEditorChange;
window.welcomeSaveChannel = welcomeSaveChannel;



document.addEventListener('visibilitychange', async () => {
  if (document.hidden) {
    saveUiState();
    return;
  }
  try {
    await refreshSections(['core','media_inbox','drafts'], { silent: true });
    await consumeAwaitedMediaIfAny();
    render();
  } catch (e) {
    console.error('visibility refresh failed', e);
  }
});

window.addEventListener('pagehide', saveUiState);
window.addEventListener('beforeunload', saveUiState);

bindViewportObservers();
loadBootstrap();


// ===== UX ULTRA PATCH v2 =====

// Auto resize textarea
document.addEventListener("input", function(e){
  if(e.target.tagName === "TEXTAREA"){
    e.target.style.height = "auto";
    e.target.style.height = (e.target.scrollHeight) + "px";
  }
});

// Smooth scroll preview into view
function focusPreview(){
  const el = document.querySelector("#editor-live-preview");
  if(el){
    el.scrollIntoView({behavior:"smooth", block:"center"});
  }
}

// Better media preview sizing
function fixPreviewMedia(){
  document.querySelectorAll(".preview-media img, .preview-media video").forEach(m=>{
    m.style.width="100%";
    m.style.maxHeight="320px";
    m.style.objectFit= m.tagName === 'VIDEO' ? "contain" : "cover";
    m.style.borderRadius="var(--radius-sm, 14px)";
    m.style.display="block";
  });
  document.querySelectorAll(".editor-media-preview img, .editor-media-preview video").forEach(m=>{
    m.style.width="100%";
    m.style.maxHeight="360px";
    m.style.objectFit= m.tagName === 'VIDEO' ? "contain" : "cover";
    m.style.borderRadius="var(--radius-sm, 14px)";
    m.style.display="block";
  });
}

window.addEventListener("load", ()=>{
  fixPreviewMedia();
});

