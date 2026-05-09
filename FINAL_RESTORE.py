import os

# Пути (проверяем обе возможные локации)
locations = ['./', './miniapp/']
files = ['app.js', 'styles.css', 'index.html']

# --- КОНТЕНТ СОВРЕМЕННОГО CSS (Стекло, градиенты, темная тема) ---
MODERN_CSS = """
:root {
    --bg: #0f172a;
    --glass: rgba(30, 41, 59, 0.7);
    --accent: #00d4aa;
    --text: #f8fafc;
    --text-dim: #94a3b8;
    --card-border: rgba(255, 255, 255, 0.1);
}
body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 16px; }
.stat-item { padding: 16px; border-radius: 16px; background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); text-align: center; }
.stat-value { font-size: 24px; font-weight: 800; color: var(--accent); }
.stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; }
#gen-toast { position:fixed; top:20px; left:50%; transform:translateX(-50%); width:92%; max-width:400px; z-index:10000; padding:16px; border-radius:16px; background:rgba(15,23,42,0.95); border:1px solid var(--accent); display:none; flex-direction:column; gap:8px; box-shadow: 0 0 20px rgba(0,212,170,0.2); }
"""

# --- ФУНКЦИЯ ПЕРЕЗАПИСИ ---
def rewrite():
    for loc in locations:
        js_path = os.path.join(loc, 'app.js')
        css_path = os.path.join(loc, 'styles.css')
        html_path = os.path.join(loc, 'index.html')

        if os.path.exists(css_path):
            with open(css_path, 'w', encoding='utf-8') as f:
                f.write(MODERN_CSS)
            print(f"✅ Перезаписан {css_path}")

        if os.path.exists(js_path):
            with open(js_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Если в JS нет нашей логики — добавляем в самый конец
            if 'BACKGROUND GENERATION' not in content:
                logic = """
// ===== FINAL BG LOGIC & UI FIX =====
window.isGenerating = false;
window.showGenToast = function(msg) {
    if (typeof closeModal === 'function') closeModal();
    window.isGenerating = true;
    let t = document.getElementById('gen-toast');
    if(!t) {
        t = document.createElement('div'); t.id = 'gen-toast'; document.body.appendChild(t);
    }
    t.innerHTML = `<div style="display:flex;justify-content:space-between;font-weight:bold;font-size:13px"><span>✨ ${msg||'Работаю...'}</span><span id="gen-pct">0%</span></div><div style="width:100%;height:4px;background:rgba(255,255,255,0.1);margin-top:8px;border-radius:2px;overflow:hidden"><div id="gen-fill" style="width:0;height:100%;background:#00d4aa;transition:width 1s"></div></div>`;
    t.style.display = 'flex';
    let p=0; window.gInt = setInterval(()=>{ if(p<98){p+=2; document.getElementById('gen-pct').innerText=p+'%'; document.getElementById('gen-fill').style.width=p+'%';}}, 2000);
};
window.hideGenToast = function() { 
    clearInterval(window.gInt); document.getElementById('gen-toast').style.display='none'; window.isGenerating=false; 
};
const _api = window.api;
window.api = async function(path, opt) {
    const gen = path.includes('/generate');
    if(gen) window.showGenToast('Пишу пост...');
    try {
        const r = await _api(path, opt);
        if(gen) {
            document.getElementById('gen-pct').innerText='100%'; 
            document.getElementById('gen-fill').style.width='100%';
            setTimeout(async ()=>{
                await refreshSections(['core','drafts'],{silent:true});
                window.hideGenToast();
                const id = r.id || (r.draft && r.draft.id);
                if(id) openDraftEditor(id);
            }, 800);
        }
        return r;
    } catch(e) { if(gen) window.hideGenToast(); throw e; }
};
const _busy = window.showBusy;
window.showBusy = function(m) {
    if(String(m).includes('генер')) { window.showGenToast(m); return; }
    if(_busy) _busy(m);
};
"""
                with open(js_path, 'a', encoding='utf-8') as f:
                    f.write(logic)
            print(f"✅ Пропатчен {js_path}")

        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                h = f.read()
            import re
            h = re.sub(r'design\d+', 'design' + str(os.getpid()), h)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(h)
            print(f"✅ Обновлен кэш в {html_path}")

rewrite()
