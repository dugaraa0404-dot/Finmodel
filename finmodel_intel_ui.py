"""Front-end for finmodel_intel: three dashboard views injected into the main page.

inject(page) rewrites the main dashboard HTML:
  * the old "Macro Calendar & News" view becomes "Macro & News Intelligence"
  * new views: "Geopolitical Simulator" and "Indicator Lab"
  * extra CSS and one extra <script> that hooks into switchView / renderNewsCalendar
Every server-provided string is HTML-escaped before it is inserted.
"""
import re

_NAV_OLD = '<div class="navlink" data-view="news" onclick="switchView(\'news\', this)">◉ <span>Macro Calendar & News</span></div>'
_NAV_NEW = ('<div class="navlink" data-view="news" onclick="switchView(\'news\', this)">◉ <span>Macro & News Intelligence</span></div>\n'
            '        <div class="navlink" data-view="geo" onclick="switchView(\'geo\', this)">⊕ <span>Geopolitical Simulator</span></div>\n'
            '        <div class="navlink" data-view="indicators" onclick="switchView(\'indicators\', this)">≋ <span>Indicator Lab</span></div>')

CSS = r'''
.chip{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:10px;font-family:"JetBrains Mono",monospace;white-space:nowrap}
.chip.pos{border-color:#14532d;color:#34d399}.chip.neg{border-color:#7f1d1d;color:#fb7185}.chip.neu{color:#94a3b8}
.chip.vh{border-color:#7f1d1d;color:#fda4af}.chip.hi{border-color:#78350f;color:#fbbf24}.chip.md{color:#94a3b8}
.chip.btn{cursor:pointer}.chip.btn.on{background:#0a2d23;border-color:#155d49;color:var(--green)}
.intel-card{border:1px solid var(--line);border-radius:12px;padding:14px;background:rgba(7,12,18,.6)}
.intel-bar{height:5px;border-radius:999px;background:#17212d;overflow:hidden}.intel-bar>span{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}
.intel-ev summary{cursor:pointer;list-style:none}.intel-ev summary::-webkit-details-marker{display:none}
.intel-ev[open]{background:rgba(13,24,34,.75)}
.intel-day{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px}
.intel-link{color:var(--cyan);font-size:11px}.intel-link:hover{text-decoration:underline}
.intel-p{font-size:12.5px;line-height:1.65;color:#cbd5e1}.intel-p b{color:#fff}
.intel-table td,.intel-table th{padding:7px 8px;font-size:11px}
.intel-input{background:#070c12;border:1px solid #263444;color:#fff;border-radius:8px;padding:6px 9px;font-size:12px;outline:none;width:100%}
.intel-check{display:flex;align-items:center;gap:6px;font-size:11px;color:#cbd5e1;cursor:pointer}
#geoNetwork svg text{font-family:Inter,sans-serif}
'''

HTML_NEWS = r'''
        <div id="view-news" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Macro &amp; News Intelligence</div>
                    <div class="text-[10px] muted mono mt-1">THEMES · TONE · FACTOR LINKS · CALENDAR · SOURCES</div></div>
                    <div class="flex items-center gap-2"><span id="intelNewsStatus" class="badge">OPEN TO LOAD</span>
                    <button class="btn px-3 py-1.5 text-xs" onclick="loadIntelNews(true)">Refresh ↻</button></div>
                </div>
                <div id="intelBrief" class="mt-4 space-y-3"></div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-3 gap-5">
                <div class="panel p-5 xl:col-span-2">
                    <div class="font-semibold">Themes driving the news</div>
                    <div class="text-[10px] muted mono mt-1 mb-4">RANKED BY RECENCY-WEIGHTED COVERAGE · EACH WITH ITS MECHANISM AND LINK TO THE ASSET</div>
                    <div id="intelThemes" class="space-y-3"></div>
                </div>
                <div class="space-y-5">
                    <div class="panel p-5"><div class="font-semibold">What moved the asset last week</div>
                        <div class="text-[10px] muted mono mt-1 mb-3">FACTOR ATTRIBUTION (WEEKLY REGRESSION)</div>
                        <div class="h-[230px]"><canvas id="intelAttribChart"></canvas></div>
                        <div id="intelAttribText" class="text-xs muted mt-3 leading-5"></div></div>
                    <div class="panel p-5"><div class="font-semibold">Macro exposures</div>
                        <div class="text-[10px] muted mono mt-1 mb-3">3Y WEEKLY BETAS · t-STATS</div>
                        <div class="table-wrap"><table class="intel-table" id="intelBetaTable"><tbody></tbody></table></div></div>
                </div>
            </section>
            <section class="panel p-5">
                <div class="flex flex-col xl:flex-row xl:items-end justify-between gap-3">
                    <div><div class="font-semibold text-lg">Economic Calendar</div>
                    <div class="text-[10px] muted mono mt-1">AUTOMATIC · OFFICIAL DATES (FRED) · CONSENSUS (WEEKLY FEED) · YOUR LOCAL TIME</div>
                    <div id="calStatus" class="text-[10px] muted mt-1"></div></div>
                    <div class="flex flex-wrap gap-2 items-center">
                        <select id="calDays" class="intel-input" style="width:auto" onchange="loadCalendar(true)"><option value="7">7 days</option><option value="21" selected>21 days</option><option value="45">45 days</option><option value="90">90 days</option></select>
                        <select id="calImportance" class="intel-input" style="width:auto" onchange="loadCalendar(true)"><option value="MEDIUM">Medium+</option><option value="HIGH" selected>High+</option><option value="VERY HIGH">Very high</option></select>
                        <select id="calRegion" class="intel-input" style="width:auto" onchange="loadCalendar(true)"><option value="all">All regions</option><option value="us">United States</option><option value="global">Outside US</option></select>
                        <label class="intel-check"><input type="checkbox" id="calSensitive" onchange="renderCalendar()"> Only events that move this asset</label>
                        <a id="calIcs" class="btn px-3 py-1.5 text-xs" href="#" title="Download or subscribe in Google/Apple/Outlook calendar">Add to my calendar (.ics)</a>
                    </div>
                </div>
                <div id="calFocus" class="mt-4"></div>
                <div id="calList" class="mt-2"></div>
            </section>
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Headline stream</div><div class="text-[10px] muted mono mt-1">DE-DUPLICATED ACROSS OUTLETS · TONE-SCORED · LINKED</div></div>
                    <div id="intelFilters" class="flex flex-wrap gap-2"></div>
                </div>
                <div id="intelHeadlines" class="mt-4 space-y-2"></div>
                <div id="intelMethod" class="text-[10px] muted mt-4 leading-5"></div>
            </section>
        </div>

        <div id="view-geo" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Geopolitical Shock Simulator</div>
                    <div class="text-[10px] muted mono mt-1">SCENARIO → FACTOR SPILLOVERS → ASSET · MONTE CARLO ON MEASURED EXPOSURES</div></div>
                    <span id="geoStatus" class="badge">OPEN TO LOAD</span>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-4 gap-3 mt-5">
                    <label class="text-xs muted lg:col-span-2">Scenario<select id="geoScenario" class="intel-input mt-1" onchange="geoFillShocks(); runGeo()"></select></label>
                    <label class="text-xs muted">Horizon<select id="geoHorizon" class="intel-input mt-1" onchange="runGeo()"><option value="1">1 week</option><option value="2">2 weeks</option><option value="4" selected>1 month</option><option value="8">2 months</option><option value="13">3 months</option></select></label>
                    <label class="text-xs muted">Correlations<select id="geoRegime" class="intel-input mt-1" onchange="loadGeo(true)"><option value="stress" selected>Stress regime (crisis weeks)</option><option value="normal">Full history</option></select></label>
                </div>
                <div class="text-[10px] muted mt-4 mb-2">Primary shocks (edit to build your own; leave blank to let a factor follow its historical co-movement)</div>
                <div id="geoShockInputs" class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3"></div>
                <div class="flex flex-wrap gap-2 mt-4 items-center">
                    <button class="btn px-5 py-2.5" onclick="runGeo()">Run simulation →</button>
                    <button class="btn px-4 py-2.5 text-xs" onclick="geoFillShocks(); runGeo()">Reset to scenario</button>
                    <label class="text-xs muted flex items-center gap-2 ml-2">Shock uncertainty <input id="geoUnc" type="range" min="0" max="0.8" step="0.05" value="0.35" onchange="runGeo()"> <span id="geoUncVal" class="mono">35%</span></label>
                </div>
            </section>
            <section class="grid grid-cols-2 md:grid-cols-6 gap-3" id="geoKpis"></section>
            <section class="panel p-5">
                <div class="font-semibold">Propagation network</div>
                <div class="text-[10px] muted mono mt-1 mb-3">TRIGGER → SHOCKED FACTORS → SPILLOVERS (CONDITIONAL ON HISTORY) → ASSET · EDGE LABELS = EXPECTED EFFECT</div>
                <div id="geoNetwork" class="w-full overflow-x-auto"></div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Outcome distribution</div><div class="text-[10px] muted mono mt-1 mb-3">SIMULATED RETURN OVER THE HORIZON</div><div class="h-[300px]"><canvas id="geoHist"></canvas></div></div>
                <div class="panel p-5"><div class="font-semibold">Contribution by channel</div><div class="text-[10px] muted mono mt-1 mb-3">BETA × EXPECTED FACTOR MOVE</div><div class="h-[300px]"><canvas id="geoContrib"></canvas></div></div>
            </section>
            <section class="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <div class="panel p-5"><div class="font-semibold">Analysis</div><div id="geoNarrative" class="space-y-3 mt-3"></div>
                    <div class="font-semibold mt-5">Causal chain</div><ol id="geoChain" class="list-decimal ml-5 mt-2 space-y-1 intel-p"></ol>
                    <div id="geoAnalogues" class="text-xs muted mt-3"></div></div>
                <div class="panel p-5"><div class="font-semibold">Factor detail</div><div class="table-wrap mt-3"><table class="intel-table" id="geoFactorTable"><thead><tr><th>Factor</th><th>Scenario move</th><th>Beta</th><th>t</th><th>Contribution</th></tr></thead><tbody></tbody></table></div></div>
            </section>
            <section class="panel p-5">
                <div class="font-semibold">Stress-test matrix</div>
                <div class="text-[10px] muted mono mt-1 mb-3">EVERY SCENARIO FOR THIS ASSET · HEADLINE RELEVANCE FROM LIVE NEWS · CLICK A ROW TO SIMULATE</div>
                <div class="table-wrap"><table class="intel-table" id="geoMatrix"><thead><tr><th>Scenario</th><th>Region</th><th>Headline relevance</th><th>Primary shocks</th><th>Expected move</th></tr></thead><tbody></tbody></table></div>
            </section>
        </div>

        <div id="view-indicators" class="view-section space-y-6">
            <section class="panel p-5">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div><div class="font-semibold text-lg">Indicator Lab</div><div class="text-[10px] muted mono mt-1">20+ INDICATORS · SIGNAL SUMMARY · RELIABILITY ON THIS ASSET</div></div>
                    <span id="indStatus" class="badge">OPEN TO LOAD</span>
                </div>
                <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3 mt-5">
                    <label class="text-xs muted">Interval<select id="indInterval" class="intel-input mt-1" onchange="loadIndicators(true)"><option value="1d" selected>Daily</option><option value="1wk">Weekly</option><option value="1h">Hourly</option></select></label>
                    <label class="text-xs muted">Window<select id="indPeriod" class="intel-input mt-1" onchange="loadIndicators(true)"><option value="3mo">3 months</option><option value="6mo">6 months</option><option value="1y" selected>1 year</option><option value="2y">2 years</option><option value="5y">5 years</option></select></label>
                    <label class="text-xs muted">Oscillator<select id="indOsc" class="intel-input mt-1" onchange="renderIndicators()"><option>RSI</option><option>MACD</option><option>Stochastic</option><option>ADX / DMI</option><option>CCI</option><option>Williams %R</option><option>MFI</option><option>OBV</option><option>ATR %</option></select></label>
                    <label class="text-xs muted">SMA periods<input id="indSma" class="intel-input mt-1" value="50,200"></label>
                    <label class="text-xs muted">EMA periods<input id="indEma" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">Forward test (bars)<input id="indFwd" class="intel-input mt-1" type="number" value="10" min="2" max="120"></label>
                </div>
                <details class="mt-3"><summary class="text-xs cyan cursor-pointer">More parameters</summary>
                <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3 mt-3">
                    <label class="text-xs muted">BB length<input id="ind_bb_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">BB std dev<input id="ind_bb_k" class="intel-input mt-1" value="2"></label>
                    <label class="text-xs muted">Supertrend ATR<input id="ind_st_n" class="intel-input mt-1" value="10"></label>
                    <label class="text-xs muted">Supertrend mult<input id="ind_st_mult" class="intel-input mt-1" value="3"></label>
                    <label class="text-xs muted">RSI length<input id="ind_rsi_n" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">MACD fast<input id="ind_macd_fast" class="intel-input mt-1" value="12"></label>
                    <label class="text-xs muted">MACD slow<input id="ind_macd_slow" class="intel-input mt-1" value="26"></label>
                    <label class="text-xs muted">MACD signal<input id="ind_macd_signal" class="intel-input mt-1" value="9"></label>
                    <label class="text-xs muted">Stochastic %K<input id="ind_stoch_k" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">ADX length<input id="ind_adx_n" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">CCI length<input id="ind_cci_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">Donchian length<input id="ind_dc_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">Keltner length<input id="ind_kc_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">VWAP window<input id="ind_vwap_n" class="intel-input mt-1" value="20"></label>
                    <label class="text-xs muted">MFI length<input id="ind_mfi_n" class="intel-input mt-1" value="14"></label>
                    <label class="text-xs muted">Williams length<input id="ind_wr_n" class="intel-input mt-1" value="14"></label>
                </div></details>
                <div class="flex flex-wrap gap-3 mt-4" id="indOverlays"></div>
                <div class="mt-4"><button class="btn px-5 py-2.5" onclick="loadIndicators(true)">Apply parameters →</button></div>
            </section>
            <section class="panel p-4 md:p-5"><div class="h-[420px]"><canvas id="indPriceChart"></canvas></div><div class="h-[200px] mt-4"><canvas id="indOscChart"></canvas></div></section>
            <section class="grid grid-cols-1 xl:grid-cols-3 gap-5">
                <div class="panel p-5"><div class="font-semibold">Technical summary</div><div id="indSummary" class="mt-4"></div></div>
                <div class="panel p-5 xl:col-span-2"><div class="font-semibold">Signals &amp; reliability on this asset</div>
                    <div class="text-[10px] muted mono mt-1 mb-3">AVERAGE FORWARD RETURN AFTER EACH SIGNAL VS THE ASSET'S NORMAL DRIFT</div>
                    <div class="table-wrap"><table class="intel-table" id="indSignals"><thead><tr><th>Indicator</th><th>Now</th><th>Reading</th><th>After bullish signals</th><th>After bearish signals</th></tr></thead><tbody></tbody></table></div>
                    <div id="indNote" class="text-[10px] muted mt-3 leading-5"></div></div>
            </section>
        </div>
'''

JS = r'''
<script>
/* ===================== finmodel intelligence layer ===================== */
const INTEL = {ticker: null, news: null, cal: null, geo: null, sim: null, ind: null, headFilter: 'all', charts: {}};
function iesc(s){ return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function iurl(u){ return /^https?:\/\//i.test(String(u||'')) ? iesc(u) : '#'; }
function itk(){ return String((typeof currentData !== 'undefined' && currentData && currentData.ticker) || (document.getElementById('tickerInput')||{}).value || '').trim().toUpperCase(); }
async function ifetch(path){ const r = await fetch(API_BASE + path); let j = {}; try { j = await r.json(); } catch(e) {} if(!r.ok) throw new Error(j.detail || ('HTTP ' + r.status)); return j; }
function iLocal(iso){ try { return new Date(iso).toLocaleString([], {weekday:'short', month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}); } catch(e){ return iso; } }
function iDay(iso){ try { return new Date(iso).toLocaleDateString([], {weekday:'long', month:'long', day:'numeric'}); } catch(e){ return iso.slice(0,10); } }
function iCountdown(iso){ const ms = new Date(iso) - new Date(); const a = Math.abs(ms)/36e5; const txt = a < 1 ? Math.round(a*60)+'m' : a < 48 ? Math.round(a)+'h' : Math.round(a/24)+'d'; return ms >= 0 ? 'in '+txt : txt+' ago'; }
function iToneChip(t){ const v = Number(t||0); const cls = v > 0.08 ? 'pos' : v < -0.08 ? 'neg' : 'neu'; return `<span class="chip ${cls}">tone ${v>=0?'+':''}${v.toFixed(2)}</span>`; }
function iImpChip(i){ const c = i==='VERY HIGH'?'vh':i==='HIGH'?'hi':'md'; return `<span class="chip ${c}">${iesc(i)}</span>`; }
function iChart(id, cfg){ if(INTEL.charts[id]) INTEL.charts[id].destroy(); const el = document.getElementById(id); if(!el) return null; INTEL.charts[id] = new Chart(el, cfg); return INTEL.charts[id]; }
const ICOL = {grid:'#17212d', tick:'#64748b', green:'#19d58b', red:'#fb7185', cyan:'#22d3ee', amber:'#fbbf24', violet:'#a78bfa', slate:'#94a3b8'};
function iAxes(extra){ return Object.assign({x:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick,maxTicksLimit:10}}, y:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick}}}, extra||{}); }

/* ---- hooks into the existing dashboard ---- */
const _intelOrigSwitch = switchView;
switchView = function(v, el){ _intelOrigSwitch(v, el); if(v==='news') loadIntelNews(); if(v==='geo') loadGeo(); if(v==='indicators') loadIndicators(); };
renderNewsCalendar = function(data){
    const t = (data && data.ticker) ? String(data.ticker).toUpperCase() : itk();
    if (t && t !== INTEL.ticker) { INTEL.ticker = t; INTEL.news = INTEL.cal = INTEL.geo = INTEL.sim = INTEL.ind = null; }
    const active = (document.querySelector('.view-section.active-view')||{}).id;
    if (active === 'view-news') loadIntelNews(); else if (active === 'view-geo') loadGeo(); else if (active === 'view-indicators') loadIndicators();
};

/* ===================== Macro & News ===================== */
async function loadIntelNews(force){
    const t = itk(); if(!t) return;
    INTEL.ticker = t;
    loadCalendar(force);
    if (INTEL.news && INTEL.news.ticker === t && !force) return renderIntelNews();
    const st = document.getElementById('intelNewsStatus'); st.innerText = 'READING THE NEWS…';
    try { INTEL.news = await ifetch('/api/intel/' + encodeURIComponent(t)); st.innerText = INTEL.news.story_count + ' STORIES'; renderIntelNews(); }
    catch(e){ st.innerText = 'UNAVAILABLE'; document.getElementById('intelBrief').innerHTML = `<div class="text-sm red">News intelligence unavailable: ${iesc(e.message)}</div>`; }
}
function renderIntelNews(){
    const d = INTEL.news; if(!d) return;
    document.getElementById('intelBrief').innerHTML = (d.brief||[]).map(p => {
        const m = String(p).match(/^([^:]{3,40}):\s(.*)$/);
        return `<p class="intel-p">${m ? `<b>${iesc(m[1])}:</b> ${iesc(m[2])}` : iesc(p)}</p>`;
    }).join('');
    document.getElementById('intelThemes').innerHTML = (d.themes||[]).map(th => `
        <div class="intel-card">
            <div class="flex flex-wrap items-center justify-between gap-2"><strong>${iesc(th.label)}</strong>
              <div class="flex gap-2">${iToneChip(th.tone)}<span class="chip">${th.count} stories · ${th.outlets} outlets</span></div></div>
            <div class="intel-bar mt-2"><span style="width:${Math.max(4, th.intensity_pct)}%"></span></div>
            <div class="mt-3 space-y-2">${(th.paragraphs||[]).map(p => `<p class="intel-p">${iesc(p)}</p>`).join('')}</div>
            <div class="mt-3 space-y-1">${(th.stories||[]).map(s => `<div class="text-xs"><a class="intel-link" href="${iurl(s.url)}" target="_blank" rel="noopener">${iesc(s.title)}</a>
                <span class="muted"> — ${iesc(s.source)}, ${Math.round(s.age_h)}h ago${s.outlets>1?`, ${s.outlets} outlets`:''}</span></div>`).join('')}</div>
        </div>`).join('') || '<div class="text-sm muted">No themed coverage returned by the feeds.</div>';

    // filters + headline stream
    const themes = [...new Set((d.headlines||[]).flatMap(h => h.themes||[]))].slice(0, 8);
    const filters = [['all','All'],['asset','About '+d.ticker],['macro','Macro'],['geo','Geopolitical'], ...themes.map(t => ['theme:'+t, t])];
    document.getElementById('intelFilters').innerHTML = filters.map(([k,l]) => `<span class="chip btn ${INTEL.headFilter===k?'on':''}" onclick="INTEL.headFilter='${iesc(k).replace(/'/g,'')}';renderIntelNews()">${iesc(l)}</span>`).join('');
    const f = INTEL.headFilter;
    const rows = (d.headlines||[]).filter(h => f==='all' || (f==='asset' && h.about_asset) || (f==='macro' && h.origin==='macro') || (f==='geo' && h.origin==='geo') || (f.startsWith('theme:') && (h.themes||[]).includes(f.slice(6))));
    document.getElementById('intelHeadlines').innerHTML = rows.map(h => `
        <div class="scenario news-row"><div class="flex flex-col md:flex-row md:justify-between gap-2">
            <a class="text-sm intel-link" style="font-size:13px" href="${iurl(h.url)}" target="_blank" rel="noopener">${iesc(h.title)}</a>
            <div class="flex gap-1 flex-wrap">${iToneChip(h.tone)}${(h.themes||[]).map(t => `<span class="chip">${iesc(t)}</span>`).join('')}</div></div>
            <div class="text-[10px] muted mt-1">${iesc(h.source)} · ${Math.round(h.age_h)}h ago${h.about_asset?' · mentions '+iesc(d.ticker):''}
            ${(h.also||[]).length ? ' · also: ' + h.also.map(a => `<a class="intel-link" href="${iurl(a.url)}" target="_blank" rel="noopener">${iesc(a.source)}</a>`).join(', ') : ''}</div></div>`).join('') || '<div class="text-sm muted">No headlines for this filter.</div>';
    document.getElementById('intelMethod').innerText = d.method || '';

    // factor attribution + betas
    const fm = d.factor_model;
    const bt = document.querySelector('#intelBetaTable tbody');
    if (fm && fm.ok) {
        const lw = fm.last_week || {};
        const keys = Object.keys(lw.contributions||{});
        const labels = keys.map(k => (fm.labels||{})[k] || k).concat(['Stock-specific', 'Actual']);
        const vals = keys.map(k => lw.contributions[k]).concat([lw.stock_specific, lw.actual]);
        iChart('intelAttribChart', {type:'bar', data:{labels, datasets:[{data:vals, backgroundColor: vals.map((v,i) => i===vals.length-1 ? ICOL.cyan : (v>=0?ICOL.green:ICOL.red))}]},
            options:{indexAxis:'y', responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}}, scales:iAxes({x:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick,callback:v=>v+'%'}}})}});
        document.getElementById('intelAttribText').innerText = `Week ${lw.start} → ${lw.end}. Factors explain ${Math.round((fm.r2||0)*100)}% of ${fm.ticker}'s weekly moves over ${fm.weeks} weeks (${fm.start} → ${fm.end}).`;
        bt.innerHTML = '<tr><th>Factor</th><th>Effect</th><th>t</th></tr>' + Object.entries(fm.betas||{}).map(([k,b]) => {
            if (b.self) return `<tr><td>${iesc(b.label)}</td><td colspan="2">this asset</td></tr>`;
            const per = b.unit==='bp' ? 10 : 1; const unit = b.unit==='bp' ? 'per +10bp' : b.unit==='pts' ? 'per +1 pt' : 'per +1%';
            const v = (b.beta||0)*per; const sig = Math.abs(b.t||0) >= 2;
            return `<tr><td>${iesc(b.label)}</td><td class="${v>=0?'green':'red'}">${v>=0?'+':''}${v.toFixed(2)}% <span class="muted">${unit}</span></td><td class="${sig?'':'muted'}">${b.t==null?'–':Number(b.t).toFixed(1)}</td></tr>`;
        }).join('');
    } else {
        bt.innerHTML = `<tr><td class="muted">${iesc((fm && fm.reason) || 'Factor model unavailable')}</td></tr>`;
    }
}

async function loadCalendar(force){
    const t = itk();
    const days = document.getElementById('calDays').value, imp = document.getElementById('calImportance').value, reg = document.getElementById('calRegion').value;
    const key = [t, days, imp, reg].join('|');
    document.getElementById('calIcs').href = API_BASE + '/api/calendar.ics?days=60&importance=HIGH';
    if (INTEL.cal && INTEL.cal._key === key && !force) return renderCalendar();
    document.getElementById('calList').innerHTML = '<div class="text-sm muted">Loading calendar…</div>';
    try { INTEL.cal = await ifetch(`/api/calendar?ticker=${encodeURIComponent(t)}&days=${days}&importance=${encodeURIComponent(imp)}&region=${reg}`); INTEL.cal._key = key; renderCalendar(); }
    catch(e){ document.getElementById('calList').innerHTML = `<div class="text-sm red">Calendar unavailable: ${iesc(e.message)}</div>`; }
}
function renderCalendar(){
    const d = INTEL.cal; if(!d) return;
    const s = d.status || {};
    document.getElementById('calStatus').innerHTML = `Official schedule: <b>${iesc(s.fred_schedule||'?')}</b> · Consensus feed: <b>${iesc(s.weekly_feed||'?')}</b>` +
        (d.fred_key_configured ? '' : ' · <span class="amber">Set FRED_API_KEY for official dates and event-day statistics on every release</span>');
    const f = d.focus;
    document.getElementById('calFocus').innerHTML = f ? `<div class="intel-card"><span class="chip hi">MOST RELEVANT FOR ${iesc(d.ticker||'')}</span>
        <div class="mt-2"><b>${iesc(f.title)}</b> <span class="muted text-xs">${iLocal(f.datetime_utc)} (${iCountdown(f.datetime_utc)})</span></div><div class="text-xs muted mt-1">${iesc(f.why)}</div></div>` : '';
    const only = document.getElementById('calSensitive').checked;
    let events = (d.events||[]);
    if (only) events = events.filter(e => e.sensitivity && e.sensitivity.ratio >= 1.15);
    let html = '', lastDay = '';
    for (const e of events) {
        const day = iDay(e.datetime_utc);
        if (day !== lastDay) { html += `<div class="intel-day">${iesc(day)}</div>`; lastDay = day; }
        const sens = e.sensitivity;
        const sensChip = sens && sens.ratio ? `<span class="chip ${sens.ratio>=1.3?'hi':'md'}" title="Average move on past release days vs other days">${iesc(d.ticker)} ${sens.ratio}× normal</span>` : '';
        const cons = (e.consensus||[]).slice(0,3).map(c => `<span class="chip" title="${iesc(c.label)}">${iesc(c.label.length>22?c.label.slice(0,22)+'…':c.label)}: f ${iesc(c.forecast||'–')} · p ${iesc(c.previous||'–')}${c.actual?' · a '+iesc(c.actual):''}</span>`).join('');
        const a = e.analysis || {};
        html += `<details class="intel-ev scenario mb-2"><summary><div class="flex flex-col md:flex-row md:items-center gap-3">
            <div class="w-40 mono text-xs">${iLocal(e.datetime_utc)}<div class="muted text-[10px]">${iesc(e.time)} ET · ${iCountdown(e.datetime_utc)}</div></div>
            <div class="flex-1"><strong class="text-sm">${iesc(e.title)}</strong>${e.estimated?' <span class="chip md">est.</span>':''}
                <div class="flex flex-wrap gap-1 mt-1">${iImpChip(e.importance)}<span class="chip">${iesc(e.region||'')}</span><span class="chip">${iesc(a.category||'')}</span>${sensChip}${cons}</div></div>
            <div class="text-[10px] cyan">Analysis ▾</div></div></summary>
            <div class="mt-3 space-y-2">${(a.paragraphs||[]).map(p => { const m = String(p).match(/^([^:]{3,32}):\s(.*)$/); return `<p class="intel-p">${m?`<b>${iesc(m[1])}:</b> ${iesc(m[2])}`:iesc(p)}</p>`; }).join('')}
            <div class="flex flex-wrap gap-3 mt-2">${(a.links||[]).map(l => `<a class="intel-link" href="${iurl(l.url)}" target="_blank" rel="noopener">${iesc(l.label)} ↗</a>`).join('')}
            ${(a.readings||[]).map(r => `<a class="intel-link" href="${iurl(r.url)}" target="_blank" rel="noopener">${iesc(r.label)} data (FRED) ↗</a>`).join('')}</div>
            <div class="text-[10px] muted">Sources: ${iesc((e.sources||[]).join(', '))}</div></div></details>`;
    }
    document.getElementById('calList').innerHTML = html || '<div class="text-sm muted">No events match these filters.</div>';
}

/* ===================== Geopolitical simulator ===================== */
async function loadGeo(force){
    const t = itk(); if(!t) return;
    const regime = document.getElementById('geoRegime').value;
    const st = document.getElementById('geoStatus');
    if (!INTEL.geo || INTEL.geo.ticker !== t || INTEL.geo.regime !== regime || force) {
        st.innerText = 'ESTIMATING EXPOSURES…';
        try { INTEL.geo = await ifetch(`/api/geo/scenarios?ticker=${encodeURIComponent(t)}&regime=${regime}`); }
        catch(e){ st.innerText = 'UNAVAILABLE'; document.getElementById('geoNarrative').innerHTML = `<div class="text-sm red">${iesc(e.message)}</div>`; return; }
        const sel = document.getElementById('geoScenario');
        const prev = sel.value;
        const sorted = [...INTEL.geo.scenarios].sort((a,b) => b.relevance_pct - a.relevance_pct);
        sel.innerHTML = sorted.map(s => `<option value="${iesc(s.key)}">${iesc(s.name)}${s.relevance_pct>=60?'  — in the news':''}</option>`).join('') + '<option value="custom">Custom scenario (set shocks below)</option>';
        sel.value = prev && [...sel.options].some(o => o.value===prev) ? prev : sorted[0].key;
        geoFillShocks();
        renderGeoMatrix();
    }
    if (!INTEL.sim || INTEL.sim.ticker !== t || force) runGeo();
}
function geoFillShocks(){
    const g = INTEL.geo; if(!g) return;
    const sc = g.scenarios.find(s => s.key === document.getElementById('geoScenario').value) || {shock_values:{}};
    document.getElementById('geoShockInputs').innerHTML = g.factors.map(f => {
        const v = (sc.shock_values||{})[f.symbol];
        const unit = f.unit==='bp' ? 'bp' : f.unit==='pts' ? 'pts' : '%';
        return `<label class="text-xs muted">${iesc(f.label)} (${unit})<input class="intel-input mt-1 mono" data-f="${iesc(f.symbol)}" value="${v==null?'':v}" placeholder="follows history"></label>`;
    }).join('');
}
function renderGeoMatrix(){
    const g = INTEL.geo; if(!g) return;
    document.querySelector('#geoMatrix tbody').innerHTML = [...g.scenarios].sort((a,b) => (a.expected_move??0) - (b.expected_move??0)).map(s => `
        <tr style="cursor:pointer" onclick="document.getElementById('geoScenario').value='${iesc(s.key)}';geoFillShocks();runGeo();window.scrollTo({top:0,behavior:'smooth'})">
        <td>${iesc(s.name)}</td><td>${iesc(s.region)}</td>
        <td><div class="intel-bar" style="width:90px;display:inline-block;vertical-align:middle"><span style="width:${s.relevance_pct}%"></span></div> <span class="muted">${s.headline_hits} stories</span></td>
        <td class="muted">${Object.entries(s.shocks).map(([k,v]) => iesc(((g.factors.find(f=>f.symbol===k)||{}).label||k) + ' ' + v)).join(', ')}</td>
        <td class="${(s.expected_move??0)>=0?'green':'red'}">${s.expected_move==null?'–':(s.expected_move>=0?'+':'')+s.expected_move.toFixed(2)+'%'}</td></tr>`).join('');
}
async function runGeo(){
    const t = itk(); if(!t || !INTEL.geo) return;
    const st = document.getElementById('geoStatus'); st.innerText = 'SIMULATING…';
    const unc = document.getElementById('geoUnc').value; document.getElementById('geoUncVal').innerText = Math.round(unc*100)+'%';
    const shocks = [...document.querySelectorAll('#geoShockInputs input')].filter(i => i.value.trim()!=='' && !isNaN(Number(i.value))).map(i => i.dataset.f + ':' + Number(i.value)).join(',');
    const q = new URLSearchParams({scenario: document.getElementById('geoScenario').value, horizon_weeks: document.getElementById('geoHorizon').value,
        regime: document.getElementById('geoRegime').value, uncertainty: unc});
    if (shocks) q.set('shocks', shocks);
    try { INTEL.sim = await ifetch(`/api/geo/simulate/${encodeURIComponent(t)}?` + q.toString()); st.innerText = 'MODEL R² ' + Math.round((INTEL.sim.model.r2||0)*100) + '%'; renderGeo(); }
    catch(e){ st.innerText = 'ERROR'; document.getElementById('geoNarrative').innerHTML = `<div class="text-sm red">${iesc(e.message)}</div>`; }
}
function renderGeo(){
    const d = INTEL.sim; if(!d) return;
    const s = d.stats;
    const k = [['Expected', s.mean], ['Median', s.median], ['5th pct', s.p5], ['95th pct', s.p95], ['Chance of loss', s.prob_loss, '%'], ['Avg worst 5%', s.es5]];
    document.getElementById('geoKpis').innerHTML = k.map(([l,v,u]) => `<div class="panel p-4"><div class="text-[10px] muted">${l}</div><div class="metric-big mt-2 ${u?'':(v>=0?'green':'red')}" style="font-size:22px">${u ? Number(v).toFixed(0)+'%' : (v>=0?'+':'')+Number(v).toFixed(2)+'%'}</div></div>`).join('');
    document.getElementById('geoNarrative').innerHTML = (d.narrative||[]).map(p => { const m = String(p).match(/^([^:]{3,32}):\s(.*)$/); return `<p class="intel-p">${m?`<b>${iesc(m[1])}:</b> ${iesc(m[2])}`:iesc(p)}</p>`; }).join('');
    document.getElementById('geoChain').innerHTML = (d.scenario.chain||[]).map(c => `<li>${iesc(c)}</li>`).join('') || '<li class="muted">Custom shocks: no predefined chain.</li>';
    document.getElementById('geoAnalogues').innerText = d.scenario.analogues ? 'Historical analogues: ' + d.scenario.analogues : '';
    document.querySelector('#geoFactorTable tbody').innerHTML = (d.factor_table||[]).map(r => `<tr><td>${iesc(r.label)}${r.fixed?' <span class="chip hi">shocked</span>':''}</td>
        <td class="${(r.expected_move||0)>=0?'green':'red'}">${iesc(r.move_text)}</td><td>${r.beta==null?'–':Number(r.beta).toFixed(4)}</td><td class="muted">${r.t==null?'–':Number(r.t).toFixed(1)}</td>
        <td class="${(r.contribution||0)>=0?'green':'red'}">${r.contribution==null?'–':(r.contribution>=0?'+':'')+Number(r.contribution).toFixed(2)+'%'}</td></tr>`).join('');
    const e = d.histogram.edges, c = d.histogram.counts;
    const mids = c.map((_,i) => ((e[i]+e[i+1])/2));
    iChart('geoHist', {type:'bar', data:{labels: mids.map(v => v.toFixed(1)+'%'), datasets:[{data:c, backgroundColor: mids.map(v => v<0?ICOL.red:ICOL.green), borderWidth:0, barPercentage:1, categoryPercentage:1}]},
        options:{responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}}, scales:iAxes({y:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick},title:{display:true,text:'paths',color:ICOL.tick}}})}});
    const cb = Object.entries(d.contributions||{}).filter(([,v]) => v!=null);
    iChart('geoContrib', {type:'bar', data:{labels: cb.map(([k]) => ((d.factor_table.find(r=>r.factor===k)||{}).label||k)), datasets:[{data: cb.map(([,v]) => v), backgroundColor: cb.map(([,v]) => v>=0?ICOL.green:ICOL.red)}]},
        options:{indexAxis:'y', responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false}}, scales:iAxes({x:{grid:{color:ICOL.grid},ticks:{color:ICOL.tick,callback:v=>v+'%'}}})}});
    drawGeoNetwork(d.network);
}
function drawGeoNetwork(net){
    const box = document.getElementById('geoNetwork'); if(!net) { box.innerHTML=''; return; }
    const cols = {trigger:0, primary:1, secondary:2, asset:3};
    const byCol = [[],[],[],[]]; net.nodes.forEach(n => byCol[cols[n.kind]].push(n));
    const W = 980, rowH = 74, H = Math.max(260, Math.max(...byCol.map(c => c.length)) * rowH + 40);
    const X = [20, 270, 540, 800], NW = [190, 170, 170, 160], NH = 48;
    const pos = {};
    byCol.forEach((col, ci) => { const total = col.length * rowH; col.forEach((n, i) => { pos[n.id] = {x: X[ci], y: (H - total)/2 + i*rowH + (rowH-NH)/2, w: NW[ci]}; }); });
    const maxAsset = Math.max(0.01, ...net.edges.filter(e => e.to==='asset').map(e => Math.abs(e.value||0)));
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="min-width:760px" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>`;
    for (const e of net.edges) {
        const a = pos[e.from], b = pos[e.to]; if(!a || !b) continue;
        const x1 = a.x + a.w, y1 = a.y + NH/2, x2 = b.x, y2 = b.y + NH/2, mx = (x1+x2)/2;
        const v = Number(e.value||0); const col = v>=0 ? ICOL.green : ICOL.red;
        const w = e.to==='asset' ? 1.5 + 6*Math.abs(v)/maxAsset : 2;
        svg += `<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2-4},${y2}" fill="none" stroke="${col}" stroke-opacity=".55" stroke-width="${w.toFixed(1)}" marker-end="url(#arr)"/>`;
        svg += `<text x="${mx}" y="${(y1+y2)/2 - 4}" fill="${col}" font-size="10" text-anchor="middle">${iesc(e.label)}</text>`;
    }
    for (const n of net.nodes) {
        const p = pos[n.id];
        const stroke = n.kind==='trigger' ? ICOL.amber : n.kind==='asset' ? ICOL.cyan : n.kind==='primary' ? '#f59e0b' : '#334155';
        const mv = n.move_text ? `<text x="${p.x+10}" y="${p.y+37}" fill="${(n.move||0)>=0?ICOL.green:ICOL.red}" font-size="12" font-family="JetBrains Mono,monospace">${iesc(n.move_text)}${n.z!=null && n.kind!=='asset' ? ' ('+Number(n.z).toFixed(1)+'σ)' : ''}</text>` : '';
        const label = String(n.label); const short = label.length > 28 ? label.slice(0,27)+'…' : label;
        svg += `<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${NH}" rx="10" fill="#0b1119" stroke="${stroke}" stroke-width="1.4"/>`;
        svg += `<text x="${p.x+10}" y="${p.y+18}" fill="#e7edf5" font-size="11" font-weight="600">${iesc(short)}</text>${mv}`;
    }
    ['Trigger','Shocked factors','Spillovers','Asset'].forEach((t,i) => { svg += `<text x="${X[i]}" y="14" fill="#64748b" font-size="9" letter-spacing="1.5">${t.toUpperCase()}</text>`; });
    box.innerHTML = svg + '</svg>';
}

/* ===================== Indicator Lab ===================== */
const IND_OVERLAYS = [
    ['ma','Moving averages',true], ['bb','Bollinger Bands',true], ['kc','Keltner Channels',false], ['dc','Donchian Channel',false],
    ['st','Supertrend',false], ['psar','Parabolic SAR',false], ['vwap','VWAP (rolling)',false], ['ichi','Ichimoku',false]];
INTEL.indOn = Object.fromEntries(IND_OVERLAYS.map(([k,,on]) => [k, on]));
function indOverlayControls(){
    document.getElementById('indOverlays').innerHTML = IND_OVERLAYS.map(([k,l]) => `<label class="intel-check"><input type="checkbox" ${INTEL.indOn[k]?'checked':''} onchange="INTEL.indOn['${k}']=this.checked;renderIndicators()"> ${l}</label>`).join('');
}
async function loadIndicators(force){
    const t = itk(); if(!t) return;
    indOverlayControls();
    const q = new URLSearchParams({period: document.getElementById('indPeriod').value, interval: document.getElementById('indInterval').value,
        sma: document.getElementById('indSma').value, ema: document.getElementById('indEma').value, fwd: document.getElementById('indFwd').value});
    ['bb_n','bb_k','st_n','st_mult','rsi_n','macd_fast','macd_slow','macd_signal','stoch_k','adx_n','cci_n','dc_n','kc_n','vwap_n','mfi_n','wr_n'].forEach(k => { const v = (document.getElementById('ind_'+k)||{}).value; if (v) q.set(k, v); });
    const key = t + '?' + q.toString();
    if (INTEL.ind && INTEL.ind._key === key && !force) return renderIndicators();
    const st = document.getElementById('indStatus'); st.innerText = 'COMPUTING…';
    try { INTEL.ind = await ifetch(`/api/indicators/${encodeURIComponent(t)}?` + q.toString()); INTEL.ind._key = key; st.innerText = INTEL.ind.dates.length + ' BARS'; renderIndicators(); }
    catch(e){ st.innerText = 'ERROR'; document.getElementById('indSummary').innerHTML = `<div class="text-sm red">${iesc(e.message)}</div>`; }
}
function renderIndicators(){
    const d = INTEL.ind; if(!d) return;
    const L = d.dates, O = d.overlays, S = d.oscillators;
    const ds = [{label:'Close', data:d.ohlc.close, borderColor:'#e7edf5', borderWidth:1.6, pointRadius:0}];
    const line = (label, data, color, dash, fill) => ds.push({label, data, borderColor:color, borderWidth:1.2, pointRadius:0, borderDash:dash||[], fill:fill||false, backgroundColor: fill ? 'rgba(34,211,238,.06)' : undefined, spanGaps:true});
    const palette = [ICOL.amber, ICOL.violet, ICOL.green, '#f472b6'];
    if (INTEL.indOn.ma) Object.keys(O).filter(k => /^(SMA|EMA) /.test(k)).forEach((k,i) => line(k, O[k], palette[i%4], k.startsWith('EMA')?[4,3]:[]));
    if (INTEL.indOn.bb) { line('BB upper', O['BB upper'], ICOL.cyan, [2,2]); line('BB lower', O['BB lower'], ICOL.cyan, [2,2], '-1'); line('BB middle', O['BB middle'], 'rgba(34,211,238,.5)', [1,3]); }
    if (INTEL.indOn.kc) { line('Keltner upper', O['Keltner upper'], '#f472b6', [5,3]); line('Keltner lower', O['Keltner lower'], '#f472b6', [5,3]); }
    if (INTEL.indOn.dc) { line('Donchian upper', O['Donchian upper'], ICOL.slate, [1,2]); line('Donchian lower', O['Donchian lower'], ICOL.slate, [1,2]); }
    if (INTEL.indOn.st) line('Supertrend', O['Supertrend'], ICOL.green);
    if (INTEL.indOn.vwap) line('VWAP (rolling)', O['VWAP (rolling)'], '#fb923c');
    if (INTEL.indOn.ichi) { line('Tenkan', O['Ichimoku tenkan'], '#60a5fa'); line('Kijun', O['Ichimoku kijun'], '#f87171'); line('Span A', O['Ichimoku span A'], 'rgba(52,211,153,.6)'); line('Span B', O['Ichimoku span B'], 'rgba(251,113,133,.6)', [], '-1'); }
    if (INTEL.indOn.psar) ds.push({label:'Parabolic SAR', data:O['Parabolic SAR'], showLine:false, pointRadius:1.6, pointBackgroundColor:ICOL.amber, borderColor:ICOL.amber});
    iChart('indPriceChart', {type:'line', data:{labels:L, datasets:ds}, options:{responsive:true, maintainAspectRatio:false, animation:false, interaction:{mode:'index', intersect:false},
        plugins:{legend:{labels:{color:'#8492a3', boxWidth:12, font:{size:10}}}}, scales:iAxes()}});

    const osc = document.getElementById('indOsc').value;
    const od = [], flat = (v, c) => od.push({label:String(v), data:L.map(() => v), borderColor:c||'#334155', borderWidth:1, pointRadius:0, borderDash:[3,3]});
    const add = (label, data, color, type) => od.push({label, data, borderColor:color, backgroundColor: type==='bar' ? (data||[]).map(v => (v||0)>=0?'rgba(25,213,139,.5)':'rgba(251,113,133,.5)') : undefined, type: type||'line', borderWidth:1.3, pointRadius:0, spanGaps:true});
    if (osc==='RSI') { add('RSI', S['RSI'], ICOL.amber); flat(70, ICOL.red); flat(30, ICOL.green); }
    if (osc==='MACD') { add('MACD', S['MACD'], ICOL.cyan); add('Signal', S['MACD signal'], ICOL.amber); add('Histogram', S['MACD histogram'], ICOL.slate, 'bar'); }
    if (osc==='Stochastic') { add('%K', S['Stochastic %K'], ICOL.cyan); add('%D', S['Stochastic %D'], ICOL.amber); flat(80, ICOL.red); flat(20, ICOL.green); }
    if (osc==='ADX / DMI') { add('ADX', S['ADX'], '#e7edf5'); add('+DI', S['+DI'], ICOL.green); add('-DI', S['-DI'], ICOL.red); flat(25); }
    if (osc==='CCI') { add('CCI', S['CCI'], ICOL.violet); flat(100, ICOL.red); flat(-100, ICOL.green); }
    if (osc==='Williams %R') { add('Williams %R', S['Williams %R'], ICOL.cyan); flat(-20, ICOL.red); flat(-80, ICOL.green); }
    if (osc==='MFI') { add('MFI', S['MFI'], ICOL.green); flat(80, ICOL.red); flat(20, ICOL.green); }
    if (osc==='OBV') add('OBV', S['OBV'], ICOL.cyan);
    if (osc==='ATR %') add('ATR % of price', S['ATR %'], ICOL.amber);
    iChart('indOscChart', {type:'line', data:{labels:L, datasets:od}, options:{responsive:true, maintainAspectRatio:false, animation:false, interaction:{mode:'index', intersect:false},
        plugins:{legend:{labels:{color:'#8492a3', boxWidth:12, font:{size:10}, filter: it => isNaN(Number(it.text))}}}, scales:iAxes({x:{display:false}})}});

    const sm = d.summary, b = d.baseline;
    const col = /bullish/i.test(sm.label) ? 'green' : /bearish/i.test(sm.label) ? 'red' : 'cyan';
    document.getElementById('indSummary').innerHTML = `<div class="metric-big ${col}">${iesc(sm.label)}</div>
        <div class="intel-bar mt-3" style="height:8px"><span style="width:${Math.round((sm.score+1)*50)}%"></span></div>
        <div class="flex gap-2 mt-3"><span class="chip pos">${sm.buy} bullish</span><span class="chip neu">${sm.neutral} neutral</span><span class="chip neg">${sm.sell} bearish</span></div>
        <p class="intel-p mt-4">Baseline: over ${b.bars} bars this asset rose ${b.hit_up}% of the time across ${b.forward_bars}-bar windows, averaging ${b.avg_fwd>=0?'+':''}${b.avg_fwd}%. A signal is only useful if its "edge" beats that.</p>`;
    const rel = (r, side) => r ? `<span class="${(r.edge||0)>0?'green':'muted'}">${r.avg_fwd>=0?'+':''}${r.avg_fwd}% · hit ${r.hit}% · edge ${r.edge>=0?'+':''}${r.edge}%</span> <span class="muted">(n=${r.n})</span>` : '<span class="muted">too few signals</span>';
    document.querySelector('#indSignals tbody').innerHTML = d.signals.map(s => `<tr><td>${iesc(s.indicator)}</td>
        <td><span class="chip ${s.signal==='BUY'?'pos':s.signal==='SELL'?'neg':'neu'}">${s.signal==='BUY'?'BULLISH':s.signal==='SELL'?'BEARISH':'NEUTRAL'}</span></td>
        <td class="muted" style="font-family:Inter,sans-serif">${iesc(s.reason)}</td><td>${rel(s.reliability.bullish)}</td><td>${rel(s.reliability.bearish)}</td></tr>`).join('');
    document.getElementById('indNote').innerText = d.note;
}
</script>
'''


def inject(page: str) -> str:
    """Return the dashboard page with the intelligence views wired in."""
    out = page
    if _NAV_OLD in out:
        out = out.replace(_NAV_OLD, _NAV_NEW, 1)
    out = re.sub(r'\n\s*<div id="view-news" class="view-section space-y-6">.*?\n        </div>\n',
                 lambda m: "\n" + HTML_NEWS, out, count=1, flags=re.S)
    out = out.replace("</style>", CSS + "\n</style>", 1)
    idx = out.rfind("</body>")
    if idx != -1:
        out = out[:idx] + JS + "\n" + out[idx:]
    return out
