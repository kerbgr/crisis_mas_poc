#!/usr/bin/env python3
"""
Crisis MAS — HTML Run Report Generator

Combines scenario metadata and run results into a single self-contained HTML
file that can be opened in any browser without a web server.

Usage
-----
    python scripts/generate_report.py
    python scripts/generate_report.py --scenario santorini_volcanic_seismic
    python scripts/generate_report.py --scenario santorini_volcanic_seismic \
        --run results/santorini_volcanic_seismic/run_1_lmstudio \
        --output reports/santorini_report.html
"""

import argparse
import base64
import json
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Offline web-asset bundler
# ---------------------------------------------------------------------------
# All CDN resources are downloaded once and cached in scripts/.asset_cache.json.
# Every subsequent report generation reads from the cache — no internet needed.

_ASSET_CACHE_PATH = Path(__file__).parent / ".asset_cache.json"

_CDN = {
    "bootstrap_css":  "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css",
    "bootstrap_js":   "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js",
    "bi_css":         "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
    "bi_font_woff2":  "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2",
    "leaflet_css":    "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css",
    "leaflet_js":     "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js",
    # Leaflet marker images used by the default icon
    "leaflet_marker":      "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    "leaflet_marker_2x":   "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    "leaflet_marker_shadow":"https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
}


def _fetch(url: str, binary: bool = False):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read() if binary else r.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def _build_asset_cache() -> dict:
    """Download all CDN assets, patch font/image URLs to base64 data URIs, cache."""
    print("  Downloading web assets for offline embedding...", flush=True)
    cache = {}

    # Plain text resources
    for key in ("bootstrap_css", "bootstrap_js", "leaflet_js"):
        print(f"    {key}...", end="", flush=True)
        cache[key] = _fetch(_CDN[key])
        print(f" {len(cache[key])//1024} KB")

    # Bootstrap Icons font (woff2) -> base64
    print("    bi_font_woff2...", end="", flush=True)
    font_bytes = _fetch(_CDN["bi_font_woff2"], binary=True)
    font_b64   = base64.b64encode(font_bytes).decode()
    cache["bi_font_woff2_b64"] = font_b64
    print(f" {len(font_bytes)//1024} KB")

    # Bootstrap Icons CSS — patch relative font URL to embedded data URI
    print("    bi_css (patching font URLs)...", end="", flush=True)
    bi_css = _fetch(_CDN["bi_css"])
    # Replace both woff2 and woff references; we only have woff2 so drop the woff fallback
    bi_css = re.sub(
        r'url\("[^"]*\.woff2[^"]*"\)\s*format\("woff2"\)',
        f'url("data:font/woff2;base64,{font_b64}") format("woff2")',
        bi_css,
    )
    # Remove the woff fallback entirely (we embedded woff2 which all modern browsers support)
    bi_css = re.sub(r',\s*\n?\s*url\("[^"]*\.woff[^"]*"\)\s*format\("woff"\)', "", bi_css)
    cache["bi_css"] = bi_css
    print(f" {len(bi_css)//1024} KB (fonts inlined)")

    # Leaflet CSS — patch marker image URLs to base64 data URIs
    print("    leaflet_css (patching marker images)...", end="", flush=True)
    leaflet_css = _fetch(_CDN["leaflet_css"])
    for img_key, img_url in (
        ("leaflet_marker",       _CDN["leaflet_marker"]),
        ("leaflet_marker_2x",    _CDN["leaflet_marker_2x"]),
        ("leaflet_marker_shadow",_CDN["leaflet_marker_shadow"]),
    ):
        img_b64 = base64.b64encode(_fetch(img_url, binary=True)).decode()
        cache[img_key + "_b64"] = img_b64
        fname = img_url.split("/")[-1]
        leaflet_css = leaflet_css.replace(
            f"images/{fname}",
            f"data:image/png;base64,{img_b64}",
        )
    cache["leaflet_css"] = leaflet_css
    print(f" {len(leaflet_css)//1024} KB (images inlined)")

    _ASSET_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    print(f"  Assets cached -> {_ASSET_CACHE_PATH}")
    return cache


_ASSETS: dict = {}


def get_assets() -> dict:
    """Return cached assets, downloading if the cache file does not exist."""
    global _ASSETS
    if _ASSETS:
        return _ASSETS
    if _ASSET_CACHE_PATH.exists():
        _ASSETS = json.loads(_ASSET_CACHE_PATH.read_text(encoding="utf-8"))
    else:
        _ASSETS = _build_asset_cache()
    return _ASSETS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_img(path: Path) -> str:
    """Return a data-URI string for a PNG, or empty string if file missing."""
    if not path or not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _load_json(path: Path) -> dict:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _severity_class(s: float) -> str:
    if s >= 0.9:
        return "danger"
    if s >= 0.7:
        return "warning"
    return "success"


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


METHOD_COLORS = {
    "er":          ("primary",  "ER",          "bi-diagram-2"),
    "gat":         ("danger",   "GAT",         "bi-graph-up"),
    "gat_trained": ("success",  "GAT-Trained", "bi-stars"),
    "mcda":        ("warning",  "MCDA",        "bi-bar-chart-steps"),
}

CRITERIA_ICONS = {
    "effectiveness": "bi-bullseye",
    "safety":        "bi-shield-check",
    "speed":         "bi-lightning-charge",
    "cost":          "bi-currency-euro",
    "public_acceptance": "bi-people",
}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_scenario(sc: dict) -> str:
    loc = sc.get("location", {})
    lat = loc.get("coordinates", {}).get("lat", 37.9)
    lon = loc.get("coordinates", {}).get("lon", 23.7)
    sev = sc.get("severity", 0)
    sev_cls = _severity_class(sev)
    tags_html = " ".join(
        f'<span class="badge bg-secondary me-1">{t.replace("_"," ").title()}</span>'
        for t in sc.get("tags", [])
    )

    rt = sc.get("real_time_factors", {})
    rt_rows = "".join(
        f'<tr><td class="text-muted small">{k.replace("_"," ").title()}</td>'
        f'<td class="small">{v}</td></tr>'
        for k, v in rt.items()
    )

    constraints = sc.get("constraints", {})
    res_lims = constraints.get("resource_limitations", [])
    res_html = "".join(
        f'<li class="small text-muted">{r.replace("_", " ")}</li>'
        for r in res_lims
    )
    add_concerns = constraints.get("additional_concerns", [])
    concerns_html = "".join(
        f'<li class="small">{c}</li>' for c in add_concerns
    )

    # Map JS returned separately so it runs after Leaflet library loads
    map_js = f"""
(function() {{
  var map = L.map('scenario-map').setView([{lat}, {lon}], 11);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{attribution:'© OpenStreetMap contributors', maxZoom:18}}).addTo(map);
  L.marker([{lat},{lon}])
   .addTo(map)
   .bindPopup('<b>{sc.get("name","").replace("'","&#39;")}</b><br>Severity: {_pct(sev)}')
   .openPopup();
}})();"""

    html = f"""
<section id="scenario" class="mb-5">
  <div class="card shadow-sm border-{sev_cls}">
    <div class="card-header bg-{sev_cls} {'text-dark' if sev_cls == 'warning' else 'text-white'}">
      <h4 class="mb-0"><i class="bi bi-map-fill me-2"></i>{sc.get('name','Scenario')}</h4>
    </div>
    <div class="card-body">

      <!-- Key stats row -->
      <div class="row g-3 mb-4">
        <div class="col-6 col-md-3">
          <div class="card text-center border-0 bg-light h-100">
            <div class="card-body py-2">
              <div class="display-6 fw-bold text-{sev_cls}">{_pct(sev)}</div>
              <small class="text-muted">Severity</small>
            </div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="card text-center border-0 bg-light h-100">
            <div class="card-body py-2">
              <div class="display-6 fw-bold text-dark">{sc.get('affected_population',0):,}</div>
              <small class="text-muted">Affected Population</small>
            </div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="card text-center border-0 bg-light h-100">
            <div class="card-body py-2">
              <div class="display-6 fw-bold text-danger">{sc.get('casualties',0)}</div>
              <small class="text-muted">Casualties</small>
            </div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="card text-center border-0 bg-light h-100">
            <div class="card-body py-2">
              <div class="display-6 fw-bold text-info">{len(sc.get('available_actions', sc.get('alternatives',[])))}</div>
              <small class="text-muted">Response Alternatives</small>
            </div>
          </div>
        </div>
      </div>

      <!-- Map (full width) -->
      <div class="mb-3">
        <div id="scenario-map" style="height:340px;border-radius:8px;overflow:hidden;border:1px solid #dee2e6;"></div>
        <p class="text-muted small mt-1 mb-0">
          <i class="bi bi-geo-alt-fill text-danger me-1"></i>
          {loc.get('affected_area_description', loc.get('region',''))}
        </p>
      </div>

      <!-- Description -->
      <div class="mb-3">
        <p class="mb-2">{sc.get('description','')}</p>
        <div class="mb-2">{tags_html}</div>
        <p class="small text-muted mb-0">
          <i class="bi bi-link-45deg me-1"></i>
          <em>{sc.get('event_reference','')}</em>
        </p>
      </div>

      <!-- Real-time factors -->
      {'<hr><h6 class="text-muted"><i class="bi bi-clock-history me-1"></i>Real-Time Factors at Incident Time</h6>' if rt_rows else ''}
      {f'<div class="table-responsive"><table class="table table-sm table-bordered mb-0"><tbody>{rt_rows}</tbody></table></div>' if rt_rows else ''}

      <!-- Resource limitations & concerns -->
      {'<hr><div class="row g-3">' if res_html or concerns_html else ''}
      {f'<div class="col-md-6"><h6 class="text-muted"><i class="bi bi-exclamation-triangle me-1"></i>Resource Limitations</h6><ul class="mb-0">{res_html}</ul></div>' if res_html else ''}
      {f'<div class="col-md-6"><h6 class="text-muted"><i class="bi bi-chat-square-warning me-1"></i>Additional Concerns</h6><ul class="mb-0">{concerns_html}</ul></div>' if concerns_html else ''}
      {'</div>' if res_html or concerns_html else ''}
    </div>
  </div>
</section>
"""
    return html, map_js


def _section_alternatives(sc: dict) -> str:
    alts = sc.get("available_actions", sc.get("alternatives", []))
    if not alts:
        return ""

    cards = []
    for alt in alts:
        crit = alt.get("criteria_scores", {})
        crit_bars = ""
        for k, v in crit.items():
            icon = CRITERIA_ICONS.get(k, "bi-check2")
            pct = int(v * 100)
            color = "success" if pct >= 70 else "warning" if pct >= 40 else "danger"
            crit_bars += f"""
            <div class="d-flex align-items-center mb-1">
              <i class="bi {icon} me-2 text-muted small" style="width:16px"></i>
              <span class="small text-muted me-2" style="width:120px">{k.replace('_',' ').title()}</span>
              <div class="progress flex-grow-1" style="height:14px">
                <div class="progress-bar bg-{color}" style="width:{pct}%">
                  <span class="small">{pct}%</span>
                </div>
              </div>
            </div>"""

        risk = alt.get("risk_level", 0)
        risk_cls = "danger" if risk >= 0.6 else "warning" if risk >= 0.35 else "success"
        resources = alt.get("required_resources", [])
        res_badges = " ".join(
            f'<span class="badge bg-light text-dark border me-1 mb-1 small">'
            f'<i class="bi bi-gear-fill me-1 text-muted"></i>'
            f'{r.replace("_"," ")}</span>'
            for r in resources[:6]
        ) + (f' <span class="text-muted small">+{len(resources)-6} more</span>' if len(resources) > 6 else "")

        cards.append(f"""
      <div class="col-md-6 col-xl-4 mb-3">
        <div class="card h-100 shadow-sm">
          <div class="card-header d-flex justify-content-between align-items-start">
            <h6 class="mb-0 me-2 small fw-bold">{alt.get('name', alt.get('id',''))}</h6>
            <span class="badge bg-{risk_cls} text-nowrap">Risk {int(risk*100)}%</span>
          </div>
          <div class="card-body">
            <p class="small text-muted mb-2">{alt.get('description','')[:180]}{'…' if len(alt.get('description',''))>180 else ''}</p>
            {crit_bars}
            <div class="mt-2">
              <i class="bi bi-clock text-muted me-1 small"></i>
              <span class="small text-muted">{alt.get('estimated_duration','')}</span>
            </div>
          </div>
          {'<div class="card-footer bg-transparent py-2">'+res_badges+'</div>' if res_badges else ''}
        </div>
      </div>""")

    return f"""
<section id="alternatives" class="mb-5">
  <h4 class="section-title"><i class="bi bi-list-check me-2"></i>Response Alternatives
    <span class="badge bg-secondary ms-2">{len(alts)}</span>
  </h4>
  <div class="row g-2">
    {"".join(cards)}
  </div>
</section>
"""


def _embed_img(path: Path, alt: str = "", css_class: str = "img-fluid rounded shadow-sm") -> str:
    src = _b64_img(path)
    if not src:
        return f'<p class="text-muted small"><i class="bi bi-image"></i> {alt} not available</p>'
    return f'<img src="{src}" class="{css_class}" alt="{alt}" style="max-width:100%">'


def _section_run_results(run_dir: Path, comp: dict) -> str:
    """Build the per-method tabbed results section for one run."""

    methods = list(comp.get("methods", {}).keys()) or ["er", "gat", "gat_trained", "mcda"]
    # Tabs header
    tab_nav = ""
    tab_content = ""

    for i, method in enumerate(methods):
        m_lower = method.lower()
        color, label, icon = METHOD_COLORS.get(m_lower, ("secondary", method, "bi-gear"))
        active = "active" if i == 0 else ""
        show = "show active" if i == 0 else ""
        tab_id = f"tab-{m_lower}"

        m_data = comp.get("methods", {}).get(method, {})
        rec = m_data.get("recommended_alternative", "—")
        rec_display = rec.replace("action_", "").replace("_", " ").title()
        conf = m_data.get("confidence", 0)
        cons = m_data.get("consensus_level", 0)
        dqs  = m_data.get("decision_quality_score", 0)
        time_ms = m_data.get("processing_time_ms", 0)

        method_dir = run_dir / m_lower
        imgs = {
            "belief": method_dir / "belief_distributions.png",
            "dqs":    method_dir / "dqs_breakdown.png",
            "network":method_dir / "agent_network.png",
            "consensus": method_dir / "consensus_evolution.png",
            "criteria":  method_dir / "criteria_importance.png",
            "decision":  method_dir / "decision_comparison.png",
        }

        tab_nav += f"""
          <li class="nav-item" role="presentation">
            <button class="nav-link {active}" id="{tab_id}-btn"
              data-bs-toggle="tab" data-bs-target="#{tab_id}"
              type="button" role="tab">
              <i class="bi {icon} me-1"></i>{label}
            </button>
          </li>"""

        # Metric cards
        metric_cards = f"""
          <div class="row g-2 mb-3">
            <div class="col-6 col-md-3">
              <div class="card text-center border-0 bg-light">
                <div class="card-body py-2">
                  <div class="h4 fw-bold text-{color}">{conf:.3f}</div>
                  <small class="text-muted">Confidence</small>
                </div>
              </div>
            </div>
            <div class="col-6 col-md-3">
              <div class="card text-center border-0 bg-light">
                <div class="card-body py-2">
                  <div class="h4 fw-bold text-{color}">{cons:.3f}</div>
                  <small class="text-muted">Consensus</small>
                </div>
              </div>
            </div>
            <div class="col-6 col-md-3">
              <div class="card text-center border-0 bg-light">
                <div class="card-body py-2">
                  <div class="h4 fw-bold text-{color}">{dqs:.3f}</div>
                  <small class="text-muted">DQS</small>
                </div>
              </div>
            </div>
            <div class="col-6 col-md-3">
              <div class="card text-center border-0 bg-light">
                <div class="card-body py-2">
                  <div class="h4 fw-bold text-{color}">{time_ms/1000:.1f}s</div>
                  <small class="text-muted">Processing</small>
                </div>
              </div>
            </div>
          </div>"""

        # Recommendation badge
        rec_html = f"""
          <div class="alert alert-{color} d-flex align-items-center py-2 mb-3">
            <i class="bi bi-award-fill me-2 fs-5"></i>
            <div>
              <strong>Recommended:</strong>
              <span class="ms-2 fw-bold">{rec_display}</span>
              <code class="ms-2 small text-muted">({rec})</code>
            </div>
          </div>"""

        # Images grid
        img_pairs = [
            ("Agent Belief Distributions", imgs["belief"]),
            ("Alternative DQS Decomposition", imgs["dqs"]),
            ("Agent Network", imgs["network"]),
            ("Consensus Evolution", imgs["consensus"]),
            ("Criteria Importance", imgs["criteria"]),
            ("Decision Comparison", imgs["decision"]),
        ]
        img_grid = '<div class="row g-3">'
        for img_label, img_path in img_pairs:
            src = _b64_img(img_path)
            if src:
                img_grid += f"""
          <div class="col-md-6">
            <div class="card border-0 shadow-sm h-100">
              <div class="card-header py-1 bg-light">
                <small class="text-muted fw-semibold">{img_label}</small>
              </div>
              <div class="card-body p-2 text-center">
                <img src="{src}" class="img-fluid rounded" alt="{img_label}">
              </div>
            </div>
          </div>"""
        img_grid += "</div>"

        tab_content += f"""
          <div class="tab-pane fade {show}" id="{tab_id}" role="tabpanel">
            {rec_html}
            {metric_cards}
            {img_grid}
          </div>"""

    return f"""
  <ul class="nav nav-tabs mb-0" id="methodTabs" role="tablist">
    {tab_nav}
  </ul>
  <div class="tab-content border border-top-0 rounded-bottom p-3 bg-white shadow-sm">
    {tab_content}
  </div>"""


def _section_comparison(run_dir: Path, comp: dict) -> str:
    """Embedded comparison charts + summary table."""
    comp_imgs = [
        ("Method Comparison",      run_dir / "er_vs_gat_metrics.png"),
        ("Recommendations",        run_dir / "er_vs_gat_recommendations.png"),
        ("Summary",                run_dir / "er_vs_gat_summary.png"),
        ("DQS Deviation",          run_dir / "er_vs_gat_dqs_deviation.png"),
    ]

    methods = comp.get("methods", {})
    tbl_rows = ""
    for method, mdata in methods.items():
        m_lower = method.lower()
        color, label, icon = METHOD_COLORS.get(m_lower, ("secondary", method, "bi-gear"))
        rec = mdata.get("recommended_alternative", "—").replace("action_", "").replace("_", " ").title()
        tbl_rows += f"""
        <tr>
          <td><span class="badge bg-{color}"><i class="bi {icon} me-1"></i>{label}</span></td>
          <td class="fw-semibold">{rec}</td>
          <td>{mdata.get('confidence', 0):.3f}</td>
          <td>{mdata.get('consensus_level', 0):.3f}</td>
          <td>{mdata.get('decision_quality_score', 0):.3f}</td>
          <td>{mdata.get('processing_time_ms', 0)/1000:.1f}s</td>
        </tr>"""

    same_all = comp.get("comparison", {}).get("same_recommendation_all", False)
    agreement_badge = (
        '<span class="badge bg-success"><i class="bi bi-check-circle me-1"></i>All methods agree</span>'
        if same_all else
        '<span class="badge bg-warning text-dark"><i class="bi bi-exclamation-triangle me-1"></i>Methods diverge</span>'
    )

    img_grid = '<div class="row g-3">'
    for label, path in comp_imgs:
        src = _b64_img(path)
        if src:
            img_grid += f"""
      <div class="col-md-6">
        <div class="card border-0 shadow-sm">
          <div class="card-header py-1 bg-light">
            <small class="text-muted fw-semibold">{label}</small>
          </div>
          <div class="card-body p-2 text-center">
            <img src="{src}" class="img-fluid rounded" alt="{label}">
          </div>
        </div>
      </div>"""
    img_grid += "</div>"

    return f"""
<section id="comparison" class="mb-5">
  <h4 class="section-title"><i class="bi bi-columns-gap me-2"></i>4-Method Comparative Analysis
    <span class="ms-2">{agreement_badge}</span>
  </h4>

  <div class="table-responsive mb-4">
    <table class="table table-bordered table-hover align-middle">
      <thead class="table-dark">
        <tr>
          <th>Method</th><th>Recommendation</th><th>Confidence</th>
          <th>Consensus</th><th>DQS</th><th>Time</th>
        </tr>
      </thead>
      <tbody>{tbl_rows}</tbody>
    </table>
  </div>

  {img_grid}
</section>
"""


DOMAIN_COLORS = {
    "flood":           ("primary",  "bi-water"),
    "wildfire":        ("danger",   "bi-fire"),
    "hazmat":          ("warning",  "bi-radioactive"),
    "volcanic_seismic":("dark",     "bi-tornado"),
}

LEVEL_META = {
    "gold":   ("Gold Strategic",   "warning text-dark", "bi-star-fill"),
    "silver": ("Silver Tactical",  "secondary",         "bi-person-badge"),
}


def _section_vision_subsystem(run_dir: Path) -> str:
    """Geospatial terrain analysis + camera feed reports from the vision subsystem."""
    # Pull vision data from the first available method result
    decision = {}
    for method_dir in ("er", "gat", "gat_trained"):
        result_path = run_dir / method_dir / "results.json"
        if result_path.exists():
            raw = _load_json(result_path)
            decision = raw.get("decision", {})
            if "geospatial_context" in decision or "camera_feed_reports" in decision:
                break

    geo = decision.get("geospatial_context")
    feeds = decision.get("camera_feed_reports", [])
    excluded = decision.get("agents_excluded_by_terrain", [])

    if not geo and not feeds:
        return ""

    # --- Geospatial card ---
    terrain_icons = {
        "island":           ("bi-water",         "primary"),
        "coastal_mainland": ("bi-tsunami",        "info"),
        "inland":           ("bi-mountains",      "secondary"),
        "unknown":          ("bi-question-circle","secondary"),
    }
    terrain_type = geo.get("terrain_type", "unknown") if geo else "unknown"
    t_icon, t_color = terrain_icons.get(terrain_type, ("bi-geo", "secondary"))
    island_name = (geo.get("island_name") or "").replace("_", " ").title() if geo else ""
    water_access = geo.get("has_water_access", True) if geo else True
    method_used = (geo.get("method_used") or "—") if geo else "—"
    coords = geo.get("coordinates", {}) if geo else {}
    lat = coords.get("lat", "—")
    lon = coords.get("lon", "—")

    water_badge = (
        '<span class="badge bg-success"><i class="bi bi-water me-1"></i>Maritime access</span>'
        if water_access else
        '<span class="badge bg-danger"><i class="bi bi-x-circle me-1"></i>No maritime access</span>'
    )
    excl_html = ""
    if excluded:
        excl_html = (
            '<div class="mt-2"><span class="text-danger small fw-semibold">'
            '<i class="bi bi-slash-circle me-1"></i>Excluded by terrain:</span> '
            + " ".join(f'<code class="small">{a}</code>' for a in excluded)
            + "</div>"
        )

    geo_html = f"""
    <div class="col-md-4">
      <div class="card border-{t_color} h-100 shadow-sm">
        <div class="card-header bg-{t_color} {'text-dark' if t_color == 'info' else 'text-white'} py-2">
          <i class="bi {t_icon} me-2"></i><strong>Terrain Classification</strong>
        </div>
        <div class="card-body">
          <div class="d-flex align-items-center gap-2 mb-2">
            <span class="badge bg-{t_color} fs-6">{terrain_type.replace('_',' ').title()}</span>
            {water_badge}
          </div>
          {'<div class="small text-muted mb-1"><i class="bi bi-geo-alt me-1"></i><strong>Island:</strong> '+island_name+'</div>' if island_name else ''}
          <div class="small text-muted mb-1">
            <i class="bi bi-crosshair me-1"></i>
            <strong>Coordinates:</strong> {lat}, {lon}
          </div>
          <div class="small text-muted mb-1">
            <i class="bi bi-cpu me-1"></i>
            <strong>Method:</strong> <code>{method_used}</code>
          </div>
          {excl_html}
        </div>
      </div>
    </div>""" if geo else ""

    # --- Camera feed cards ---
    status_meta = {
        "ok":                      ("success",   "bi-camera-video-fill",  "Online"),
        "feed_unavailable":        ("secondary", "bi-camera-video-off",   "Unavailable"),
        "vision_model_unavailable":("warning",   "bi-cpu-fill",           "Vision offline"),
        "parse_failed":            ("danger",    "bi-exclamation-circle", "Parse error"),
    }
    severity_colors = {"normal": "success", "elevated": "warning", "high": "danger", "critical": "danger"}

    feed_cards = ""
    for feed in feeds:
        status = feed.get("status", "unknown")
        label = feed.get("label", feed.get("url", "Camera"))
        mode = feed.get("mode", "general")
        s_color, s_icon, s_label = status_meta.get(status, ("secondary", "bi-camera", status))
        sev = feed.get("situation_severity", "")
        sev_color = severity_colors.get(sev, "secondary")
        summary = feed.get("situation_summary", "")

        # Critical indicator pills
        indicators = []
        if feed.get("tsunami_indicators_present"):   indicators.append(("danger",  "bi-exclamation-triangle-fill", "TSUNAMI"))
        if feed.get("water_withdrawal_observed"):    indicators.append(("danger",  "bi-arrow-down-circle-fill",    "Sea withdrawal"))
        if feed.get("wave_detected"):                indicators.append(("danger",  "bi-water",                    "Wave detected"))
        if feed.get("panic_indicators"):             indicators.append(("danger",  "bi-person-exclamation",       "Panic"))
        if feed.get("stampede_risk"):                indicators.append(("danger",  "bi-people-fill",              "Stampede risk"))
        if feed.get("fire_visible"):                 indicators.append(("danger",  "bi-fire",                     "Fire visible"))
        if feed.get("smoke_visible"):                indicators.append(("warning", "bi-cloud-haze-fill",          "Smoke"))
        ind_html = "".join(
            f'<span class="badge bg-{c} me-1"><i class="bi {i} me-1"></i>{t}</span>'
            for c, i, t in indicators
        )
        warning_level = feed.get("warning_level", "")
        if warning_level and warning_level != "none":
            wl_color = {"watch": "warning", "warning": "danger", "emergency": "danger"}.get(warning_level, "secondary")
            ind_html += f'<span class="badge bg-{wl_color} me-1">WARNING LEVEL: {warning_level.upper()}</span>'

        feed_cards += f"""
    <div class="col-md-4">
      <div class="card h-100 shadow-sm">
        <div class="card-header d-flex justify-content-between align-items-start py-2">
          <span class="small fw-semibold text-truncate me-2" title="{label}">{label[:60]}</span>
          <span class="badge bg-{s_color} text-nowrap">
            <i class="bi {s_icon} me-1"></i>{s_label}
          </span>
        </div>
        <div class="card-body py-2">
          <div class="d-flex gap-1 mb-2 flex-wrap">
            <span class="badge bg-light text-dark border"><i class="bi bi-eye me-1"></i>{mode}</span>
            {f'<span class="badge bg-{sev_color}">{sev}</span>' if sev else ''}
          </div>
          {f'<div class="alert alert-danger py-1 px-2 mb-2 small">{ind_html}</div>' if ind_html else ''}
          {f'<p class="small text-muted mb-0">{summary}</p>' if summary else '<p class="small text-muted mb-0 fst-italic">No data</p>'}
        </div>
      </div>
    </div>"""

    feeds_html = f"""
    <div class="row g-3 mt-0">
      {feed_cards}
    </div>""" if feed_cards else ""

    return f"""
<section id="vision" class="mb-5">
  <h4 class="section-title"><i class="bi bi-camera-fill me-2"></i>Vision Subsystem — Pre-Assessment Layer</h4>
  <div class="row g-3">
    {geo_html}
    {'<div class="col-md-8"><h6 class="text-muted mb-2"><i class="bi bi-camera-video me-1"></i>Camera Feed Intelligence</h6>' + feeds_html + '</div>' if feeds_html else ''}
  </div>
</section>
"""


def _section_agent_reliability(run_dir: Path) -> str:
    """Per-agent reliability from results/reliability/ JSON files."""
    rel_dir = _ROOT / "results" / "reliability"
    if not rel_dir.exists():
        return ""

    agents = []
    for f in sorted(rel_dir.glob("*.json")):
        try:
            d       = json.loads(f.read_text())
            metrics = d.get("metrics", d)          # support both flat and nested formats
            aid     = d.get("agent_id", f.stem)

            score   = metrics.get("overall_reliability",  d.get("overall_reliability",  0))
            recent  = metrics.get("recent_reliability",   d.get("recent_reliability",   score))
            cons    = metrics.get("consistency_score",    d.get("consistency_score",    0))
            n_total = metrics.get("total_assessments",    d.get("total_assessments",    0))
            n_acc   = metrics.get("accurate_assessments", d.get("accurate_assessments", 0))
            acc_rt  = metrics.get("accuracy_rate",        d.get("accuracy_rate",        n_acc / n_total if n_total else 0))
            domain  = metrics.get("domain_reliability",   d.get("domain_reliability",   {}))
            updated = metrics.get("last_updated",         d.get("last_updated",         ""))[:10]
            tags    = d.get("expertise_tags", [])

            # Infer GOLD/SILVER from agent id
            lvl_key = "gold" if "gold" in aid else "silver"
            lvl_label, lvl_badge_cls, lvl_icon = LEVEL_META[lvl_key]

            score_col = "success" if score >= 0.60 else "warning" if score >= 0.45 else "danger"
            agents.append(dict(
                score=score, aid=aid, lvl_label=lvl_label,
                lvl_badge_cls=lvl_badge_cls, lvl_icon=lvl_icon,
                score_col=score_col, recent=recent, cons=cons,
                n_total=n_total, acc_rt=acc_rt, domain=domain,
                updated=updated, tags=tags,
            ))
        except Exception:
            continue

    agents.sort(key=lambda x: x["score"], reverse=True)
    if not agents:
        return ""

    rows = []
    for i, a in enumerate(agents):
        # Domain reliability mini-bars
        domain_html = ""
        for dname, dval in a["domain"].items():
            d_col, d_icon = DOMAIN_COLORS.get(dname, ("secondary", "bi-globe"))
            d_pct = int(dval * 100)
            d_bar_col = "success" if dval >= 0.60 else "warning" if dval >= 0.40 else "danger"
            domain_html += f"""
              <div class="d-flex align-items-center gap-1 mb-1">
                <i class="bi {d_icon} text-{d_col}" style="width:14px;font-size:0.75rem"></i>
                <span class="text-muted" style="width:88px;font-size:0.72rem">{dname.replace('_',' ').title()}</span>
                <div class="progress flex-grow-1" style="height:10px;min-width:60px">
                  <div class="progress-bar bg-{d_bar_col}" style="width:{d_pct}%"></div>
                </div>
                <span style="font-size:0.72rem;width:32px;text-align:right">{dval:.2f}</span>
              </div>"""

        # Expertise tags (collapsed, show first 4)
        tag_html = ""
        if a["tags"]:
            visible = a["tags"][:4]
            extra   = a["tags"][4:]
            tag_html = "".join(
                f'<span class="badge bg-light text-dark border me-1 mb-1" style="font-size:0.68rem">'
                f'{t.replace("_"," ")}</span>'
                for t in visible
            )
            if extra:
                tag_html += (
                    f'<a class="small text-muted" data-bs-toggle="collapse" '
                    f'href="#tags-{i}" style="cursor:pointer">+{len(extra)} more</a>'
                    f'<div class="collapse" id="tags-{i}">'
                    + "".join(
                        f'<span class="badge bg-light text-dark border me-1 mb-1" style="font-size:0.68rem">'
                        f'{t.replace("_"," ")}</span>'
                        for t in extra
                    )
                    + "</div>"
                )

        rows.append(f"""
        <tr>
          <td>
            <div class="fw-semibold small">{a['aid'].replace('_',' ').title()}</div>
            <div class="mt-1">{tag_html}</div>
          </td>
          <td class="text-center">
            <span class="badge bg-{a['lvl_badge_cls']}">
              <i class="bi {a['lvl_icon']} me-1"></i>{a['lvl_label']}
            </span>
          </td>
          <td>
            <div class="d-flex align-items-center gap-2">
              <div class="progress flex-grow-1" style="height:14px">
                <div class="progress-bar bg-{a['score_col']}" style="width:{int(a['score']*100)}%"></div>
              </div>
              <span class="fw-bold small text-{a['score_col']}">{a['score']:.3f}</span>
            </div>
          </td>
          <td class="text-center small">{a['recent']:.3f}</td>
          <td class="text-center small">{a['cons']:.3f}</td>
          <td class="text-center small">{int(a['acc_rt']*100)}%<br>
            <span class="text-muted" style="font-size:0.7rem">{a['n_total']} runs</span>
          </td>
          <td>{domain_html if domain_html else '<span class="text-muted small">n/a</span>'}</td>
          <td class="text-muted" style="font-size:0.72rem">{a['updated']}</td>
        </tr>""")

    return f"""
<section id="agents" class="mb-5">
  <h4 class="section-title"><i class="bi bi-people-fill me-2"></i>Agent Reliability Panel
    <span class="badge bg-secondary ms-2">{len(agents)} agents</span>
  </h4>
  <div class="table-responsive">
    <table class="table table-sm table-bordered table-hover align-middle">
      <thead class="table-dark">
        <tr>
          <th style="min-width:200px">Agent / Expertise</th>
          <th class="text-center">Level</th>
          <th style="min-width:160px">Overall Reliability</th>
          <th class="text-center">Recent</th>
          <th class="text-center">Consistency</th>
          <th class="text-center">Accuracy</th>
          <th style="min-width:180px">Domain Reliability</th>
          <th class="text-center">Updated</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</section>
"""


# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------

def generate_html(scenario_path: Path, run_dir: Path) -> str:
    sc  = _load_json(scenario_path)
    comp = _load_json(run_dir / "comparative_analysis.json")

    run_label = run_dir.name
    provider  = run_label.split("_")[-1].upper() if "_" in run_label else run_label
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M")

    s_scenario, map_js = _section_scenario(sc)
    s_alts        = _section_alternatives(sc)
    s_vision      = _section_vision_subsystem(run_dir)
    s_run_results = _section_run_results(run_dir, comp) if comp else ""
    s_comparison  = _section_comparison(run_dir, comp) if comp else ""
    s_agents      = _section_agent_reliability(run_dir)

    # Load offline assets; fall back to CDN links if unavailable
    try:
        assets = get_assets()
    except Exception as e:
        print(f"\n  Warning: offline assets unavailable ({e}), falling back to CDN", flush=True)
        assets = {}

    if assets.get("bootstrap_css"):
        # Inline all CSS/JS - fully offline, no CDN needed
        # Assets are computed as variables first because f-strings cannot contain
        # arbitrary {}-laden CSS/JS text as direct expressions.
        _css_bootstrap = assets["bootstrap_css"]
        _css_bi        = assets["bi_css"]
        _css_leaflet   = assets["leaflet_css"]
        _js_bootstrap  = assets["bootstrap_js"]
        _js_leaflet    = assets["leaflet_js"]
        head_css   = f"  <style>{_css_bootstrap}</style>\n  <style>{_css_bi}</style>\n  <style>{_css_leaflet}</style>"
        foot_libs  = f"<script>{_js_bootstrap}</script>\n<script>{_js_leaflet}</script>"
    else:
        head_css  = (
            '  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">\n'
            '  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">\n'
            '  <link href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" rel="stylesheet">'
        )
        foot_libs = (
            '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>\n'
            '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{sc.get('name', 'Crisis MAS Report')} - Run Report</title>

{head_css}

  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; }}
    .sidebar {{
      position: sticky; top: 70px; height: calc(100vh - 90px);
      overflow-y: auto; font-size: 0.875rem;
    }}
    .nav-pills .nav-link {{ color: #495057; border-radius: 6px; padding: 6px 12px; }}
    .nav-pills .nav-link.active {{ background: #0d6efd; color: #fff; }}
    .section-title {{
      font-size: 1.1rem; font-weight: 700; color: #343a40;
      border-left: 4px solid #0d6efd; padding-left: 10px; margin-bottom: 1rem;
    }}
    .main-content {{ max-width: 1200px; }}
    section {{ scroll-margin-top: 80px; }}
    .navbar-brand {{ font-weight: 700; letter-spacing: -0.5px; }}
    @media (max-width: 768px) {{ .sidebar {{ display: none; }} }}
  </style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-dark bg-primary sticky-top shadow-sm">
  <div class="container-fluid">
    <span class="navbar-brand">
      <i class="bi bi-shield-fill-check me-2"></i>Crisis MAS
    </span>
    <span class="navbar-text text-white-50 small">
      Run Report &nbsp;·&nbsp; {run_label} &nbsp;·&nbsp; Generated {ts}
    </span>
  </div>
</nav>

<div class="container-fluid py-4">
  <div class="row g-4">

    <!-- Sidebar nav -->
    <div class="col-md-2 col-lg-2 d-none d-md-block">
      <div class="sidebar">
        <ul class="nav nav-pills flex-column gap-1">
          <li class="nav-item"><a class="nav-link active" href="#scenario">
            <i class="bi bi-map-fill me-1"></i>Scenario</a></li>
          <li class="nav-item"><a class="nav-link" href="#alternatives">
            <i class="bi bi-list-check me-1"></i>Alternatives</a></li>
          {"<li class='nav-item'><a class='nav-link' href='#vision'><i class='bi bi-camera-fill me-1'></i>Vision</a></li>" if s_vision else ""}
          {"<li class='nav-item'><a class='nav-link' href='#run-results'><i class='bi bi-activity me-1'></i>Run Results</a></li>" if s_run_results else ""}
          {"<li class='nav-item'><a class='nav-link' href='#comparison'><i class='bi bi-columns-gap me-1'></i>Comparison</a></li>" if s_comparison else ""}
          {"<li class='nav-item'><a class='nav-link' href='#agents'><i class='bi bi-people-fill me-1'></i>Agent Panel</a></li>" if s_agents else ""}
        </ul>
        <hr>
        <div class="text-muted small px-2">
          <div class="mb-1"><i class="bi bi-calendar2 me-1"></i>{ts[:10]}</div>
          <div class="mb-1"><i class="bi bi-cpu me-1"></i>{provider}</div>
          <div><i class="bi bi-folder2-open me-1"></i>{run_label}</div>
        </div>
      </div>
    </div>

    <!-- Main content -->
    <div class="col-md-10 col-lg-10 main-content">

      {s_scenario}

      {s_alts}

      {s_vision}

      {"<section id='run-results' class='mb-5'><h4 class='section-title'><i class='bi bi-activity me-2'></i>Aggregation Method Results</h4>" + s_run_results + "</section>" if s_run_results else ""}

      {s_comparison}

      {s_agents}

    </div>
  </div>
</div>

{foot_libs}
<script>{map_js}</script>

<!-- Smooth scroll & sidebar highlight -->
<script>
document.querySelectorAll('.sidebar .nav-link').forEach(function(link) {{
  link.addEventListener('click', function(e) {{
    document.querySelectorAll('.sidebar .nav-link').forEach(l => l.classList.remove('active'));
    this.classList.add('active');
  }});
}});
const sections = document.querySelectorAll('section[id]');
const sideLinks = document.querySelectorAll('.sidebar .nav-link');
window.addEventListener('scroll', function() {{
  let cur = '';
  sections.forEach(s => {{ if (window.scrollY >= s.offsetTop - 90) cur = s.id; }});
  sideLinks.forEach(l => {{
    l.classList.toggle('active', l.getAttribute('href') === '#' + cur);
  }});
}});
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a standalone HTML run report")
    p.add_argument("--scenario",  default="santorini_volcanic_seismic",
                   help="Scenario ID or JSON filename (default: santorini_volcanic_seismic)")
    p.add_argument("--run",       default=None,
                   help="Run directory path (auto-detected if omitted)")
    p.add_argument("--output",    default=None,
                   help="Output HTML path (default: <run_dir>/report.html)")
    p.add_argument("--scenarios-dir", default="scenarios",
                   help="Scenarios directory (default: scenarios)")
    p.add_argument("--results-dir",   default="results",
                   help="Results root directory (default: results)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve scenario JSON
    sc_name = args.scenario.replace(".json", "")
    sc_path = _ROOT / args.scenarios_dir / f"{sc_name}.json"
    if not sc_path.exists():
        print(f"ERROR: scenario not found: {sc_path}")
        return 1

    # Resolve run directory
    if args.run:
        run_dir = Path(args.run)
    else:
        scenario_results = _ROOT / args.results_dir / sc_name
        if not scenario_results.exists():
            print(f"ERROR: no results found for scenario '{sc_name}' in {args.results_dir}/")
            return 1
        run_dirs = sorted(
            [d for d in scenario_results.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime, reverse=True
        )
        if not run_dirs:
            print(f"ERROR: no run directories found in {scenario_results}")
            return 1
        run_dir = run_dirs[0]
        print(f"  Auto-selected run: {run_dir.name}")

    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}")
        return 1

    # Output path
    out_path = Path(args.output) if args.output else run_dir / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Scenario : {sc_path.name}")
    print(f"  Run      : {run_dir}")
    print(f"  Output   : {out_path}")
    print("  Generating...", end="", flush=True)

    html = generate_html(sc_path, run_dir)
    out_path.write_text(html, encoding="utf-8")

    size_kb = out_path.stat().st_size // 1024
    print(f" done ({size_kb} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
