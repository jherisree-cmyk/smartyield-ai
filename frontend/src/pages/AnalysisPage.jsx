import { useEffect, useMemo, useState } from 'react'
import { Bar } from 'react-chartjs-2'
import { Link, useParams } from 'react-router-dom'
import { api, fileUrl, fmt } from '../api'
import { axis, C, HEALTH } from '../components/charts'
import { useT } from '../i18n'

const PRIORITY = { high: { label: 'High', color: C.critical, icon: '▲' }, medium: { label: 'Medium', color: C.warning, icon: '●' },
  info: { label: 'Info', color: C.neutral, icon: 'i' } }

export default function AnalysisPage() {
  const { id } = useParams()
  const { t } = useT()
  const [a, setA] = useState(null)
  const [err, setErr] = useState('')
  const [view, setView] = useState('segmentation')
  const [actual, setActual] = useState('')
  const [msg, setMsg] = useState('')

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const r = await api.analysis(id)
        if (!alive) return
        setA(r)
        if (r.actual_t_ha) setActual(String(r.actual_t_ha))
        if (r.status === 'queued' || r.status === 'processing') setTimeout(load, 1500)
      } catch (e) { setErr(e.message) }
    }
    load()
    return () => { alive = false }
  }, [id])

  const s = a?.summary
  const hasYield = s?.yield_model_available !== false
  const varKey = hasYield ? 'pred_t_ha' : 'rel_productivity'

  const hist = useMemo(() => {
    const z = (a?.zones || []).filter(z => z.is_crop && z[varKey] != null)
    if (!z.length) return null
    const vals = z.map(x => x[varKey])
    const lo = Math.min(...vals), hi = Math.max(...vals)
    const n = 12, w = Math.max((hi - lo) / n, hasYield ? 0.25 : 0.02)
    const bins = Array.from({ length: n }, (_, i) => ({ from: lo + i * w, to: lo + (i + 1) * w, n: 0 }))
    vals.forEach(v => { bins[Math.min(n - 1, Math.floor((v - lo) / w))].n++ })
    return bins.filter((b, i) => b.n > 0 || i < bins.findLastIndex(x => x.n > 0))
  }, [a, varKey, hasYield])

  if (err) return <div className="error">{err}</div>
  if (!a) return <p><span className="spinner" />Loading…</p>
  if (a.status === 'failed') return <div className="error">Analysis failed: {a.error}</div>
  if (a.status !== 'done') return <p><span className="spinner" />Analysis {a.status}…</p>

  const fr = s.health.fractions
  const views = [['segmentation', t('segmentation')], ['health', t('health_map')],
    ['yield_map', hasYield ? t('yield_map') : t('rel_productivity')], ['mask', 'Binary mask'], ['enhanced', 'Enhanced (CLAHE)']]

  async function saveActual(e) {
    e.preventDefault()
    try { await api.setActual(id, Number(actual)); setMsg('✓') } catch (ex) { setMsg(ex.message) }
  }

  return (
    <>
      <div className="toolbar">
        <div>
          <h1>{a.field_name} – {a.capture_date}</h1>
          <p className="sub">{s.crop_label || s.crop} · {a.image_name} · {s.track_label} · segmentation: {s.segmentation_method}</p>
        </div>
        <div className="spacer" />
        <Link className="btn secondary" to={`/fields/${a.field_id}`}>{t('field_history')}</Link>
        <a className="btn secondary" href={fileUrl(id, 'zones.csv')}>Zones CSV</a>
        <a className="btn" href={fileUrl(id, 'report.pdf')}>{t('download_pdf')}</a>
      </div>

      {!hasYield && <div className="warn">ℹ {t('no_yield_model')}</div>}
      {s.ood_warning && <div className="warn">⚠ This image is outside the range of the training data (on {fmt(100 * s.ood_fraction_mean, 0)} % of features). The yield estimate is an extrapolation.</div>}
      {s.notes && <div className="warn">ℹ {s.notes}</div>}

      <div className="grid g4">
        {hasYield ? (
          <div className="card stat"><div className="label">{t('predicted_yield')}</div><div className="value">{fmt(s.predicted_yield_t_ha, 2, ' t/ha')}</div>
            <div className="hint">90 %: {fmt(s.yield_lo90_t_ha)}–{fmt(s.yield_hi90_t_ha)} · {fmt(s.predicted_yield_kg_acre, 0)} kg/acre</div></div>
        ) : (
          <div className="card stat"><div className="label">{t('rel_productivity')}</div>
            <div className="value">{fmt(s.rel_productivity_stats?.p10, 2)} – {fmt(s.rel_productivity_stats?.p90, 2)}</div>
            <div className="hint">10th–90th percentile of zones (1.00 = field median)</div></div>
        )}
        <div className="card stat"><div className="label">{t('production')}</div><div className="value">{hasYield ? fmt(s.estimated_production_t, 3, ' t') : '–'}</div>
          <div className="hint">cropped area {fmt(s.cropped_area_ha, 4, ' ha')} ({s.area_source})</div></div>
        <div className="card stat"><div className="label">{t('crop_cover')}</div><div className="value">{fmt(s.vegetation_cover_pct, 1, ' %')}</div>
          <div className="hint">green pixels {fmt(s.green_pixel_pct, 1)} % · canopy density {fmt(s.canopy_density, 2)}</div></div>
        <div className="card stat"><div className="label">{t('poor_zones')}</div><div className="value">{s.n_poor_growth_zones} / {s.n_crop_zones}</div>
          <div className="hint">{s.n_zones} zones of {s.zone_size_m ? `${fmt(s.zone_size_m, 2)} m` : `${s.zone_size_px} px`}</div></div>
      </div>

      <div className="card mt">
        <div className="toolbar" style={{ marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>{t('uploaded_image')} ↔ {views.find(v => v[0] === view)[1]}</h2><div className="spacer" />
          <select value={view} onChange={e => setView(e.target.value)}>{views.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
        </div>
        <div className="imgpair">
          <div className="imgbox"><img src={fileUrl(id, 'original')} alt="original" /><div className="cap">{t('uploaded_image')}</div></div>
          <div className="imgbox"><img src={fileUrl(id, view)} alt={view} /><div className="cap">{views.find(v => v[0] === view)[1]}</div></div>
        </div>
        {view === 'health' && <div className="legend"><span><i style={{ background: '#787878' }} />Background</span>
          {HEALTH.slice().reverse().map(x => <span key={x.key}><i style={{ background: x.color }} />{t(x.key)}</span>)}
          <span className="muted">Basis: {s.health.basis}</span></div>}
        {view === 'yield_map' && <div className="legend"><span><i style={{ background: 'linear-gradient(90deg,#d73027,#fee08b,#1a9850)' }} />
          {hasYield ? 'low → high t/ha' : 'below → above field median (0.5 – 1.5)'}</span>
          <span><i style={{ border: '2px solid #ff00ff' }} />{t('poor_zones')}</span></div>}
      </div>

      <div className="grid g2 mt">
        <div className="card">
          <h2>Recommendations</h2>
          {(s.recommendations || []).map((r, i) => (
            <div key={i} style={{ display: 'flex', gap: 10, padding: '8px 0', borderBottom: '1px solid var(--line)' }}>
              <span title={PRIORITY[r.priority]?.label} style={{ color: PRIORITY[r.priority]?.color, fontWeight: 700, minWidth: 56 }}>
                {PRIORITY[r.priority]?.icon} {PRIORITY[r.priority]?.label}</span>
              <div><b>{r.title}</b><div className="muted" style={{ color: 'var(--ink2)' }}>{r.detail}</div></div>
            </div>))}
        </div>
        <div className="card">
          <h2>Scouting list – zones to inspect first</h2>
          {(s.scouting || []).length === 0 ? <p className="muted">No poor-growth zones.</p> : (
            <table><thead><tr><th>#</th><th>Grid (row, col)</th><th className="num">Cover %</th><th className="num">{s.scouting[0].value_name}</th><th>Location</th></tr></thead>
              <tbody>{s.scouting.map((z, i) => (
                <tr key={i}><td>{i + 1}</td><td>{z.row}, {z.col}</td><td className="num">{fmt(z.veg_cover_pct, 0)}</td><td className="num">{fmt(z.value, 2)}</td>
                  <td>{z.lat != null ? <a target="_blank" rel="noreferrer" href={`https://www.google.com/maps?q=${z.lat},${z.lon}`}>{z.lat.toFixed(6)}, {z.lon.toFixed(6)}</a> : <span className="muted">not georeferenced</span>}</td></tr>))}
              </tbody></table>)}
        </div>
      </div>

      <div className="grid g2 mt">
        <div className="card">
          <h2>{hasYield ? 'Regional yield distribution' : 'Relative productivity distribution'}</h2>
          {hist ? <div className="chart sm"><Bar data={{
            labels: hist.map(b => `${b.from.toFixed(2)}–${b.to.toFixed(2)}`),
            datasets: [{ label: 'Zones', data: hist.map(b => b.n), backgroundColor: C.series1, barPercentage: 0.95, categoryPercentage: 0.95 }],
          }} options={{ plugins: { legend: { display: false }, tooltip: { callbacks: { title: (c) => `${c[0].label}${hasYield ? ' t/ha' : ''}`, label: (c) => `${c.raw} zones` } } },
            scales: { x: axis(hasYield ? 'predicted zone yield (t/ha)' : 'relative productivity (1 = field median)', { grid: { display: false } }), y: axis('zones', { beginAtZero: true }) } }} /></div>
            : <p className="muted">No crop zones.</p>}
        </div>
        <div className="card">
          <h2>{t('health_summary')}</h2>
          <table><tbody>
            <tr><td>{s.health.index} mean (crop pixels)</td><td className="num">{fmt(s.health.index_mean_veg, 3)}</td></tr>
            {HEALTH.map(x => <tr key={x.key}><td><i style={{ display: 'inline-block', width: 10, height: 10, background: x.color, borderRadius: 2, marginRight: 6 }} />{t(x.key)}</td><td className="num">{fmt(100 * fr[x.key], 1)} %</td></tr>)}
            {hasYield && <tr><td>Zone yield range</td><td className="num">{fmt(s.zone_yield_stats.min)} – {fmt(s.zone_yield_stats.max)} t/ha</td></tr>}
            <tr><td>Model</td><td className="num">{s.model.name ? `${s.model.name} (${s.model.track})` : 'not available for this crop'}</td></tr>
            {hasYield && <tr><td>Model validation RMSE / R²</td><td className="num">{fmt(s.model.cv_rmse_t_ha, 2)} t/ha / {fmt(s.model.cv_r2, 2)}</td></tr>}
            <tr><td>Ground sampling distance</td><td className="num">{s.gsd_m ? `${fmt(100 * s.gsd_m, 2)} cm/px (${s.gsd_source})` : 'not georeferenced'}</td></tr>
          </tbody></table>
          <form className="toolbar mt" onSubmit={saveActual}>
            <label>{t('actual_yield')}</label>
            <input type="number" step="0.01" min="0.01" max="25" value={actual} onChange={e => setActual(e.target.value)} style={{ width: 110 }} required />
            <button className="secondary">{t('save')}</button>{msg && <span className="muted">{msg}</span>}
          </form>
        </div>
      </div>
    </>
  )
}
