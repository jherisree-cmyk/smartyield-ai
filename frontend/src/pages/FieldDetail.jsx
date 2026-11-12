import { useEffect, useState } from 'react'
import { Bar, Line } from 'react-chartjs-2'
import { Link, useParams } from 'react-router-dom'
import { api, fmt } from '../api'
import { axis, C, HEALTH } from '../components/charts'
import { useT } from '../i18n'

export default function FieldDetail() {
  const { id } = useParams()
  const [h, setH] = useState(null)
  const [list, setList] = useState([])
  const [season, setSeason] = useState('all')
  const [err, setErr] = useState('')
  const { t } = useT()
  const load = () => {
    api.history(id).then(setH).catch(e => setErr(e.message))
    api.analyses(id).then(setList)
  }
  useEffect(load, [id]) // eslint-disable-line react-hooks/exhaustive-deps
  if (err) return <div className="error">{err}</div>
  if (!h) return <p><span className="spinner" />Loading…</p>

  const seasons = [...new Set(h.observations.map(o => o.season))]
  const obs = h.observations.filter(o => season === 'all' || o.season === season)
  const labels = obs.map(o => o.capture_date)
  const f = h.field
  const lineOpts = (y, extra = {}) => ({
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { display: true, position: 'top', align: 'start' } },
    scales: { y: axis(y, extra), x: axis('image date', { grid: { display: false } }) },
  })

  async function remove(aid) {
    if (!window.confirm('Delete this observation?')) return
    await api.deleteAnalysis(aid); load()
  }

  return (
    <>
      <div className="toolbar">
        <div><h1>{f.name}</h1>
          <p className="sub">{f.crop} · {f.area_ha ? `${f.area_ha} ha` : 'area not set'} · {f.location || 'no location'}{f.sowing_date ? ` · sown ${f.sowing_date}` : ''}{f.variety ? ` · ${f.variety}` : ''}</p></div>
        <div className="spacer" />
        {seasons.length > 1 && (
          <select value={season} onChange={e => setSeason(e.target.value)}>
            <option value="all">All seasons</option>{seasons.map(s => <option key={s} value={s}>Season {s}</option>)}
          </select>)}
        <Link className="btn" to={`/upload?field=${f.id}`}>{t('upload_new')}</Link>
      </div>

      {obs.length === 0 ? <div className="ok">No completed observations yet.</div> : (
        <>
          <div className="grid g2">
            <div className="card"><h2>{t('predicted_yield')} – {t('field_history')}</h2>
              <div className="chart"><Line data={{
                labels,
                datasets: [
                  { label: 'Predicted (t/ha)', data: obs.map(o => o.predicted_t_ha), borderColor: C.series1, backgroundColor: C.series1 },
                  { label: '90 % upper', data: obs.map(o => o.hi90_t_ha), borderColor: 'transparent', backgroundColor: 'rgba(42,120,214,.12)', pointRadius: 0, fill: '+1' },
                  { label: '90 % lower', data: obs.map(o => o.lo90_t_ha), borderColor: 'transparent', pointRadius: 0, fill: false },
                  { label: 'Actual harvest (t/ha)', data: obs.map(o => o.actual_t_ha), borderColor: C.series2, backgroundColor: C.series2, showLine: false, pointRadius: 7, pointStyle: 'rectRot' },
                ],
              }} options={{ ...lineOpts('t/ha', { beginAtZero: true }), plugins: { legend: { labels: { filter: (i) => !i.text.startsWith('90 % lower') } } } }} /></div>
            </div>
            <div className="card"><h2>Vegetation cover progression</h2>
              <div className="chart"><Line data={{
                labels,
                datasets: [{ label: 'Crop cover (%)', data: obs.map(o => o.veg_cover_pct), borderColor: C.series1, backgroundColor: C.series1 }],
              }} options={{ ...lineOpts('% of image', { min: 0, max: 100 }), plugins: { legend: { display: false } } }} /></div>
            </div>
          </div>
          <ChangeCard obs={obs} />
          <div className="card mt"><h2>{t('health_summary')}</h2>
            <div className="chart"><Bar data={{
              labels,
              datasets: HEALTH.map(x => ({ label: t(x.key), data: obs.map(o => o[`${x.key}_pct`]), backgroundColor: x.color, borderColor: '#fcfcfb', borderWidth: { top: 2 }, borderRadius: 0, maxBarThickness: 56 })),
            }} options={{ interaction: { mode: 'index', intersect: false },
              plugins: { legend: { position: 'top', align: 'start' }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.raw, 1)} %` } } },
              scales: { x: axis('', { stacked: true, grid: { display: false } }), y: axis('% of image', { stacked: true, min: 0, max: 100 }) } }} /></div>
          </div>
        </>
      )}

      <div className="card mt"><h2>{t('observations')}</h2>
        <table>
          <thead><tr><th>Date</th><th>Image</th><th>Status</th><th className="num">Cover %</th><th className="num">Predicted</th><th className="num">Actual</th><th className="num">Error</th><th /></tr></thead>
          <tbody>{list.map(a => (
            <tr key={a.id}>
              <td>{a.capture_date}</td><td>{a.image_name}</td><td><span className={`badge ${a.status}`}>{a.status}</span></td>
              <td className="num">{fmt(a.veg_cover_pct, 1)}</td><td className="num">{fmt(a.predicted_t_ha)}</td><td className="num">{fmt(a.actual_t_ha)}</td>
              <td className="num">{a.actual_t_ha && a.predicted_t_ha ? fmt(a.predicted_t_ha - a.actual_t_ha, 2) : '–'}</td>
              <td><Link to={`/analyses/${a.id}`}>open</Link> · <button className="link" onClick={() => remove(a.id)}>delete</button></td>
            </tr>))}</tbody>
        </table>
      </div>
    </>
  )
}


// Change detection between consecutive observations (Module 9: growth progression)
function ChangeCard({ obs }) {
  if (obs.length < 2) return null
  const rows = obs.slice(1).map((o, i) => {
    const p = obs[i]
    const days = Math.round((new Date(o.capture_date) - new Date(p.capture_date)) / 86400000)
    return { from: p.capture_date, to: o.capture_date, days,
      cover: o.veg_cover_pct - p.veg_cover_pct, healthy: o.healthy_pct - p.healthy_pct, poor: o.poor_pct - p.poor_pct,
      index: o.index_mean != null && p.index_mean != null && o.index_name === p.index_name ? o.index_mean - p.index_mean : null,
      indexName: o.index_name,
      yld: o.predicted_t_ha != null && p.predicted_t_ha != null ? o.predicted_t_ha - p.predicted_t_ha : null }
  })
  const sign = (v, d = 1) => v == null ? '–' : `${v > 0 ? '▲ +' : v < 0 ? '▼ ' : ''}${v.toFixed(d)}`
  return (
    <div className="card mt"><h2>Change detection – consecutive observations</h2>
      <p className="muted">Differences between each observation and the previous one (same field). ▲ increase · ▼ decrease.</p>
      <table><thead><tr><th>From → to</th><th className="num">Days</th><th className="num">Δ crop cover (pp)</th><th className="num">Δ healthy (pp)</th>
        <th className="num">Δ poor (pp)</th><th className="num">Δ index mean</th><th className="num">Δ predicted t/ha</th></tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i}><td>{r.from} → {r.to}</td><td className="num">{r.days}</td><td className="num">{sign(r.cover)}</td>
            <td className="num">{sign(r.healthy)}</td><td className="num">{sign(r.poor)}</td>
            <td className="num">{r.index == null ? '–' : `${sign(r.index, 3)} (${r.indexName})`}</td><td className="num">{sign(r.yld, 2)}</td></tr>))}
        </tbody></table>
      <p className="muted">Note: health classes are relative within each image, so compare the index means and crop cover for absolute change; compare images from the same camera.</p>
    </div>)
}
