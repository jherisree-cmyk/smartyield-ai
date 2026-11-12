import { useEffect, useState } from 'react'
import { Bar } from 'react-chartjs-2'
import { Link } from 'react-router-dom'
import { api, fmt } from '../api'
import { axis, C, HEALTH } from '../components/charts'
import { useT } from '../i18n'

export default function Dashboard() {
  const [d, setD] = useState(null)
  const [err, setErr] = useState('')
  const { t } = useT()
  useEffect(() => { api.dashboard().then(setD).catch(e => setErr(e.message)) }, [])
  if (err) return <div className="error">{err}</div>
  if (!d) return <p><span className="spinner" />Loading…</p>

  const fields = d.latest_per_field
  const yields = fields.filter(f => f.predicted_t_ha != null)
  const meanYield = yields.length ? yields.reduce((s, f) => s + f.predicted_t_ha, 0) / yields.length : null
  const meanCover = fields.length ? fields.reduce((s, f) => s + (f.veg_cover_pct || 0), 0) / fields.length : null
  const h = d.health_mean_fractions

  return (
    <>
      <h1>{t('dashboard')}</h1>
      <p className="sub">Latest observation of every field. All crops are monitored; t/ha yield is given for crops with a harvest-trained model (currently rice).</p>
      {fields.length === 0 && (
        <div className="ok">No analyses yet. <Link to="/fields">Create a field</Link>, then <Link to="/upload">upload a drone image</Link>.</div>
      )}
      <div className="grid g4">
        <Stat label={t('fields_monitored')} value={d.n_fields} hint={`${fmt(d.total_area_ha, 2)} ha registered`} />
        <Stat label={t('mean_yield')} value={fmt(meanYield, 2, ' t/ha')} hint={meanYield ? `${fmt(meanYield * 1000 / 2.47105, 0)} kg/acre` : 'no predictions yet'} />
        <Stat label={t('production')} value={fmt(d.estimated_total_production_t, 2, ' t')} hint="yield × cropped area (fields with area)" />
        <Stat label={t('crop_cover')} value={fmt(meanCover, 1, ' %')} hint="vegetation pixels / image" />
      </div>

      <div className="grid g2 mt">
        <div className="card">
          <h2>{t('yield_by_field')}</h2>
          {yields.length ? (
            <div className="chart">
              <Bar data={{
                labels: yields.map(f => f.field_name),
                datasets: [
                  { label: 'Predicted yield (t/ha)', data: yields.map(f => f.predicted_t_ha), backgroundColor: C.series1, maxBarThickness: 48 },
                ],
              }} options={{
                plugins: {
                  legend: { display: false },
                  tooltip: { callbacks: { afterLabel: (ctx) => { const f = yields[ctx.dataIndex]; return `90% interval: ${fmt(f.lo90_t_ha)}–${fmt(f.hi90_t_ha)} t/ha\nimage: ${f.capture_date}` } } },
                },
                scales: { y: axis('t/ha', { beginAtZero: true }), x: axis('', { grid: { display: false } }) },
              }} />
            </div>
          ) : <p className="muted">No predictions yet.</p>}
        </div>
        <div className="card">
          <h2>{t('health_summary')}</h2>
          <p className="muted" style={{ marginTop: -6 }}>Mean share of image area across latest observations</p>
          <div className="chart sm">
            <Bar data={{
              labels: HEALTH.map(x => t(x.key)),
              datasets: [{ label: 'Share of image (%)', data: HEALTH.map(x => 100 * (h[x.key] || 0)), backgroundColor: HEALTH.map(x => x.color), maxBarThickness: 56 }],
            }} options={{
              indexAxis: 'y',
              plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${fmt(c.raw, 1)} % of image` } } },
              scales: { x: axis('% of image', { beginAtZero: true }), y: axis('', { grid: { display: false } }) },
            }} />
          </div>
          <div className="legend">{HEALTH.map(x => <span key={x.key}><i style={{ background: x.color }} />{t(x.key)}: {fmt(100 * (h[x.key] || 0), 1)} %</span>)}</div>
        </div>
      </div>

      <div className="card mt">
        <h2>Fields – latest observation</h2>
        <table>
          <thead><tr><th>Field</th><th>Image date</th><th>Track</th><th className="num">Cover %</th><th className="num">Yield t/ha</th>
            <th className="num">90 % interval</th><th className="num">Production t</th><th className="num">Poor zones</th><th /></tr></thead>
          <tbody>{fields.map(f => (
            <tr key={f.field_id}>
              <td><Link to={`/fields/${f.field_id}`}>{f.field_name}</Link></td><td>{f.capture_date}</td>
              <td><span className="badge">{f.track === 'ms' ? 'multispectral' : 'RGB'}</span></td>
              <td className="num">{fmt(f.veg_cover_pct, 1)}</td><td className="num">{fmt(f.predicted_t_ha)}</td>
              <td className="num">{fmt(f.lo90_t_ha)}–{fmt(f.hi90_t_ha)}</td><td className="num">{fmt(f.production_t, 2)}</td>
              <td className="num">{f.poor_zones}/{f.zones}</td><td><Link to={`/analyses/${f.analysis_id}`}>open</Link></td>
            </tr>))}
          </tbody>
        </table>
        {d.user_evaluation && (
          <p className="muted mt">Your harvest records: {d.user_evaluation.n} observation(s) with actual yield. MAE {fmt(d.user_evaluation.mae)} t/ha, RMSE {fmt(d.user_evaluation.rmse)} t/ha.</p>
        )}
      </div>
    </>
  )
}

function Stat({ label, value, hint }) {
  return <div className="card stat"><div className="label">{label}</div><div className="value">{value}</div><div className="hint">{hint}</div></div>
}
