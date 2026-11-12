import { useEffect, useState } from 'react'
import { Bar } from 'react-chartjs-2'
import { api, fmt } from '../api'
import { axis, C } from '../components/charts'

const MODELS = ['Mean baseline', 'Linear Regression', 'Ridge Regression', 'Random Forest', 'SVR', 'XGBoost']
const TRACKS = [['yield_results_ms', 'Multispectral', C.series1], ['yield_results_rgb', 'RGB only', C.series2]]

export default function Models() {
  const [m, setM] = useState(null)
  const [err, setErr] = useState('')
  const [metric, setMetric] = useState('RMSE')
  useEffect(() => { api.models().then(setM).catch(e => setErr(e.message)) }, [])
  if (err) return <div className="error">{err}</div>
  if (!m) return <p><span className="spinner" />Loading…</p>

  const tracks = TRACKS.filter(([k]) => m[k])
  const seg = m.segmentation_metrics
  const imp = m.yield_results_ms?.permutation_importance || m.yield_results_rgb?.permutation_importance
  const impTop = imp ? Object.entries(imp).slice(0, 12) : []

  return (
    <>
      <h1>Model performance</h1>
      <p className="sub">Real results from the training scripts on held-out data. Nothing on this page is illustrative.</p>

      <div className="card">
        <div className="toolbar"><h2 style={{ margin: 0 }}>Yield regression – grouped 5-fold CV (validation)</h2><div className="spacer" />
          <select value={metric} onChange={e => setMetric(e.target.value)}>{['RMSE', 'MAE', 'R2', 'MAPE_%'].map(x => <option key={x}>{x}</option>)}</select></div>
        <p className="muted">Folds are whole replicate blocks, so no plot is in both training and test data. Hyper-parameters were tuned by nested CV. Image level = one prediction per drone image of a plot.</p>
        <div className="chart"><Bar data={{
          labels: MODELS,
          datasets: tracks.map(([k, label, color]) => ({ label, backgroundColor: color, maxBarThickness: 40,
            data: MODELS.map(n => m[k].P1_grouped_cv[n]?.image_level[metric]) })),
        }} options={{ plugins: { legend: { position: 'top', align: 'start' }, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${fmt(c.raw, 3)}` } } },
          scales: { y: axis(metric === 'R2' ? 'R²' : metric.replace('_%', ' (%)') + (metric.startsWith('M') && metric !== 'MAPE_%' || metric === 'RMSE' ? ' (t/ha)' : ''), { beginAtZero: metric !== 'R2' }), x: axis('', { grid: { display: false } }) } }} /></div>
        {tracks.map(([k, label]) => (
          <div key={k} className="mt">
            <h3>{label} – {m[k].n_rows} images of {m[k].n_plots} plots · {m[k].features.length} features · selected: <b>{m[k].selected_model}</b></h3>
            <table>
              <thead><tr><th>Model</th><th className="num">MAE</th><th className="num">MSE</th><th className="num">RMSE</th><th className="num">R²</th><th className="num">MAPE %</th><th className="num">Plot-level RMSE</th><th className="num">Plot-level R²</th></tr></thead>
              <tbody>{MODELS.map(n => { const r = m[k].P1_grouped_cv[n]; if (!r) return null; const i = r.image_level, p = r.plot_level
                return <tr key={n} className={n === m[k].selected_model ? 'best' : ''}><td>{n}</td><td className="num">{fmt(i.MAE, 3)}</td><td className="num">{fmt(i.MSE, 3)}</td><td className="num">{fmt(i.RMSE, 3)}</td>
                  <td className="num">{fmt(i.R2, 3)}</td><td className="num">{fmt(i['MAPE_%'], 1)}</td><td className="num">{fmt(p.RMSE, 3)}</td><td className="num">{fmt(p.R2, 3)}</td></tr> })}</tbody>
            </table>
          </div>))}
      </div>

      {tracks.some(([k]) => Object.keys(m[k].P2_leave_site_season_out || {}).length) && (
        <div className="card mt"><h2>Generalisation – leave one site-season out</h2>
          <p className="muted">The model is trained on the other site-seasons and tested on a field and season it has never seen. This is the hardest and most realistic test.</p>
          {tracks.map(([k, label]) => { const p2 = m[k].P2_leave_site_season_out; const ss = Object.keys(p2[MODELS[0]] || {}); return (
            <div key={k} className="mt"><h3>{label}</h3><table>
              <thead><tr><th>Model</th>{ss.map(s => <th key={s} className="num">{s} RMSE / R²</th>)}</tr></thead>
              <tbody>{MODELS.map(n => p2[n] && <tr key={n}><td>{n}</td>{ss.map(s => <td key={s} className="num">{fmt(p2[n][s].image_level.RMSE, 2)} / {fmt(p2[n][s].image_level.R2, 2)}</td>)}</tr>)}</tbody>
            </table></div>) })}
        </div>)}

      <div className="grid g2 mt">
        {impTop.length > 0 && <div className="card"><h2>Feature importance (permutation, held-out block)</h2>
          <div className="chart" style={{ height: 360 }}><Bar data={{ labels: impTop.map(([k]) => k), datasets: [{ label: 'RMSE increase when shuffled (t/ha)', data: impTop.map(([, v]) => v), backgroundColor: C.series1 }] }}
            options={{ indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: axis('RMSE increase (t/ha)'), y: axis('', { grid: { display: false } }) } }} /></div></div>}
        {seg && <div className="card"><h2>Crop segmentation (Experiment 2)</h2>
          <p className="muted">Held-out test set: {seg.test_unet.n_images} human-annotated UAV images ({seg.dataset}).</p>
          <table><thead><tr><th>Method</th><th className="num">IoU</th><th className="num">Dice</th><th className="num">Precision</th><th className="num">Recall</th></tr></thead>
            <tbody>{[['U-Net (CNN)', seg.test_unet], ['ExG + Otsu (baseline)', seg.test_exg_otsu_baseline]].map(([n, r]) => (
              <tr key={n}><td>{n}</td><td className="num">{fmt(r.iou, 3)}</td><td className="num">{fmt(r.dice, 3)}</td><td className="num">{fmt(r.precision, 3)}</td><td className="num">{fmt(r.recall, 3)}</td></tr>))}</tbody></table>
        </div>}
      </div>
    </>
  )
}
