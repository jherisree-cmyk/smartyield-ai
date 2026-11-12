import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, fmt } from '../api'
import { useT } from '../i18n'

const EMPTY = { name: '', crop: 'rice', area_ha: '', location: '', sowing_date: '', variety: '', notes: '' }

export default function Fields() {
  const [fields, setFields] = useState([])
  const [crops, setCrops] = useState([])
  const [form, setForm] = useState(EMPTY)
  const [err, setErr] = useState('')
  const [msg, setMsg] = useState('')
  const { t } = useT()
  const load = () => api.fields().then(setFields).catch(e => setErr(e.message))
  useEffect(() => { load(); api.crops().then(setCrops) }, [])
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  async function submit(e) {
    e.preventDefault()
    setErr(''); setMsg('')
    const body = { ...form, area_ha: form.area_ha === '' ? null : Number(form.area_ha), sowing_date: form.sowing_date || null }
    try { await api.createField(body); setForm(EMPTY); setMsg('Field created'); load() } catch (ex) { setErr(ex.message) }
  }
  async function remove(f) {
    if (!window.confirm(`Delete field "${f.name}" and all its analyses?`)) return
    await api.deleteField(f.id); load()
  }

  return (
    <>
      <h1>{t('fields')}</h1>
      <p className="sub">Register each field once, then upload drone images of it over the season.</p>
      <div className="grid g3">
        <form className="card" onSubmit={submit}>
          <h2>{t('new_field')}</h2>
          {err && <div className="error">{err}</div>}
          {msg && <div className="ok">{msg}</div>}
          <div className="row"><label>{t('field')} *</label><input value={form.name} onChange={set('name')} required /></div>
          <div className="row"><label>{t('crop')} *</label>
            <select value={form.crop} onChange={set('crop')}>{crops.map(c => <option key={c.key} value={c.key}>{c.label}{c.trained ? ' – t/ha model ✓' : ''}</option>)}</select></div>
          <div className="row"><label>{t('area_ha')}</label><input type="number" step="0.0001" min="0" value={form.area_ha} onChange={set('area_ha')} placeholder="needed for production (t) of non-georeferenced photos" /></div>
          <div className="row"><label>{t('location')}</label><input value={form.location} onChange={set('location')} /></div>
          <div className="row"><label>{t('sowing_date')}</label><input type="date" value={form.sowing_date} onChange={set('sowing_date')} /></div>
          <div className="row"><label>{t('variety')}</label><input value={form.variety} onChange={set('variety')} /></div>
          <div className="row"><label>{t('notes')}</label><textarea rows={2} value={form.notes} onChange={set('notes')} /></div>
          <button>{t('create_field')}</button>
        </form>
        <div className="card" style={{ gridColumn: 'span 2' }}>
          <h2>{t('your_fields')}</h2>
          {fields.length === 0 ? <p className="muted">No fields yet.</p> : (
            <table>
              <thead><tr><th>Name</th><th>Crop</th><th className="num">Area ha</th><th>Location</th><th className="num">Observations</th><th>Last image</th><th /></tr></thead>
              <tbody>{fields.map(f => (
                <tr key={f.id}>
                  <td><Link to={`/fields/${f.id}`}>{f.name}</Link></td><td>{f.crop}</td><td className="num">{fmt(f.area_ha, 2)}</td>
                  <td>{f.location || '–'}</td><td className="num">{f.n_analyses}</td><td>{f.last_capture || '–'}</td>
                  <td><button className="link" onClick={() => remove(f)}>{t('delete')}</button></td>
                </tr>))}</tbody>
            </table>)}
        </div>
      </div>
    </>
  )
}
