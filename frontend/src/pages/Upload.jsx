import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { useT } from '../i18n'

const OK_EXT = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
const MAX_MB = 300

export default function Upload() {
  const [params] = useSearchParams()
  const [fields, setFields] = useState([])
  const [fieldId, setFieldId] = useState(params.get('field') || '')
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10))
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [drag, setDrag] = useState(false)
  const [err, setErr] = useState('')
  const [status, setStatus] = useState('')
  const input = useRef()
  const nav = useNavigate()
  const { t } = useT()

  useEffect(() => {
    api.fields().then(fs => { setFields(fs); if (!fieldId && fs.length) setFieldId(String(fs[0].id)) })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  function pick(f) {
    setErr('')
    if (!f) return
    const ext = f.name.split('.').pop().toLowerCase()
    if (!OK_EXT.includes(ext)) return setErr('Only JPEG, JPG, PNG or GeoTIFF (.tif) drone images are accepted')
    if (f.size > MAX_MB * 1024 * 1024) return setErr(`File is larger than ${MAX_MB} MB`)
    setFile(f)
    setPreview(['jpg', 'jpeg', 'png'].includes(ext) ? URL.createObjectURL(f) : null)
  }

  async function submit(e) {
    e.preventDefault()
    if (!file || !fieldId) return
    setErr(''); setStatus('Uploading…')
    const form = new FormData()
    form.append('field_id', fieldId)
    form.append('capture_date', date)
    form.append('file', file)
    try {
      const { id } = await api.upload(form)
      setStatus('Analysing: segmentation, health map, features, yield model…')
      for (let i = 0; i < 600; i++) {
        const a = await api.analysis(id)
        if (a.status === 'done') return nav(`/analyses/${id}`)
        if (a.status === 'failed') throw new Error(`Analysis failed: ${a.error}`)
        await new Promise(r => setTimeout(r, 1500))
      }
      throw new Error('Analysis is taking unusually long; check the field history later')
    } catch (ex) { setErr(ex.message); setStatus('') }
  }

  if (!fields.length) return <div className="ok">Create a field first on the <Link to="/fields">Fields</Link> page.</div>

  return (
    <>
      <h1>{t('upload')}</h1>
      <p className="sub">Upload a nadir (top-down) drone image of the field. GeoTIFF orthomosaics give real areas; 9-band MAIA multispectral GeoTIFFs enable true NDVI.</p>
      <form className="card" onSubmit={submit} style={{ maxWidth: 760 }}>
        {err && <div className="error">{err}</div>}
        <div className="grid g2">
          <div className="row"><label>{t('field')} *</label>
            <select value={fieldId} onChange={e => setFieldId(e.target.value)}>
              {fields.map(f => <option key={f.id} value={f.id}>{f.name} ({f.crop}{f.area_ha ? `, ${f.area_ha} ha` : ''})</option>)}
            </select></div>
          <div className="row"><label>{t('capture_date')} *</label>
            <input type="date" value={date} max={new Date().toISOString().slice(0, 10)} onChange={e => setDate(e.target.value)} required /></div>
        </div>
        <div className={`dropzone ${drag ? 'drag' : ''}`} onClick={() => input.current.click()}
          onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]) }}>
          {file ? <><b>{file.name}</b> — {(file.size / 1e6).toFixed(1)} MB</> : <>{t('drop_here')}<br /><small>JPEG · PNG · GeoTIFF, max {MAX_MB} MB</small></>}
          <input ref={input} type="file" hidden accept=".jpg,.jpeg,.png,.tif,.tiff" onChange={e => pick(e.target.files[0])} />
        </div>
        {preview && <img className="preview" src={preview} alt="preview" />}
        <div className="toolbar mt">
          <button disabled={!file || !!status}>{t('run_analysis')}</button>
          {status && <span><span className="spinner" />{status}</span>}
        </div>
      </form>
    </>
  )
}
