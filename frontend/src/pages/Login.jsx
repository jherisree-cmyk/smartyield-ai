import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, auth } from '../api'
import { LanguageSelect, useT } from '../i18n'

export default function Login() {
  const [mode, setMode] = useState('login')
  const [form, setForm] = useState({ name: '', email: '', password: '' })
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const nav = useNavigate()
  const { t } = useT()
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  async function submit(e) {
    e.preventDefault()
    setErr('')
    if (mode === 'register' && form.password.length < 8) return setErr('Password must be at least 8 characters')
    setBusy(true)
    try {
      const res = mode === 'login'
        ? await api.login({ email: form.email, password: form.password })
        : await api.register(form)
      auth.save(res)
      nav('/')
    } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>🌾 SmartYield AI</h1>
        <p className="sub">{t('tagline')}</p>
        <LanguageSelect style={{ marginBottom: 14 }} />
        <div className="tabs">
          <button type="button" className={mode === 'login' ? 'on' : ''} onClick={() => setMode('login')}>{t('login')}</button>
          <button type="button" className={mode === 'register' ? 'on' : ''} onClick={() => setMode('register')}>{t('register')}</button>
        </div>
        {err && <div className="error">{err}</div>}
        {mode === 'register' && (
          <div className="row"><label>{t('name')}</label><input value={form.name} onChange={set('name')} required /></div>
        )}
        <div className="row"><label>{t('email')}</label><input type="email" value={form.email} onChange={set('email')} required /></div>
        <div className="row"><label>{t('password')}</label>
          <input type="password" value={form.password} onChange={set('password')} required minLength={mode === 'register' ? 8 : 1} /></div>
        <button disabled={busy} style={{ width: '100%' }}>{busy ? 'Please wait…' : mode === 'login' ? t('login') : t('create_account')}</button>
      </form>
    </div>
  )
}
