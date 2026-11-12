// Thin wrapper around fetch: adds the JWT, parses JSON, and turns API errors into readable messages.
const TOKEN_KEY = 'smartyield_token'
const USER_KEY = 'smartyield_user'

export const auth = {
  get token() { return localStorage.getItem(TOKEN_KEY) },
  get user() { try { return JSON.parse(localStorage.getItem(USER_KEY)) } catch { return null } },
  save(res) { localStorage.setItem(TOKEN_KEY, res.access_token); localStorage.setItem(USER_KEY, JSON.stringify(res.user)) },
  clear() { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY) },
}

async function request(path, { method = 'GET', body, form } = {}) {
  const headers = {}
  if (auth.token) headers.Authorization = `Bearer ${auth.token}`
  let payload
  if (form) payload = form
  else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body) }
  const res = await fetch(path, { method, headers, body: payload })
  if (res.status === 401 && !path.startsWith('/api/auth/')) {
    auth.clear()
    window.location.href = '/login'
    throw new Error('Session expired, please log in again')
  }
  if (res.status === 204) return null
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const d = data.detail
    const msg = Array.isArray(d) ? d.map(e => `${e.loc?.slice(-1)[0]}: ${e.msg}`).join('; ') : d
    throw new Error(msg || `Request failed (${res.status})`)
  }
  return data
}

export const api = {
  register: (b) => request('/api/auth/register', { method: 'POST', body: b }),
  login: (b) => request('/api/auth/login', { method: 'POST', body: b }),
  crops: () => request('/api/fields/crops'),
  fields: () => request('/api/fields'),
  field: (id) => request(`/api/fields/${id}`),
  createField: (b) => request('/api/fields', { method: 'POST', body: b }),
  updateField: (id, b) => request(`/api/fields/${id}`, { method: 'PUT', body: b }),
  deleteField: (id) => request(`/api/fields/${id}`, { method: 'DELETE' }),
  history: (id) => request(`/api/fields/${id}/history`),
  analyses: (fieldId) => request(`/api/analyses${fieldId ? `?field_id=${fieldId}` : ''}`),
  analysis: (id) => request(`/api/analyses/${id}`),
  upload: (form) => request('/api/analyses', { method: 'POST', form }),
  setActual: (id, v) => request(`/api/analyses/${id}/actual`, { method: 'PUT', body: { actual_t_ha: v } }),
  deleteAnalysis: (id) => request(`/api/analyses/${id}`, { method: 'DELETE' }),
  dashboard: () => request('/api/dashboard'),
  models: () => request('/api/models'),
}

// <img>/<a> cannot send headers, so the token goes in the query string for file links.
export const fileUrl = (id, name) => `/api/analyses/${id}/files/${name}?token=${encodeURIComponent(auth.token || '')}`

export const fmt = (v, d = 2, unit = '') =>
  v === null || v === undefined || Number.isNaN(v) ? '–' : `${Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })}${unit}`
