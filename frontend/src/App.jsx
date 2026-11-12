import { Navigate, NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import { auth } from './api'
import { LanguageSelect, useT } from './i18n'
import AnalysisPage from './pages/AnalysisPage'
import Dashboard from './pages/Dashboard'
import FieldDetail from './pages/FieldDetail'
import Fields from './pages/Fields'
import Login from './pages/Login'
import Models from './pages/Models'
import Upload from './pages/Upload'

function Protected({ children }) {
  return auth.token ? children : <Navigate to="/login" replace />
}

function Shell({ children }) {
  const nav = useNavigate()
  const user = auth.user
  const { t } = useT()
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="logo">🌾</span> SmartYield <b>AI</b></div>
        <nav>
          <NavLink to="/" end>{t('dashboard')}</NavLink>
          <NavLink to="/upload">{t('upload')}</NavLink>
          <NavLink to="/fields">{t('fields')}</NavLink>
          <NavLink to="/models">{t('models')}</NavLink>
        </nav>
        <LanguageSelect />
        <div className="user">
          <div>{user?.name}</div>
          <small>{user?.email}</small>
          <button className="link" onClick={() => { auth.clear(); nav('/login') }}>{t('logout')}</button>
        </div>
      </aside>
      <main className="content">{children}</main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/*" element={
        <Protected>
          <Shell>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/fields" element={<Fields />} />
              <Route path="/fields/:id" element={<FieldDetail />} />
              <Route path="/analyses/:id" element={<AnalysisPage />} />
              <Route path="/models" element={<Models />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Shell>
        </Protected>
      } />
    </Routes>
  )
}
