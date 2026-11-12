// Shared Chart.js setup + the validated reference palette (dataviz skill, light mode).
import {
  ArcElement, BarElement, CategoryScale, Chart, Filler, Legend, LinearScale, LineElement, PointElement, Tooltip,
} from 'chart.js'

Chart.register(ArcElement, BarElement, CategoryScale, LinearScale, LineElement, PointElement, Tooltip, Legend, Filler)

export const C = {
  series1: '#2a78d6', // blue   - multispectral track / predicted
  series2: '#eb6834', // orange - RGB track / actual
  series3: '#1baf7a', // aqua
  good: '#0ca30c', warning: '#fab219', critical: '#d03b3b', neutral: '#898781',
  ink: '#0b0b0b', ink2: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7',
}

Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif'
Chart.defaults.font.size = 12
Chart.defaults.color = C.ink2
Chart.defaults.borderColor = C.grid
Chart.defaults.plugins.tooltip.backgroundColor = '#0b0b0b'
Chart.defaults.plugins.tooltip.padding = 8
Chart.defaults.plugins.legend.labels.boxWidth = 12
Chart.defaults.plugins.legend.labels.boxHeight = 12
Chart.defaults.elements.bar.borderRadius = 4
Chart.defaults.elements.bar.borderSkipped = 'start'
Chart.defaults.elements.line.borderWidth = 2
Chart.defaults.elements.point.radius = 4
Chart.defaults.elements.point.hoverRadius = 6
Chart.defaults.maintainAspectRatio = false

export const axis = (title, extra = {}) => ({
  title: { display: !!title, text: title, color: C.muted },
  grid: { color: C.grid },
  border: { color: C.axis },
  ticks: { color: C.muted },
  ...extra,
})

export const HEALTH = [
  { key: 'healthy', label: 'Healthy', color: C.good },  // labels are translated via t(key) in the pages
  { key: 'moderate', label: 'Moderate', color: C.warning },
  { key: 'poor', label: 'Poor', color: C.critical },
]
