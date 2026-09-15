import { useEffect, useState, useCallback } from 'react'
import DocumentLibrary from './components/DocumentLibrary.jsx'
import QuestionBar from './components/QuestionBar.jsx'
import AnswerPanel from './components/AnswerPanel.jsx'
import { fetchDocuments, uploadDocument, deleteDocument, askQuestion } from './api.js'
import './App.css'

export default function App() {
  const [documents, setDocuments] = useState([])
  const [selectedDocIds, setSelectedDocIds] = useState([])
  const [docsLoading, setDocsLoading] = useState(true)
  const [uploadError, setUploadError] = useState(null)
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [askError, setAskError] = useState(null)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [libraryOpen, setLibraryOpen] = useState(false)
  const loadDocuments = useCallback(async () => { setDocsLoading(true); try { setDocuments(await fetchDocuments()) } catch (err) { setUploadError(err.message) } finally { setDocsLoading(false) } }, [])
  useEffect(() => { loadDocuments() }, [loadDocuments])
  async function handleUpload(file) { setUploadError(null); try { await uploadDocument(file); await loadDocuments() } catch (err) { setUploadError(err.message) } }
  async function handleDelete(docId) { try { await deleteDocument(docId); setSelectedDocIds((prev) => prev.filter((id) => id !== docId)); await loadDocuments() } catch (err) { setUploadError(err.message) } }
  function toggleDocSelection(docId) { setSelectedDocIds((prev) => prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]) }
  async function handleAsk() { if (!question.trim() || documents.length === 0) return; setAsking(true); setAskError(null); try { const res = await askQuestion(question.trim(), selectedDocIds); setResult(res); setHistory((prev) => [{ question: question.trim(), ...res }, ...prev].slice(0, 8)) } catch (err) { setAskError(err.message); setResult(null) } finally { setAsking(false) } }
  return <div className="app">
    <aside className={`library-rail ${libraryOpen ? 'library-rail--open' : ''}`}><div className="rail-header"><div className="brand-mark">Q</div><div><span className="rail-eyebrow">Knowledge workspace</span><h1>Query<span>base</span></h1></div><button className="mobile-close" onClick={() => setLibraryOpen(false)} aria-label="Close library">×</button></div><div className="library-title-row"><span>Your library</span><strong>{documents.length}</strong></div><DocumentLibrary documents={documents} loading={docsLoading} selectedDocIds={selectedDocIds} onToggleSelect={toggleDocSelection} onUpload={handleUpload} onDelete={handleDelete} uploadError={uploadError} /></aside>
    <main className="reading-room"><header className="topbar"><button className="library-toggle" onClick={() => setLibraryOpen(true)} aria-label="Open document library"><span>☰</span> Library</button><div className="workspace-status"><i /> Private workspace</div><div className="avatar">YK</div></header><section className="hero"><div><span className="section-kicker">Document intelligence</span><h2>Find clarity in<br /><em>every document.</em></h2><p>Ask precise questions. Get answers grounded in the source material.</p></div><div className="hero-orb" aria-hidden="true"><span>✦</span></div></section><QuestionBar question={question} onQuestionChange={setQuestion} onAsk={handleAsk} asking={asking} disabled={documents.length === 0} selectedCount={selectedDocIds.length} totalCount={documents.length} onSuggestion={setQuestion} /><AnswerPanel result={result} error={askError} asking={asking} hasDocuments={documents.length > 0} history={history} /></main>
  </div>
}
