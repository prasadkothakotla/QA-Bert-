// Use IPv4 explicitly: on some Windows setups `localhost` resolves to IPv6
// first while Uvicorn is only listening on 127.0.0.1, causing connection resets.
// VITE_API_URL lets deployments point the UI at a remote API without edits.
const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  return res.json()
}

async function request(url, options) {
  try {
    return await fetch(url, options)
  } catch (error) {
    throw new Error(`Cannot reach the API at ${BASE_URL}. Start the backend with: uvicorn main:app --reload --host 127.0.0.1 --port 8000`)
  }
}

export async function fetchDocuments() {
  const res = await request(`${BASE_URL}/documents`)
  return handle(res)
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await request(`${BASE_URL}/documents/upload`, {
    method: 'POST',
    body: formData,
  })
  return handle(res)
}

export async function deleteDocument(docId) {
  const res = await request(`${BASE_URL}/documents/${docId}`, { method: 'DELETE' })
  return handle(res)
}

export async function askQuestion(question, docIds) {
  const res = await request(`${BASE_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, doc_ids: docIds && docIds.length ? docIds : null }),
  })
  return handle(res)
}
