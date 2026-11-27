// doc-chat-ai frontend: upload documents, list/delete them, chat with RAG.

const healthEl = document.getElementById("health");
const dropEl = document.getElementById("drop");
const fileInput = document.getElementById("file-input");
const docListEl = document.getElementById("doc-list");
const chatLogEl = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");

// -- health ----------------------------------------------------------
async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    healthEl.innerHTML = data.api_key_configured
      ? `<span class="ok">● API key configured</span> — model ${data.chat_model}, ` +
        `${data.documents} document(s) loaded.`
      : `<span class="warn">● No API key configured</span> — uploads work, but ` +
        `chat needs OPENAI_API_KEY in .env.`;
  } catch (err) {
    healthEl.textContent = "Server unreachable.";
  }
}

// -- documents -------------------------------------------------------
async function refreshDocs() {
  const res = await fetch("/api/documents");
  const data = await res.json();
  docListEl.innerHTML = "";
  for (const doc of data.documents) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${doc.filename} — ${doc.chunks} chunk(s)`;
    const delBtn = document.createElement("button");
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      await fetch(`/api/documents/${doc.id}`, { method: "DELETE" });
      refreshDocs();
      refreshHealth();
    };
    li.append(label, delBtn);
    docListEl.appendChild(li);
  }
}

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/documents/upload", {
    method: "POST", body: form,
  });
  const data = await res.json();
  if (!res.ok) {
    addMessage("error", data.detail || "Upload failed.");
    return;
  }
  if (data.warning) {
    addMessage("error", data.warning);
  }
  refreshDocs();
  refreshHealth();
}

dropEl.onclick = () => fileInput.click();
fileInput.onchange = () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  fileInput.value = "";
};
dropEl.ondragover = (e) => { e.preventDefault(); };
dropEl.ondrop = (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
};

// -- chat ------------------------------------------------------------
function addMessage(role, text, sources) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  if (sources && sources.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${sources.length})`;
    details.appendChild(summary);
    for (const src of sources) {
      const quote = document.createElement("blockquote");
      quote.textContent = `${src.filename}: ${src.excerpt}`;
      details.appendChild(quote);
    }
    div.appendChild(details);
  }
  chatLogEl.appendChild(div);
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
  return div;
}

chatForm.onsubmit = async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  addMessage("user", question);
  const thinking = addMessage("assistant", "Thinking…");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    thinking.remove();
    if (!res.ok) {
      addMessage("error", data.detail || "Chat request failed.");
      return;
    }
    addMessage("assistant", data.answer, data.sources);
  } catch (err) {
    thinking.remove();
    addMessage("error", "Could not reach the server.");
  }
};

refreshHealth();
refreshDocs();
