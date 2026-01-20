const BACKEND = (window.BACKEND_URL || "https://jnu-admission-frontend.onrender.com").replace(/\/$/, "");

const QUESTIONS = {
  circular: [
    "Has the undergraduate admission Test Circular been published for the 2025-2026 academic year?",
    "What undergraduate programs are available in JNU?",
    "What is the minimum GPA required for selection?",
    "What is the minimum GPA required for applying to A unit (Science)?",
    "What is the minimum GPA required for applying to B unit (Arts and Law)?",
    "What is the minimum GPA required for applying to C unit (Business Education)?",
    "What is the minimum GPA required for applying to D unit (Faculty of Social Sciences)?",
    "What is the minimum GPA required for applying to E unit (Fine Arts)?",
    "Can I change the department and take the exam in which unit?",
    "Is there a chance for second time admission in Jagannath University?"
  ],
  apply: [
    "Application procedure for Undergraduate Admission Test (2025-2026)."
  ],
  payment: [
    "Payment process for the Undergraduate (Honours) Admission Test 2025-2026"
  ]
};

const LINKS = {
  website: "https://jnu.ac.bd/",
  admissionMessage: "https://jnuadmission.com/"
};

let activeCategory = null;
const STORAGE_KEY = "jnu_helpdesk_chat_v2";
const SESSION_KEY = "jnu_helpdesk_session_id_v1";

function getSessionId(){
  return localStorage.getItem(SESSION_KEY) || "";
}

function setSessionId(sid){
  if (!sid) return;
  localStorage.setItem(SESSION_KEY, String(sid));
}

function qs(id){ return document.getElementById(id); }

/* =========================================================
   ✅ Empty-state controller
   ========================================================= */
function updateEmptyState(){
  const box = qs("answerBox");
  const chat = qs("chatMessages");
  if (!box || !chat) return;

  const hasMsgs = chat.children.length > 0;
  box.classList.toggle("is-empty", !hasMsgs);
}

/* ✅ Sidebar QA container show/hide */
function updateSidebarQaVisibility(){
  const wrap = document.querySelector(".sidebar-qa");
  const list = qs("questionList");
  if (!wrap || !list) return;

  const has = list.children.length > 0;
  wrap.classList.toggle("hidden", !has);
}

/* =============== Sidebar Dropdown =============== */
function toggleDropdown(id){
  const el = qs(id);
  if (!el) return;

  const willOpen = el.classList.contains("hidden");
  el.classList.toggle("hidden");

  // ✅ Dropdown বন্ধ করলে section title + questions সব reset (আপনার চাওয়া অনুযায়ী)
  if (!willOpen){
    clearQuestions();
  }
}
window.toggleDropdown = toggleDropdown;

function clearQuestions(){
  const title = qs("panelTitle");
  const list  = qs("questionList");

  if (title){
    title.innerText = "";
    title.classList.add("hidden");
  }
  if (list){
    list.innerHTML = "";
  }

  activeCategory = null;
  updateSidebarQaVisibility(); // ✅ fully hide sidebar-qa
}

function showQuestions(type){
  const list = qs("questionList");
  const title = qs("panelTitle");
  if (!list || !title) return;

  // ✅ same section clicked again -> hide (toggle)
  if(activeCategory === type){
    clearQuestions();
    return;
  }

  activeCategory = type;

  const titleMap = {
    circular: "Admission Circular",
    apply: "Application Procedure",
    payment: "Payment Process"
  };

  // আপনি title দেখাতে না চাইলে hidden থাকবে (CSS এ hidden করা আছে)
  title.innerText = titleMap[type] || "Questions";
  title.classList.add("hidden"); // ✅ title UI এ না দেখানোর জন্য

  list.innerHTML = "";

  (QUESTIONS[type] || []).forEach((text) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.innerText = text;
    btn.onclick = () => {
      const input = qs("input");
      if (input){
        input.value = text;
        input.focus();
      }
    };
    li.appendChild(btn);
    list.appendChild(li);
  });

  updateSidebarQaVisibility(); // ✅ show panel when has questions
}
window.showQuestions = showQuestions;

/* =============== Chat memory =============== */
function getChatBox(){ return qs("chatMessages"); }

function scrollChatToBottom(){
  const chat = getChatBox();
  if (!chat) return;
  chat.scrollTop = chat.scrollHeight;
}

function saveChat(){
  const chat = getChatBox();
  if (!chat) return;
  localStorage.setItem(STORAGE_KEY, chat.innerHTML);
}

function loadChat(){
  const chat = getChatBox();
  if (!chat) return;
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved && saved.trim()){
    chat.innerHTML = saved;
    scrollChatToBottom();
  }
  updateEmptyState();
}

function clearChat(){
  const chat = getChatBox();
  if (!chat) return;
  chat.innerHTML = "";
  localStorage.removeItem(STORAGE_KEY);
  updateEmptyState();
}
window.clearChat = clearChat;

/* =============== Helpers =============== */
function escapeHtml(s){
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;",
    "<":"&lt;",
    ">":"&gt;",
    '"':"&quot;",
    "'":"&#39;"
  }[c]));
}

function linkifySafe(text){
  let escaped = escapeHtml(text || "");
  const urlRe = /(https?:\/\/[^\s)]+)(\)?)/g;
  escaped = escaped.replace(urlRe, (match, url, trailingParen) => {
    const cleanUrl = url.replace(/&amp;/g, "&");
    return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${url}</a>${trailingParen || ""}`;
  });
  return escaped;
}

function formatISODate(iso){
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;

  const dd = String(d.getUTCDate()).padStart(2, "0");
  const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
  const yyyy = String(d.getUTCFullYear());
  return `${dd}-${mm}-${yyyy}`;
}

/* =============== Meta render (sources + updated) =============== */
function renderBotMeta(meta){
  const wrap = document.createElement("div");
  wrap.className = "msg-meta";

  const sources = Array.isArray(meta.sources) ? meta.sources : [];
  const updated = formatISODate(meta.updated_at);

  sources.slice(0, 5).forEach((s) => {
    const title = (s && s.title) ? String(s.title) : "Source";
    const url = (s && s.url) ? String(s.url) : "";
    if (!url) return;

    const a = document.createElement("a");
    a.className = "meta-badge";
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = title;
    wrap.appendChild(a);
  });

  if (updated){
    const sp = document.createElement("span");
    sp.className = "meta-badge";
    sp.textContent = `Updated: ${updated}`;
    wrap.appendChild(sp);
  }

  return wrap;
}

/* =============== Message render =============== */
function appendUserMessage(text){
  const chat = getChatBox();
  if (!chat) return;

  const row = document.createElement("div");
  row.className = "msg-row msg-row-user";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble msg-user";
  bubble.textContent = text;

  row.appendChild(bubble);
  chat.appendChild(row);

  updateEmptyState();
  scrollChatToBottom();
  saveChat();
}

function appendBotMessage(answerText, meta){
  const chat = getChatBox();
  if (!chat) return;

  const row = document.createElement("div");
  row.className = "msg-row msg-row-bot";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble msg-bot";
  bubble.innerHTML = linkifySafe(answerText || "");

  row.appendChild(bubble);

  const metaWrap = renderBotMeta(meta || {});
  if (metaWrap && metaWrap.childNodes.length > 0){
    row.appendChild(metaWrap);
  }

  chat.appendChild(row);

  updateEmptyState();
  scrollChatToBottom();
  saveChat();
}

/* ✅ Typing indicator bubble */
function appendTypingBubble(){
  const chat = getChatBox();
  if (!chat) return;

  removeTypingBubble();

  const row = document.createElement("div");
  row.className = "msg-row msg-row-bot";
  row.id = "typingRow";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble msg-bot";
  bubble.innerHTML = `
    <span class="typing" aria-label="Typing">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </span>
  `;

  row.appendChild(bubble);
  chat.appendChild(row);

  updateEmptyState();
  scrollChatToBottom();
}

function removeTypingBubble(){
  const chat = getChatBox();
  if (!chat) return;
  const row = qs("typingRow");
  if (row && row.parentElement === chat){
    chat.removeChild(row);
  }
  updateEmptyState();
}

/* =============== Backend status =============== */
function setStatus(kind, text){
  const st = qs("status");
  if (!st) return;

  st.classList.remove("hidden", "ok", "err");
  st.classList.add(kind);
  st.innerText = text;
}

let connectedHideTimer = null;

async function checkBackend(){
  const btn = qs("sendBtn");
  try{
    const r = await fetch(BACKEND + "/health", { method: "GET" });
    if(!r.ok) throw new Error("health not ok");

    setStatus("ok", "✅ Backend connected");
    if (btn) btn.disabled = false;

    if (connectedHideTimer) clearTimeout(connectedHideTimer);
    connectedHideTimer = setTimeout(() => {
      const st = qs("status");
      if (st) st.classList.add("hidden");
    }, 60000);

  }catch(e){
    setStatus("err", "⚠️ Backend সংযুক্ত নেই। Backend চালু করুন (cd backend && uvicorn main:app --reload --port 8000)।");
    if (btn) btn.disabled = true;

    if (connectedHideTimer) clearTimeout(connectedHideTimer);
  }
}

/* =============== Send message =============== */
async function sendMessage(){
  const input = qs("input");
  const msg = (input.value || "").trim();
  if(!msg) return;

  const box = qs("answerBox");
  if (box) box.classList.remove("hidden");

  appendUserMessage(msg);
  input.value = "";

  appendTypingBubble();

  try{
    const res = await fetch(BACKEND + "/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message:msg, session_id: getSessionId()})
    });

    if(!res.ok){
      const err = await res.json().catch(()=> ({}));
      throw new Error(err.detail || ("HTTP " + res.status));
    }

    const data = await res.json();
    setSessionId(data.session_id || "");
    removeTypingBubble();

    appendBotMessage(data.answer || "No answer", {
      sources: data.sources || [],
      updated_at: data.updated_at || ""
    });

  }catch(e){
    removeTypingBubble();

    appendBotMessage("দুঃখিত—এখন সার্ভিস পাওয়া যাচ্ছে না। Backend চালু আছে কিনা চেক করুন।", {
      sources: [],
      updated_at: ""
    });

    setStatus("err", "⚠️ Backend error: " + e.message);
  }
}
window.sendMessage = sendMessage;

/* =============== Links init =============== */
function initLinks(){
  const w = qs("linkWebsite");
  const a = qs("linkAdmission");

  if (w){
    w.textContent = "Website";
    w.href = LINKS.website;
    w.target = "_blank";
    w.rel = "noopener noreferrer";
  }

  if (a){
    a.textContent = "Admission Message & Date";
    a.href = LINKS.admissionMessage;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
  }
}

/* Enter key */
document.addEventListener("keydown", (e) => {
  if(e.key === "Enter"){
    const active = document.activeElement;
    if(active && active.id === "input"){
      sendMessage();
    }
  }
});

/* Boot */
clearQuestions();
initLinks();
checkBackend();
setInterval(checkBackend, 30000);
loadChat();
updateEmptyState();
updateSidebarQaVisibility();
