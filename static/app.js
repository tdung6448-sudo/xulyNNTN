// Session management
let currentSessionId = null;
const sessionLabels = {}; // {id: label}

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function saveSessionsToStorage() {
  localStorage.setItem("sessions", JSON.stringify(Object.keys(sessionLabels)));
  localStorage.setItem("sessionLabels", JSON.stringify(sessionLabels));
  localStorage.setItem("currentSession", currentSessionId || "");
}

function loadSessionsFromStorage() {
  const ids = JSON.parse(localStorage.getItem("sessions") || "[]");
  const labels = JSON.parse(localStorage.getItem("sessionLabels") || "{}");
  Object.assign(sessionLabels, labels);
  return ids;
}

function renderSessionList() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  const ids = Object.keys(sessionLabels);
  ids.reverse().forEach((id) => {
    const li = document.createElement("li");
    li.textContent = sessionLabels[id] || "Hội thoại";
    li.dataset.id = id;
    if (id === currentSessionId) li.classList.add("active");
    li.addEventListener("click", () => switchSession(id));
    list.appendChild(li);
  });
}

async function switchSession(id) {
  currentSessionId = id;
  saveSessionsToStorage();
  renderSessionList();

  const messagesEl = document.getElementById("messages");
  messagesEl.innerHTML = "";

  // Load history từ server
  try {
    const resp = await fetch(`/api/history/${id}`);
    if (resp.ok) {
      const history = await resp.json();
      history.forEach((msg) => appendMessage(msg.role, msg.content));
    }
  } catch (e) {
    console.error("Không load được lịch sử:", e);
  }
}

function newSession() {
  const id = generateId();
  sessionLabels[id] = "Hội thoại mới";
  currentSessionId = id;
  saveSessionsToStorage();
  renderSessionList();
  document.getElementById("messages").innerHTML = "";
}

// Message rendering
function appendMessage(role, text) {
  const messagesEl = document.getElementById("messages");

  const wrapper = document.createElement("div");
  wrapper.classList.add("message", role);

  const avatar = document.createElement("div");
  avatar.classList.add("avatar");
  avatar.textContent = role === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  // Tách tool notices ra khỏi bubble chính
  const lines = text.split("\n");
  const toolLines = [];
  const contentLines = [];
  lines.forEach((line) => {
    if (line.startsWith("[Đang dùng tool:")) {
      toolLines.push(line);
    } else {
      contentLines.push(line);
    }
  });

  if (toolLines.length > 0) {
    toolLines.forEach((t) => {
      const notice = document.createElement("div");
      notice.classList.add("tool-notice");
      notice.textContent = t;
      wrapper.appendChild(notice);
    });
  }

  bubble.textContent = contentLines.join("\n").trim();
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

// Chat form submit
const form = document.getElementById("chat-form");
const input = document.getElementById("input");
const btnSend = document.getElementById("btn-send");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  if (!currentSessionId) newSession();

  // Update session label with first message
  if (sessionLabels[currentSessionId] === "Hội thoại mới") {
    sessionLabels[currentSessionId] = message.slice(0, 30) + (message.length > 30 ? "..." : "");
    saveSessionsToStorage();
    renderSessionList();
  }

  appendMessage("user", message);
  input.value = "";
  input.style.height = "auto";
  btnSend.disabled = true;

  // Create assistant bubble (streaming)
  const messagesEl = document.getElementById("messages");
  const wrapper = document.createElement("div");
  wrapper.classList.add("message", "assistant", "typing");

  const avatar = document.createElement("div");
  avatar.classList.add("avatar");
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: currentSessionId, message }),
    });

    if (!resp.ok) {
      bubble.textContent = "Lỗi: " + resp.statusText;
      wrapper.classList.remove("typing");
      btnSend.disabled = false;
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;

      // Render: tách tool notices từ phần text đang stream
      const lines = fullText.split("\n");
      const toolLines = [];
      const contentLines = [];
      lines.forEach((line) => {
        if (line.startsWith("[Đang dùng tool:")) {
          toolLines.push(line);
        } else {
          contentLines.push(line);
        }
      });

      // Xóa tool notices cũ và render lại
      Array.from(wrapper.querySelectorAll(".tool-notice")).forEach((n) => n.remove());
      toolLines.forEach((t) => {
        const notice = document.createElement("div");
        notice.classList.add("tool-notice");
        notice.textContent = t;
        wrapper.insertBefore(notice, bubble);
      });

      bubble.textContent = contentLines.join("\n").trim();
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  } catch (err) {
    bubble.textContent = "Lỗi kết nối: " + err.message;
  }

  wrapper.classList.remove("typing");
  btnSend.disabled = false;
  input.focus();
});

// Auto-resize textarea
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 150) + "px";
});

// Shift+Enter = newline, Enter = submit
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// New session button
document.getElementById("btn-new").addEventListener("click", newSession);

// Init
(function init() {
  const savedIds = loadSessionsFromStorage();
  const savedCurrent = localStorage.getItem("currentSession");

  if (savedIds.length > 0) {
    if (savedCurrent && sessionLabels[savedCurrent]) {
      currentSessionId = savedCurrent;
    } else {
      currentSessionId = savedIds[savedIds.length - 1];
    }
    renderSessionList();
    switchSession(currentSessionId);
  } else {
    newSession();
  }
})();
