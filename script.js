/* Shared behavior for CloudFlow's static pages. */
const menuButton = document.querySelector('.menu-toggle');
const navigation = document.querySelector('.main-nav');

if (menuButton && navigation) {
  menuButton.addEventListener('click', () => {
    const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!isOpen));
    menuButton.setAttribute('aria-label', isOpen ? 'Open navigation' : 'Close navigation');
    navigation.classList.toggle('is-open', !isOpen);
  });

  navigation.addEventListener('click', (event) => {
    if (event.target.closest('a')) {
      menuButton.setAttribute('aria-expanded', 'false');
      menuButton.setAttribute('aria-label', 'Open navigation');
      navigation.classList.remove('is-open');
    }
  });
}

document.querySelectorAll('[data-year]').forEach((year) => {
  year.textContent = new Date().getFullYear();
});

// The HTML chatbot control is shared by all six pages. Build its panel here so
// the static pages stay simple and only this script needs the API integration.
const chatButton = document.querySelector('.chat-placeholder');

if (chatButton) {
  const HISTORY_KEY = 'cloudflow-chat-history-v1';
  const MAX_HISTORY_MESSAGES = 6;

  function getChatApiUrl() {
    if (window.CLOUDFLOW_API_URL) return window.CLOUDFLOW_API_URL;

    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1') {
      // Run the static site on port 8000 and FastAPI on port 8001 locally.
      return `${window.location.protocol}//${host}:8001/api/chat`;
    }
    return '/api/chat';
  }

  function loadHistory() {
    try {
      const saved = JSON.parse(window.sessionStorage.getItem(HISTORY_KEY) || '[]');
      if (!Array.isArray(saved)) return [];
      return saved
        .filter((message) => (
          message
          && (message.role === 'user' || message.role === 'assistant')
          && typeof message.content === 'string'
        ))
        .slice(-MAX_HISTORY_MESSAGES);
    } catch {
      // Storage may be disabled by the browser; the chat still works in memory.
      return [];
    }
  }

  let conversationHistory = loadHistory();

  const panel = document.createElement('section');
  panel.className = 'chat-panel';
  panel.id = 'cloudflow-chat-panel';
  panel.hidden = true;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-modal', 'false');
  panel.setAttribute('aria-labelledby', 'cloudflow-chat-title');

  const panelHeader = document.createElement('div');
  panelHeader.className = 'chat-panel-header';
  const headingWrap = document.createElement('div');
  const title = document.createElement('h2');
  title.id = 'cloudflow-chat-title';
  title.textContent = 'CloudFlow support';
  const subtitle = document.createElement('p');
  subtitle.textContent = 'Ask a question about CloudFlow';
  headingWrap.append(title, subtitle);

  const closeButton = document.createElement('button');
  closeButton.className = 'chat-close';
  closeButton.type = 'button';
  closeButton.setAttribute('aria-label', 'Close chat');
  closeButton.textContent = '×';
  panelHeader.append(headingWrap, closeButton);

  const messages = document.createElement('div');
  messages.className = 'chat-messages';
  messages.setAttribute('role', 'log');
  messages.setAttribute('aria-live', 'polite');
  messages.setAttribute('aria-relevant', 'additions text');

  const form = document.createElement('form');
  form.className = 'chat-form';
  const inputLabel = document.createElement('label');
  inputLabel.className = 'chat-input-label';
  inputLabel.htmlFor = 'cloudflow-chat-input';
  inputLabel.textContent = 'Your question';
  const inputRow = document.createElement('div');
  inputRow.className = 'chat-input-row';
  const input = document.createElement('input');
  input.id = 'cloudflow-chat-input';
  input.name = 'question';
  input.type = 'text';
  input.maxLength = 4000;
  input.autocomplete = 'off';
  input.placeholder = 'Type your question…';
  input.required = true;
  const sendButton = document.createElement('button');
  sendButton.className = 'chat-send';
  sendButton.type = 'submit';
  sendButton.textContent = 'Send';
  inputRow.append(input, sendButton);
  const formNote = document.createElement('p');
  formNote.className = 'chat-form-note';
  formNote.textContent = 'Conversation history lasts for this browser tab.';
  form.append(inputLabel, inputRow, formNote);

  panel.append(panelHeader, messages, form);
  document.body.append(panel);

  function scrollMessagesToBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function addMessage(role, text, sources = []) {
    const message = document.createElement('article');
    message.className = `chat-message chat-message--${role}`;
    const bubble = document.createElement('p');
    bubble.className = 'chat-message-bubble';
    bubble.textContent = text;
    message.append(bubble);

    if (sources.length) {
      const sourceHeading = document.createElement('p');
      sourceHeading.className = 'chat-sources-heading';
      sourceHeading.textContent = 'Sources';
      const sourceList = document.createElement('ul');
      sourceList.className = 'chat-sources';
      sources.forEach((source) => {
        const item = document.createElement('li');
        item.textContent = source;
        sourceList.append(item);
      });
      message.append(sourceHeading, sourceList);
    }

    messages.append(message);
    scrollMessagesToBottom();
  }

  function splitAnswer(answer) {
    let visibleAnswer = answer.trim();
    const timingsMarker = '\n\nTempi:';
    const timingsIndex = visibleAnswer.lastIndexOf(timingsMarker);
    if (timingsIndex >= 0) visibleAnswer = visibleAnswer.slice(0, timingsIndex);

    const sourcesMarker = '\n\nFonti utilizzate:\n';
    const sourcesIndex = visibleAnswer.lastIndexOf(sourcesMarker);
    if (sourcesIndex < 0) return { text: visibleAnswer, sources: [] };

    const sourceText = visibleAnswer.slice(sourcesIndex + sourcesMarker.length);
    const sources = sourceText
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => /^\[S\d+\]\s+/.test(line));
    return { text: visibleAnswer.slice(0, sourcesIndex).trim(), sources };
  }

  function persistHistory() {
    conversationHistory = conversationHistory.slice(-MAX_HISTORY_MESSAGES);
    try {
      window.sessionStorage.setItem(HISTORY_KEY, JSON.stringify(conversationHistory));
    } catch {
      // Retain the current conversation in memory if sessionStorage is blocked.
    }
  }

  function setBusy(isBusy) {
    input.disabled = isBusy;
    sendButton.disabled = isBusy;
    sendButton.textContent = isBusy ? 'Sending…' : 'Send';
    form.setAttribute('aria-busy', String(isBusy));
  }

  function openChat() {
    panel.hidden = false;
    panel.classList.add('is-open');
    panel.setAttribute('aria-hidden', 'false');
    chatButton.setAttribute('aria-expanded', 'true');
    chatButton.setAttribute('aria-label', 'Close CloudFlow support chat');
    chatButton.title = 'Close chat';
    const tooltip = chatButton.querySelector('.chat-tooltip');
    if (tooltip) tooltip.textContent = 'Close chat';
    if (!messages.childElementCount) {
      addMessage('assistant', 'Hello! Ask me a question about CloudFlow.');
    }
    input.focus();
  }

  function closeChat() {
    panel.hidden = true;
    panel.classList.remove('is-open');
    panel.setAttribute('aria-hidden', 'true');
    chatButton.setAttribute('aria-expanded', 'false');
    chatButton.setAttribute('aria-label', 'Open CloudFlow support chat');
    chatButton.title = 'Open chat';
    const tooltip = chatButton.querySelector('.chat-tooltip');
    if (tooltip) tooltip.textContent = 'Open chat';
    chatButton.focus();
  }

  chatButton.setAttribute('aria-controls', panel.id);
  chatButton.setAttribute('aria-expanded', 'false');
  chatButton.setAttribute('aria-label', 'Open CloudFlow support chat');
  chatButton.title = 'Open chat';
  const chatTooltip = chatButton.querySelector('.chat-tooltip');
  if (chatTooltip) chatTooltip.textContent = 'Open chat';
  chatButton.addEventListener('click', () => {
    if (panel.hidden) openChat();
    else closeChat();
  });
  closeButton.addEventListener('click', closeChat);
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !panel.hidden) closeChat();
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || sendButton.disabled) return;

    addMessage('user', question);
    input.value = '';
    setBusy(true);

    try {
      const response = await window.fetch(getChatApiUrl(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, conversation_history: conversationHistory }),
      });

      let payload;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        if (response.status === 502) {
          throw new Error('CloudFlow support is temporarily unavailable. Please try again shortly.');
        }
        const detail = payload && typeof payload.detail === 'string' ? payload.detail : '';
        throw new Error(detail || `We could not send your question (HTTP ${response.status}).`);
      }

      if (!payload || typeof payload.answer !== 'string') {
        throw new Error('The support service returned an invalid response. Please try again.');
      }

      const parsed = splitAnswer(payload.answer);
      addMessage('assistant', parsed.text || payload.answer, parsed.sources);
      if (Array.isArray(payload.conversation_history)) {
        conversationHistory = payload.conversation_history
          .filter((message) => (
            message
            && (message.role === 'user' || message.role === 'assistant')
            && typeof message.content === 'string'
          ))
          .slice(-MAX_HISTORY_MESSAGES);
      } else {
        conversationHistory = [
          ...conversationHistory,
          { role: 'user', content: question },
          { role: 'assistant', content: payload.answer },
        ].slice(-MAX_HISTORY_MESSAGES);
      }
      persistHistory();
    } catch (error) {
      const message = error instanceof Error
        ? error.message
        : 'We could not reach CloudFlow support. Please try again.';
      addMessage('error', message);
    } finally {
      setBusy(false);
      input.focus();
    }
  });

  conversationHistory.forEach((message) => {
    if (message.role === 'user') addMessage('user', message.content);
    else {
      const parsed = splitAnswer(message.content);
      addMessage('assistant', parsed.text || message.content, parsed.sources);
    }
  });
}
