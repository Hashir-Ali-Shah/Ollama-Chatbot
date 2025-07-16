// Chatbot.jsx
import React, { useState, useRef } from 'react';

const Chatbot = () => {
  const [messages, setMessages] = useState([]); // [{role: 'user'|'assistant', content: '...'}]
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    const assistantMessage = { role: 'assistant', content: '' };
    setMessages((prev) => [...prev, assistantMessage]);

    const response = await fetch(`http://localhost:8000/invoke?content=${encodeURIComponent(input)}`, {
      method: 'POST',
      headers: {
        Accept: 'text/event-stream'
      }
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });

      setMessages((prev) =>
        prev.map((msg, idx) =>
          idx === prev.length - 1
            ? { ...msg, content: msg.content + chunk }
            : msg
        )
      );
      scrollToBottom();
    }

    setLoading(false);
  };

  return (
    <div className="chat-container" style={styles.container}>
      <div className="chat-box" style={styles.chatBox}>
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.message,
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              background: msg.role === 'user' ? '#007bff' : '#e0e0e0',
              color: msg.role === 'user' ? 'white' : 'black'
            }}
          >
            {msg.content}
          </div>
        ))}
        {loading && <div style={styles.typing}>Assistant is typing...</div>}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          style={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
        />
        <button type="submit" style={styles.button} disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
};

const styles = {
  container: {
    maxWidth: '700px',
    margin: '0 auto',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    fontFamily: 'sans-serif'
  },
  chatBox: {
    flex: 1,
    overflowY: 'auto',
    padding: '1rem',
    border: '1px solid #ccc',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    backgroundColor: '#f9f9f9'
  },
  message: {
    maxWidth: '60%',
    padding: '0.75rem',
    borderRadius: '12px',
    fontSize: '0.95rem'
  },
  form: {
    display: 'flex',
    padding: '1rem',
    borderTop: '1px solid #ccc',
    gap: '0.5rem'
  },
  input: {
    flex: 1,
    padding: '0.75rem',
    fontSize: '1rem'
  },
  button: {
    padding: '0.75rem 1.5rem',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    cursor: 'pointer',
    borderRadius:'10px'
  },
  typing: {
    fontStyle: 'italic',
    color: '#666',
    fontSize: '0.9rem'
  }
};

export default Chatbot;
