//! pdfcraft-ai — Groq API client for PdfCraft's AI assistant.
//!
//! Manages multiple API keys with round-robin rotation, automatic failover on
//! rate limits (HTTP 429), sends chat completions to Groq, and streams back
//! responses token by token via an `mpsc` channel.
//!
//! # Privacy
//! No telemetry. The only network traffic is what the user explicitly triggers
//! by asking the AI with at least one key configured. Selected PDF text is sent
//! as part of the prompt context; nothing else leaves the machine.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fmt;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum AiError {
    NoApiKeys,
    AllKeysFailed(String),
    Http(String),
    Json(String),
    Io(String),
}

impl fmt::Display for AiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AiError::NoApiKeys => write!(f, "No Groq API keys configured. Add one or more keys in Preferences › AI."),
            AiError::AllKeysFailed(e) => write!(f, "All API keys failed or rate-limited: {e}"),
            AiError::Http(e) => write!(f, "Groq API error: {e}"),
            AiError::Json(e) => write!(f, "Failed to parse API response: {e}"),
            AiError::Io(e) => write!(f, "Network connection error: {e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Groq Models and Constants
// ---------------------------------------------------------------------------

/// Default model: Llama 3.3 70B (best balance of quality and speed on Groq).
pub const DEFAULT_MODEL: &str = "llama-3.3-70b-versatile";

pub const AVAILABLE_MODELS: &[(&str, &str)] = &[
    ("llama-3.3-70b-versatile", "Llama 3.3 70B (Versatile — recommended)"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B (Fast, multimodal)"),
    ("qwen/qwen3-32b", "Qwen3 32B (Multilingual, agentic)"),
    ("qwen-qwq-32b", "QwQ 32B (Deep reasoning)"),
    ("llama-3.1-8b-instant", "Llama 3.1 8B (Ultra-fast)"),
];

const GROQ_CHAT_URL: &str = "https://api.groq.com/openai/v1/chat/completions";

// ---------------------------------------------------------------------------
// Groq API Request & Response Types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct StreamDelta {
    content: Option<String>,
}

// ---------------------------------------------------------------------------
// Key pool: multiple keys with round-robin rotation & fallback
// ---------------------------------------------------------------------------

/// A pool of Groq API keys with round-robin rotation and concurrency safety.
#[derive(Clone, Default)]
pub struct KeyPool {
    inner: Arc<KeyPoolInner>,
}

#[derive(Default)]
struct KeyPoolInner {
    keys: Mutex<Vec<String>>,
    next: AtomicUsize,
}

impl KeyPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace all keys in the pool (e.g. from Preferences).
    pub fn set_keys(&self, keys: Vec<String>) {
        let cleaned: Vec<String> = keys
            .into_iter()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .collect();
        let mut guard = self.inner.keys.lock().unwrap_or_else(|e| e.into_inner());
        *guard = cleaned;
        self.inner.next.store(0, Ordering::Relaxed);
    }

    /// Return all keys currently stored.
    pub fn keys(&self) -> Vec<String> {
        self.inner.keys.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Number of configured keys.
    pub fn count(&self) -> usize {
        self.inner.keys.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Returns `true` if no keys are configured.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Pick the next key in round-robin order.
    pub fn pick(&self) -> Option<String> {
        let guard = self.inner.keys.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_empty() {
            return None;
        }
        let idx = self.inner.next.fetch_add(1, Ordering::Relaxed) % guard.len();
        guard.get(idx).cloned()
    }

    /// Pick a key with an offset (for retrying the next available keys).
    fn pick_offset(&self, offset: usize) -> Option<String> {
        let guard = self.inner.keys.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_empty() {
            return None;
        }
        let cur = self.inner.next.load(Ordering::Relaxed);
        let idx = (cur.saturating_add(offset)) % guard.len();
        guard.get(idx).cloned()
    }
}

// ---------------------------------------------------------------------------
// Streaming Events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum AiEvent {
    /// A text delta from the assistant.
    Token(String),
    /// Stream ended normally.
    Done,
    /// An error occurred during streaming.
    Error(String),
}

// ---------------------------------------------------------------------------
// Streaming Chat Execution
// ---------------------------------------------------------------------------

/// Send a chat completion request to Groq and stream tokens back through `sender`.
/// Rotates keys automatically and retries on rate limits (HTTP 429).
pub fn stream_chat(
    pool: KeyPool,
    model: String,
    messages: Vec<Message>,
    sender: std::sync::mpsc::Sender<AiEvent>,
) {
    let key_count = pool.count();
    if key_count == 0 {
        let _ = sender.send(AiEvent::Error(AiError::NoApiKeys.to_string()));
        return;
    }

    let model_to_use = if model.is_empty() { DEFAULT_MODEL } else { &model };
    let body = ChatRequest {
        model: model_to_use,
        messages: &messages,
        stream: true,
        max_tokens: 2048,
        temperature: 0.7,
    };

    let body_json = match serde_json::to_string(&body) {
        Ok(j) => j,
        Err(e) => {
            let _ = sender.send(AiEvent::Error(AiError::Json(e.to_string()).to_string()));
            return;
        }
    };

    // Try keys sequentially on failure / rate limit
    let mut last_error = String::new();
    let max_tries = key_count.min(5);

    for attempt in 0..max_tries {
        let key = match pool.pick_offset(attempt) {
            Some(k) => k,
            None => break,
        };

        match try_stream_request(&key, &body_json, &sender) {
            Ok(()) => return,
            Err(e) => {
                log::warn!("pdfcraft-ai: attempt {} failed: {}", attempt + 1, e);
                last_error = e.to_string();
                // If it's not a rate-limit/auth error (e.g. client canceled or channel closed), exit
                if !last_error.contains("429") && !last_error.contains("401") && !last_error.contains("limit") {
                    let _ = sender.send(AiEvent::Error(last_error));
                    return;
                }
            }
        }
    }

    let _ = sender.send(AiEvent::Error(AiError::AllKeysFailed(last_error).to_string()));
}

fn try_stream_request(
    key: &str,
    body_json: &str,
    sender: &std::sync::mpsc::Sender<AiEvent>,
) -> Result<(), AiError> {
    let agent_builder = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(45)));

    // Try loading OS native certificates
    let agent_builder = if let Ok(roots) = os_roots() {
        agent_builder.tls_config(ureq::tls::TlsConfig::builder().root_certs(roots).build())
    } else {
        agent_builder
    };

    let agent = agent_builder.build().new_agent();

    let mut response = agent
        .post(GROQ_CHAT_URL)
        .header("Authorization", &format!("Bearer {key}"))
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .header("User-Agent", "PdfCraft-AI/0.10.1")
        .config()
        .http_status_as_error(false)
        .build()
        .send(body_json.as_bytes())
        .map_err(|e| AiError::Io(e.to_string()))?;

    let status = response.status().as_u16();
    if status == 429 {
        return Err(AiError::Http("Rate limit reached (429)".into()));
    }
    if status == 401 {
        return Err(AiError::Http("Invalid API key (401)".into()));
    }
    if !(200..300).contains(&status) {
        let err_body = response
            .body_mut()
            .with_config()
            .limit(4096)
            .read_to_string()
            .unwrap_or_default();
        return Err(AiError::Http(format!("Status {status}: {err_body}")));
    }

    // Read SSE lines
    let mut reader = BufReader::new(response.body_mut().as_reader());
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).map_err(|e| AiError::Io(e.to_string()))?;
        if bytes_read == 0 {
            break;
        }

        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() || trimmed.starts_with(':') {
            continue;
        }

        if let Some(data) = trimmed.strip_prefix("data: ") {
            let data = data.trim();
            if data == "[DONE]" {
                let _ = sender.send(AiEvent::Done);
                return Ok(());
            }

            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                for choice in chunk.choices {
                    if let Some(token) = choice.delta.content {
                        if !token.is_empty() {
                            let _ = sender.send(AiEvent::Token(token));
                        }
                    }
                    if choice.finish_reason.as_deref() == Some("stop") {
                        let _ = sender.send(AiEvent::Done);
                        return Ok(());
                    }
                }
            }
        }
    }

    let _ = sender.send(AiEvent::Done);
    Ok(())
}

fn os_roots() -> Result<ureq::tls::RootCerts, String> {
    let found = rustls_native_certs::load_native_certs();
    let certs: Vec<ureq::tls::Certificate<'static>> =
        found.certs.iter().map(|c| ureq::tls::Certificate::from_der(c.as_ref()).to_owned()).collect();
    if certs.is_empty() {
        return Err("No native certificates found".into());
    }
    Ok(ureq::tls::RootCerts::new_with_certs(&certs))
}

// ---------------------------------------------------------------------------
// System Prompt Builder
// ---------------------------------------------------------------------------

/// Build a contextual system prompt for the PDF assistant.
pub fn system_prompt(selected_text: &str) -> Message {
    let context = if selected_text.trim().is_empty() {
        String::new()
    } else {
        format!(
            "\n\nContext excerpt from the open PDF:\n\"\"\"\n{}\n\"\"\"\n\
             Answer questions regarding this excerpt accurately and concisely.",
            selected_text.trim()
        )
    };

    Message::system(format!(
        "You are an expert AI assistant embedded in PdfCraft, an open-source PDF application.\
         Help the user understand, summarize, explain, and extract insights from their document.\
         Be concise, precise, and polite. Respond in the language used by the user.{context}"
    ))
}
