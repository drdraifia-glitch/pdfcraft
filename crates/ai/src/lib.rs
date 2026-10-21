//! pdfcraft-ai — Groq API client for PdfCraft's AI assistant.
//!
//! Manages multiple API keys with round-robin rotation, 60-second cooldown on
//! rate limits (HTTP 429), instant failover, cancellation signals, and streaming
//! token delivery with <think> tag sanitization.
//!
//! # Privacy
//! No telemetry. The only network traffic is what the user explicitly triggers
//! by asking the AI with at least one key configured. Selected PDF text is sent
//! as part of the prompt context; nothing else leaves the machine.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    ("llama-3.3-70b-versatile", "Llama 3.3 70B (Versatile — Recommended)"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B (Fast, multimodal)"),
    ("qwen/qwen3-32b", "Qwen3 32B (Multilingual, accurate)"),
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
// Thinking Block Sanitizer (strips <think>...</think> from reasoning models)
// ---------------------------------------------------------------------------

pub fn strip_thinking_blocks(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result[start..].find("</think>") {
            result.replace_range(start..start + end + 8, "");
        } else {
            result.truncate(start);
            break;
        }
    }
    while let Some(start) = result.find("<thinking>") {
        if let Some(end) = result[start..].find("</thinking>") {
            result.replace_range(start..start + end + 11, "");
        } else {
            result.truncate(start);
            break;
        }
    }
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// Key pool: multiple keys with cooldown & automatic failover
// ---------------------------------------------------------------------------

/// A pool of Groq API keys with rate-limit cooldown and concurrency safety.
#[derive(Clone, Default)]
pub struct KeyPool {
    inner: Arc<KeyPoolInner>,
}

#[derive(Default)]
struct KeyPoolInner {
    keys: Mutex<Vec<String>>,
    cooldowns: Mutex<HashMap<String, Instant>>,
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
        let mut cd_guard = self.inner.cooldowns.lock().unwrap_or_else(|e| e.into_inner());
        cd_guard.clear();
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

    /// Put a key into temporary cooldown (e.g. 60 seconds on HTTP 429).
    pub fn mark_cooldown(&self, key: &str, duration: Duration) {
        let mut guard = self.inner.cooldowns.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(key.to_string(), Instant::now() + duration);
    }

    /// Pick the next available key that is not in cooldown.
    /// If all keys are in cooldown, falls back to picking round-robin.
    pub fn pick_next_available(&self) -> Option<String> {
        let keys_guard = self.inner.keys.lock().unwrap_or_else(|e| e.into_inner());
        if keys_guard.is_empty() {
            return None;
        }

        let cd_guard = self.inner.cooldowns.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        let available: Vec<&String> = keys_guard
            .iter()
            .filter(|k| match cd_guard.get(*k) {
                Some(expiry) => now >= *expiry,
                None => true,
            })
            .collect();

        if !available.is_empty() {
            let idx = self.inner.next.fetch_add(1, Ordering::Relaxed) % available.len();
            available.get(idx).map(|s| (*s).clone())
        } else {
            // All in cooldown: round-robin fallback across all keys
            let idx = self.inner.next.fetch_add(1, Ordering::Relaxed) % keys_guard.len();
            keys_guard.get(idx).cloned()
        }
    }

    /// Pick an available key with an offset to avoid trying the same key again in a retry loop.
    pub fn pick_offset(&self, offset: usize) -> Option<String> {
        let keys_guard = self.inner.keys.lock().unwrap_or_else(|e| e.into_inner());
        if keys_guard.is_empty() {
            return None;
        }

        let cd_guard = self.inner.cooldowns.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        let available: Vec<&String> = keys_guard
            .iter()
            .filter(|k| match cd_guard.get(*k) {
                Some(expiry) => now >= *expiry,
                None => true,
            })
            .collect();

        let pool_to_use = if available.is_empty() {
            &*keys_guard
        } else {
            let cloned: Vec<String> = available.into_iter().cloned().collect();
            return {
                let cur = self.inner.next.load(Ordering::Relaxed);
                let idx = cur.saturating_add(offset) % cloned.len();
                cloned.get(idx).cloned()
            };
        };

        let cur = self.inner.next.load(Ordering::Relaxed);
        let idx = cur.saturating_add(offset) % pool_to_use.len();
        pool_to_use.get(idx).cloned()
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

/// Spawn a background thread that calls Groq and returns the receiving end
/// of the event channel and a cancellation flag.
/// Setting the cancellation flag to `true` stops streaming immediately.
pub fn stream_completion(
    pool: KeyPool,
    model: String,
    messages: Vec<Message>,
) -> (std::sync::mpsc::Receiver<AiEvent>, Arc<AtomicBool>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_thread = Arc::clone(&cancel);
    std::thread::spawn(move || {
        stream_chat(pool, model, messages, tx, cancel_thread);
    });
    (rx, cancel)
}

/// Send a chat completion request to Groq and stream tokens back through `sender`.
/// Rotates keys automatically, cools down rate-limited keys for 60s, and checks `cancel`.
pub fn stream_chat(
    pool: KeyPool,
    model: String,
    messages: Vec<Message>,
    sender: std::sync::mpsc::Sender<AiEvent>,
    cancel: Arc<AtomicBool>,
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
        max_tokens: 3072,
        temperature: 0.3,
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
        if cancel.load(Ordering::Relaxed) {
            let _ = sender.send(AiEvent::Done);
            return;
        }

        let key = match pool.pick_offset(attempt) {
            Some(k) => k,
            None => break,
        };

        match try_stream_request(&key, &body_json, &sender, &cancel) {
            Ok(()) => return,
            Err(e) => {
                log::warn!("pdfcraft-ai: attempt {} with key failed: {}", attempt + 1, e);
                last_error = e.to_string();

                if last_error.contains("429") || last_error.contains("limit") {
                    pool.mark_cooldown(&key, Duration::from_secs(60));
                    continue;
                }

                if !last_error.contains("401") {
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
    cancel: &Arc<AtomicBool>,
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
        if cancel.load(Ordering::Relaxed) {
            let _ = sender.send(AiEvent::Done);
            return Ok(());
        }

        line.clear();
        let bytes_read = reader.read_line(&mut line).map_err(|e| AiError::Io(e.to_string()))?;
        if bytes_read == 0 {
            break;
        }

        if cancel.load(Ordering::Relaxed) {
            let _ = sender.send(AiEvent::Done);
            return Ok(());
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
             Answer questions regarding this excerpt accurately, thoroughly, and concisely.",
            selected_text.trim()
        )
    };

    Message::system(format!(
        "You are an expert, multilingual academic and educational AI assistant embedded in PdfCraft.\
         Help the user understand, summarize, explain, and extract insights from their PDF document.\
         Be concise, precise, and polite. Always respond in the exact language used by the user (Arabic, French, or English).\
         Format mathematical and scientific expressions cleanly using standard Markdown and LaTeX math delimiters.\
         Do not include meta-reasoning, scratchpads, or <think> tags in your final answer.{context}"
    ))
}
