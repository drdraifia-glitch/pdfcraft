//! AI assistant panel, document summarization, interactive QCM/Quiz runner,
//! and Groq preferences for PdfCraft, modeled after the books assistant.
//!
//! Provides:
//! - Docked right side panel matching the books reader.
//! - 💬 **Chat Tab:** Context-aware conversations with sanitized PDF excerpts,
//!   rich Markdown rendering, mathematical symbols/LaTeX equations, tables,
//!   stop-generation controls, and instant model switching.
//! - 📄 **Summary Tab:** Smart comparison tables and structured executive summaries
//!   with custom page ranges (From page X to page Y).
//! - 🎯 **QCM / Quiz Tab:** Interactive step-by-step quiz runner (Question-by-Question)
//!   matching AIQuiz.tsx with instant green/red answer states, per-option explanations
//!   (A-E breakdown), comprehensive Review Mode, and final score card with performance grade.
//! - ⚙ **Preferences Section:** Multiple Groq API keys with round-robin rotation,
//!   60-second cooldown on rate limits (HTTP 429), and verified model options.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use egui::{Align, Color32, CornerRadius, Layout, Margin, Response, Stroke, vec2};
use pdfcraft_ai::{strip_thinking_blocks, AiEvent, KeyPool, Message};
use serde::{Deserialize, Serialize};

use crate::theme::{self, Tokens};
use crate::{PdfCraftApp, icons};

// ---------------------------------------------------------------------------
// Text Sanitization & LaTeX Math Preprocessor
// ---------------------------------------------------------------------------

/// Cleans raw text extracted from PDF streams to eliminate control characters,
/// broken ligatures, private use characters, and excessive spacing.
pub fn clean_and_sanitize_pdf_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '\n' | '\t' | '\r' => out.push(' '),
            c if c.is_control() => continue,
            // Reconstruct common PDF ligatures
            '\u{FB00}' => out.push_str("ff"),
            '\u{FB01}' => out.push_str("fi"),
            '\u{FB02}' => out.push_str("fl"),
            '\u{FB03}' => out.push_str("ffi"),
            '\u{FB04}' => out.push_str("ffl"),
            '\u{FB05}' | '\u{FB06}' => out.push_str("st"),
            // Zero-width & soft-hyphen characters
            '\u{00AD}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => continue,
            // Replacement or Private-Use characters
            '\u{FFFD}' | '\u{E000}'..='\u{F8FF}' => out.push(' '),
            c => out.push(c),
        }
    }
    // Collapse consecutive spaces
    let mut collapsed = String::with_capacity(out.len());
    let mut prev_space = false;
    for ch in out.chars() {
        if ch == ' ' {
            if !prev_space {
                collapsed.push(' ');
                prev_space = true;
            }
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }
    collapsed.trim().to_string()
}

/// Converts LaTeX delimiters, HTML entity tags, and mathematical symbols
/// into clean, readable Unicode math equivalents (modeled after preprocessMarkdown in books).
pub fn preprocess_markdown_math(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut s = text.to_string();

    // Strip raw HTML tags like <span> or &lt;span&gt;
    s = s.replace("&lt;span&gt;", "")
        .replace("&lt;/span&gt;", "")
        .replace("<span>", "")
        .replace("</span>", "");

    // Convert display math delimiters \[ ... \] -> $$ ... $$
    s = s.replace("\\[", "\n$$\n").replace("\\]", "\n$$\n");
    // Convert inline math delimiters \( ... \) -> $ ... $
    s = s.replace("\\(", "$").replace("\\)", "$");

    // Mathematical symbols and Greek alphabet
    s = s.replace("\\alpha", "α")
        .replace("\\beta", "β")
        .replace("\\gamma", "γ")
        .replace("\\delta", "δ")
        .replace("\\epsilon", "ε")
        .replace("\\theta", "θ")
        .replace("\\lambda", "λ")
        .replace("\\mu", "μ")
        .replace("\\pi", "π")
        .replace("\\sigma", "σ")
        .replace("\\tau", "τ")
        .replace("\\phi", "φ")
        .replace("\\omega", "ω")
        .replace("\\Delta", "Δ")
        .replace("\\Omega", "Ω")
        .replace("\\Sigma", "Σ")
        // Comparison & Operators
        .replace("\\leq", "≤")
        .replace("\\le", "≤")
        .replace("\\geq", "≥")
        .replace("\\ge", "≥")
        .replace("\\neq", "≠")
        .replace("\\ne", "≠")
        .replace("\\approx", "≈")
        .replace("\\pm", "±")
        .replace("\\times", "×")
        .replace("\\div", "÷")
        .replace("\\cdot", "·")
        // Arrows
        .replace("\\rightarrow", "→")
        .replace("\\to", "→")
        .replace("\\leftarrow", "←")
        .replace("\\Rightarrow", "⇒")
        .replace("\\Leftarrow", "⇐")
        .replace("\\Leftrightarrow", "⇔")
        .replace("-->", "→")
        .replace("<--", "←")
        .replace("==>", "⇒")
        // Calculus & Sets
        .replace("\\infty", "∞")
        .replace("\\sum", "∑")
        .replace("\\prod", "∏")
        .replace("\\int", "∫")
        .replace("\\partial", "∂")
        .replace("\\in", "∈")
        .replace("\\subset", "⊂")
        .replace("\\cap", "∩")
        .replace("\\cup", "∪")
        // Common Fractions & Roots
        .replace("\\sqrt", "√")
        .replace("\\circ", "°")
        // Superscripts
        .replace("^2", "²")
        .replace("^3", "³")
        .replace("^n", "ⁿ")
        .replace("^+", "⁺")
        .replace("^-", "⁻");

    s
}

/// Cleans leading choice letters like "A. ", "A) ", "1. " from option strings so they don't double.
fn clean_option_prefix(opt: &str) -> String {
    let trimmed = opt.trim();
    if trimmed.len() > 3 {
        let first = trimmed.chars().next().unwrap_or(' ');
        let second = trimmed.chars().nth(1).unwrap_or(' ');
        if (first.is_ascii_alphabetic() || first.is_ascii_digit()) && (second == '.' || second == ')' || second == '-') {
            return trimmed[2..].trim().to_string();
        }
    }
    trimmed.to_string()
}

// ---------------------------------------------------------------------------
// Tabs & Configuration Enums
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum AiTab {
    #[default]
    Chat,
    Summary,
    Quiz,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum QuizKind {
    #[default]
    Qcs, // Single correct answer
    Qcm, // Multiple correct answers
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum QuizDifficulty {
    Easy,
    #[default]
    Medium,
    Hard,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum QuizReviewFilter {
    #[default]
    All,
    MistakesOnly,
    CorrectOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuizQuestion {
    pub question: String,
    pub options: Vec<String>,
    #[serde(deserialize_with = "deserialize_correct_indices")]
    pub correct: Vec<usize>,
    #[serde(default)]
    pub explanation: String,
    #[serde(default)]
    pub question_type: Option<String>,
}

fn deserialize_correct_indices<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Indices {
        SingleNum(usize),
        SingleStr(String),
        MultipleNum(Vec<usize>),
        MultipleStr(Vec<String>),
    }
    match Indices::deserialize(deserializer)? {
        Indices::SingleNum(idx) => Ok(vec![idx]),
        Indices::SingleStr(s) => {
            let n = s.trim().parse::<usize>().unwrap_or(0);
            Ok(vec![n])
        }
        Indices::MultipleNum(v) => Ok(v),
        Indices::MultipleStr(v) => {
            let nums = v.into_iter().filter_map(|s| s.trim().parse::<usize>().ok()).collect();
            Ok(nums)
        }
    }
}

// ---------------------------------------------------------------------------
// AI Assistant State (lives inside PdfCraftApp)
// ---------------------------------------------------------------------------

pub struct AiState {
    pub open: bool,
    pub active_tab: AiTab,

    // Chat state
    pub history: Vec<Message>,
    pub context_text: String,
    pub draft: String,
    pub selected_model: String,
    pub keys_input: String,
    receiver: Option<mpsc::Receiver<AiEvent>>,
    pub streaming: Option<String>,
    pub error: Option<String>,
    chat_cancel: Option<Arc<AtomicBool>>,

    // Summary state
    pub summary_from_page: usize,
    pub summary_to_page: usize,
    pub summary_result: String,
    pub is_summarizing: bool,
    summary_receiver: Option<mpsc::Receiver<AiEvent>>,
    summary_cancel: Option<Arc<AtomicBool>>,

    // Quiz generator & runner state (AIQuiz.tsx model)
    pub quiz_from_page: usize,
    pub quiz_to_page: usize,
    pub quiz_count: usize,
    pub quiz_kind: QuizKind,
    pub quiz_difficulty: QuizDifficulty,
    pub is_generating_quiz: bool,
    quiz_receiver: Option<mpsc::Receiver<AiEvent>>,
    quiz_cancel: Option<Arc<AtomicBool>>,
    quiz_buffer: String,
    pub quiz_notice: Option<String>,

    // Step-by-step Interactive Runner
    pub quiz_questions: Vec<QuizQuestion>,
    pub quiz_current_index: usize,
    pub quiz_selected_options: Vec<usize>,
    pub quiz_is_confirmed: bool,
    pub quiz_user_answers: Vec<Vec<usize>>,
    pub quiz_show_results: bool,
    pub quiz_show_review: bool,
    pub quiz_review_filter: QuizReviewFilter,
}

impl Default for AiState {
    fn default() -> Self {
        Self {
            open: false,
            active_tab: AiTab::Chat,
            history: Vec::new(),
            context_text: String::new(),
            draft: String::new(),
            selected_model: pdfcraft_ai::DEFAULT_MODEL.to_string(),
            keys_input: String::new(),
            receiver: None,
            streaming: None,
            error: None,
            chat_cancel: None,
            summary_from_page: 1,
            summary_to_page: 1,
            summary_result: String::new(),
            is_summarizing: false,
            summary_receiver: None,
            summary_cancel: None,
            quiz_from_page: 1,
            quiz_to_page: 1,
            quiz_count: 5,
            quiz_kind: QuizKind::Qcs,
            quiz_difficulty: QuizDifficulty::Medium,
            is_generating_quiz: false,
            quiz_receiver: None,
            quiz_cancel: None,
            quiz_buffer: String::new(),
            quiz_notice: None,
            quiz_questions: Vec::new(),
            quiz_current_index: 0,
            quiz_selected_options: Vec::new(),
            quiz_is_confirmed: false,
            quiz_user_answers: Vec::new(),
            quiz_show_results: false,
            quiz_show_review: false,
            quiz_review_filter: QuizReviewFilter::All,
        }
    }
}

impl AiState {
    /// Open the assistant panel with the given excerpt as context.
    pub fn open_with_context(&mut self, text: String) {
        self.open = true;
        self.active_tab = AiTab::Chat;
        let clean = clean_and_sanitize_pdf_text(&text);
        if !clean.is_empty() {
            self.context_text = clean;
        }
    }

    /// Whether any streaming operation is in progress.
    pub fn is_streaming(&self) -> bool {
        self.receiver.is_some() || self.summary_receiver.is_some() || self.quiz_receiver.is_some()
    }

    /// Stop all active generations immediately (like StopGeneration in books).
    pub fn stop_generation(&mut self) {
        if let Some(c) = self.chat_cancel.take() {
            c.store(true, Ordering::Relaxed);
        }
        if let Some(c) = self.summary_cancel.take() {
            c.store(true, Ordering::Relaxed);
        }
        if let Some(c) = self.quiz_cancel.take() {
            c.store(true, Ordering::Relaxed);
        }

        if let Some(stream_text) = self.streaming.take() {
            let cleaned = strip_thinking_blocks(&stream_text);
            if !cleaned.is_empty() {
                self.history.push(Message::assistant(cleaned));
            }
        }

        self.receiver = None;
        self.summary_receiver = None;
        self.quiz_receiver = None;
        self.is_summarizing = false;
        self.is_generating_quiz = false;
    }

    /// Poll stream events from worker threads; call every frame.
    pub fn poll(&mut self) {
        // 1. Poll chat stream
        if let Some(rx) = &self.receiver {
            loop {
                match rx.try_recv() {
                    Ok(AiEvent::Token(t)) => {
                        self.streaming.get_or_insert_with(String::new).push_str(&t);
                    }
                    Ok(AiEvent::Done) => {
                        if let Some(text) = self.streaming.take() {
                            let cleaned = strip_thinking_blocks(&text);
                            if !cleaned.is_empty() {
                                self.history.push(Message::assistant(cleaned));
                            }
                        }
                        self.receiver = None;
                        self.chat_cancel = None;
                        break;
                    }
                    Ok(AiEvent::Error(err)) => {
                        self.error = Some(err);
                        self.streaming = None;
                        self.receiver = None;
                        self.chat_cancel = None;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if let Some(text) = self.streaming.take() {
                            let cleaned = strip_thinking_blocks(&text);
                            if !cleaned.is_empty() {
                                self.history.push(Message::assistant(cleaned));
                            }
                        }
                        self.receiver = None;
                        self.chat_cancel = None;
                        break;
                    }
                }
            }
        }

        // 2. Poll summary stream
        if let Some(rx) = &self.summary_receiver {
            loop {
                match rx.try_recv() {
                    Ok(AiEvent::Token(t)) => {
                        self.summary_result.push_str(&t);
                    }
                    Ok(AiEvent::Done) => {
                        self.summary_result = strip_thinking_blocks(&self.summary_result);
                        self.is_summarizing = false;
                        self.summary_receiver = None;
                        self.summary_cancel = None;
                        break;
                    }
                    Ok(AiEvent::Error(err)) => {
                        self.error = Some(err);
                        self.is_summarizing = false;
                        self.summary_receiver = None;
                        self.summary_cancel = None;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.summary_result = strip_thinking_blocks(&self.summary_result);
                        self.is_summarizing = false;
                        self.summary_receiver = None;
                        self.summary_cancel = None;
                        break;
                    }
                }
            }
        }

        // 3. Poll quiz generation stream
        if let Some(rx) = &self.quiz_receiver {
            loop {
                match rx.try_recv() {
                    Ok(AiEvent::Token(t)) => {
                        self.quiz_buffer.push_str(&t);
                    }
                    Ok(AiEvent::Done) => {
                        self.parse_and_set_quiz();
                        self.is_generating_quiz = false;
                        self.quiz_receiver = None;
                        self.quiz_cancel = None;
                        break;
                    }
                    Ok(AiEvent::Error(err)) => {
                        self.error = Some(err);
                        self.is_generating_quiz = false;
                        self.quiz_receiver = None;
                        self.quiz_cancel = None;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.parse_and_set_quiz();
                        self.is_generating_quiz = false;
                        self.quiz_receiver = None;
                        self.quiz_cancel = None;
                        break;
                    }
                }
            }
        }
    }

    /// Robust multi-tier parser for generated QCM/QCS questions (modeled after extractJson in books).
    fn parse_and_set_quiz(&mut self) {
        let raw = self.quiz_buffer.trim();
        if raw.is_empty() {
            self.error = Some("Model returned an empty response. Please try again.".into());
            return;
        }

        let stripped = raw
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .trim()
            .to_string();

        let value_res: Result<serde_json::Value, String> = serde_json::from_str(&stripped)
            .map_err(|e| e.to_string())
            .or_else(|_| {
                if let (Some(s), Some(e)) = (stripped.find('{'), stripped.rfind('}')) {
                    if e > s {
                        serde_json::from_str(&stripped[s..=e]).map_err(|e| e.to_string())
                    } else {
                        Err("no json object found".to_string())
                    }
                } else if let (Some(s), Some(e)) = (stripped.find('['), stripped.rfind(']')) {
                    if e > s {
                        serde_json::from_str(&stripped[s..=e]).map_err(|e| e.to_string())
                    } else {
                        Err("no json array found".to_string())
                    }
                } else {
                    Err("no json delimiters".to_string())
                }
            });

        match value_res {
            Ok(v) => {
                if let Some(true) = v.get("insufficient_content").and_then(|b| b.as_bool()) {
                    if let Some(notice) = v.get("notice").and_then(|n| n.as_str()) {
                        self.quiz_notice = Some(notice.to_string());
                    }
                } else {
                    self.quiz_notice = None;
                }

                let questions_val = if let Some(arr) = v.get("questions").and_then(|q| q.as_array()) {
                    Some(arr)
                } else if let Some(arr) = v.as_array() {
                    Some(arr)
                } else if let Some(arr) = v.get("quiz").and_then(|q| q.as_array()) {
                    Some(arr)
                } else {
                    None
                };

                if let Some(arr) = questions_val {
                    let mut parsed_questions = Vec::new();
                    for item in arr {
                        if let Ok(mut q) = serde_json::from_value::<QuizQuestion>(item.clone()) {
                            for opt in &mut q.options {
                                *opt = clean_option_prefix(opt);
                            }
                            parsed_questions.push(q);
                        }
                    }

                    if !parsed_questions.is_empty() {
                        let count = parsed_questions.len();
                        self.quiz_questions = parsed_questions;
                        self.quiz_current_index = 0;
                        self.quiz_selected_options.clear();
                        self.quiz_is_confirmed = false;
                        self.quiz_show_results = false;
                        self.quiz_show_review = false;
                        self.quiz_review_filter = QuizReviewFilter::All;
                        self.quiz_user_answers = vec![Vec::new(); count];
                        self.error = None;
                    } else {
                        self.error = Some("Could not extract any valid questions from the response.".into());
                    }
                } else {
                    self.error = Some("Model response did not contain a recognized questions array.".into());
                }
            }
            Err(e) => {
                self.error = Some(format!("Could not parse generated quiz questions: {e}"));
            }
        }

        self.quiz_buffer.clear();
    }

    /// Submit a user prompt in the chat.
    pub fn submit_prompt(&mut self, pool: &KeyPool, prompt: &str) {
        let text = prompt.trim().to_string();
        if text.is_empty() || self.is_streaming() {
            return;
        }
        self.error = None;
        self.history.push(Message::user(text));

        let mut messages = vec![pdfcraft_ai::system_prompt(&self.context_text)];
        messages.extend(self.history.iter().cloned());

        let model = self.selected_model.clone();
        let (rx, cancel) = pdfcraft_ai::stream_completion(pool.clone(), model, messages);
        self.receiver = Some(rx);
        self.chat_cancel = Some(cancel);
        self.streaming = Some(String::new());
    }

    /// Request a smart standard summary (overview + core takeaways).
    pub fn request_standard_summary(&mut self, pool: &KeyPool, extracted_text: &str) {
        if self.is_streaming() {
            return;
        }
        self.error = None;
        self.summary_result.clear();
        self.is_summarizing = true;

        let prompt = format!(
            "You are an expert academic and educational summarizer.\
             Summarize the following PDF excerpt thoroughly, accurately, and concisely.\n\
             Structure your response with:\n\
             1. **Executive Overview (نظرة عامة شاملة)**\n\
             2. **Key Concepts & Definitions (المفاهيم والتعاريف الأساسية)**\n\
             3. **High-Yield Takeaways (أهم النقاط والاستنتاجات)**\n\n\
             Text:\n{extracted_text}"
        );

        let messages = vec![
            Message::system("You are an expert academic summarizer. Output clean, beautifully structured Markdown."),
            Message::user(prompt),
        ];

        let model = self.selected_model.clone();
        let (rx, cancel) = pdfcraft_ai::stream_completion(pool.clone(), model, messages);
        self.summary_receiver = Some(rx);
        self.summary_cancel = Some(cancel);
    }

    /// Request a summary as smart comparison tables (matching books summarizeTable prompt).
    pub fn request_summary_tables(&mut self, pool: &KeyPool, extracted_text: &str) {
        if self.is_streaming() {
            return;
        }
        self.error = None;
        self.summary_result.clear();
        self.is_summarizing = true;

        let prompt = format!(
            "You are an expert academic summarizer. Transform the following text into structured, clean Markdown comparison tables and concise key points.\n\
            Rules:\n\
            1. Use Markdown tables with clear column headers: | Concept / Criterion | Details & Mechanism | Clinical / Practical Notes |\n\
            2. Break complex topics into dedicated comparative tables.\n\
            3. Include key bullet takeaways below the tables.\n\
            4. Keep mathematical and scientific notation clean and precise.\n\n\
            Text:\n\
            {extracted_text}"
        );

        let messages = vec![
            Message::system("You are an expert summarizer that outputs well-structured Markdown tables and key takeaways."),
            Message::user(prompt),
        ];

        let model = self.selected_model.clone();
        let (rx, cancel) = pdfcraft_ai::stream_completion(pool.clone(), model, messages);
        self.summary_receiver = Some(rx);
        self.summary_cancel = Some(cancel);
    }

    /// Request generating QCM/QCS questions from context text (matching books generateQuiz prompt).
    pub fn request_quiz(&mut self, pool: &KeyPool, context: &str) {
        if self.is_streaming() {
            return;
        }
        self.error = None;
        self.quiz_buffer.clear();
        self.quiz_notice = None;
        self.quiz_questions.clear();
        self.quiz_current_index = 0;
        self.quiz_selected_options.clear();
        self.quiz_is_confirmed = false;
        self.quiz_show_results = false;
        self.quiz_show_review = false;
        self.is_generating_quiz = true;

        let count = self.quiz_count;
        let is_qcm = self.quiz_kind == QuizKind::Qcm;
        let diff_str = match self.quiz_difficulty {
            QuizDifficulty::Easy => "direct recall, foundational definitions, and straightforward principles",
            QuizDifficulty::Medium => "conceptual understanding, logical application, and diagnostic reasoning",
            QuizDifficulty::Hard => "advanced synthesis, subtle distinctions, and critical problem solving",
        };

        let type_rules = if is_qcm {
            "QCM (Multiple Choice): Each question MUST have MULTIPLE correct answers (2 or 3 correct answers). 'correct' field MUST be an array of indices like [0, 2]."
        } else {
            "QCS (Single Choice): Each question MUST have EXACTLY 1 correct answer (single best answer). 'correct' field MUST be an array of a single index like [1]."
        };

        let prompt = format!(
            "You are an expert educational and exam question author. Create {count} {diff_str} questions from the text below.\n\
            {type_rules}\n\
            \n\
            STRICT RULES:\n\
            1. Output ONLY a valid JSON object matching the schema below. No markdown fences outside the JSON, no preamble text.\n\
            2. Each question MUST have EXACTLY 5 options (A, B, C, D, E) — never 4, never 6.\n\
            3. The 'explanation' field format is MANDATORY — one line per option in this exact format:\n\
               \"A ✓ - reason why option A is correct\\nB ✗ - reason why option B is incorrect\\nC ✓ - reason...\\nD ✗ - reason...\\nE ✗ - reason...\"\n\
            4. If the text does not contain enough material for {count} questions, output as many as possible (minimum 3) and set \"insufficient_content\": true.\n\
            \n\
            Schema:\n\
            {{\n\
              \"questions\": [\n\
                {{\n\
                  \"question\": \"Question text here?\",\n\
                  \"options\": [\"Option A text\", \"Option B text\", \"Option C text\", \"Option D text\", \"Option E text\"],\n\
                  \"correct\": {}, \n\
                  \"explanation\": \"A ✓ - reason\\nB ✗ - reason\\nC ✗ - reason\\nD ✗ - reason\\nE ✗ - reason\",\n\
                  \"type\": \"{}\"\n\
                }}\n\
              ]\n\
            }}\n\n\
            Text excerpt:\n\
            {context}",
            if is_qcm { "[0, 2]" } else { "[0]" },
            if is_qcm { "qcm" } else { "qcs" }
        );

        let messages = vec![
            Message::system("You are an exam generator. Output ONLY a raw JSON object matching the requested schema."),
            Message::user(prompt),
        ];

        let model = self.selected_model.clone();
        let (rx, cancel) = pdfcraft_ai::stream_completion(pool.clone(), model, messages);
        self.quiz_receiver = Some(rx);
        self.quiz_cancel = Some(cancel);
    }
}

// ---------------------------------------------------------------------------
// Markdown & Table & Math Renderer
// ---------------------------------------------------------------------------

fn danger_color(t: &Tokens) -> Color32 {
    if t.dark() {
        Color32::from_rgb(0xF8, 0x71, 0x71)
    } else {
        Color32::from_rgb(0xDC, 0x26, 0x26)
    }
}

fn success_color(t: &Tokens) -> Color32 {
    if t.dark() {
        Color32::from_rgb(0x34, 0xD3, 0x99)
    } else {
        Color32::from_rgb(0x05, 0x96, 0x69)
    }
}

/// Renders rich formatted text: Markdown tables, math equations, code blocks,
/// headers, bold text, and bullet lists.
pub fn render_formatted_markdown(ui: &mut egui::Ui, text: &str, t: &Tokens) {
    let processed = preprocess_markdown_math(text);
    let lines: Vec<&str> = processed.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // 1. Math Display Block ($$)
        if line == "$$" {
            let mut math_content = String::new();
            i += 1;
            while i < lines.len() && lines[i].trim() != "$$" {
                math_content.push_str(lines[i]);
                math_content.push('\n');
                i += 1;
            }
            if i < lines.len() {
                i += 1; // skip closing $$
            }
            egui::Frame::new()
                .fill(t.hover)
                .stroke(Stroke::new(1.0, t.accent))
                .corner_radius(CornerRadius::same(6))
                .inner_margin(Margin::symmetric(10, 8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new(math_content.trim())
                                .font(theme::semibold(13.5))
                                .color(t.accent),
                        );
                    });
                });
            ui.add_space(4.0);
            continue;
        }

        // 2. Markdown Table Detection (consecutive lines with '|')
        if line.starts_with('|') && line.ends_with('|') {
            let mut table_lines = Vec::new();
            while i < lines.len() && lines[i].trim().starts_with('|') && lines[i].trim().ends_with('|') {
                table_lines.push(lines[i].trim());
                i += 1;
            }
            render_markdown_table(ui, &table_lines, t);
            ui.add_space(4.0);
            continue;
        }

        // 3. Code Block (```)
        if line.starts_with("```") {
            let mut code_content = String::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with("```") {
                code_content.push_str(lines[i]);
                code_content.push('\n');
                i += 1;
            }
            if i < lines.len() {
                i += 1; // skip closing ```
            }
            let code_to_copy = code_content.clone();
            egui::Frame::new()
                .fill(if t.dark() { Color32::from_rgb(18, 18, 22) } else { Color32::from_rgb(240, 240, 244) })
                .stroke(Stroke::new(1.0, t.divider))
                .corner_radius(CornerRadius::same(6))
                .inner_margin(Margin::same(8))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Code").small().color(t.text_faint));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui.small_button("📋 Copy").clicked() {
                                ui.ctx().copy_text(code_to_copy);
                            }
                        });
                    });
                    ui.label(
                        egui::RichText::new(code_content.trim())
                            .font(theme::regular(12.0))
                            .monospace()
                            .color(t.text),
                    );
                });
            ui.add_space(4.0);
            continue;
        }

        // 4. Headings
        if let Some(h) = line.strip_prefix("### ") {
            ui.label(egui::RichText::new(h).font(theme::semibold(14.0)).color(t.text));
            ui.add_space(2.0);
            i += 1;
            continue;
        }
        if let Some(h) = line.strip_prefix("## ") {
            ui.label(egui::RichText::new(h).font(theme::semibold(15.0)).color(t.accent));
            ui.add_space(2.0);
            i += 1;
            continue;
        }
        if let Some(h) = line.strip_prefix("# ") {
            ui.label(egui::RichText::new(h).font(theme::semibold(16.0)).color(t.accent));
            ui.add_space(4.0);
            i += 1;
            continue;
        }

        // 5. Bullet Lists
        if let Some(item) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")).or_else(|| line.strip_prefix("• ")) {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("•").color(t.accent).font(theme::semibold(13.0)));
                render_inline_spans(ui, item, t);
            });
            ui.add_space(1.0);
            i += 1;
            continue;
        }

        // 6. Numbered Lists
        if line.len() > 2 && line.chars().next().is_some_and(|c| c.is_ascii_digit()) && line.contains(". ") {
            if let Some(dot_idx) = line.find(". ") {
                let num_str = &line[..=dot_idx];
                let rest = &line[dot_idx + 2..];
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new(num_str).color(t.accent).font(theme::semibold(12.5)));
                    render_inline_spans(ui, rest, t);
                });
                ui.add_space(1.0);
                i += 1;
                continue;
            }
        }

        // 7. Regular paragraph
        if !line.is_empty() {
            ui.horizontal_wrapped(|ui| {
                render_inline_spans(ui, line, t);
            });
            ui.add_space(2.0);
        } else {
            ui.add_space(4.0);
        }

        i += 1;
    }
}

/// Renders a detected Markdown table as a visually clean, structured table.
fn render_markdown_table(ui: &mut egui::Ui, lines: &[&str], t: &Tokens) {
    if lines.is_empty() {
        return;
    }

    let mut rows: Vec<Vec<String>> = Vec::new();
    for l in lines {
        let trimmed = l.trim().trim_start_matches('|').trim_end_matches('|');
        if trimmed.chars().all(|c| c == '-' || c == '|' || c == ':' || c == ' ') {
            continue;
        }
        let cells: Vec<String> = trimmed.split('|').map(|c| c.trim().to_string()).collect();
        if !cells.is_empty() {
            rows.push(cells);
        }
    }

    if rows.is_empty() {
        return;
    }

    let table_raw = lines.join("\n");
    egui::Frame::new()
        .fill(t.hover)
        .stroke(Stroke::new(1.0, t.divider))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            // Header bar of the table with copy button
            ui.horizontal(|ui| {
                ui.add(icons::image("table", 13.0, t.accent));
                ui.label(egui::RichText::new("Table").small().font(theme::semibold(11.0)).color(t.accent));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.small_button("📋 Copy Table").clicked() {
                        ui.ctx().copy_text(table_raw);
                    }
                });
            });
            ui.add_space(2.0);

            // Table Grid
            egui::ScrollArea::horizontal().show(ui, |ui| {
                let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(1);
                egui::Grid::new(ui.id().with("ai_md_table"))
                    .striped(true)
                    .spacing(vec2(10.0, 6.0))
                    .min_col_width(70.0)
                    .show(ui, |ui| {
                        for (row_idx, row) in rows.iter().enumerate() {
                            let is_header = row_idx == 0;
                            for col_idx in 0..col_count {
                                let cell_text = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
                                if is_header {
                                    ui.label(
                                        egui::RichText::new(cell_text)
                                            .font(theme::semibold(12.5))
                                            .color(t.accent),
                                    );
                                } else {
                                    ui.label(
                                        egui::RichText::new(cell_text)
                                            .font(theme::regular(12.0))
                                            .color(t.text),
                                    );
                                }
                            }
                            ui.end_row();
                        }
                    });
            });
        });
}

/// Helper to render inline formatting like **bold** text and `inline code`.
fn render_inline_spans(ui: &mut egui::Ui, text: &str, t: &Tokens) {
    let mut rest = text;
    while !rest.is_empty() {
        if let Some(start) = rest.find("**") {
            if start > 0 {
                ui.label(egui::RichText::new(&rest[..start]).color(t.text));
            }
            let after_start = &rest[start + 2..];
            if let Some(end) = after_start.find("**") {
                let bold_text = &after_start[..end];
                ui.label(egui::RichText::new(bold_text).font(theme::semibold(12.5)).color(t.text));
                rest = &after_start[end + 2..];
            } else {
                ui.label(egui::RichText::new(&rest[start..]).color(t.text));
                break;
            }
        } else if let Some(code_start) = rest.find('`') {
            if code_start > 0 {
                ui.label(egui::RichText::new(&rest[..code_start]).color(t.text));
            }
            let after_code = &rest[code_start + 1..];
            if let Some(code_end) = after_code.find('`') {
                let code_str = &after_code[..code_end];
                egui::Frame::new()
                    .fill(t.hover)
                    .corner_radius(CornerRadius::same(4))
                    .inner_margin(Margin::symmetric(4, 1))
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new(code_str).monospace().small().color(t.accent));
                    });
                rest = &after_code[code_end + 1..];
            } else {
                ui.label(egui::RichText::new(&rest[code_start..]).color(t.text));
                break;
            }
        } else {
            ui.label(egui::RichText::new(rest).color(t.text));
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Document Text Extraction Helper
// ---------------------------------------------------------------------------

pub fn extract_document_text(app: &mut PdfCraftApp, start_page: usize, end_page: usize) -> Result<String, String> {
    let Some((index, _)) = app.active_ids() else {
        return Err("No active PDF document open".to_string());
    };
    let view = &mut app.views[index];
    let total_pages = view.page_count();

    if total_pages == 0 {
        return Err("Document is empty".to_string());
    }

    let s = start_page.max(1).min(total_pages);
    let e = end_page.max(s).min(total_pages);

    let mut combined = String::new();

    for p in s..=e {
        let page_index = p - 1;
        if let Some(text_layer) = view.texts.get(&page_index) {
            combined.push_str(&text_layer.plain_text());
            combined.push('\n');
        }
    }

    let clean = clean_and_sanitize_pdf_text(&combined);
    if clean.is_empty() {
        return Err("No readable text found on the selected pages".to_string());
    }

    // Cap context length at 35,000 characters to match books and avoid token overflow
    if clean.chars().count() > 35_000 {
        let truncated: String = clean.chars().take(35_000).collect();
        Ok(format!("{truncated}\n...[Document excerpt truncated for model context limit]"))
    } else {
        Ok(clean)
    }
}

// ---------------------------------------------------------------------------
// Right Side Panel UI (The core user experience matching books AIPanel)
// ---------------------------------------------------------------------------

/// Main entry point called by `panels::right_panel` when `RightPanel::Ai` is open.
pub fn render_right_panel(app: &mut PdfCraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());

    // 1. Navigation Tabs Header (Chat, Summary, Quiz) with Model Switcher & Stop Button
    ui.horizontal(|ui| {
        let tabs = [
            (AiTab::Chat, "message-square-text", "Chat (محادثة)"),
            (AiTab::Summary, "file-text", "Summary (ملخص)"),
            (AiTab::Quiz, "help-circle", "Quiz (اختبار)"),
        ];

        for (tab, icon, label) in tabs {
            let selected = app.ai.active_tab == tab;
            let (bg, stroke, text_color) = if selected {
                (t.accent_soft, Stroke::new(1.0, t.accent), t.accent)
            } else {
                (Color32::TRANSPARENT, Stroke::new(1.0, t.divider), t.text_muted)
            };

            let resp = egui::Frame::new()
                .fill(bg)
                .stroke(stroke)
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::symmetric(8, 5))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.add(icons::image(icon, 13.0, text_color));
                        ui.label(egui::RichText::new(label).font(theme::semibold(11.5)).color(text_color));
                    });
                })
                .response;

            if resp.interact(egui::Sense::click()).clicked() {
                app.ai.active_tab = tab;
            }
        }
    });

    ui.add_space(2.0);

    // Sub-header bar: Quick Model Selector + Stop Button if streaming
    ui.horizontal(|ui| {
        // Model Selector
        egui::ComboBox::from_id_salt("ai_model_quick_selector")
            .selected_text(
                pdfcraft_ai::AVAILABLE_MODELS
                    .iter()
                    .find(|(id, _)| *id == app.ai.selected_model)
                    .map(|(_, label)| *label)
                    .unwrap_or("Llama 3.3 70B"),
            )
            .width(ui.available_width() - if app.ai.is_streaming() { 70.0 } else { 0.0 })
            .show_ui(ui, |ui| {
                for (id, label) in pdfcraft_ai::AVAILABLE_MODELS {
                    let is_active = app.ai.selected_model == *id;
                    if ui.selectable_label(is_active, *label).clicked() {
                        app.ai.selected_model = (*id).to_string();
                    }
                }
            });

        // Prominent Stop Generation Button
        if app.ai.is_streaming() {
            let stop_btn = egui::Button::new(
                egui::RichText::new("⏹ Stop").small().font(theme::semibold(11.0)).color(Color32::WHITE),
            )
            .fill(danger_color(&t));

            if ui.add(stop_btn).on_hover_text("Stop generating immediately").clicked() {
                app.ai.stop_generation();
            }
        }
    });

    ui.add_space(2.0);
    ui.separator();
    ui.add_space(2.0);

    // Error banner if any
    let mut clear_error = false;
    if let Some(err) = &app.ai.error {
        let err_text = err.clone();
        egui::Frame::new()
            .fill(Color32::from_rgba_unmultiplied(239, 68, 68, 25))
            .stroke(Stroke::new(1.0, danger_color(&t)))
            .corner_radius(CornerRadius::same(6))
            .inner_margin(Margin::same(8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(icons::image("alert-circle", 14.0, danger_color(&t)));
                    ui.label(egui::RichText::new(&err_text).small().color(danger_color(&t)));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("✕").clicked() {
                            clear_error = true;
                        }
                    });
                });
            });
        ui.add_space(4.0);
    }
    if clear_error {
        app.ai.error = None;
    }

    // 2. Dispatch Active Tab
    match app.ai.active_tab {
        AiTab::Chat => render_chat_tab(ui, app, &t),
        AiTab::Summary => render_summary_tab(ui, app, &t),
        AiTab::Quiz => render_quiz_tab(ui, app, &t),
    }
}

// ---------------------------------------------------------------------------
// 1. Chat Tab
// ---------------------------------------------------------------------------

fn render_chat_tab(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    // Context Excerpt Card
    if !app.ai.context_text.is_empty() {
        egui::Frame::new()
            .fill(t.hover)
            .stroke(Stroke::new(1.0, t.divider))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.add(icons::image("file-text", 12.0, t.accent));
                    ui.label(egui::RichText::new("Selected PDF Excerpt").font(theme::semibold(11.0)).color(t.accent));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("✕").clicked() {
                            app.ai.context_text.clear();
                        }
                    });
                });
                let excerpt = if app.ai.context_text.chars().count() > 160 {
                    let mut s: String = app.ai.context_text.chars().take(160).collect();
                    s.push_str("…");
                    s
                } else {
                    app.ai.context_text.clone()
                };
                ui.label(egui::RichText::new(excerpt).small().color(t.text_muted).italics());
            });
        ui.add_space(2.0);
    }

    // Quick Action Chips (Explain, Summarize as Table, Key Takeaways, Quiz)
    if !app.ai.context_text.is_empty() && !app.ai.is_streaming() {
        ui.horizontal_wrapped(|ui| {
            let pool = app.groq_keys.clone();
            if ui.small_button("💡 Explain this").clicked() {
                app.ai.submit_prompt(&pool, "Explain this selected excerpt clearly and simply.");
            }
            if ui.small_button("📊 Summarize as Table").clicked() {
                app.ai.submit_prompt(&pool, "Summarize this excerpt as a clean Markdown comparison table.");
            }
            if ui.small_button("🔍 Key Takeaways").clicked() {
                app.ai.submit_prompt(&pool, "Extract the core key takeaways from this excerpt.");
            }
            if ui.small_button("❓ Quiz on this").clicked() {
                let ctx = app.ai.context_text.clone();
                app.ai.active_tab = AiTab::Quiz;
                app.ai.request_quiz(&pool, &ctx);
            }
        });
        ui.add_space(2.0);
    }

    // Scrollable Chat History
    let input_area_height = 80.0;
    let scroll_h = (ui.available_height() - input_area_height).max(100.0);

    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            if app.ai.history.is_empty() && app.ai.streaming.is_none() {
                ui.vertical_centered(|ui| {
                    ui.add_space(30.0);
                    ui.add(icons::image("sparkles", 32.0, t.text_faint));
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Ask any question about your PDF").font(theme::semibold(13.0)).color(t.text_muted));
                    ui.label(egui::RichText::new("Select text in the document or type below").small().color(t.text_faint));
                });
            }

            for msg in &app.ai.history {
                render_chat_bubble(ui, msg, t);
            }

            if let Some(stream_text) = &app.ai.streaming {
                let cleaned = strip_thinking_blocks(stream_text);
                let stream_msg = Message::assistant(cleaned);
                render_chat_bubble(ui, &stream_msg, t);
            }
        });

    ui.separator();

    // Input Bar (Fixed at bottom)
    ui.horizontal(|ui| {
        let is_loading = app.ai.is_streaming();
        let edit = egui::TextEdit::multiline(&mut app.ai.draft)
            .hint_text("Ask AI a question… (Enter to send)")
            .desired_rows(2)
            .desired_width(ui.available_width() - 40.0);

        let resp = ui.add_enabled(!is_loading, edit);

        // Submit on Enter without Shift
        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift) {
            let pool = app.groq_keys.clone();
            let draft = std::mem::take(&mut app.ai.draft);
            app.ai.submit_prompt(&pool, &draft);
        }

        let send_btn = egui::Button::new(egui::RichText::new("➤").font(theme::semibold(14.0)))
            .min_size(vec2(32.0, 32.0));

        if ui.add_enabled(!is_loading && !app.ai.draft.trim().is_empty(), send_btn).clicked() {
            let pool = app.groq_keys.clone();
            let draft = std::mem::take(&mut app.ai.draft);
            app.ai.submit_prompt(&pool, &draft);
        }
    });
}

fn render_chat_bubble(ui: &mut egui::Ui, msg: &Message, t: &Tokens) {
    let is_user = msg.role == "user";
    ui.add_space(3.0);

    if is_user {
        ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
            egui::Frame::new()
                .fill(t.accent)
                .corner_radius(CornerRadius::same(10))
                .inner_margin(Margin::symmetric(10, 6))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(&msg.content).font(theme::regular(12.5)).color(Color32::WHITE));
                });
        });
    } else {
        let copy_content = msg.content.clone();
        egui::Frame::new()
            .fill(t.hover)
            .stroke(Stroke::new(1.0, t.divider))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.add(icons::image("sparkles", 13.0, t.accent));
                    ui.label(egui::RichText::new("Assistant").small().font(theme::semibold(11.0)).color(t.accent));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("📋").on_hover_text("Copy message").clicked() {
                            ui.ctx().copy_text(copy_content);
                        }
                    });
                });
                ui.add_space(4.0);
                render_formatted_markdown(ui, &msg.content, t);
            });
    }
}

// ---------------------------------------------------------------------------
// 2. Summary Tab (Tables & Executive Summary)
// ---------------------------------------------------------------------------

fn render_summary_tab(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    let active_pages = app.active_ids().and_then(|(i, _)| app.views.get(i)).map_or(1, |v| v.page_count());

    // Page Range & Dual Mode Trigger Controls
    egui::Frame::new()
        .fill(t.hover)
        .stroke(Stroke::new(1.0, t.divider))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Pages:").font(theme::semibold(12.0)));
                ui.add(egui::DragValue::new(&mut app.ai.summary_from_page).range(1..=active_pages).prefix("From p. "));
                ui.add(egui::DragValue::new(&mut app.ai.summary_to_page).range(1..=active_pages).prefix("To p. "));

                if ui.small_button("Current Page").clicked() {
                    let cur = app.active_ids().and_then(|(i, _)| app.views.get(i)).map_or(1, |v| v.current + 1);
                    app.ai.summary_from_page = cur;
                    app.ai.summary_to_page = cur;
                }
            });

            ui.add_space(6.0);

            let is_loading = app.ai.is_summarizing;

            ui.horizontal(|ui| {
                // Button 1: Executive Summary
                let btn_sum = egui::Button::new(
                    egui::RichText::new("📄 Smart Summary").font(theme::semibold(12.0)).color(Color32::WHITE),
                )
                .fill(t.accent)
                .min_size(vec2((ui.available_width() - 8.0) / 2.0, 32.0));

                if ui.add_enabled(!is_loading, btn_sum).clicked() {
                    let s = app.ai.summary_from_page;
                    let e = app.ai.summary_to_page;
                    match extract_document_text(app, s, e) {
                        Ok(text) => {
                            let pool = app.groq_keys.clone();
                            app.ai.request_standard_summary(&pool, &text);
                        }
                        Err(err) => {
                            app.ai.error = Some(err);
                        }
                    }
                }

                // Button 2: Comparison Tables
                let btn_tbl = egui::Button::new(
                    egui::RichText::new("📊 Comparison Tables").font(theme::semibold(12.0)).color(Color32::WHITE),
                )
                .fill(t.accent)
                .min_size(vec2(ui.available_width(), 32.0));

                if ui.add_enabled(!is_loading, btn_tbl).clicked() {
                    let s = app.ai.summary_from_page;
                    let e = app.ai.summary_to_page;
                    match extract_document_text(app, s, e) {
                        Ok(text) => {
                            let pool = app.groq_keys.clone();
                            app.ai.request_summary_tables(&pool, &text);
                        }
                        Err(err) => {
                            app.ai.error = Some(err);
                        }
                    }
                }
            });
        });

    ui.add_space(4.0);

    // Summary Result Output with Copy & Clear actions
    if !app.ai.summary_result.is_empty() {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Summary & Synthesis").font(theme::semibold(12.5)));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.small_button("🗑 Clear").clicked() {
                    app.ai.summary_result.clear();
                }
                if ui.small_button("📋 Copy All").clicked() {
                    ui.ctx().copy_text(app.ai.summary_result.clone());
                }
            });
        });
        ui.add_space(2.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::Frame::new()
                .fill(t.hover)
                .stroke(Stroke::new(1.0, t.divider))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::same(10))
                .show(ui, |ui| {
                    render_formatted_markdown(ui, &app.ai.summary_result, t);
                });
        });
    }
}

// ---------------------------------------------------------------------------
// 3. Quiz Tab (AIQuiz.tsx model: Interactive Question-by-Question runner)
// ---------------------------------------------------------------------------

fn render_quiz_tab(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    // State 1: Generating Questions Loading
    if app.ai.is_generating_quiz {
        ui.vertical_centered(|ui| {
            ui.add_space(40.0);
            ui.add(icons::image("sparkles", 32.0, t.accent));
            ui.add_space(10.0);
            ui.label(egui::RichText::new("Generating Interactive Quiz...").font(theme::semibold(14.0)).color(t.accent));
            ui.label(egui::RichText::new("Analyzing document concepts & extracting rigorous exam questions").small().color(t.text_muted));
        });
        return;
    }

    // Notice for insufficient content if applicable
    if let Some(notice) = &app.ai.quiz_notice {
        egui::Frame::new()
            .fill(t.accent_soft)
            .stroke(Stroke::new(1.0, t.accent))
            .corner_radius(CornerRadius::same(6))
            .inner_margin(Margin::same(8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(icons::image("info", 14.0, t.accent));
                    ui.label(egui::RichText::new(notice).small().color(t.accent));
                });
            });
        ui.add_space(4.0);
    }

    // State 2: No Quiz Generated Yet -> Settings Card
    if app.ai.quiz_questions.is_empty() {
        render_quiz_setup(ui, app, t);
        return;
    }

    // State 3: Quiz in Review Mode
    if app.ai.quiz_show_review {
        render_quiz_review(ui, app, t);
        return;
    }

    // State 4: Quiz Finished -> Show Final Score Card
    if app.ai.quiz_show_results {
        render_quiz_score_card(ui, app, t);
        return;
    }

    // State 5: Interactive Question Runner (Question by Question)
    render_quiz_runner(ui, app, t);
}

fn render_quiz_setup(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    let active_pages = app.active_ids().and_then(|(i, _)| app.views.get(i)).map_or(1, |v| v.page_count());

    egui::Frame::new()
        .fill(t.hover)
        .stroke(Stroke::new(1.0, t.divider))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.label(egui::RichText::new("Generate Interactive Quiz (اختبار تفاعلي)").font(theme::semibold(13.5)).color(t.accent));
            ui.label(egui::RichText::new("Test your comprehension and retention from this PDF").small().color(t.text_muted));
            ui.add_space(6.0);

            // Question Type (QCS vs QCM)
            ui.horizontal(|ui| {
                ui.label("Format:");
                if ui.selectable_label(app.ai.quiz_kind == QuizKind::Qcs, "QCS (Single Choice)").clicked() {
                    app.ai.quiz_kind = QuizKind::Qcs;
                }
                if ui.selectable_label(app.ai.quiz_kind == QuizKind::Qcm, "QCM (Multiple Choices)").clicked() {
                    app.ai.quiz_kind = QuizKind::Qcm;
                }
            });
            ui.add_space(4.0);

            // Question Count
            ui.horizontal(|ui| {
                ui.label("Questions:");
                for count in [3, 5, 10] {
                    if ui.selectable_label(app.ai.quiz_count == count, format!("{count}")).clicked() {
                        app.ai.quiz_count = count;
                    }
                }
            });
            ui.add_space(4.0);

            // Difficulty Level
            ui.horizontal(|ui| {
                ui.label("Difficulty:");
                if ui.selectable_label(app.ai.quiz_difficulty == QuizDifficulty::Easy, "Easy").clicked() {
                    app.ai.quiz_difficulty = QuizDifficulty::Easy;
                }
                if ui.selectable_label(app.ai.quiz_difficulty == QuizDifficulty::Medium, "Medium").clicked() {
                    app.ai.quiz_difficulty = QuizDifficulty::Medium;
                }
                if ui.selectable_label(app.ai.quiz_difficulty == QuizDifficulty::Hard, "Hard").clicked() {
                    app.ai.quiz_difficulty = QuizDifficulty::Hard;
                }
            });
            ui.add_space(4.0);

            // Page Range
            ui.horizontal(|ui| {
                ui.label("From:");
                ui.add(egui::DragValue::new(&mut app.ai.quiz_from_page).range(1..=active_pages).prefix("p. "));
                ui.label("To:");
                ui.add(egui::DragValue::new(&mut app.ai.quiz_to_page).range(1..=active_pages).prefix("p. "));

                if ui.small_button("Current Page").clicked() {
                    let cur = app.active_ids().and_then(|(i, _)| app.views.get(i)).map_or(1, |v| v.current + 1);
                    app.ai.quiz_from_page = cur;
                    app.ai.quiz_to_page = cur;
                }
            });
            ui.add_space(8.0);

            let btn = egui::Button::new(egui::RichText::new("🎯 Generate Questions").font(theme::semibold(13.5)).color(Color32::WHITE))
                .fill(t.accent)
                .min_size(vec2(ui.available_width(), 34.0));

            if ui.add(btn).clicked() {
                let s = app.ai.quiz_from_page;
                let e = app.ai.quiz_to_page;
                match extract_document_text(app, s, e) {
                    Ok(text) => {
                        let pool = app.groq_keys.clone();
                        app.ai.request_quiz(&pool, &text);
                    }
                    Err(err) => {
                        app.ai.error = Some(err);
                    }
                }
            }
        });
}

fn render_quiz_runner(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    let q_idx = app.ai.quiz_current_index;
    let total = app.ai.quiz_questions.len();

    let Some(q) = app.ai.quiz_questions.get(q_idx).cloned() else {
        app.ai.quiz_show_results = true;
        return;
    };

    // Header: Question Counter & Type Badge
    ui.horizontal(|ui| {
        let badge = if app.ai.quiz_kind == QuizKind::Qcm { "QCM (Multiple)" } else { "QCS (Single)" };
        ui.label(egui::RichText::new(format!("Question {} of {}", q_idx + 1, total)).font(theme::semibold(14.0)).color(t.text));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.small_button("✕ Exit Quiz").clicked() {
                app.ai.quiz_questions.clear();
            }
            ui.label(egui::RichText::new(badge).small().color(t.accent).font(theme::semibold(11.0)));
        });
    });

    // Progress Bar
    let progress = (q_idx + 1) as f32 / total as f32;
    ui.add(egui::ProgressBar::new(progress).show_percentage());
    ui.add_space(6.0);

    // Question Prompt Box
    egui::Frame::new()
        .fill(t.hover)
        .stroke(Stroke::new(1.0, t.divider))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(egui::RichText::new(&q.question).font(theme::semibold(14.0)).color(t.text));
        });
    ui.add_space(6.0);

    // Option Cards (A, B, C, D, E)
    let is_confirmed = app.ai.quiz_is_confirmed;

    for (opt_idx, opt_text) in q.options.iter().enumerate() {
        let is_selected = app.ai.quiz_selected_options.contains(&opt_idx);
        let is_correct_ans = q.correct.contains(&opt_idx);

        let (bg, border, text_color) = if is_confirmed {
            if is_correct_ans {
                (Color32::from_rgba_unmultiplied(16, 185, 129, 35), Stroke::new(1.5, Color32::from_rgb(16, 185, 129)), Color32::from_rgb(16, 185, 129))
            } else if is_selected {
                (Color32::from_rgba_unmultiplied(239, 68, 68, 35), Stroke::new(1.5, Color32::from_rgb(239, 68, 68)), Color32::from_rgb(239, 68, 68))
            } else {
                (Color32::TRANSPARENT, Stroke::new(1.0, t.divider), t.text_muted)
            }
        } else if is_selected {
            (t.accent_soft, Stroke::new(1.5, t.accent), t.accent)
        } else {
            (Color32::TRANSPARENT, Stroke::new(1.0, t.divider), t.text)
        };

        let letter = match opt_idx {
            0 => "A",
            1 => "B",
            2 => "C",
            3 => "D",
            _ => "E",
        };

        let mark = if is_confirmed {
            if is_correct_ans { "  ✓" } else if is_selected { "  ✗" } else { "" }
        } else {
            ""
        };

        let resp = egui::Frame::new()
            .fill(bg)
            .stroke(border)
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("[{letter}]")).font(theme::semibold(12.5)).color(text_color));
                    ui.label(egui::RichText::new(format!("{opt_text}{mark}")).font(theme::regular(13.0)).color(t.text));
                });
            })
            .response;

        if !is_confirmed && resp.interact(egui::Sense::click()).clicked() {
            if app.ai.quiz_kind == QuizKind::Qcs {
                app.ai.quiz_selected_options.clear();
                app.ai.quiz_selected_options.push(opt_idx);
            } else if let Some(pos) = app.ai.quiz_selected_options.iter().position(|&x| x == opt_idx) {
                app.ai.quiz_selected_options.remove(pos);
            } else {
                app.ai.quiz_selected_options.push(opt_idx);
            }
        }
        ui.add_space(2.0);
    }

    ui.add_space(6.0);

    // Explanation Box (Shown after confirmation with structured A-E lines)
    if is_confirmed {
        egui::Frame::new()
            .fill(t.hover)
            .stroke(Stroke::new(1.0, t.accent))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::same(10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.add(icons::image("info", 14.0, t.accent));
                    ui.label(egui::RichText::new("Explanation / التوضيح").font(theme::semibold(12.0)).color(t.accent));
                });
                ui.add_space(4.0);
                render_structured_explanation(ui, &q.explanation, t);
            });
        ui.add_space(6.0);
    }

    // Action Button (Confirm -> Next Question / Results)
    if !is_confirmed {
        let has_selection = !app.ai.quiz_selected_options.is_empty();
        let btn = egui::Button::new(egui::RichText::new("Confirm Answer ✓").font(theme::semibold(13.5)).color(Color32::WHITE))
            .fill(t.accent)
            .min_size(vec2(ui.available_width(), 34.0));

        if ui.add_enabled(has_selection, btn).clicked() {
            app.ai.quiz_is_confirmed = true;
            if let Some(slot) = app.ai.quiz_user_answers.get_mut(q_idx) {
                *slot = app.ai.quiz_selected_options.clone();
            }
        }
    } else {
        let is_last = q_idx + 1 >= total;
        let next_label = if is_last { "Show Final Score 🏆" } else { "Next Question ➔" };
        let btn = egui::Button::new(egui::RichText::new(next_label).font(theme::semibold(13.5)).color(Color32::WHITE))
            .fill(success_color(t))
            .min_size(vec2(ui.available_width(), 34.0));

        if ui.add(btn).clicked() {
            if is_last {
                app.ai.quiz_show_results = true;
            } else {
                app.ai.quiz_current_index += 1;
                app.ai.quiz_selected_options.clear();
                app.ai.quiz_is_confirmed = false;
            }
        }
    }
}

/// Renders structured option explanations matching ExplanationLines in books AIQuiz.tsx.
fn render_structured_explanation(ui: &mut egui::Ui, explanation: &str, t: &Tokens) {
    if explanation.trim().is_empty() {
        ui.label(egui::RichText::new("Refer to the lesson text for detailed analysis.").small().color(t.text_muted));
        return;
    }

    let lines: Vec<&str> = explanation.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let has_structure = lines.iter().any(|l| {
        l.starts_with('A') || l.starts_with('B') || l.starts_with('C') || l.starts_with('D') || l.starts_with('E')
    });

    if !has_structure {
        render_formatted_markdown(ui, explanation, t);
        return;
    }

    for line in lines {
        let first_char = line.chars().next().unwrap_or(' ');
        let is_letter = matches!(first_char, 'A'..='E');

        if is_letter {
            let is_correct = line.contains('✓') || line.contains("correct") || line.contains("صحيح");
            let is_wrong = line.contains('✗') || line.contains('x') || line.contains("خطأ") || line.contains("incorrect");

            let badge_color = if is_correct {
                success_color(t)
            } else if is_wrong {
                danger_color(t)
            } else {
                t.accent
            };

            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(format!("[{first_char}]"))
                        .font(theme::semibold(12.0))
                        .color(badge_color),
                );
                let content = if line.len() > 2 {
                    line[2..].trim().trim_start_matches(['✓', '✗', '-', ':']).trim()
                } else {
                    line
                };
                ui.label(egui::RichText::new(content).font(theme::regular(12.0)).color(t.text));
            });
            ui.add_space(2.0);
        } else {
            ui.label(egui::RichText::new(line).font(theme::regular(12.0)).color(t.text_muted));
        }
    }
}

fn render_quiz_score_card(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    let total = app.ai.quiz_questions.len();
    let mut score = 0;

    for (idx, q) in app.ai.quiz_questions.iter().enumerate() {
        if let Some(user_ans) = app.ai.quiz_user_answers.get(idx) {
            let mut correct_sorted = q.correct.clone();
            correct_sorted.sort();
            let mut user_sorted = user_ans.clone();
            user_sorted.sort();
            if correct_sorted == user_sorted {
                score += 1;
            }
        }
    }

    let percentage = if total > 0 { (score * 100) / total } else { 0 };

    ui.vertical_centered(|ui| {
        ui.add_space(20.0);
        egui::Frame::new()
            .fill(t.hover)
            .stroke(Stroke::new(1.5, t.accent))
            .corner_radius(CornerRadius::same(12))
            .inner_margin(Margin::same(16))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(egui::RichText::new("🏆 Quiz Completed!").font(theme::semibold(18.0)).color(t.accent));
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!("{score} / {total} Correct ({percentage}%)"))
                        .font(theme::semibold(20.0))
                        .color(if percentage >= 70 { success_color(t) } else { t.text }),
                );
                ui.add_space(6.0);

                let verdict = if percentage >= 80 {
                    "🌟 Excellent mastery of document content!"
                } else if percentage >= 50 {
                    "👍 Good job! Review the explanations to reinforce key concepts."
                } else {
                    "📚 Needs revision. Review your mistakes below to solidify your understanding!"
                };
                ui.label(egui::RichText::new(verdict).color(t.text_muted));
            });

        ui.add_space(14.0);

        // Review Answers Button
        if ui.button(egui::RichText::new("📖 Review All Answers & Explanations").font(theme::semibold(13.5))).clicked() {
            app.ai.quiz_show_review = true;
        }

        ui.add_space(6.0);

        if ui.button(egui::RichText::new("🔄 Retake Quiz").font(theme::semibold(13.0))).clicked() {
            app.ai.quiz_current_index = 0;
            app.ai.quiz_selected_options.clear();
            app.ai.quiz_is_confirmed = false;
            app.ai.quiz_show_results = false;
            app.ai.quiz_show_review = false;
            app.ai.quiz_user_answers = vec![Vec::new(); total];
        }

        ui.add_space(4.0);

        if ui.button(egui::RichText::new("📝 Setup New Quiz").font(theme::semibold(13.0))).clicked() {
            app.ai.quiz_questions.clear();
            app.ai.quiz_show_results = false;
            app.ai.quiz_show_review = false;
        }
    });
}

fn render_quiz_review(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Detailed Review (مراجعة الإجابات)").font(theme::semibold(14.0)).color(t.text));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.small_button("← Back to Score").clicked() {
                app.ai.quiz_show_review = false;
            }
        });
    });

    ui.add_space(4.0);

    // Filters: All, Mistakes Only, Correct Only
    ui.horizontal(|ui| {
        if ui.selectable_label(app.ai.quiz_review_filter == QuizReviewFilter::All, "All Questions").clicked() {
            app.ai.quiz_review_filter = QuizReviewFilter::All;
        }
        if ui.selectable_label(app.ai.quiz_review_filter == QuizReviewFilter::MistakesOnly, "Mistakes Only ✗").clicked() {
            app.ai.quiz_review_filter = QuizReviewFilter::MistakesOnly;
        }
        if ui.selectable_label(app.ai.quiz_review_filter == QuizReviewFilter::CorrectOnly, "Correct Only ✓").clicked() {
            app.ai.quiz_review_filter = QuizReviewFilter::CorrectOnly;
        }
    });

    ui.add_space(6.0);
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.set_width(ui.available_width());

        for (idx, q) in app.ai.quiz_questions.iter().enumerate() {
            let user_ans = app.ai.quiz_user_answers.get(idx).cloned().unwrap_or_default();
            let mut correct_sorted = q.correct.clone();
            correct_sorted.sort();
            let mut user_sorted = user_ans.clone();
            user_sorted.sort();
            let is_correct = correct_sorted == user_sorted;

            if app.ai.quiz_review_filter == QuizReviewFilter::MistakesOnly && is_correct {
                continue;
            }
            if app.ai.quiz_review_filter == QuizReviewFilter::CorrectOnly && !is_correct {
                continue;
            }

            let card_border = if is_correct { success_color(t) } else { danger_color(t) };
            let card_mark = if is_correct { "✓ Correct" } else { "✗ Incorrect" };

            egui::Frame::new()
                .fill(t.hover)
                .stroke(Stroke::new(1.0, card_border))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(Margin::same(10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("Q{}: {}", idx + 1, q.question)).font(theme::semibold(13.0)).color(t.text));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label(egui::RichText::new(card_mark).small().font(theme::semibold(11.0)).color(card_border));
                        });
                    });

                    ui.add_space(4.0);

                    for (opt_idx, opt_text) in q.options.iter().enumerate() {
                        let is_ans = q.correct.contains(&opt_idx);
                        let was_picked = user_ans.contains(&opt_idx);

                        let opt_color = if is_ans {
                            success_color(t)
                        } else if was_picked {
                            danger_color(t)
                        } else {
                            t.text_muted
                        };

                        let mark = if is_ans && was_picked {
                            " (Your correct choice ✓)"
                        } else if is_ans {
                            " (Correct answer ✓)"
                        } else if was_picked {
                            " (Your wrong choice ✗)"
                        } else {
                            ""
                        };

                        let letter = match opt_idx {
                            0 => "A",
                            1 => "B",
                            2 => "C",
                            3 => "D",
                            _ => "E",
                        };

                        ui.label(egui::RichText::new(format!("[{letter}] {opt_text}{mark}")).small().color(opt_color));
                    }

                    if !q.explanation.is_empty() {
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(2.0);
                        render_structured_explanation(ui, &q.explanation, t);
                    }
                });

            ui.add_space(6.0);
        }
    });
}

// ---------------------------------------------------------------------------
// Preferences Section & Ask AI Button Component
// ---------------------------------------------------------------------------

pub(crate) fn ask_ai_button(ui: &mut egui::Ui) -> Response {
    let t = Tokens::get(ui.ctx());
    ui.add(
        egui::Button::image_and_text(icons::image("sparkles", 13.0, t.accent), "Ask AI")
            .fill(t.accent_soft)
            .stroke(Stroke::new(1.0, t.accent)),
    )
}

pub(crate) fn preferences_section(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    ui.collapsing(egui::RichText::new("✨ AI Assistant (Groq)").font(theme::semibold(14.0)), |ui| {
        ui.add_space(4.0);

        egui::Frame::new()
            .fill(t.hover)
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::same(12))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());

                // Model Selection Dropdown
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Default Model:").font(theme::semibold(12.5)));
                });
                ui.add_space(2.0);
                for (id, label) in pdfcraft_ai::AVAILABLE_MODELS {
                    let is_active = app.ai.selected_model == *id;
                    if ui.selectable_label(is_active, *label).clicked() {
                        app.ai.selected_model = (*id).to_string();
                    }
                }

                ui.add_space(8.0);
                ui.label(egui::RichText::new("Groq API Keys (one key per line for automatic round-robin rotation & 60s 429 cooldown):").small());
                ui.add_space(2.0);

                let edit = egui::TextEdit::multiline(&mut app.ai.keys_input)
                    .hint_text("gsk_...\ngsk_...")
                    .desired_rows(3)
                    .desired_width(ui.available_width());

                if ui.add(edit).changed() {
                    let keys: Vec<String> = app.ai.keys_input
                        .lines()
                        .map(|l| l.trim().to_string())
                        .filter(|l| !l.is_empty())
                        .collect();
                    app.groq_keys.set_keys(keys);
                }

                ui.add_space(6.0);

                let key_count = app.groq_keys.count();
                ui.horizontal(|ui| {
                    ui.add(icons::image("shield-check", 14.0, t.accent));
                    ui.add_space(4.0);
                    if key_count == 0 {
                        ui.label(egui::RichText::new("No Groq API keys configured").color(danger_color(t)));
                    } else {
                        ui.label(
                            egui::RichText::new(format!(
                                "{key_count} active API key{} · Round-robin & 429 failover active",
                                if key_count > 1 { "s" } else { "" }
                            ))
                            .color(t.accent_text),
                        );
                    }
                });
            });
    });
}

pub(crate) fn show(ui_ctx: &egui::Context, app: &mut PdfCraftApp) {
    app.ai.poll();
    if app.ai.is_streaming() {
        ui_ctx.request_repaint();
    }
}
