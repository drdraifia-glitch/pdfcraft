//! AI assistant panel, document summarization, interactive QCMs/quiz generator,
//! and Groq preferences for PdfCraft.
//!
//! Provides:
//! - Floating multi-tab assistant window (Chat, Summary, QCM / Quiz).
//! - 💬 **Chat Tab:** Context-aware conversations with selected PDF excerpts.
//! - 📄 **Summary Tab:** Full document or page-range executive summaries and notes.
//! - 🎯 **QCM / Quiz Tab:** Interactive clinical/academic QCM generator and grader
//!   with custom question count, difficulty levels, instantaneous scoring, and
//!   per-option explanations.
//! - ⚙ **Preferences Section:** Multiple Groq API keys with round-robin rotation,
//!   failover, and model selection (GPT-OSS 120B, GPT-OSS 20B, Llama 3.3, Qwen 3.8).

use std::sync::mpsc;
use std::sync::Arc;

use egui::{Color32, CornerRadius, Margin, Pos2, Response, Stroke, Vec2};
use pdfcraft_ai::{AiEvent, KeyPool, Message};
use serde::{Deserialize, Serialize};

use crate::theme::{self, Tokens};
use crate::{PdfCraftApp, icons, widgets};

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuizQuestion {
    pub question: String,
    pub options: Vec<String>,
    #[serde(deserialize_with = "deserialize_correct_indices")]
    pub correct: Vec<usize>,
    #[serde(default)]
    pub explanation: String,
}

fn deserialize_correct_indices<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Indices {
        Single(usize),
        Multiple(Vec<usize>),
    }
    match Indices::deserialize(deserializer)? {
        Indices::Single(idx) => Ok(vec![idx]),
        Indices::Multiple(v) => Ok(v),
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

    // Summary state
    pub summary_from_page: usize,
    pub summary_to_page: usize,
    pub summary_style: usize, // 0 = Bullet Points, 1 = Executive, 2 = In-depth Study Notes
    pub summary_result: String,
    pub is_summarizing: bool,
    summary_receiver: Option<mpsc::Receiver<AiEvent>>,

    // Quiz state
    pub quiz_from_page: usize,
    pub quiz_to_page: usize,
    pub quiz_count: usize,
    pub quiz_kind: QuizKind,
    pub quiz_difficulty: QuizDifficulty,
    pub quiz_questions: Vec<QuizQuestion>,
    pub quiz_selected_answers: Vec<Vec<usize>>,
    pub quiz_checked: bool,
    pub is_generating_quiz: bool,
    quiz_receiver: Option<mpsc::Receiver<AiEvent>>,
    quiz_buffer: String,
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
            summary_from_page: 1,
            summary_to_page: 1,
            summary_style: 1,
            summary_result: String::new(),
            is_summarizing: false,
            summary_receiver: None,
            quiz_from_page: 1,
            quiz_to_page: 1,
            quiz_count: 5,
            quiz_kind: QuizKind::Qcs,
            quiz_difficulty: QuizDifficulty::Medium,
            quiz_questions: Vec::new(),
            quiz_selected_answers: Vec::new(),
            quiz_checked: false,
            is_generating_quiz: false,
            quiz_receiver: None,
            quiz_buffer: String::new(),
        }
    }
}

impl AiState {
    /// Open the assistant panel with the given excerpt as context.
    pub fn open_with_context(&mut self, text: String) {
        self.open = true;
        self.active_tab = AiTab::Chat;
        if !text.trim().is_empty() {
            self.context_text = text.trim().to_string();
        }
    }

    /// Whether any streaming operation is in progress.
    pub fn is_streaming(&self) -> bool {
        self.receiver.is_some() || self.summary_receiver.is_some() || self.quiz_receiver.is_some()
    }

    /// Poll stream events from worker threads; call every frame.
    pub fn poll(&mut self) {
        // Poll chat stream
        if let Some(rx) = &self.receiver {
            loop {
                match rx.try_recv() {
                    Ok(AiEvent::Token(t)) => {
                        self.streaming.get_or_insert_with(String::new).push_str(&t);
                    }
                    Ok(AiEvent::Done) => {
                        if let Some(text) = self.streaming.take() {
                            if !text.is_empty() {
                                self.history.push(Message::assistant(text));
                            }
                        }
                        self.receiver = None;
                        break;
                    }
                    Ok(AiEvent::Error(err)) => {
                        self.error = Some(err);
                        self.streaming = None;
                        self.receiver = None;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if let Some(text) = self.streaming.take() {
                            if !text.is_empty() {
                                self.history.push(Message::assistant(text));
                            }
                        }
                        self.receiver = None;
                        break;
                    }
                }
            }
        }

        // Poll summary stream
        if let Some(rx) = &self.summary_receiver {
            loop {
                match rx.try_recv() {
                    Ok(AiEvent::Token(t)) => {
                        self.summary_result.push_str(&t);
                    }
                    Ok(AiEvent::Done) => {
                        self.is_summarizing = false;
                        self.summary_receiver = None;
                        break;
                    }
                    Ok(AiEvent::Error(err)) => {
                        self.error = Some(err);
                        self.is_summarizing = false;
                        self.summary_receiver = None;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.is_summarizing = false;
                        self.summary_receiver = None;
                        break;
                    }
                }
            }
        }

        // Poll quiz generation stream
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
                        break;
                    }
                    Ok(AiEvent::Error(err)) => {
                        self.error = Some(err);
                        self.is_generating_quiz = false;
                        self.quiz_receiver = None;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.parse_and_set_quiz();
                        self.is_generating_quiz = false;
                        self.quiz_receiver = None;
                        break;
                    }
                }
            }
        }
    }

    fn parse_and_set_quiz(&mut self) {
        let raw = self.quiz_buffer.trim();
        let cleaned = if let Some(start) = raw.find('[') {
            if let Some(end) = raw.rfind(']') {
                &raw[start..=end]
            } else {
                raw
            }
        } else {
            raw
        };

        match serde_json::from_str::<Vec<QuizQuestion>>(cleaned) {
            Ok(questions) => {
                let count = questions.len();
                self.quiz_questions = questions;
                self.quiz_selected_answers = vec![Vec::new(); count];
                self.quiz_checked = false;
                self.error = None;
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

        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.streaming = None;

        let pool_clone = pool.clone();
        let model = self.selected_model.clone();

        std::thread::spawn(move || {
            pdfcraft_ai::stream_chat(pool_clone, model, messages, tx);
        });
    }

    /// Submit the draft in the chat tab.
    pub fn submit_draft(&mut self, pool: &KeyPool) {
        let text = std::mem::take(&mut self.draft);
        self.submit_prompt(pool, &text);
    }

    /// Start document summarization for the extracted text.
    pub fn start_summary(&mut self, pool: &KeyPool, document_text: &str) {
        if document_text.trim().is_empty() || self.is_streaming() {
            return;
        }
        self.error = None;
        self.summary_result.clear();
        self.is_summarizing = true;

        let style_desc = match self.summary_style {
            0 => "concise bullet points highlighting key facts and outcomes",
            2 => "in-depth analytical study notes with deep breakdowns of each topic",
            _ => "a well-rounded executive summary with overview, key takeaways, and conclusions",
        };

        let sys_prompt = Message::system(
            "You are an expert document summarizer and analyst embedded in PdfCraft.\
             Summarize the provided PDF excerpt thoroughly, accurately, and clearly.\
             Use structured markdown headers (##), bullet points, and bold text.\
             Respond in the language of the document (Arabic, English, or French).",
        );

        let user_prompt = Message::user(format!(
            "Please provide {style_desc} for the following text from the PDF:\n\n\"\"\"\n{}\n\"\"\"",
            document_text.trim()
        ));

        let (tx, rx) = mpsc::channel();
        self.summary_receiver = Some(rx);

        let pool_clone = pool.clone();
        let model = self.selected_model.clone();

        std::thread::spawn(move || {
            pdfcraft_ai::stream_chat(pool_clone, model, vec![sys_prompt, user_prompt], tx);
        });
    }

    /// Start generating QCMs/Quiz based on the document text.
    pub fn start_quiz_generation(&mut self, pool: &KeyPool, document_text: &str) {
        if document_text.trim().is_empty() || self.is_streaming() {
            return;
        }
        self.error = None;
        self.quiz_buffer.clear();
        self.is_generating_quiz = true;

        let kind_str = match self.quiz_kind {
            QuizKind::Qcs => "QCS (Single Correct Answer per question)",
            QuizKind::Qcm => "QCM (Multiple Correct Answers per question)",
        };

        let diff_str = match self.quiz_difficulty {
            QuizDifficulty::Easy => "Easy (direct recall, definitions, key facts)",
            QuizDifficulty::Medium => "Medium (comprehension, application, concept linkage)",
            QuizDifficulty::Hard => "Hard (clinical cases, detailed discrimination between close options)",
        };

        let count = self.quiz_count.clamp(1, 15);

        let sys_prompt = Message::system(
            "You are an academic and clinical examination expert.\
             Generate rigorous multiple-choice questions based STRICTLY on the provided text.\
             Return ONLY a raw JSON array of objects without Markdown formatting or backticks.\
             Each object MUST have:\
             - 'question': the question text\
             - 'options': an array of exactly 5 options (strings)\
             - 'correct': array of integers (0-based indices of the correct options)\
             - 'explanation': detailed rationale for why each option is correct or incorrect (e.g. 'A ✓ - ...\\nB ✗ - ...')\
             Respond in the language of the document (Arabic, English, or French).",
        );

        let user_prompt = Message::user(format!(
            "Generate {count} {kind_str} questions with difficulty '{diff_str}' from this text:\n\n\"\"\"\n{}\n\"\"\"",
            document_text.trim()
        ));

        let (tx, rx) = mpsc::channel();
        self.quiz_receiver = Some(rx);

        let pool_clone = pool.clone();
        let model = self.selected_model.clone();

        std::thread::spawn(move || {
            pdfcraft_ai::stream_chat(pool_clone, model, vec![sys_prompt, user_prompt], tx);
        });
    }

    /// Clear chat history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.streaming = None;
        self.receiver = None;
        self.error = None;
        self.draft.clear();
    }

    pub fn clear_summary(&mut self) {
        self.summary_result.clear();
        self.is_summarizing = false;
        self.summary_receiver = None;
    }

    pub fn clear_quiz(&mut self) {
        self.quiz_questions.clear();
        self.quiz_selected_answers.clear();
        self.quiz_checked = false;
        self.is_generating_quiz = false;
        self.quiz_receiver = None;
    }
}

// ---------------------------------------------------------------------------
// Text Extraction Helper from Active Document
// ---------------------------------------------------------------------------

pub fn extract_document_text(app: &PdfCraftApp, from_page_1based: usize, to_page_1based: usize) -> String {
    let Some(view_idx) = app.active else { return String::new() };
    let view = &app.views[view_idx];
    let doc_id = view.id;
    let Some(doc) = app.session.get(doc_id) else { return String::new() };
    let total_pages = doc.info.pages.len();
    if total_pages == 0 {
        return String::new();
    }

    let from = from_page_1based.clamp(1, total_pages).saturating_sub(1);
    let to = to_page_1based.clamp(1, total_pages).max(from_page_1based).saturating_sub(1);

    let mut full_text = String::new();
    let config = pdfcraft_render::RenderConfig {
        password: doc.password.as_deref().map(Arc::from),
        ..Default::default()
    };
    let mut renderer = pdfcraft_render::PageRenderer::new(doc.bytes.clone(), config);

    for p in from..=to {
        let page_content = if let Some(t) = view.texts.get(&p) {
            t.plain_text()
        } else {
            let req = pdfcraft_render::RenderRequest {
                page: p,
                kind: pdfcraft_render::RequestKind::Text,
                tile: None,
                scale: 1.0,
                tag: 0,
                preview: false,
            };
            let out = renderer.render(req);
            out.text.map(|t| t.plain_text()).unwrap_or_default()
        };

        let trimmed = page_content.trim();
        if !trimmed.is_empty() {
            if !full_text.is_empty() {
                full_text.push_str("\n\n");
            }
            full_text.push_str(&format!("--- Page {} ---\n", p + 1));
            full_text.push_str(trimmed);
        }

        // Cap extraction at 35,000 characters to stay within Groq context bounds
        if full_text.len() >= 35_000 {
            break;
        }
    }

    full_text
}

// ---------------------------------------------------------------------------
// Floating AI Assistant Window
// ---------------------------------------------------------------------------

pub(crate) fn show(ui_ctx: &egui::Context, app: &mut PdfCraftApp) {
    if !app.ai.open {
        return;
    }

    app.ai.poll();
    if app.ai.is_streaming() {
        ui_ctx.request_repaint();
    }

    let t = Tokens::get(ui_ctx);

    let mut is_open = app.ai.open;
    egui::Window::new("Ask AI")
        .id(egui::Id::new("pdfcraft-ai-floating-window"))
        .default_pos(Pos2::new(100.0, 100.0))
        .default_size(Vec2::new(450.0, 580.0))
        .min_width(340.0)
        .min_height(340.0)
        .collapsible(false)
        .open(&mut is_open)
        .show(ui_ctx, |ui| {
            render_window_body(ui, app, &t);
        });
    app.ai.open = is_open;
}

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

fn render_window_body(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    // Header & Tab Bar
    ui.horizontal(|ui| {
        ui.add(icons::image("sparkles", 16.0, t.accent));
        ui.add_space(4.0);
        ui.label(egui::RichText::new("PdfCraft AI").font(theme::semibold(15.0)));

        let key_count = app.groq_keys.count();
        let badge_text = if key_count == 0 {
            "No keys (Add in Settings)".to_string()
        } else {
            format!("{key_count} key{}", if key_count > 1 { "s" } else { "" })
        };
        ui.label(egui::RichText::new(badge_text).small().color(if key_count == 0 { danger_color(t) } else { t.text_muted }));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icons::button(ui, "trash-2", 24.0, false, "Reset current tab").clicked() {
                match app.ai.active_tab {
                    AiTab::Chat => app.ai.clear(),
                    AiTab::Summary => app.ai.clear_summary(),
                    AiTab::Quiz => app.ai.clear_quiz(),
                }
            }
        });
    });

    ui.add_space(4.0);

    // Navigation Tabs
    ui.horizontal(|ui| {
        for (tab, label) in [
            (AiTab::Chat, "💬 Chat"),
            (AiTab::Summary, "📄 Summary"),
            (AiTab::Quiz, "🎯 QCM / Quiz"),
        ] {
            let is_sel = app.ai.active_tab == tab;
            if widgets::mode_tab(ui, label, is_sel).clicked() {
                app.ai.active_tab = tab;
            }
        }
    });

    ui.separator();

    // Body based on active tab
    match app.ai.active_tab {
        AiTab::Chat => render_chat_tab(ui, app, t),
        AiTab::Summary => render_summary_tab(ui, app, t),
        AiTab::Quiz => render_quiz_tab(ui, app, t),
    }
}

// ---------------------------------------------------------------------------
// TAB 1: Chat
// ---------------------------------------------------------------------------

fn render_chat_tab(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    // Context preview card (if text is selected)
    if !app.ai.context_text.is_empty() {
        egui::Frame::new()
            .fill(t.hover)
            .corner_radius(CornerRadius::same(6))
            .inner_margin(Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.add(icons::image("file-text", 12.0, t.accent));
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new("PDF Context Excerpt").small().font(theme::semibold(11.0)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("✕").clicked() {
                            app.ai.context_text.clear();
                        }
                    });
                });
                let excerpt = if app.ai.context_text.chars().count() > 180 {
                    let mut s: String = app.ai.context_text.chars().take(180).collect();
                    s.push('…');
                    s
                } else {
                    app.ai.context_text.clone()
                };
                ui.label(egui::RichText::new(excerpt).small().color(t.text_muted).italics());
            });
        ui.add_space(4.0);
    }

    // Quick action pills
    if !app.ai.context_text.is_empty() && !app.ai.is_streaming() {
        ui.horizontal(|ui| {
            let pool = app.groq_keys.clone();
            if ui.small_button("💡 Explain this").clicked() {
                app.ai.submit_prompt(&pool, "Explain this selected excerpt clearly and simply.");
            }
            if ui.small_button("📝 Summarize").clicked() {
                app.ai.submit_prompt(&pool, "Summarize the key points of this excerpt.");
            }
            if ui.small_button("🔍 Key takeaways").clicked() {
                app.ai.submit_prompt(&pool, "Extract the main insights and takeaways from this excerpt.");
            }
        });
        ui.add_space(4.0);
    }

    // Chat history area
    let bottom_height = 80.0;
    let scroll_height = (ui.available_height() - bottom_height).max(100.0);
    egui::ScrollArea::vertical()
        .max_height(scroll_height)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            if app.ai.history.is_empty() && app.ai.streaming.is_none() {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.add(icons::image("sparkles", 28.0, t.text_muted));
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Select any text in the PDF and click 'Ask AI'").color(t.text_muted).italics());
                    ui.label(egui::RichText::new("or type a question below to analyze your document.").small().color(t.text_muted));
                });
            }

            for msg in &app.ai.history {
                render_bubble(ui, msg, t);
            }

            // Real-time streaming bubble
            if let Some(partial) = &app.ai.streaming {
                let streaming_msg = Message::assistant(partial.clone());
                render_bubble(ui, &streaming_msg, t);
                let cursor = if (ui.input(|i| i.time) * 2.0) as i64 % 2 == 0 { "▋" } else { " " };
                ui.label(egui::RichText::new(cursor).color(t.accent));
            }
        });

    // Error banner
    if let Some(err) = &app.ai.error.clone() {
        render_error_banner(ui, err);
    }

    ui.separator();

    // Input row
    ui.horizontal(|ui| {
        let input_id = egui::Id::new("ai-draft-input-field");
        let available_w = ui.available_width() - 40.0;
        let resp = ui.add_sized(
            Vec2::new(available_w, 30.0),
            egui::TextEdit::singleline(&mut app.ai.draft)
                .id(input_id)
                .hint_text("Ask a question about the PDF…")
                .desired_width(available_w),
        );

        let is_streaming = app.ai.is_streaming();
        let should_send = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

        if should_send && !is_streaming {
            let pool = app.groq_keys.clone();
            app.ai.submit_draft(&pool);
            ui.memory_mut(|m| m.request_focus(input_id));
        }

        let send_btn = icons::button(ui, if is_streaming { "clock" } else { "send" }, 30.0, false, "Send");
        if send_btn.clicked() && !is_streaming {
            let pool = app.groq_keys.clone();
            app.ai.submit_draft(&pool);
            ui.memory_mut(|m| m.request_focus(input_id));
        }
    });

    ui.add_space(2.0);
    render_footer(ui, app, t);
}

// ---------------------------------------------------------------------------
// TAB 2: Document Summary
// ---------------------------------------------------------------------------

fn render_summary_tab(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    let total_pages = app
        .active
        .and_then(|i| app.session.get(app.views[i].id))
        .map_or(1, |d| d.info.pages.len());
    let current_page = app.active.map_or(1, |i| app.views[i].current + 1);

    // Page Range & Settings Header
    egui::Frame::new()
        .fill(t.hover)
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Pages:").font(theme::semibold(12.0)));
                ui.add(egui::DragValue::new(&mut app.ai.summary_from_page).range(1..=total_pages).prefix("p. "));
                ui.label("to");
                ui.add(egui::DragValue::new(&mut app.ai.summary_to_page).range(1..=total_pages).prefix("p. "));

                if ui.small_button("Current").clicked() {
                    app.ai.summary_from_page = current_page;
                    app.ai.summary_to_page = current_page;
                }
                if ui.small_button("All Pages").clicked() {
                    app.ai.summary_from_page = 1;
                    app.ai.summary_to_page = total_pages;
                }
            });

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Style:").font(theme::semibold(12.0)));
                for (style_idx, label) in [(0, "Bullets"), (1, "Executive"), (2, "Study Notes")] {
                    if ui.selectable_label(app.ai.summary_style == style_idx, label).clicked() {
                        app.ai.summary_style = style_idx;
                    }
                }
            });

            ui.add_space(6.0);
            let is_running = app.ai.is_summarizing;
            let btn_label = if is_running { "⏳ Generating Summary…" } else { "⚡ Generate Summary" };

            ui.horizontal(|ui| {
                let pool = app.groq_keys.clone();
                let from = app.ai.summary_from_page;
                let to = app.ai.summary_to_page;

                if ui.add_enabled(!is_running, egui::Button::new(btn_label)).clicked() {
                    let doc_text = extract_document_text(app, from, to);
                    app.ai.start_summary(&pool, &doc_text);
                }

                if !app.ai.summary_result.is_empty() {
                    if ui.button("📋 Copy").clicked() {
                        ui.ctx().copy_text(app.ai.summary_result.clone());
                    }
                    if ui.button("🗑 Clear").clicked() {
                        app.ai.summary_result.clear();
                    }
                }
            });
        });

    ui.add_space(6.0);

    // Summary Text Output Area
    let bottom_h = 30.0;
    let scroll_h = (ui.available_height() - bottom_h).max(120.0);
    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            if app.ai.summary_result.is_empty() && !app.ai.is_summarizing {
                ui.vertical_centered(|ui| {
                    ui.add_space(40.0);
                    ui.add(icons::image("file-text", 28.0, t.text_muted));
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Choose a page range above and click 'Generate Summary'").color(t.text_muted).italics());
                });
            } else {
                egui::Frame::new()
                    .fill(t.hover)
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::same(12))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(egui::RichText::new(&app.ai.summary_result).color(t.text));
                        if app.ai.is_summarizing {
                            let cursor = if (ui.input(|i| i.time) * 2.0) as i64 % 2 == 0 { "▋" } else { " " };
                            ui.label(egui::RichText::new(cursor).color(t.accent));
                        }
                    });
            }
        });

    if let Some(err) = &app.ai.error.clone() {
        render_error_banner(ui, err);
    }

    render_footer(ui, app, t);
}

// ---------------------------------------------------------------------------
// TAB 3: Interactive QCMs / Quiz
// ---------------------------------------------------------------------------

fn render_quiz_tab(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    let total_pages = app
        .active
        .and_then(|i| app.session.get(app.views[i].id))
        .map_or(1, |d| d.info.pages.len());
    let current_page = app.active.map_or(1, |i| app.views[i].current + 1);

    // If no questions generated yet, show setup screen
    if app.ai.quiz_questions.is_empty() {
        egui::Frame::new()
            .fill(t.hover)
            .corner_radius(CornerRadius::same(6))
            .inner_margin(Margin::same(12))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(egui::RichText::new("Configure QCM / Quiz Generator").font(theme::semibold(13.0)));
                ui.add_space(6.0);

                // Scope
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Pages:").font(theme::semibold(12.0)));
                    ui.add(egui::DragValue::new(&mut app.ai.quiz_from_page).range(1..=total_pages).prefix("p. "));
                    ui.label("to");
                    ui.add(egui::DragValue::new(&mut app.ai.quiz_to_page).range(1..=total_pages).prefix("p. "));
                    if ui.small_button("Current").clicked() {
                        app.ai.quiz_from_page = current_page;
                        app.ai.quiz_to_page = current_page;
                    }
                    if ui.small_button("All").clicked() {
                        app.ai.quiz_from_page = 1;
                        app.ai.quiz_to_page = total_pages;
                    }
                });

                ui.add_space(6.0);

                // Type: QCS vs QCM
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Type:").font(theme::semibold(12.0)));
                    if ui.selectable_label(app.ai.quiz_kind == QuizKind::Qcs, "QCS (Single Choice)").clicked() {
                        app.ai.quiz_kind = QuizKind::Qcs;
                    }
                    if ui.selectable_label(app.ai.quiz_kind == QuizKind::Qcm, "QCM (Multiple Choices)").clicked() {
                        app.ai.quiz_kind = QuizKind::Qcm;
                    }
                });

                ui.add_space(6.0);

                // Difficulty
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Difficulty:").font(theme::semibold(12.0)));
                    for (diff, label) in [
                        (QuizDifficulty::Easy, "Easy (سهل)"),
                        (QuizDifficulty::Medium, "Medium (متوسط)"),
                        (QuizDifficulty::Hard, "Hard (صعب)"),
                    ] {
                        if ui.selectable_label(app.ai.quiz_difficulty == diff, label).clicked() {
                            app.ai.quiz_difficulty = diff;
                        }
                    }
                });

                ui.add_space(6.0);

                // Question count
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Questions:").font(theme::semibold(12.0)));
                    for count in [3, 5, 10] {
                        if ui.selectable_label(app.ai.quiz_count == count, format!("{count}")).clicked() {
                            app.ai.quiz_count = count;
                        }
                    }
                });

                ui.add_space(10.0);

                let is_running = app.ai.is_generating_quiz;
                let btn_text = if is_running { "⏳ Generating Questions with Groq…" } else { "🎯 Generate QCM / Quiz" };

                let pool = app.groq_keys.clone();
                let from = app.ai.quiz_from_page;
                let to = app.ai.quiz_to_page;

                if ui.add_enabled(!is_running, egui::Button::new(btn_text)).clicked() {
                    let doc_text = extract_document_text(app, from, to);
                    app.ai.start_quiz_generation(&pool, &doc_text);
                }
            });

        if let Some(err) = &app.ai.error.clone() {
            render_error_banner(ui, err);
        }

        render_footer(ui, app, t);
        return;
    }

    // Interactive Quiz Session View
    let question_count = app.ai.quiz_questions.len();

    // Score header if checked
    if app.ai.quiz_checked {
        let mut score = 0;
        for (q_idx, q) in app.ai.quiz_questions.iter().enumerate() {
            let user_sel = app.ai.quiz_selected_answers.get(q_idx).cloned().unwrap_or_default();
            let mut correct_sorted = q.correct.clone();
            let mut user_sorted = user_sel.clone();
            correct_sorted.sort_unstable();
            user_sorted.sort_unstable();
            if correct_sorted == user_sorted {
                score += 1;
            }
        }

        let percentage = (score as f32 / question_count.max(1) as f32) * 100.0;
        let score_color = if percentage >= 80.0 {
            success_color(t)
        } else if percentage >= 50.0 {
            t.accent
        } else {
            danger_color(t)
        };

        egui::Frame::new()
            .fill(t.hover)
            .corner_radius(CornerRadius::same(6))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("🎯 Score: {score} / {question_count} ({percentage:.0}%)"))
                            .font(theme::semibold(14.0))
                            .color(score_color),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("✨ New Quiz").clicked() {
                            app.ai.clear_quiz();
                        }
                        if ui.small_button("🔄 Retake").clicked() {
                            for answers in &mut app.ai.quiz_selected_answers {
                                answers.clear();
                            }
                            app.ai.quiz_checked = false;
                        }
                    });
                });
            });

        ui.add_space(6.0);
    }

    // Scrollable questions list
    let bottom_h = 45.0;
    let scroll_h = (ui.available_height() - bottom_h).max(120.0);
    egui::ScrollArea::vertical()
        .max_height(scroll_h)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            for q_idx in 0..question_count {
                let q = match app.ai.quiz_questions.get(q_idx) {
                    Some(q) => q.clone(),
                    None => continue,
                };

                egui::Frame::new()
                    .fill(t.hover)
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::same(10))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());

                        ui.label(
                            egui::RichText::new(format!("Q{}: {}", q_idx + 1, q.question))
                                .font(theme::semibold(13.0))
                                .color(t.text),
                        );
                        ui.add_space(4.0);

                        for (opt_idx, opt_text) in q.options.iter().enumerate() {
                            let is_selected = app.ai.quiz_selected_answers
                                .get(q_idx)
                                .map_or(false, |ans| ans.contains(&opt_idx));

                            let is_correct_ans = q.correct.contains(&opt_idx);

                            let (bg_color, border_stroke) = if app.ai.quiz_checked {
                                if is_correct_ans {
                                    (Color32::from_rgba_unmultiplied(16, 185, 129, 45), Stroke::new(1.0, Color32::from_rgb(16, 185, 129)))
                                } else if is_selected {
                                    (Color32::from_rgba_unmultiplied(239, 68, 68, 45), Stroke::new(1.0, Color32::from_rgb(239, 68, 68)))
                                } else {
                                    (Color32::TRANSPARENT, Stroke::NONE)
                                }
                            } else if is_selected {
                                (t.accent_soft, Stroke::new(1.0, t.accent))
                            } else {
                                (Color32::TRANSPARENT, Stroke::NONE)
                            };

                            let prefix = match opt_idx {
                                0 => "A",
                                1 => "B",
                                2 => "C",
                                3 => "D",
                                _ => "E",
                            };

                            let mark = if app.ai.quiz_checked {
                                if is_correct_ans {
                                    " ✓"
                                } else if is_selected {
                                    " ✗"
                                } else {
                                    ""
                                }
                            } else {
                                ""
                            };

                            egui::Frame::new()
                                .fill(bg_color)
                                .stroke(border_stroke)
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    let btn = ui.selectable_label(is_selected, format!("{prefix}. {opt_text}{mark}"));
                                    if !app.ai.quiz_checked && btn.clicked() {
                                        if let Some(ans) = app.ai.quiz_selected_answers.get_mut(q_idx) {
                                            if app.ai.quiz_kind == QuizKind::Qcs {
                                                ans.clear();
                                                ans.push(opt_idx);
                                            } else if let Some(pos) = ans.iter().position(|&x| x == opt_idx) {
                                                ans.remove(pos);
                                            } else {
                                                ans.push(opt_idx);
                                            }
                                        }
                                    }
                                });
                            ui.add_space(2.0);
                        }

                        // Detailed explanation when checked
                        if app.ai.quiz_checked && !q.explanation.is_empty() {
                            ui.add_space(4.0);
                            egui::Frame::new()
                                .fill(Color32::from_rgba_unmultiplied(100, 100, 100, 20))
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::same(8))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.label(egui::RichText::new("💡 Explanation:").small().font(theme::semibold(11.0)));
                                    ui.label(egui::RichText::new(&q.explanation).small().color(t.text_muted));
                                });
                        }
                    });
                ui.add_space(8.0);
            }
        });

    ui.separator();

    // Bottom action bar
    ui.horizontal(|ui| {
        if !app.ai.quiz_checked {
            if ui.button("✅ Check Answers / تصحيح").clicked() {
                app.ai.quiz_checked = true;
            }
        } else {
            if ui.button("🔄 Retake Quiz").clicked() {
                for answers in &mut app.ai.quiz_selected_answers {
                    answers.clear();
                }
                app.ai.quiz_checked = false;
            }
            if ui.button("✨ Generate New Quiz").clicked() {
                app.ai.clear_quiz();
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Shared UI Elements
// ---------------------------------------------------------------------------

fn render_bubble(ui: &mut egui::Ui, msg: &Message, t: &Tokens) {
    let is_user = msg.role == "user";
    let (fill, text_color, layout) = if is_user {
        (t.accent_soft, t.accent_text, egui::Layout::right_to_left(egui::Align::TOP))
    } else {
        (t.hover, t.text, egui::Layout::left_to_right(egui::Align::TOP))
    };

    ui.with_layout(layout, |ui| {
        egui::Frame::new()
            .fill(fill)
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.set_max_width(ui.available_width() * 0.88);
                ui.label(egui::RichText::new(&msg.content).color(text_color));
            });
    });
    ui.add_space(4.0);
}

fn render_error_banner(ui: &mut egui::Ui, err: &str) {
    egui::Frame::new()
        .fill(Color32::from_rgb(0x7a, 0x1f, 0x1f))
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(egui::RichText::new(err).small().color(Color32::from_rgb(0xff, 0xdd, 0xdd)));
        });
    ui.add_space(4.0);
}

fn render_footer(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("Model: {}", app.ai.selected_model))
                .small()
                .color(t.text_muted),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("⚙ Settings").clicked() {
                app.dialog = Some(crate::Dialog::Preferences);
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Preferences Section for AI & Groq Keys
// ---------------------------------------------------------------------------

pub(crate) fn preferences_section(ui: &mut egui::Ui, app: &mut PdfCraftApp, t: &Tokens) {
    ui.label(egui::RichText::new("AI Assistant (Groq & GPT-OSS)").font(theme::semibold(13.0)));

    egui::Frame::new()
        .fill(t.hover)
        .corner_radius(CornerRadius::same(6))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            // Model selector
            ui.horizontal(|ui| {
                ui.label("Active Model:");
                let current_label = pdfcraft_ai::AVAILABLE_MODELS
                    .iter()
                    .find(|(id, _)| *id == app.ai.selected_model.as_str())
                    .map(|(_, l)| *l)
                    .unwrap_or("GPT-OSS 120B");

                egui::ComboBox::from_id_salt("ai-model-selector")
                    .selected_text(current_label)
                    .width(300.0)
                    .show_ui(ui, |ui| {
                        for &(id, label) in pdfcraft_ai::AVAILABLE_MODELS {
                            if ui.selectable_label(app.ai.selected_model == id, label).clicked() {
                                app.ai.selected_model = id.to_string();
                                ui.close();
                            }
                        }
                    });
            });

            ui.add_space(6.0);

            // API keys summary
            let key_count = app.groq_keys.count();
            ui.horizontal(|ui| {
                ui.add(icons::image("shield-check", 14.0, t.accent));
                ui.add_space(4.0);
                if key_count == 0 {
                    ui.label(egui::RichText::new("No Groq API keys configured").color(danger_color(t)));
                } else {
                    ui.label(
                        egui::RichText::new(format!(
                            "{key_count} active API key{} · Round-robin rotation active",
                            if key_count > 1 { "s" } else { "" }
                        ))
                        .color(t.accent_text),
                    );
                }
            });

            ui.add_space(6.0);
            ui.label(egui::RichText::new("Add Groq API Keys (paste one key per line, or comma-separated):").small());

            let text_edit = egui::TextEdit::multiline(&mut app.ai.keys_input)
                .hint_text("gsk_...\ngsk_...\ngsk_...")
                .desired_rows(3)
                .desired_width(ui.available_width());
            ui.add(text_edit);

            ui.horizontal(|ui| {
                if ui.button("Save & Update Keys").clicked() {
                    let mut existing = app.groq_keys.keys();
                    for part in app.ai.keys_input.split(['\n', ',', ';', ' ']) {
                        let trimmed = part.trim();
                        if trimmed.starts_with("gsk_") && !existing.contains(&trimmed.to_string()) {
                            existing.push(trimmed.to_string());
                        }
                    }
                    app.groq_keys.set_keys(existing);
                    app.ai.keys_input.clear();
                }

                if key_count > 0 && ui.button("Clear All Keys").clicked() {
                    app.groq_keys.set_keys(Vec::new());
                }
            });

            // Display list of configured keys (masked)
            if key_count > 0 {
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Configured Keys:").small().font(theme::semibold(11.0)));
                let keys = app.groq_keys.keys();
                for (idx, k) in keys.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let masked = if k.len() > 12 {
                            let head = &k[..6];
                            let tail = &k[k.len().saturating_sub(4)..];
                            format!("{head}...{tail}")
                        } else {
                            "gsk_******".to_string()
                        };
                        ui.label(egui::RichText::new(format!("Key #{}: {}", idx + 1, masked)).font(egui::FontId::monospace(11.0)));
                        if ui.small_button("✕ Remove").clicked() {
                            let mut updated = keys.clone();
                            if idx < updated.len() {
                                updated.remove(idx);
                                app.groq_keys.set_keys(updated);
                            }
                        }
                    });
                }
            }

            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(
                    "All keys are kept locally on your machine and rotated automatically per query to prevent rate limits.\
                     Get free API keys at https://console.groq.com/keys"
                )
                .small()
                .color(t.text_muted),
            );
        });
}

// ---------------------------------------------------------------------------
// Text Selection "Ask AI" Button
// ---------------------------------------------------------------------------

pub(crate) fn ask_ai_button(ui: &mut egui::Ui) -> Response {
    ui.button("✨ Ask AI")
}
