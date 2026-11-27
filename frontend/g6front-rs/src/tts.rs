//! Drama 3 TTS prompt construction, text chunking, WAV decode/resample/encode, and voice store.
//!
//! Prompt formatting and validation mirror `fish_speech/models/text2semantic/fish_omni/generate.py`
//! (`build_drama3_prompt`, `validate_drama3_text`, `with_speaker_tag`) and long-text chunking
//! mirrors `fish_speech/models/text2semantic/fish_omni/chunking.py` at commit `0f8a12e8`.

use std::collections::VecDeque;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
pub const NUM_CODEBOOKS: usize = 10;
#[allow(dead_code)]
pub const ROW_WIDTH: usize = 1 + NUM_CODEBOOKS;
pub const CODEBOOK_SIZE: i32 = 4096;

/// Verified Drama 3 special token IDs (`fishaudio/drama-3` @ `78e5ab79`, `drama3-calib/contracts.json`).
#[allow(dead_code)]
pub const ENDOFTEXT_TOKEN_ID: i32 = 248_044;
#[allow(dead_code)]
pub const IM_START_TOKEN_ID: i32 = 248_045;
pub const EOS_TOKEN_ID: i32 = 248_046;
#[allow(dead_code)]
pub const VOICE_TOKEN_ID: i32 = 248_081;
#[allow(dead_code)]
pub const SPEAKER_0_TOKEN_ID: i32 = 248_083;
pub const SEMANTIC_START_ID: i32 = 248_099;
pub const SEMANTIC_END_ID: i32 = 252_194;

pub const MAX_SPEAKER_ID: u32 = 15;
pub const DRAMA3_MAX_PROMPT_ROWS: usize = 8192;
pub const DRAMA3_CHUNK_BYTES: usize = 300;
pub const DRAMA3_CHUNK_MAX_SPEAKERS: usize = 3;
pub const DRAMA3_CHUNK_MAX_NEW_TOKENS: u64 = 600;
pub const DRAMA3_HISTORY_RECENT_TURNS: usize = 1;

pub const TARGET_SAMPLE_RATE: u32 = 44_100;
pub const MAX_VOICE_DURATION_SECS: f64 = 60.0;

pub const DRAMA3_TTS_INSTRUCTION: &str = "convert the provided text to speech";
pub const DRAMA3_REFERENCE_INSTRUCTION: &str = "convert the provided text to speech reference to the following:";
pub const DRAMA3_MESSAGE_END: &str = "<|im_end|>\n";
pub const DRAMA3_GENERATION_START: &str = "<|im_start|>assistant\n<|voice|>";

/// Only the prompt builder may emit these structural markers.
pub const DRAMA3_STRUCTURAL_TOKENS: &[&str] =
    &["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"];

/// Unicode `White_Space` property, matching `UNICODE_WHITE_SPACE` in fish-speech `generate.py`.
/// Deliberately excludes U+001C..=U+001F (which Python `str.strip()` strips but Unicode White_Space does not).
#[inline]
pub fn is_unicode_whitespace(c: char) -> bool {
    matches!(
        c as u32,
        0x09..=0x0D
            | 0x20
            | 0x85
            | 0xA0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

#[inline]
pub fn strip_uws(s: &str) -> &str {
    s.trim_matches(is_unicode_whitespace)
}

#[inline]
pub fn lstrip_uws(s: &str) -> &str {
    s.trim_start_matches(is_unicode_whitespace)
}

/// Match `<|speaker:([0-9]+)|>` at the start of `s`, returning `(digits_slice, byte_len)`.
pub fn match_speaker_tag(s: &str) -> Option<(&str, usize)> {
    let rest = s.strip_prefix("<|speaker:")?;
    let digits_len = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits_len == 0 {
        return None;
    }
    let after = &rest[digits_len..];
    if !after.starts_with("|>") {
        return None;
    }
    Some((&rest[..digits_len], "<|speaker:".len() + digits_len + "|>".len()))
}

/// Non-empty, no structural tokens, and only canonical speaker tags `<|speaker:0|>`..`<|speaker:15|>`.
pub fn validate_drama3_text(text: &str, location: &str) -> Result<(), String> {
    if strip_uws(text).is_empty() {
        return Err(format!("{location}: text must not be empty"));
    }
    for &tok in DRAMA3_STRUCTURAL_TOKENS {
        if text.contains(tok) {
            return Err(format!("{location}: {tok} is written by the prompt builder"));
        }
    }
    let mut search = 0usize;
    while let Some(rel) = text[search..].find("<|speaker") {
        let pos = search + rel;
        let ok = match match_speaker_tag(&text[pos..]) {
            Some((digits, _)) => {
                let canonical = digits == "0" || !digits.starts_with('0');
                canonical && digits.parse::<u32>().is_ok_and(|n| n <= MAX_SPEAKER_ID)
            }
            None => false,
        };
        if !ok {
            return Err(format!(
                "{location}: speaker tags must be <|speaker:0|> to <|speaker:{MAX_SPEAKER_ID}|>; got {text:?}"
            ));
        }
        search = pos + "<|speaker".len();
    }
    Ok(())
}

/// Prefix `<|speaker:0|> ` unless a speaker tag already leads `text` after Unicode whitespace.
pub fn with_speaker_tag(text: &str) -> String {
    if match_speaker_tag(lstrip_uws(text)).is_none() {
        format!("<|speaker:0|> {text}")
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Chunking (`fish_omni/chunking.py` @ 0f8a12e8)
// ---------------------------------------------------------------------------------------------------------------

#[inline]
fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | ';' | '。' | '！' | '？' | '；' | '…' | '\n')
}

#[inline]
fn is_clause_mark(c: char) -> bool {
    matches!(c, ',' | ':' | '，' | '：' | '、')
}

#[inline]
fn is_space_mark(c: char) -> bool {
    c == ' '
}

#[inline]
fn is_py_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Match `PROTECTED_SPAN = r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]"` (re.S)
/// starting at byte index 0 of `s`. Returns matched byte length if found.
fn match_protected_span(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    // 1. `<|[^|>]*|>`
    if let Some(rest) = s.strip_prefix("<|") {
        let mut off = 0usize;
        for (i, c) in rest.char_indices() {
            if c == '|' {
                if rest[i..].starts_with("|>") {
                    return Some(2 + i + 2);
                }
                return None;
            }
            if c == '>' {
                return None;
            }
            off = i + c.len_utf8();
        }
        let _ = off;
        return None;
    }
    // 2. `<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>` (with Python regex backtracking on group 1)
    if bytes[0] == b'<' && bytes.len() >= 2 && bytes[1].is_ascii_alphabetic() {
        let name_start = 1usize;
        let mut ends: Vec<usize> = Vec::new();
        let mut cursor = 2usize; // after first ASCII letter
        ends.push(cursor);
        for (rel, c) in s[2..].char_indices() {
            if is_py_word_char(c) || c == '-' {
                cursor = 2 + rel + c.len_utf8();
                ends.push(cursor);
            } else {
                break;
            }
        }
        // Greedy: try longest tag name first, backtracking shorter if \b or closing tag fails.
        for &name_end in ends.iter().rev() {
            let tag_name = &s[name_start..name_end];
            let last_char = tag_name.chars().next_back().unwrap();
            let next_char = s[name_end..].chars().next();
            let prev_is_w = is_py_word_char(last_char);
            let next_is_w = next_char.is_some_and(is_py_word_char);
            // Word boundary `\b` holds iff exactly one side is a word char.
            if prev_is_w == next_is_w {
                continue;
            }
            // Scan `[^>]*>` from `name_end`.
            let after_name = &s[name_end..];
            let Some(gt_rel) = after_name.find('>') else {
                continue;
            };
            let body_start = name_end + gt_rel + 1;
            let close_tag = format!("</{tag_name}>");
            if let Some(close_rel) = s[body_start..].find(&close_tag) {
                return Some(body_start + close_rel + close_tag.len());
            }
        }
        return None;
    }
    // 3. `\[[^\[\]]*\]`
    if bytes[0] == b'[' {
        for (i, c) in s[1..].char_indices() {
            if c == '[' {
                return None;
            }
            if c == ']' {
                return Some(1 + i + 1);
            }
        }
        return None;
    }
    None
}

fn protected_pieces(text: &str) -> Vec<(String, bool)> {
    let mut pieces = Vec::new();
    let mut last = 0usize;
    let mut pos = 0usize;
    while pos < text.len() {
        if let Some(m_len) = match_protected_span(&text[pos..]) {
            if pos > last {
                pieces.push((text[last..pos].to_string(), false));
            }
            pieces.push((text[pos..pos + m_len].to_string(), true));
            pos += m_len;
            last = pos;
        } else {
            let ch = text[pos..].chars().next().unwrap();
            pos += ch.len_utf8();
        }
    }
    if last < text.len() {
        pieces.push((text[last..].to_string(), false));
    }
    pieces
}

fn breaks_at(chars: &[char], index: usize, pred: fn(char) -> bool) -> bool {
    let c = chars[index];
    if !pred(c) {
        return false;
    }
    // Keep decimals such as 3.14 in one piece.
    !(c == '.'
        && index > 0
        && index + 1 < chars.len()
        && chars[index - 1].is_numeric()
        && chars[index + 1].is_numeric())
}

fn break_at_marks(pieces: Vec<(String, bool)>, length: usize, pred: fn(char) -> bool) -> Vec<(String, bool)> {
    let mut out = Vec::with_capacity(pieces.len());
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            out.push((text, protected));
            continue;
        }
        let chars: Vec<char> = text.chars().collect();
        let byte_offsets: Vec<usize> = text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect();
        let mut start = 0usize;
        for i in 0..chars.len() {
            if breaks_at(&chars, i, pred) {
                out.push((text[byte_offsets[start]..byte_offsets[i + 1]].to_string(), false));
                start = i + 1;
            }
        }
        if start < chars.len() {
            out.push((text[byte_offsets[start]..].to_string(), false));
        }
    }
    out
}

fn break_by_length(pieces: Vec<(String, bool)>, length: usize) -> Vec<(String, bool)> {
    let mut out = Vec::with_capacity(pieces.len());
    for (text, protected) in pieces {
        if protected || text.len() <= length {
            out.push((text, protected));
            continue;
        }
        let mut current = String::new();
        for ch in text.chars() {
            if !current.is_empty() && current.len() + ch.len_utf8() > length {
                out.push((std::mem::take(&mut current), false));
            }
            current.push(ch);
        }
        out.push((current, false));
    }
    out
}

pub fn split_by_length(text: &str, length: usize) -> Vec<String> {
    if strip_uws(text).is_empty() {
        return Vec::new();
    }
    if text.len() <= length {
        return vec![text.to_string()];
    }
    let mut pieces = break_at_marks(protected_pieces(text), 0, is_sentence_end);
    pieces = break_at_marks(pieces, length, is_clause_mark);
    pieces = break_at_marks(pieces, length, is_space_mark);
    pieces = break_by_length(pieces, length);

    let mut chunks = Vec::new();
    let mut current = String::new();
    for (piece, _) in pieces {
        if !current.is_empty() && current.len() + piece.len() > length {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(&piece);
    }
    chunks.push(current);

    chunks
        .into_iter()
        .filter_map(|c| {
            let trimmed = strip_uws(&c);
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect()
}

/// Find every `<|speaker:([0-9]+)|>` match in `text`, returning `(byte_start, byte_end, digits)`.
fn find_speaker_tags(text: &str) -> Vec<(usize, usize, &str)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = text[pos..].find("<|speaker:") {
        let start = pos + rel;
        if let Some((digits, len)) = match_speaker_tag(&text[start..]) {
            out.push((start, start + len, digits));
            pos = start + len;
        } else {
            pos = start + "<|speaker:".len();
        }
    }
    out
}

/// `(speaker_id, text)` per tagged segment; text before the first tag is speaker `"0"`.
pub fn parse_speaker_segments(text: &str) -> Vec<(String, String)> {
    let tags = find_speaker_tags(text);
    let mut segments = Vec::new();
    let first_end = tags.first().map_or(text.len(), |t| t.0);
    if !strip_uws(&text[..first_end]).is_empty() {
        segments.push(("0".to_string(), text[..first_end].to_string()));
    }
    for (idx, &(_, end, digits)) in tags.iter().enumerate() {
        let body_end = tags.get(idx + 1).map_or(text.len(), |t| t.0);
        let body = &text[end..body_end];
        if !strip_uws(body).is_empty() {
            segments.push((digits.to_string(), body.to_string()));
        }
    }
    segments
}

/// Generation chunks for `text`, in order (`fish_omni/chunking.py:split_drama3_text`).
pub fn split_drama3_text(text: &str, max_bytes: usize, max_speakers: usize) -> Vec<String> {
    if find_speaker_tags(text).is_empty() {
        return split_by_length(text, max_bytes);
    }

    let mut segments: Vec<(String, String)> = Vec::new();
    for (speaker, body) in parse_speaker_segments(text) {
        for piece in split_by_length(&body, max_bytes) {
            segments.push((speaker.clone(), format!("<|speaker:{speaker}|> {piece}")));
        }
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut batch: Vec<String> = Vec::new();
    let mut speakers: Vec<String> = Vec::new();
    let mut size = 0usize;

    for (speaker, segment) in segments {
        let mut new_speaker = !speakers.contains(&speaker);
        let too_many_speakers = new_speaker && speakers.len() >= max_speakers;
        let too_long = !batch.is_empty() && size + segment.len() > max_bytes;
        if too_many_speakers || too_long {
            chunks.push(batch.join("\n"));
            batch.clear();
            speakers.clear();
            size = 0;
            new_speaker = true;
        }
        size += segment.len();
        batch.push(segment);
        if new_speaker {
            speakers.push(speaker);
        }
    }
    if !batch.is_empty() {
        chunks.push(batch.join("\n"));
    }
    chunks
}

/// History turn `(chunk text, codes [F * 10] frame-major)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryTurn {
    pub text: String,
    pub codes: Vec<i32>,
}

/// Context for the next chunk: the first chunk stays pinned, plus the latest `recent_turns` chunks.
#[derive(Clone, Debug)]
pub struct Drama3History {
    pinned: Option<HistoryTurn>,
    recent: VecDeque<HistoryTurn>,
    max_recent: usize,
}

impl Drama3History {
    pub fn new(recent_turns: usize) -> Self {
        Drama3History { pinned: None, recent: VecDeque::with_capacity(recent_turns), max_recent: recent_turns }
    }

    pub fn push(&mut self, text: String, codes: Vec<i32>) {
        let turn = HistoryTurn { text, codes };
        if self.pinned.is_none() {
            self.pinned = Some(turn);
        } else if self.max_recent > 0 {
            if self.recent.len() == self.max_recent {
                self.recent.pop_front();
            }
            self.recent.push_back(turn);
        }
    }

    /// Histories to try, longest first: drop oldest recent turns, then the pinned one, ending with empty.
    pub fn candidates(&self) -> Vec<Vec<HistoryTurn>> {
        let pinned: Vec<HistoryTurn> = self.pinned.iter().cloned().collect();
        let recent: Vec<HistoryTurn> = self.recent.iter().cloned().collect();
        let mut out = Vec::with_capacity(recent.len() + 2);
        for start in 0..=recent.len() {
            let mut c = pinned.clone();
            c.extend_from_slice(&recent[start..]);
            out.push(c);
        }
        if !pinned.is_empty() {
            out.push(Vec::new());
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Prompt row builder (`fish_omni/generate.py:build_drama3_prompt`)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceItem {
    pub text: String,
    /// Frame-major `[F, 10]` (`f * 10 + c`).
    pub codes: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromptSegment {
    Text(String),
    /// Frame-major `[F, 10]`.
    Audio(Vec<i32>),
}

pub fn check_codes(codes: &[i32], num_codebooks: usize, location: &str) -> Result<usize, String> {
    if codes.is_empty() || codes.len() % num_codebooks != 0 {
        return Err(format!(
            "{location}: codes must be non-empty multiples of {num_codebooks}; got {}",
            codes.len()
        ));
    }
    for &c in codes {
        if !(0..CODEBOOK_SIZE).contains(&c) {
            return Err(format!("{location}: codes must be in [0, {CODEBOOK_SIZE}); got {c}"));
        }
    }
    Ok(codes.len() / num_codebooks)
}

/// Build the ordered text/audio segments of `build_drama3_prompt` (before tokenization).
pub fn build_drama3_segments(
    text: &str,
    refs: &[ReferenceItem],
    history: &[HistoryTurn],
    num_codebooks: usize,
) -> Result<Vec<PromptSegment>, String> {
    validate_drama3_text(text, "text")?;
    let mut segments = Vec::with_capacity(6 + refs.len() * 2 + history.len() * 6);
    segments.push(PromptSegment::Text("<|im_start|>system\n".to_string()));
    segments.push(PromptSegment::Text(
        if refs.is_empty() { DRAMA3_TTS_INSTRUCTION } else { DRAMA3_REFERENCE_INSTRUCTION }.to_string(),
    ));

    for (i, r) in refs.iter().enumerate() {
        let index = i + 1;
        let location = format!("reference {index}");
        validate_drama3_text(&r.text, &location)?;
        check_codes(&r.codes, num_codebooks, &location)?;
        // The tag must follow "Text: " directly; leading white space moves behind it.
        let tagged = with_speaker_tag(&r.text);
        let body = lstrip_uws(&tagged);
        let leading = &tagged[..tagged.len() - body.len()];
        let (_, tag_end) = match_speaker_tag(body).ok_or_else(|| format!("{location}: missing speaker tag"))?;
        let fixed = format!("{}{}{}", &body[..tag_end], leading, &body[tag_end..]);
        segments.push(PromptSegment::Text(format!("\n\n[Reference {index}]\nText: {fixed}\nSpeech: ")));
        segments.push(PromptSegment::Audio(r.codes.clone()));
    }
    segments.push(PromptSegment::Text(DRAMA3_MESSAGE_END.to_string()));

    for (i, h) in history.iter().enumerate() {
        let index = i + 1;
        let location = format!("history {index}");
        validate_drama3_text(&h.text, &location)?;
        check_codes(&h.codes, num_codebooks, &location)?;
        segments.push(PromptSegment::Text("<|im_start|>user\n".to_string()));
        segments.push(PromptSegment::Text(with_speaker_tag(&h.text)));
        segments.push(PromptSegment::Text(DRAMA3_MESSAGE_END.to_string()));
        segments.push(PromptSegment::Text(DRAMA3_GENERATION_START.to_string()));
        segments.push(PromptSegment::Audio(h.codes.clone()));
        segments.push(PromptSegment::Text(DRAMA3_MESSAGE_END.to_string()));
    }

    segments.push(PromptSegment::Text("<|im_start|>user\n".to_string()));
    segments.push(PromptSegment::Text(with_speaker_tag(text)));
    segments.push(PromptSegment::Text(DRAMA3_MESSAGE_END.to_string()));
    segments.push(PromptSegment::Text(DRAMA3_GENERATION_START.to_string()));
    Ok(segments)
}

/// Flatten segments into frame-major `[T, 1 + num_codebooks]` rows (`(rows_flat, num_rows)`).
pub fn build_drama3_prompt_rows_with<F>(
    text: &str,
    refs: &[ReferenceItem],
    history: &[HistoryTurn],
    num_codebooks: usize,
    semantic_start_id: i32,
    max_prompt_rows: usize,
    mut encode_segment: F,
) -> Result<(Vec<i32>, usize), String>
where
    F: FnMut(&str) -> Result<Vec<u32>, String>,
{
    let segments = build_drama3_segments(text, refs, history, num_codebooks)?;
    let row_width = 1 + num_codebooks;
    let mut rows: Vec<i32> = Vec::new();
    let mut num_rows = 0usize;

    for seg in segments {
        match seg {
            PromptSegment::Text(s) => {
                let ids = encode_segment(&s)?;
                rows.reserve(ids.len() * row_width);
                for id in ids {
                    rows.push(id as i32);
                    rows.resize(rows.len() + num_codebooks, 0);
                    num_rows += 1;
                }
            }
            PromptSegment::Audio(codes) => {
                let frames = codes.len() / num_codebooks;
                rows.reserve(frames * row_width);
                for frame in codes.chunks_exact(num_codebooks) {
                    rows.push(frame[0] + semantic_start_id);
                    rows.extend_from_slice(frame);
                    num_rows += 1;
                }
            }
        }
    }

    if num_rows > max_prompt_rows {
        return Err(format!("prompt has {num_rows} rows; at most {max_prompt_rows}"));
    }
    Ok((rows, num_rows))
}

/// Build Drama 3 prompt rows using a loaded HF `tokenizers::Tokenizer`.
pub fn build_drama3_prompt_rows(
    text: &str,
    tok: &tokenizers::Tokenizer,
    refs: &[ReferenceItem],
    history: &[HistoryTurn],
    num_codebooks: usize,
    max_prompt_rows: usize,
) -> Result<(Vec<i32>, usize), String> {
    let semantic_start = tok.token_to_id("<|semantic:0|>").map_or(SEMANTIC_START_ID, |id| id as i32);
    if let Some(end_id) = tok.token_to_id(&format!("<|semantic:{}|>", CODEBOOK_SIZE - 1)) {
        if end_id as i32 != semantic_start + CODEBOOK_SIZE - 1 || (semantic_start == SEMANTIC_START_ID && end_id as i32 != SEMANTIC_END_ID) {
            return Err(format!(
                "tokenizer semantic range mismatch: start={semantic_start}, end={end_id}"
            ));
        }
    }
    if let Some(eos_id) = tok.token_to_id("<|im_end|>") {
        if semantic_start == SEMANTIC_START_ID && tok.token_to_id("<|semantic:0|>").is_some() && eos_id as i32 != EOS_TOKEN_ID {
            return Err(format!("tokenizer <|im_end|> mismatch: got {eos_id}, expected {EOS_TOKEN_ID}"));
        }
    }
    build_drama3_prompt_rows_with(text, refs, history, num_codebooks, semantic_start, max_prompt_rows, |seg| {
        tok.encode(seg, true).map(|enc| enc.get_ids().to_vec()).map_err(|e| format!("tokenize: {e}"))
    })
}

// ---------------------------------------------------------------------------------------------------------------
// WAV decode / mono mixdown / 44.1 kHz resample / WAV encode (`hound` + `rubato`)
// ---------------------------------------------------------------------------------------------------------------

/// Decode a WAV buffer into mono `f32` samples at `TARGET_SAMPLE_RATE` (44.1 kHz).
/// Rejects invalid WAVs, empty audio, and clips longer than `MAX_VOICE_DURATION_SECS` (60 s).
pub fn decode_wav_to_f32_44100(wav_bytes: &[u8]) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::new(Cursor::new(wav_bytes)).map_err(|e| format!("invalid WAV audio: {e}"))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err("invalid WAV audio: zero channels".into());
    }
    if spec.sample_rate == 0 {
        return Err("invalid WAV audio: zero sample rate".into());
    }
    let header_dur_s = reader.duration() as f64 / spec.sample_rate as f64;
    if header_dur_s > MAX_VOICE_DURATION_SECS + 1e-6 {
        return Err(format!(
            "audio duration {header_dur_s:.2}s exceeds maximum of {MAX_VOICE_DURATION_SECS:.0}s"
        ));
    }

    let channels = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| format!("invalid WAV float sample: {e}")))
            .collect::<Result<Vec<f32>, _>>()?,
        hound::SampleFormat::Int => {
            if spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
                return Err(format!("unsupported WAV bits_per_sample: {}", spec.bits_per_sample));
            }
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f32) / scale).map_err(|e| format!("invalid WAV int sample: {e}")))
                .collect::<Result<Vec<f32>, _>>()?
        }
    };

    if interleaved.is_empty() {
        return Err("invalid WAV audio: empty waveform".into());
    }
    let frames = interleaved.len() / channels;
    if frames == 0 {
        return Err("invalid WAV audio: empty waveform".into());
    }
    let dur_s = frames as f64 / spec.sample_rate as f64;
    if dur_s > MAX_VOICE_DURATION_SECS + 1e-6 {
        return Err(format!("audio duration {dur_s:.2}s exceeds maximum of {MAX_VOICE_DURATION_SECS:.0}s"));
    }

    let mut mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        let inv = 1.0f32 / channels as f32;
        interleaved[..frames * channels]
            .chunks_exact(channels)
            .map(|ch| ch.iter().copied().sum::<f32>() * inv)
            .collect()
    };

    if spec.sample_rate == TARGET_SAMPLE_RATE {
        for s in &mut mono {
            if !s.is_finite() {
                *s = 0.0;
            } else {
                *s = s.clamp(-1.0, 1.0);
            }
        }
        return Ok(mono);
    }

    resample_mono_f32(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)
}

fn resample_mono_f32(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>, String> {
    let ratio = out_rate as f64 / in_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk_size = 1024usize.min(input.len().max(1));
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
        .map_err(|e| format!("resampler init failed: {e}"))?;

    let expected_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let delay = resampler.output_delay();
    let mut raw_out = Vec::with_capacity(expected_len + delay + 256);

    let mut pos = 0usize;
    while pos + chunk_size <= input.len() {
        let block = [&input[pos..pos + chunk_size]];
        let out = resampler.process(&block, None).map_err(|e| format!("resample failed: {e}"))?;
        raw_out.extend_from_slice(&out[0]);
        pos += chunk_size;
    }
    if pos < input.len() {
        let tail = [&input[pos..]];
        let out = resampler.process_partial(Some(&tail), None).map_err(|e| format!("resample tail failed: {e}"))?;
        raw_out.extend_from_slice(&out[0]);
    }
    // Flush internal filter delay so we have enough samples after trimming `delay`.
    while raw_out.len() < delay + expected_len {
        let out = resampler
            .process_partial::<&[f32]>(None, None)
            .map_err(|e| format!("resample flush failed: {e}"))?;
        if out[0].is_empty() {
            break;
        }
        raw_out.extend_from_slice(&out[0]);
    }

    let start = delay.min(raw_out.len());
    let end = (start + expected_len).min(raw_out.len());
    let mut trimmed = raw_out[start..end].to_vec();
    for s in &mut trimmed {
        if !s.is_finite() {
            *s = 0.0;
        } else {
            *s = s.clamp(-1.0, 1.0);
        }
    }
    Ok(trimmed)
}

pub fn f32_to_le_b64(samples: &[f32]) -> String {
    let mut raw = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        raw.extend_from_slice(&s.to_le_bytes());
    }
    crate::vision::b64_encode(&raw)
}

/// Wrap raw s16le mono PCM bytes into a standard RIFF/WAVE file.
pub fn encode_s16le_pcm_to_wav(pcm_s16le: &[u8], sample_rate: u32) -> Result<Vec<u8>, String> {
    if pcm_s16le.len() % 2 != 0 {
        return Err(format!("PCM s16le byte count must be even, got {}", pcm_s16le.len()));
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cur = Cursor::new(Vec::with_capacity(44 + pcm_s16le.len()));
    {
        let mut writer = hound::WavWriter::new(&mut cur, spec).map_err(|e| format!("wav header: {e}"))?;
        for pair in pcm_s16le.chunks_exact(2) {
            let sample = i16::from_le_bytes([pair[0], pair[1]]);
            writer.write_sample(sample).map_err(|e| format!("wav write: {e}"))?;
        }
        writer.finalize().map_err(|e| format!("wav finalize: {e}"))?;
    }
    Ok(cur.into_inner())
}

// ---------------------------------------------------------------------------------------------------------------
// Voice store (`$GR8_WORK_DIR/voices/<name>.json`)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoiceRecord {
    pub name: String,
    pub text: String,
    /// Frame-major `[F, 10]` codes.
    pub codes: Vec<i32>,
    pub frames: usize,
    pub sample_rate: u32,
}

pub fn default_voices_dir() -> PathBuf {
    match std::env::var("GR8_WORK_DIR") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir).join("voices"),
        _ => std::env::temp_dir().join("g6front-voices"),
    }
}

pub fn validate_voice_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 128 {
        return Err("voice name must be 1..=128 characters".into());
    }
    if name == "default" || name == "." || name == ".." || name.starts_with('.') {
        return Err(format!("invalid voice name {name:?}"));
    }
    if !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')) {
        return Err(format!("voice name must contain only [A-Za-z0-9._-], got {name:?}"));
    }
    Ok(())
}

fn voice_path(dir: &Path, name: &str) -> Result<PathBuf, String> {
    validate_voice_name(name)?;
    Ok(dir.join(format!("{name}.json")))
}

pub fn save_voice(dir: &Path, record: &VoiceRecord) -> Result<(), String> {
    let path = voice_path(dir, &record.name)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create voices dir: {e}"))?;
    let bytes = serde_json::to_vec(record).map_err(|e| format!("serialize voice: {e}"))?;
    let tmp = dir.join(format!(".{}.{}.tmp", record.name, crate::util::new_id("v")));
    std::fs::write(&tmp, &bytes).map_err(|e| format!("write voice temp: {e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("commit voice file: {e}")
    })
}

pub fn load_voice(dir: &Path, name: &str) -> Result<Option<VoiceRecord>, String> {
    let path = voice_path(dir, name)?;
    match std::fs::read(&path) {
        Ok(bytes) => {
            let rec: VoiceRecord = serde_json::from_slice(&bytes).map_err(|e| format!("corrupt voice {name:?}: {e}"))?;
            Ok(Some(rec))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("read voice {name:?}: {e}")),
    }
}

pub fn list_voices(dir: &Path) -> Result<Vec<VoiceRecord>, String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("list voices dir: {e}")),
    };
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("read voices entry: {e}"))?;
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if fname.starts_with('.') || !fname.ends_with(".json") {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let rec: VoiceRecord = serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;
        out.push(rec);
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

pub fn delete_voice(dir: &Path, name: &str) -> Result<bool, String> {
    let path = voice_path(dir, name)?;
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(format!("delete voice {name:?}: {e}")),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::collections::HashMap;

    fn make_wav(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cur = Cursor::new(Vec::new());
        {
            let mut w = hound::WavWriter::new(&mut cur, spec).unwrap();
            for &s in samples {
                w.write_sample(s).unwrap();
            }
            w.finalize().unwrap();
        }
        cur.into_inner()
    }

    #[test]
    fn golden_50_cases_match_python_fish_omni() {
        let cases: Vec<Value> = serde_json::from_str(include_str!("tts_golden.json")).unwrap();
        assert_eq!(cases.len(), 50, "expected 50 golden cases");

        // Optional real Drama 3 tokenizer cross-check if available locally or via env.
        let real_tok = std::env::var("GR8_DRAMA3_TOKENIZER")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists())
            .and_then(|p| tokenizers::Tokenizer::from_file(p).ok());

        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let text = case["text"].as_str().unwrap();
            let max_bytes = case["max_bytes"].as_u64().unwrap() as usize;
            let max_speakers = case["max_speakers"].as_u64().unwrap() as usize;

            let expected_chunks: Vec<String> =
                case["expected_chunks"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
            let got_chunks = split_drama3_text(text, max_bytes, max_speakers);
            assert_eq!(got_chunks, expected_chunks, "chunks mismatch on case {name}");

            let refs: Vec<ReferenceItem> = case["refs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| ReferenceItem {
                    text: r["text"].as_str().unwrap().to_string(),
                    codes: r["codes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|frame| frame.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32))
                        .collect(),
                })
                .collect();
            let history: Vec<HistoryTurn> = case["history"]
                .as_array()
                .unwrap()
                .iter()
                .map(|h| HistoryTurn {
                    text: h["text"].as_str().unwrap().to_string(),
                    codes: h["codes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|frame| frame.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32))
                        .collect(),
                })
                .collect();
            let prompt_text = case["prompt_text"].as_str().unwrap();

            // 1. Check exact segments sequence
            let got_segs = build_drama3_segments(prompt_text, &refs, &history, NUM_CODEBOOKS)
                .unwrap_or_else(|e| panic!("segments error on {name}: {e}"));
            let exp_segs = case["expected_segments"].as_array().unwrap();
            assert_eq!(got_segs.len(), exp_segs.len(), "segment count mismatch on {name}");

            let mut seg_tok_table: HashMap<String, Vec<u32>> = HashMap::new();
            for (seg, exp) in got_segs.iter().zip(exp_segs.iter()) {
                match seg {
                    PromptSegment::Text(s) => {
                        let exp_s = exp["text"].as_str().unwrap_or_else(|| panic!("expected text seg in {name}"));
                        assert_eq!(s, exp_s, "text segment mismatch on {name}");
                        let ids: Vec<u32> =
                            exp["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                        seg_tok_table.insert(s.clone(), ids);
                    }
                    PromptSegment::Audio(codes) => {
                        let exp_codes: Vec<i32> = exp["audio"]
                            .as_array()
                            .unwrap_or_else(|| panic!("expected audio seg in {name}"))
                            .iter()
                            .flat_map(|f| f.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32))
                            .collect();
                        assert_eq!(codes, &exp_codes, "audio segment mismatch on {name}");
                    }
                }
            }

            // 2. Check exact rows using recorded Drama 3 token IDs
            let expected_rows: Vec<i32> = case["expected_rows"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32))
                .collect();
            let (got_rows, n_rows) = build_drama3_prompt_rows_with(
                prompt_text,
                &refs,
                &history,
                NUM_CODEBOOKS,
                SEMANTIC_START_ID,
                DRAMA3_MAX_PROMPT_ROWS,
                |s| Ok(seg_tok_table.get(s).unwrap_or_else(|| panic!("missing seg {s:?} in {name}")).clone()),
            )
            .unwrap();
            assert_eq!(n_rows * ROW_WIDTH, expected_rows.len(), "row count mismatch on {name}");
            assert_eq!(got_rows, expected_rows, "rows mismatch on {name}");

            // 3. If real Drama 3 tokenizer is present, also verify full end-to-end tokenization!
            if let Some(tok) = &real_tok {
                assert_eq!(tok.token_to_id("<|endoftext|>"), Some(ENDOFTEXT_TOKEN_ID as u32));
                assert_eq!(tok.token_to_id("<|im_start|>"), Some(IM_START_TOKEN_ID as u32));
                assert_eq!(tok.token_to_id("<|im_end|>"), Some(EOS_TOKEN_ID as u32));
                assert_eq!(tok.token_to_id("<|voice|>"), Some(VOICE_TOKEN_ID as u32));
                assert_eq!(tok.token_to_id("<|speaker:0|>"), Some(SPEAKER_0_TOKEN_ID as u32));
                assert_eq!(tok.token_to_id("<|semantic:0|>"), Some(SEMANTIC_START_ID as u32));
                assert_eq!(tok.token_to_id("<|semantic:4095|>"), Some(SEMANTIC_END_ID as u32));
                let (real_rows, _) = build_drama3_prompt_rows(
                    prompt_text,
                    tok,
                    &refs,
                    &history,
                    NUM_CODEBOOKS,
                    DRAMA3_MAX_PROMPT_ROWS,
                )
                .unwrap();
                assert_eq!(real_rows, expected_rows, "real tokenizer rows mismatch on {name}");
            }
        }
    }

    #[test]
    fn rejects_structural_tokens_and_invalid_speaker_tags() {
        for tok in DRAMA3_STRUCTURAL_TOKENS {
            let text = format!("hello {tok} world");
            assert!(validate_drama3_text(&text, "text").is_err(), "should reject {tok}");
        }
        assert!(validate_drama3_text("", "text").is_err());
        assert!(validate_drama3_text("   \t\n\u{3000}", "text").is_err());
        assert!(validate_drama3_text("<|speaker:16|> hi", "text").is_err());
        assert!(validate_drama3_text("<|speaker:00|> hi", "text").is_err());
        assert!(validate_drama3_text("<|speaker:01|> hi", "text").is_err());
        assert!(validate_drama3_text("<|speaker:-1|> hi", "text").is_err());
        assert!(validate_drama3_text("<|speaker:abc|> hi", "text").is_err());
        assert!(validate_drama3_text("<|speaker hi", "text").is_err());
        // Valid speaker tags 0..=15
        for id in 0..=15 {
            assert!(validate_drama3_text(&format!("<|speaker:{id}|> ok"), "text").is_ok());
        }
    }

    #[test]
    fn history_keeps_pinned_first_and_latest_recent() {
        let mut h = Drama3History::new(DRAMA3_HISTORY_RECENT_TURNS);
        assert_eq!(h.candidates(), vec![vec![]]);

        h.push("c0".into(), vec![0; 10]);
        assert_eq!(h.candidates().len(), 2);
        assert_eq!(h.candidates()[0].iter().map(|t| t.text.as_str()).collect::<Vec<_>>(), vec!["c0"]);
        assert!(h.candidates()[1].is_empty());

        h.push("c1".into(), vec![1; 10]);
        assert_eq!(h.candidates().len(), 3);
        assert_eq!(h.candidates()[0].iter().map(|t| t.text.as_str()).collect::<Vec<_>>(), vec!["c0", "c1"]);
        assert_eq!(h.candidates()[1].iter().map(|t| t.text.as_str()).collect::<Vec<_>>(), vec!["c0"]);
        assert!(h.candidates()[2].is_empty());

        h.push("c2".into(), vec![2; 10]);
        assert_eq!(h.candidates()[0].iter().map(|t| t.text.as_str()).collect::<Vec<_>>(), vec!["c0", "c2"]);
        assert_eq!(h.candidates()[1].iter().map(|t| t.text.as_str()).collect::<Vec<_>>(), vec!["c0"]);
        assert!(h.candidates()[2].is_empty());
    }

    #[test]
    fn wav_decode_mono_resample_and_duration_limit() {
        // 1. 44.1 kHz stereo -> mono mixdown
        let stereo = vec![16384i16, -16384, 32767, 0, 8192, 8192];
        let wav = make_wav(&stereo, 44_100, 2);
        let mono = decode_wav_to_f32_44100(&wav).unwrap();
        assert_eq!(mono.len(), 3);
        assert!(mono[0].abs() < 1e-4);
        assert!((mono[1] - 0.5).abs() < 1e-3);
        assert!((mono[2] - 0.25).abs() < 1e-3);

        // 2. 48 kHz mono sine -> resampled to 44.1 kHz (0.5 s => ~22050 samples)
        let n_in = 24_000usize;
        let sine: Vec<i16> = (0..n_in)
            .map(|i| ((2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48_000.0).sin() * 16000.0) as i16)
            .collect();
        let wav48 = make_wav(&sine, 48_000, 1);
        let resampled = decode_wav_to_f32_44100(&wav48).unwrap();
        assert!((resampled.len() as isize - 22_050).abs() <= 4, "len={}", resampled.len());
        let rms = (resampled.iter().map(|&x| x * x).sum::<f32>() / resampled.len() as f32).sqrt();
        assert!(rms > 0.2 && rms < 0.5, "unexpected rms {rms}");

        // 3. Reject > 60 seconds (e.g. 61 seconds at 8 kHz)
        let long_pcm = vec![0i16; 61 * 8_000];
        let long_wav = make_wav(&long_pcm, 8_000, 1);
        let err = decode_wav_to_f32_44100(&long_wav).unwrap_err();
        assert!(err.contains("exceeds maximum"), "{err}");

        // 4. Encode s16le PCM to RIFF/WAVE and read back
        let raw: Vec<u8> = [100i16, -200, 300, -400].iter().flat_map(|s| s.to_le_bytes()).collect();
        let out_wav = encode_s16le_pcm_to_wav(&raw, 44_100).unwrap();
        assert_eq!(&out_wav[..4], b"RIFF");
        assert_eq!(&out_wav[8..12], b"WAVE");
        let r = hound::WavReader::new(Cursor::new(&out_wav)).unwrap();
        assert_eq!(r.spec().sample_rate, 44_100);
        assert_eq!(r.spec().channels, 1);
        assert_eq!(r.spec().bits_per_sample, 16);
    }

    #[test]
    fn voices_crud_and_name_validation() {
        let dir = std::env::temp_dir().join(format!("g6front-voices-test-{}", crate::util::new_id("t")));
        assert!(validate_voice_name("default").is_err());
        assert!(validate_voice_name("../escape").is_err());
        assert!(validate_voice_name("a/b").is_err());
        assert!(validate_voice_name(".hidden").is_err());
        assert!(validate_voice_name("narrator_1").is_ok());

        assert!(list_voices(&dir).unwrap().is_empty());
        let rec = VoiceRecord {
            name: "narrator_1".into(),
            text: "Hello world.".into(),
            codes: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            frames: 1,
            sample_rate: 44_100,
        };
        save_voice(&dir, &rec).unwrap();
        assert_eq!(load_voice(&dir, "narrator_1").unwrap(), Some(rec.clone()));
        assert_eq!(list_voices(&dir).unwrap(), vec![rec]);
        assert!(delete_voice(&dir, "narrator_1").unwrap());
        assert!(!delete_voice(&dir, "narrator_1").unwrap());
        assert!(load_voice(&dir, "narrator_1").unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
