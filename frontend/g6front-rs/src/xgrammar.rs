//! The compiled-grammar cache for response_format json_schema (xgrammar).
//!
//! The tokenizer metadata is built once from this front end's own vocabulary (byte-fallback BPE, no prefix
//! space), so a schema compiles against the real vocabulary; an unsupported schema is a 400 before the engine
//! sees the request. `json_object` needs no grammar.

use xgrammar::{GrammarCompiler, TokenizerInfo, VocabType};

use crate::tok::Tok;

pub struct Grammar(pub GrammarCompiler);

impl Grammar {
    pub fn build(t: &Tok) -> Grammar {
        let n = t.vocab_size;
        // decoded single-id text (not the raw token string): the matcher reasons over output text
        let vocab: Vec<String> =
            (0..n as u32).map(|i| t.decode(&[i], false)).collect();
        let meta = format!(
            "{{\"vocab_type\": {}, \"vocab_size\": {}, \"add_prefix_space\": false}}",
            VocabType::ByteFallback as i64,
            n
        );
        let info = TokenizerInfo::from_vocab_and_metadata(&vocab, &meta)
            .unwrap_or_else(|e| panic!("xgrammar tokenizer metadata: {e}"));
        Grammar(GrammarCompiler::with_defaults(info))
    }
}
