//! The grammar compiler for `response_format` `json_schema` (XGrammar), built once from the model's tokenizer.json, so
//! a schema compiles against the vocabulary the model writes in.

use std::path::Path;

use xgrammar::{CompiledGrammar, GrammarCompiler, GrammarMatcher, TokenizerInfo};

pub struct Grammars(GrammarCompiler);

impl Grammars {
    pub fn load(model_dir: &Path, vocab_size: usize) -> Result<Grammars, String> {
        let info = TokenizerInfo::from_path(model_dir, Some(vocab_size), None)
            .map_err(|e| format!("{}: XGrammar tokenizer info: {e}", model_dir.display()))?;
        Ok(Grammars(GrammarCompiler::new(&info)))
    }

    /// A JSON schema's grammar. Err: the compiler's reason it cannot take the schema.
    pub fn json_schema(&self, schema: &str) -> Result<CompiledGrammar, String> {
        self.0
            .compile_json_schema(schema, None, None, None, None, None, None)
            .map_err(|e| reason(&e.to_string()).into())
    }
}

/// XGrammar's message without its log prefix: `[11:50:08] .../json_schema_converter.cc:4950: Cannot find field ...`
/// becomes `Cannot find field ...`, so a client's 400 carries no build path.
fn reason(msg: &str) -> &str {
    msg.find(".cc:")
        .or_else(|| msg.find(".h:"))
        .and_then(|i| msg[i..].find(": ").map(|j| i + j + 2))
        .map_or(msg, |k| &msg[k..])
}

/// `text` is one complete value of the grammar: every byte accepted, and the root rule matched to its end.
pub fn matches(g: &CompiledGrammar, text: &str) -> bool {
    let mut m = GrammarMatcher::new(g);
    m.accept_string(text, None) && m.is_completed()
}

#[cfg(test)]
mod tests {
    #[test]
    fn reason_drops_the_log_prefix() {
        let m = "[11:50:08] /x/xgrammar/cpp/json_schema_converter.cc:4950: Cannot find field nowhere in #/nowhere";
        assert_eq!(super::reason(m), "Cannot find field nowhere in #/nowhere");
        assert_eq!(super::reason("plain message"), "plain message");
    }
}
