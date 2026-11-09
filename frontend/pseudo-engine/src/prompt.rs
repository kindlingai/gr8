//! A GLM-5.3 prompt, decoded with its special tokens, cut into the parts a rule can match: the last user message, the
//! system text, the last tool result, the tools it defines, and what the reply already holds (chat_template.jinja).

use serde_json::Value;

const USER: &str = "<|user|>";
const ASSISTANT: &str = "<|assistant|>";
const SYSTEM: &str = "<|system|>";
const OBSERVATION: &str = "<|observation|>";
pub const TOOL_OPEN: &str = "<tool_call>";
pub const TOOL_CLOSE: &str = "</tool_call>";
pub const THINK_CLOSE: &str = "</think>";

/// A tool the prompt defines: its name and its parameter names, in schema order.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub params: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Prompt {
    /// The whole decoded prompt.
    pub text: String,
    /// A chat prompt (it opens an assistant turn); else raw text for /v1/completions, which the reply continues.
    pub chat: bool,
    /// The last user message.
    pub user: String,
    /// Every system message and the tool block, joined.
    pub system: String,
    /// The last tool result, when it is the prompt's last turn.
    pub tool_result: Option<String>,
    pub tools: Vec<Tool>,
    /// The reply so far already closed its think block (thinking off, or a thinking budget's continuation).
    pub think_closed: bool,
    /// The reply so far ends inside a tool call (a forced tool_choice): the text after its opener.
    pub open_call: Option<String>,
}

/// The text from `start` to the next special token (`<|`), or to the end.
fn segment(text: &str, start: usize) -> &str {
    let rest = &text[start..];
    &rest[..rest.find("<|").unwrap_or(rest.len())]
}

impl Prompt {
    pub fn parse(text: &str) -> Prompt {
        let mut p = Prompt { text: text.to_string(), ..Default::default() };
        let last_user = text.rfind(USER);
        if let Some(u) = last_user {
            p.user = segment(text, u + USER.len()).trim().to_string();
        }
        let mut sys = Vec::new();
        let mut at = 0;
        while let Some(i) = text[at..].find(SYSTEM) {
            let s = at + i + SYSTEM.len();
            sys.push(segment(text, s).trim().to_string());
            at = s;
        }
        p.system = sys.join("\n");
        if let Some(o) = text.rfind(OBSERVATION) {
            if last_user.is_none_or(|u| o > u) {
                let r = segment(text, o + OBSERVATION.len());
                p.tool_result =
                    Some(r.replace("<tool_response>", "").replace("</tool_response>", "").trim().to_string());
            }
        }
        p.tools = tools(text);
        p.chat = text.contains(ASSISTANT);
        let reply = text.rfind(ASSISTANT).map_or("", |a| &text[a + ASSISTANT.len()..]);
        p.think_closed = reply.contains(THINK_CLOSE);
        if let Some(c) = reply.rfind(TOOL_OPEN) {
            let after = &reply[c + TOOL_OPEN.len()..];
            if !after.contains(TOOL_CLOSE) {
                p.open_call = Some(after.to_string());
            }
        }
        p
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }
}

/// The tools of the prompt's last <tools> block: one JSON object a line, a function (or {"function": ...}) each.
fn tools(text: &str) -> Vec<Tool> {
    let Some(end) = text.rfind("</tools>") else { return Vec::new() };
    let Some(start) = text[..end].rfind("<tools>") else { return Vec::new() };
    text[start + "<tools>".len()..end]
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
        .filter_map(|v| {
            let f = v.get("function").unwrap_or(&v);
            let name = f.get("name")?.as_str()?.to_string();
            let params = f
                .pointer("/parameters/properties")
                .and_then(Value::as_object)
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default();
            Some(Tool { name, params })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHAT: &str = "[gMASK]<sop><|system|>\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\
{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}}}\n\
{\"type\": \"function\", \"function\": {\"name\": \"search\", \"parameters\": {\"properties\": {\"query\": {}}}}}\n\
</tools><|system|>Be brief.<|user|>What is the weather in Paris?<|assistant|><think>";

    #[test]
    fn parts_of_a_chat_prompt() {
        let p = Prompt::parse(CHAT);
        assert_eq!(p.user, "What is the weather in Paris?");
        assert!(p.system.contains("Be brief.") && p.system.contains("<tools>"));
        assert_eq!(
            p.tools,
            vec![
                Tool { name: "get_weather".into(), params: vec!["city".into(), "days".into()] },
                Tool { name: "search".into(), params: vec!["query".into()] },
            ]
        );
        assert!(p.tool_result.is_none() && !p.think_closed && p.open_call.is_none());
    }

    #[test]
    fn a_tool_result_and_what_the_reply_already_holds() {
        let after_call = format!(
            "{CHAT}</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
</tool_call><|observation|><tool_response>{{\"temp_c\": 21}}</tool_response><|assistant|><think>"
        );
        let p = Prompt::parse(&after_call);
        assert_eq!(p.tool_result.as_deref(), Some("{\"temp_c\": 21}"));
        assert!(!p.think_closed);
        let forced = format!("{CHAT}</think><tool_call>");
        let p = Prompt::parse(&forced);
        assert!(p.think_closed);
        assert_eq!(p.open_call.as_deref(), Some(""));
    }
}
