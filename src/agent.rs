use std::sync::Arc;

use claumini_core::{AgentError, MaxTurnsPolicy, ModelProvider, RuntimeLimits};
use claumini_runtime::{PromptAgent, PromptAgentBuilder, PromptSession, ReservedRuntimeTools};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::ReadSignalValueTool;

pub const SYSTEM_PROMPT: &str = "\
You are a SystemVerilog bug-localisation expert. You are given a failing \
co-simulation bug report and a single node from a backward dataflow slice \
graph centred on a suspect signal. The node represents one SystemVerilog \
code block at a specific simulation time.

Your job at each step is to:
1. Read the code block carefully in the context of the bug report.
2. Optionally use the `read_signal_value` tool to sample any hierarchical \
   signal value from the captured waveform at any time. Prefer sampling \
   inputs of the current block at its time, and outputs at one clock period \
   earlier when reasoning about sequential logic.
3. Decide if the current block is itself the root cause of the failure.

If it IS the root cause, return { \"is_root_cause\": true, \"reasoning\": ..., \
\"next_signals\": [] }.

If it is NOT the root cause, pick up to top_k incoming signals (from the \
provided list) whose driver you most want to investigate next, and return \
them in `next_signals`, ordered most-suspicious first. Use exact signal \
names from the incoming-signals list. Never invent signal names.

Only the final JSON response will be used, so make sure it matches the \
requested schema exactly.
";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDecisionInput {
    pub bug_report: String,
    pub target_signal: String,
    pub current_signal_description: String,
    pub block_scope: String,
    pub block_type: String,
    pub block_source_file: String,
    pub block_line_range: String,
    pub block_code_snippet: String,
    pub block_time: String,
    pub incoming_signals: String,
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NodeDecision {
    /// True iff the current block is the root cause of the reported failure.
    pub is_root_cause: bool,
    /// Short reasoning for the verdict, citing relevant lines / signal values.
    pub reasoning: String,
    /// Up to `top_k` incoming signal names to expand next. Empty if root cause.
    #[serde(default)]
    pub next_signals: Vec<String>,
}

pub fn render_prompt(i: &NodeDecisionInput) -> String {
    format!(
        "# Bug report\n{}\n\n\
         # Target signal being traced backward from\n{}\n\n\
         # Current slice node\nSignal under investigation at this node: {}\n\
         Time: {}\nScope: {}\nBlock type: {}\nSource file: {}\nLines: {}\n\n\
         ```systemverilog\n{}\n```\n\n\
         # Incoming signals into this block (candidates to expand BFS)\n{}\n\n\
         # Instructions\nDecide if this block is the root cause. If not, pick \
         up to top_k = {} most suspicious incoming signal names to expand next. \
         Respond with the required JSON schema only.",
        i.bug_report,
        i.target_signal,
        i.current_signal_description,
        i.block_time,
        i.block_scope,
        i.block_type,
        i.block_source_file,
        i.block_line_range,
        i.block_code_snippet,
        i.incoming_signals,
        i.top_k,
    )
}

pub fn build_agent(
    provider: Arc<dyn ModelProvider>,
    tool: ReadSignalValueTool,
) -> Result<PromptAgent<NodeDecisionInput, NodeDecision>, AgentError> {
    let reserved = ReservedRuntimeTools::default()
        .with_finish(false)
        .with_load_skill(false)
        .with_subagents(false);

    let mut limits = RuntimeLimits::default();
    limits.model_request_timeout_ms = 300_000;
    limits.max_turns_per_session = 6;
    limits.max_turns_policy = MaxTurnsPolicy::ForceFinal { nudge: None };

    PromptAgentBuilder::new(provider)
        .system_prompt(SYSTEM_PROMPT)
        .reserved_runtime_tools(reserved)
        .limits(limits)
        .tool(tool)
        .user_prompt(|i: NodeDecisionInput| render_prompt(&i))
        .json_output::<NodeDecision>()
        .build()
}

pub async fn decide(
    agent: &PromptAgent<NodeDecisionInput, NodeDecision>,
    input: NodeDecisionInput,
    session_id: String,
) -> Result<PromptSession<NodeDecision>, AgentError> {
    agent.run(input, session_id).await
}
