use std::sync::Arc;

use claumini_core::{AgentError, MaxTurnsPolicy, ModelProvider, RuntimeLimits};
use claumini_runtime::{PromptAgent, PromptAgentBuilder, PromptSession, ReservedRuntimeTools};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tools::ReadSignalValueTool;

pub const SYSTEM_PROMPT: &str = "\
You are a debugging assistant for a RISC-V microprocessor design team. You \
are given a simulation fault report and a single block from a backward \
dataflow slice graph centred on a suspect signal. The block is one \
SystemVerilog code snippet observed at a specific simulation time. Your \
task is to decide, step by step, whether this code snippet is itself the \
root cause of the fault.

# Definition of root cause
- A block IS the root cause when the block's OWN logic is wrong and that \
wrong logic produces the faulty value (e.g. wrong operator, wrong bit \
slice, wrong mux arm, wrong enable condition).
- A block IS NOT the root cause when its logic is correct but one of its \
inputs already carries the faulty value — the bug then lives upstream in \
whatever drives that input. Current block only propagates the fault.

# How to investigate
1. Read the code snippet carefully in the context of the fault report. \
Decide what this block is SUPPOSED to compute for the failing scenario.
2. When you need a concrete value, call the `read_signal_value` tool with a \
fully-qualified hierarchical signal name and a simulation time. Do not \
assume signal values — sample them. Typical patterns:
   - sample block inputs at the block's time to check if they already \
carry the fault,
   - for sequential logic, sample the driver of a register one clock \
period earlier than the observed time,
   - sample the output under investigation to confirm what the waveform \
actually shows.
   Be deliberate: a few targeted samples beat a broad scan.
3. Compare expected vs. observed. If inputs are correct but the block \
output is still wrong → the block is the root cause. If an input is \
already wrong → the block is only propagating; pick that input (and any \
co-wrong inputs) as the next BFS target.

# Response rules
- Reason step by step BEFORE emitting the final JSON.
- The final JSON must match the requested schema exactly; only the final \
JSON is consumed by the BFS driver.
- `next_signals` entries must be chosen EXACTLY from the `Incoming \
signals into this block` list — never invent, rename, or truncate a \
signal.
- At most `top_k` entries in `next_signals`, ordered most-suspicious \
first. Empty when `is_root_cause` is true.

# Example 1 — block IS the root cause
Fault: `j pc + 0xa0c0` jumps to `0x000f5fc0` instead of `0x0010a140`.
Block in `ibex_alu` at time 15:
```systemverilog
assign adder_result_ext_o = $unsigned(adder_in_a) - $unsigned(adder_in_b);
assign adder_result       = adder_result_ext_o[32:1];
assign adder_result_o     = adder_result;
```
Reasoning: sampling shows `adder_in_a = 0x00100080`, \
`adder_in_b = 0x0000a0c0` at time 15 — correct operands for `pc + 0xa0c0`. \
Expected sum is `0x0010a140`. The block uses `-` instead of `+`, and takes \
bits `[32:1]` (a shift by 1). Inputs are correct; the block's own logic is \
wrong.
Final JSON:
```json
{\"is_root_cause\": true, \"reasoning\": \"inputs sampled correct; block \
uses `-` in place of `+` and slices [32:1], producing the observed \
0x000f5fc0\", \"next_signals\": []}
```

# Example 2 — block is only a propagator
Fault: same PC mismatch as above.
Block in `ibex_if_stage` at time 19:
```systemverilog
always_ff @(posedge clk_i) begin
  if (if_id_pipe_reg_we) begin
    pc_id_o <= pc_if_o;
    // ... other non-PC assignments ...
  end
end
```
Reasoning: at time 17 (one clock earlier), sampling shows \
`if_id_pipe_reg_we = 1` and `pc_if_o = 0x000f5fc0` — the enable latched \
the already-wrong `pc_if_o`. The register logic is a plain flop and is \
correct; the fault arrived via `pc_if_o`.
Final JSON:
```json
{\"is_root_cause\": false, \"reasoning\": \"flop is correct; wrong value \
arrives on pc_if_o at t=17 and is latched at t=19\", \"next_signals\": \
[\"pc_if_o\"]}
```
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
    /// Optional corrective feedback from a previous attempt whose
    /// `next_signals` contained names outside the allowed list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
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
    let feedback_section = match i.feedback.as_deref() {
        Some(note) => format!(
            "\n# Correction required (previous attempt invalid)\n{}\n",
            note
        ),
        None => String::new(),
    };
    format!(
        "# Simulation fault information\n{}\n\n\
         # Backward-slice target\nThe BFS started from this signal: {}\n\n\
         # Current block under inspection\n\
         Signal under investigation at this block: {}\n\
         Observed time: {}\n\
         Scope: {}\n\
         Block type: {}\n\
         Source file: {}\n\
         Lines: {}\n\n\
         ```systemverilog\n{}\n```\n\n\
         # Incoming signals into this block (BFS candidates)\n\
         Signal values are NOT given here — use the `read_signal_value` tool \
         to sample any value you need (signal name + time). Pick names \
         EXACTLY from this list when populating `next_signals`:\n{}\n\
         {}\n\
         # Task\n\
         Decide whether this block is the root cause of the fault.\n\
         - If its own logic is wrong and produces the bad output → \
         `is_root_cause = true`, `next_signals = []`.\n\
         - If inputs already arrive wrong and the block only propagates → \
         `is_root_cause = false`, and return up to top_k = {} most \
         suspicious incoming signal names in `next_signals`, most-suspicious \
         first.\n\
         Reason step by step before emitting the final JSON. Only the final \
         JSON (matching the requested schema) will be consumed.",
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
        feedback_section,
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

    let limits = RuntimeLimits {
        model_request_timeout_ms: 300_000,
        max_turns_per_session: 6,
        max_turns_policy: MaxTurnsPolicy::ForceFinal { nudge: None },
        ..Default::default()
    };

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
