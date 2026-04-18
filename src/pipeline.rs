use std::collections::{HashSet, VecDeque};
use std::fs;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use claumini_core::{Message, MessageRole, Payload};
use serde::Serialize;
use serde_json::Value;
use sva_core::types::BlockJson;
use sva_core::wave::WellenReader;

use crate::agent::{NodeDecision, NodeDecisionInput, build_agent, decide, render_prompt};
use crate::config::{Cli, provider_config_from_env};
use crate::graph::SliceGraph;
use crate::tools::ReadSignalValueTool;

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallLog {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Serialize, Clone)]
pub struct TurnRecord {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallLog>,
}

fn payload_to_turn_content(p: &Payload) -> (Option<String>, Option<Value>) {
    match p {
        Payload::Text(t) => (Some(t.clone()), None),
        Payload::Json(v) => (None, Some(v.clone())),
        Payload::Artifact(id) => (Some(format!("<artifact #{}>", id.get())), None),
    }
}

fn record_turn(msg: &Message) -> TurnRecord {
    let (text, json) = payload_to_turn_content(&msg.content);
    TurnRecord {
        role: match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        }
        .to_string(),
        name: msg.name.clone(),
        text,
        json,
        tool_calls: msg
            .tool_calls
            .iter()
            .map(|tc| ToolCallLog {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            })
            .collect(),
    }
}

fn log_turn(step: usize, node_idx: usize, turn: &TurnRecord) {
    let preview = turn
        .text
        .as_deref()
        .map(|t| {
            let first_line = t.lines().next().unwrap_or("");
            if t.len() > 400 {
                format!("{}... ({} bytes)", &first_line.chars().take(200).collect::<String>(), t.len())
            } else {
                t.to_string()
            }
        })
        .or_else(|| {
            turn.json.as_ref().map(|v| {
                let s = serde_json::to_string(v).unwrap_or_default();
                if s.len() > 400 { format!("{}...", &s[..400]) } else { s }
            })
        })
        .unwrap_or_default();

    if !turn.tool_calls.is_empty() {
        for tc in &turn.tool_calls {
            tracing::info!(
                step,
                node_idx,
                role = %turn.role,
                tool = %tc.name,
                args = %tc.arguments,
                "LLM tool_call"
            );
        }
    }
    if !preview.is_empty() {
        tracing::info!(
            step,
            node_idx,
            role = %turn.role,
            name = ?turn.name,
            "LLM turn: {}",
            preview
        );
    }
}

#[derive(Debug, Serialize)]
pub struct VisitedStep {
    pub step: usize,
    pub node_idx: usize,
    pub block_id: u64,
    pub source_file: String,
    pub lines: String,
    pub time: Option<i64>,
    pub scope: String,
    pub signal_under_investigation: String,
    pub prompt: String,
    pub transcript: Vec<TurnRecord>,
    pub decision: NodeDecision,
}

#[derive(Debug, Serialize)]
pub struct RootCauseReport {
    pub node_idx: usize,
    pub block_id: u64,
    pub scope: String,
    pub source_file: String,
    pub lines: String,
    pub code_snippet: String,
    pub reasoning: String,
}

#[derive(Debug, Serialize)]
pub struct BluesReport {
    pub target: String,
    pub start_node_idx: usize,
    pub start_time: Option<i64>,
    pub visited_count: usize,
    pub root_cause: Option<RootCauseReport>,
    pub trace: Vec<VisitedStep>,
}

pub async fn run(cli: Cli) -> Result<BluesReport> {
    // ---- load inputs ----
    let graph = SliceGraph::load(&cli.slice)?;
    let reader = Arc::new(
        WellenReader::open(&cli.waveform)
            .with_context(|| format!("failed to open waveform {}", cli.waveform.display()))?,
    );
    let bug_report = fs::read_to_string(&cli.bug_report).with_context(|| {
        format!("failed to read bug report {}", cli.bug_report.display())
    })?;

    // ---- provider ----
    let config = provider_config_from_env();
    let provider: Arc<dyn claumini_core::ModelProvider> = Arc::new(
        claumini_models::OpenAiCompatibleProvider::new(config)
            .map_err(|e| anyhow!("failed to build provider: {e}"))?,
    );

    // ---- resolve start node ----
    let start_node = graph.resolve_start(
        cli.start_signal.as_deref(),
        cli.start_time,
    )?;

    let target_signal = cli
        .start_signal
        .clone()
        .unwrap_or_else(|| graph.raw.target.clone());

    tracing::info!(
        target = %target_signal,
        start_node,
        "resolved start node"
    );

    // ---- BFS ----
    let tool = ReadSignalValueTool::new(Arc::clone(&reader));
    let agent = build_agent(Arc::clone(&provider), tool)
        .map_err(|e| anyhow!("failed to build agent: {e}"))?;

    let mut queue: VecDeque<(usize, String)> = VecDeque::new();
    queue.push_back((start_node, target_signal.clone()));
    let mut visited: HashSet<usize> = HashSet::new();
    let mut trace: Vec<VisitedStep> = Vec::new();
    let mut root_cause: Option<RootCauseReport> = None;
    let mut step: usize = 0;

    while let Some((node_idx, signal_under_investigation)) = queue.pop_front() {
        if !visited.insert(node_idx) {
            continue;
        }
        if step >= cli.max_iter {
            tracing::warn!(max_iter = cli.max_iter, "reached max_iter; stopping BFS");
            break;
        }
        step += 1;

        let Some(block) = graph.block_of_node(node_idx) else {
            tracing::debug!(node_idx, "skipping non-block node");
            continue;
        };

        let incoming_signals = graph.incoming_signal_names(node_idx);
        let incoming_signals_text = if incoming_signals.is_empty() {
            "(none — this block has no incoming data dependencies in the slice)".to_string()
        } else {
            incoming_signals
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}. {}", i + 1, s))
                .collect::<Vec<_>>()
                .join("\n")
        };

        let time = graph.node_time(node_idx);
        let block_time = time
            .map(|t| t.0.to_string())
            .unwrap_or_else(|| "(untimed)".to_string());

        let input = NodeDecisionInput {
            bug_report: bug_report.clone(),
            target_signal: target_signal.clone(),
            current_signal_description: signal_under_investigation.clone(),
            block_scope: block.scope.clone(),
            block_type: block.block_type.clone(),
            block_source_file: block.source_file.clone(),
            block_line_range: format!("{}-{}", block.line_start, block.line_end),
            block_code_snippet: block.code_snippet.clone(),
            block_time: block_time.clone(),
            incoming_signals: incoming_signals_text,
            top_k: cli.top_k,
        };

        let session_id = format!("bluesfl-step-{step}-node-{node_idx}");
        let prompt_text = render_prompt(&input);
        tracing::info!(
            step,
            node_idx,
            block_id = block.id.0,
            source = %block.source_file,
            "querying LLM for node decision"
        );
            tracing::info!(
                step,
                node_idx,
                role = "user",
                "LLM prompt:\n{}",
                prompt_text
            );

        let session = decide(&agent, input, session_id)
            .await
            .map_err(|e| anyhow!("agent failed at step {step}: {e}"))?;
        let decision = session.output.clone();

        let transcript: Vec<TurnRecord> = session
            .session
            .transcript
            .iter()
            .map(record_turn)
            .collect();
        for turn in &transcript {
            log_turn(step, node_idx, turn);
        }
        tracing::info!(
            step,
            node_idx,
            is_root_cause = decision.is_root_cause,
            next_signals = ?decision.next_signals,
            "LLM decision: {}",
            decision.reasoning
        );

        let visited_step = VisitedStep {
            step,
            node_idx,
            block_id: block.id.0,
            source_file: block.source_file.clone(),
            lines: format!("{}-{}", block.line_start, block.line_end),
            time: time.map(|t| t.0),
            scope: block.scope.clone(),
            signal_under_investigation: signal_under_investigation.clone(),
            prompt: prompt_text,
            transcript,
            decision: decision.clone(),
        };
        trace.push(visited_step);

        if decision.is_root_cause {
            root_cause = Some(root_cause_report(node_idx, block, &decision.reasoning));
            tracing::info!(
                step,
                node_idx,
                block_id = block.id.0,
                "root cause identified"
            );
            break;
        }

        // enqueue top-k parents
        for sig_name in decision.next_signals.iter().take(cli.top_k) {
            let preds = graph.predecessors_for_signal(node_idx, sig_name);
            if preds.is_empty() {
                tracing::debug!(signal = %sig_name, "no predecessor node for selected signal");
            }
            for pred in preds {
                if !visited.contains(&pred) {
                    queue.push_back((pred, sig_name.clone()));
                }
            }
        }
    }

    let report = BluesReport {
        target: graph.raw.target.clone(),
        start_node_idx: start_node,
        start_time: graph.raw.start_time.map(|t| t.0),
        visited_count: trace.len(),
        root_cause,
        trace,
    };

    if let Some(out_path) = &cli.output {
        let json = serde_json::to_string_pretty(&report)?;
        fs::write(out_path, json)
            .with_context(|| format!("failed to write {}", out_path.display()))?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    Ok(report)
}

fn root_cause_report(node_idx: usize, block: &BlockJson, reasoning: &str) -> RootCauseReport {
    RootCauseReport {
        node_idx,
        block_id: block.id.0,
        scope: block.scope.clone(),
        source_file: block.source_file.clone(),
        lines: format!("{}-{}", block.line_start, block.line_end),
        code_snippet: block.code_snippet.clone(),
        reasoning: reasoning.to_string(),
    }
}
