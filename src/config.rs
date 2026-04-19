use std::env::{self, VarError};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use claumini_core::ModelProvider;
use claumini_models::{
    ClaudeConfig, ClaudeProvider, OpenAiCompatibleConfig, OpenAiCompatibleProvider,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ProviderKind {
    /// OpenAI-compatible endpoint via OPENAI_* env vars.
    Openai,
    /// Anthropic Messages API via ANTHROPIC_* env vars.
    Anthropic,
}

#[derive(Debug, Parser)]
#[command(
    name = "bluesfl",
    about = "BFS-driven LLM bug localization over a Blues dataflow slice graph"
)]
pub struct Cli {
    /// Path to the blues slice JSON produced by sv-analyzer.
    #[arg(long)]
    pub slice: PathBuf,

    /// FST/VCD waveform file from the failing simulation.
    #[arg(long)]
    pub waveform: PathBuf,

    /// Path to the bug report (its file contents are passed to the LLM verbatim).
    #[arg(long = "bug-report")]
    pub bug_report: PathBuf,

    /// Which LLM provider to use. Reads the matching env var group.
    #[arg(long = "provider", value_enum, default_value_t = ProviderKind::Openai)]
    pub provider: ProviderKind,

    /// How many parent signals the LLM may pick per node to expand BFS with.
    #[arg(long = "top-k", default_value_t = 3)]
    pub top_k: usize,

    /// Upper bound on visited BFS nodes before giving up.
    #[arg(long = "max-iter", default_value_t = 20)]
    pub max_iter: usize,

    /// Override the starting signal (otherwise uses `target` from the slice JSON).
    #[arg(long = "start-signal")]
    pub start_signal: Option<String>,

    /// Override the starting time (otherwise uses `start_time` from the slice JSON).
    #[arg(long = "start-time")]
    pub start_time: Option<i64>,

    /// Where to write the final JSON report. Prints to stdout if omitted.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn load_env() {
    for candidate in [".env", "../.env", "../../.env", "../../claumini/.env"] {
        dotenvy::from_filename(candidate).ok();
    }
}

pub fn provider_config_from_lookup(
    mut lookup: impl FnMut(&str) -> Result<String, VarError>,
) -> OpenAiCompatibleConfig {
    let base_url = lookup("OPENAI_API_BASE").unwrap_or_else(|_| panic!("OPENAI_API_BASE not set"));
    let api_key = lookup("OPENAI_API_KEY").unwrap_or_else(|_| panic!("OPENAI_API_KEY not set"));
    let model = lookup("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5-mini".to_string());

    OpenAiCompatibleConfig {
        base_url,
        api_key,
        model,
        max_tokens: None,
    }
}

pub fn anthropic_config_from_lookup(
    mut lookup: impl FnMut(&str) -> Result<String, VarError>,
) -> (ClaudeConfig, Option<String>) {
    let api_key =
        lookup("ANTHROPIC_API_KEY").unwrap_or_else(|_| panic!("ANTHROPIC_API_KEY not set"));
    let model = lookup("ANTHROPIC_MODEL").unwrap_or_else(|_| panic!("ANTHROPIC_MODEL not set"));
    let base_url = lookup("ANTHROPIC_API_BASE").ok();
    (ClaudeConfig { api_key, model }, base_url)
}

pub fn build_provider_from_env(kind: ProviderKind) -> Result<Arc<dyn ModelProvider>> {
    load_env();
    match kind {
        ProviderKind::Openai => {
            let cfg = provider_config_from_lookup(|name| env::var(name));
            Ok(Arc::new(OpenAiCompatibleProvider::new(cfg).map_err(
                |e| anyhow!("failed to build openai provider: {e}"),
            )?))
        }
        ProviderKind::Anthropic => {
            let (cfg, base_url) = anthropic_config_from_lookup(|name| env::var(name));
            let provider = match base_url {
                Some(url) => ClaudeProvider::new_with_base_url(cfg, url),
                None => ClaudeProvider::new(cfg),
            }
            .map_err(|e| anyhow!("failed to build anthropic provider: {e}"))?;
            Ok(Arc::new(provider))
        }
    }
}
