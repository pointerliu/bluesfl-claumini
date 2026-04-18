use std::env::{self, VarError};
use std::path::PathBuf;

use claumini_models::OpenAiCompatibleConfig;
use clap::Parser;

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

pub fn provider_config_from_env() -> OpenAiCompatibleConfig {
    load_env();
    provider_config_from_lookup(|name| env::var(name))
}

pub fn provider_config_from_lookup(
    mut lookup: impl FnMut(&str) -> Result<String, VarError>,
) -> OpenAiCompatibleConfig {
    let base_url = lookup("CLAUMINI_LIVE_API_BASE")
        .unwrap_or_else(|_| panic!("CLAUMINI_LIVE_API_BASE not set"));
    let api_key = lookup("CLAUMINI_LIVE_API_KEY")
        .unwrap_or_else(|_| panic!("CLAUMINI_LIVE_API_KEY not set"));
    let model = lookup("CLAUMINI_LIVE_MODEL").unwrap_or_else(|_| "gpt-5-mini".to_string());

    OpenAiCompatibleConfig {
        base_url,
        api_key,
        model,
        max_tokens: None,
    }
}
