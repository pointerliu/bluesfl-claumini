use std::sync::Arc;

use async_trait::async_trait;
use claumini_core::{Tool, ToolContext, ToolDescriptor, ToolError};
use claumini_tools::ToolMetadata;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sva_core::types::{SignalNode, Timestamp};
use sva_core::wave::{SignalValue, WaveformReader, WellenReader};
use sva_core::{FuzzyMatch, SignalNotFound};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReadSignalValueInput {
    #[schemars(
        description = "Fully scoped hierarchical signal name (e.g. TOP.ibex_simple_system.u_top.u_ibex_top.u_ibex_core.if_stage_i.pc_id_o)."
    )]
    pub signal: String,
    #[schemars(description = "Simulation time at which to sample the signal value.")]
    pub time: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReadSignalValueOutput {
    #[schemars(description = "Whether the signal was found at the requested time.")]
    pub found: bool,
    #[schemars(description = "Binary-bit string representation if found.")]
    pub raw_bits: Option<String>,
    #[schemars(description = "Hexadecimal pretty form if the width lines up.")]
    pub pretty_hex: Option<String>,
    #[schemars(description = "Error explanation and fuzzy suggestions if found=false.")]
    pub error: Option<String>,
}

pub struct ReadSignalValueTool {
    reader: Arc<WellenReader>,
}

impl ReadSignalValueTool {
    pub fn new(reader: Arc<WellenReader>) -> Self {
        Self { reader }
    }
}

#[async_trait]
impl Tool for ReadSignalValueTool {
    type Input = ReadSignalValueInput;
    type Output = ReadSignalValueOutput;

    fn descriptor(&self) -> ToolDescriptor {
        ToolMetadata::new(
            "read_signal_value",
            "Read a signal's value from the captured waveform (FST/VCD) at a given simulation time. \
             Use the fully scoped signal name as it appears in the slice graph.",
        )
        .descriptor_for::<ReadSignalValueInput, ReadSignalValueOutput>()
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        let signal = SignalNode::named(input.signal.clone());
        let ts = Timestamp(input.time);
        match self.reader.signal_value_at(&signal, ts) {
            Ok(Some(SignalValue {
                raw_bits,
                pretty_hex,
            })) => Ok(ReadSignalValueOutput {
                found: true,
                raw_bits: Some(raw_bits),
                pretty_hex,
                error: None,
            }),
            Ok(None) => {
                let candidates: Vec<String> =
                    self.reader.signal_names().map(|s| s.to_string()).collect();
                let suggestions = FuzzyMatch::find_top_n(&input.signal, &candidates);
                Ok(ReadSignalValueOutput {
                    found: false,
                    raw_bits: None,
                    pretty_hex: None,
                    error: Some(
                        SignalNotFound {
                            signal: input.signal,
                            suggestions,
                        }
                        .to_string(),
                    ),
                })
            }
            Err(e) => Ok(ReadSignalValueOutput {
                found: false,
                raw_bits: None,
                pretty_hex: None,
                error: Some(format!("waveform read failed: {e}")),
            }),
        }
    }
}
