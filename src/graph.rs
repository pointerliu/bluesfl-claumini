use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use sva_core::types::{
    BlockId, BlockJson, StableSliceEdgeJson, StableSliceGraphJson, StableSliceNodeJson, Timestamp,
};

/// Thin wrapper over `StableSliceGraphJson` with predecessor/successor indices.
pub struct SliceGraph {
    pub raw: StableSliceGraphJson,
    /// For each node index, edges whose `to` equals that index (i.e. predecessors).
    incoming: Vec<Vec<usize>>,
    /// For each node index, edges whose `from` equals that index (i.e. successors).
    outgoing: Vec<Vec<usize>>,
    blocks_by_id: HashMap<BlockId, usize>,
}

impl SliceGraph {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open slice JSON: {}", path.display()))?;
        let raw: StableSliceGraphJson = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("failed to parse slice JSON: {}", path.display()))?;

        let node_count = raw.nodes.len();
        let mut incoming = vec![Vec::new(); node_count];
        let mut outgoing = vec![Vec::new(); node_count];
        for (edge_idx, edge) in raw.edges.iter().enumerate() {
            if edge.from < node_count {
                outgoing[edge.from].push(edge_idx);
            }
            if edge.to < node_count {
                incoming[edge.to].push(edge_idx);
            }
        }

        let mut blocks_by_id = HashMap::with_capacity(raw.blocks.len());
        for (idx, block) in raw.blocks.iter().enumerate() {
            blocks_by_id.insert(block.id, idx);
        }

        Ok(Self {
            raw,
            incoming,
            outgoing,
            blocks_by_id,
        })
    }

    pub fn nodes(&self) -> &[StableSliceNodeJson] {
        &self.raw.nodes
    }

    pub fn edges(&self) -> &[StableSliceEdgeJson] {
        &self.raw.edges
    }

    pub fn block_by_id(&self, id: BlockId) -> Option<&BlockJson> {
        self.blocks_by_id.get(&id).map(|&i| &self.raw.blocks[i])
    }

    pub fn block_of_node(&self, node_idx: usize) -> Option<&BlockJson> {
        match self.raw.nodes.get(node_idx)? {
            StableSliceNodeJson::Block { block_id, .. } => self.block_by_id(*block_id),
            StableSliceNodeJson::Literal { .. } => None,
        }
    }

    pub fn node_time(&self, node_idx: usize) -> Option<Timestamp> {
        match self.raw.nodes.get(node_idx)? {
            StableSliceNodeJson::Block { time, .. } => *time,
            StableSliceNodeJson::Literal { time, .. } => *time,
        }
    }

    pub fn incoming_edges(&self, node_idx: usize) -> impl Iterator<Item = &StableSliceEdgeJson> {
        self.incoming
            .get(node_idx)
            .into_iter()
            .flat_map(move |ids| ids.iter().map(move |&i| &self.raw.edges[i]))
    }

    pub fn outgoing_edges(&self, node_idx: usize) -> impl Iterator<Item = &StableSliceEdgeJson> {
        self.outgoing
            .get(node_idx)
            .into_iter()
            .flat_map(move |ids| ids.iter().map(move |&i| &self.raw.edges[i]))
    }

    /// Choose the BFS starting node.
    ///
    /// Prefer the user-supplied override; otherwise consult the slice JSON's
    /// `target` + `start_time`. Among matching block-nodes we favour the one
    /// that has no outgoing edges (i.e. the sink of the backward slice), which
    /// is the block that drives the target signal.
    pub fn resolve_start(
        &self,
        override_signal: Option<&str>,
        override_time: Option<i64>,
    ) -> Result<usize> {
        let target = override_signal
            .map(str::to_string)
            .unwrap_or_else(|| self.raw.target.clone());
        let time = override_time
            .or(self.raw.start_time.map(|t| t.0))
            .ok_or_else(|| {
                anyhow!(
                    "slice JSON has no start_time and no --start-time override was provided"
                )
            })?;
        let ts = Timestamp(time);

        let mut candidates: Vec<usize> = (0..self.raw.nodes.len())
            .filter(|&idx| matches!(self.raw.nodes[idx], StableSliceNodeJson::Block { .. }))
            .filter(|&idx| self.node_time(idx) == Some(ts))
            .collect();

        // Prefer candidates with an outgoing edge whose signal name matches the target.
        let driving_target: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&idx| {
                self.outgoing_edges(idx).any(|e| {
                    e.signal
                        .as_ref()
                        .is_some_and(|s| s.name == target)
                })
            })
            .collect();
        if !driving_target.is_empty() {
            candidates = driving_target;
        } else {
            // Otherwise prefer nodes that are sinks (no outgoing edges at all).
            let sinks: Vec<usize> = candidates
                .iter()
                .copied()
                .filter(|&idx| self.outgoing_edges(idx).next().is_none())
                .collect();
            if !sinks.is_empty() {
                candidates = sinks;
            }
        }

        candidates.into_iter().next().ok_or_else(|| {
            anyhow!(
                "no block-node at time {} matches target signal {:?}",
                time,
                target
            )
        })
    }

    /// Deduplicated list of signal names carried by incoming edges to `node_idx`.
    pub fn incoming_signal_names(&self, node_idx: usize) -> Vec<String> {
        let mut seen: HashMap<String, ()> = HashMap::new();
        let mut names = Vec::new();
        for edge in self.incoming_edges(node_idx) {
            if let Some(sig) = &edge.signal {
                if seen.insert(sig.name.clone(), ()).is_none() {
                    names.push(sig.name.clone());
                }
            }
        }
        names
    }

    /// Returns the predecessor node(s) whose outgoing edge to `node_idx` carries `signal_name`.
    pub fn predecessors_for_signal(&self, node_idx: usize, signal_name: &str) -> Vec<usize> {
        self.incoming_edges(node_idx)
            .filter(|e| {
                e.signal
                    .as_ref()
                    .is_some_and(|s| s.name == signal_name)
            })
            .map(|e| e.from)
            .collect()
    }
}
