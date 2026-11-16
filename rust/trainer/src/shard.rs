//! Replay shards for the learner, identical in content to the ones
//! `python/train/replay_spool.py` writes: a compressed `.npz` with
//!
//! - `planes` `(N, 4, 8, 8)`, `policies` `(N, 64)`,
//! - `outcomes`, `margins`, `policy_weights`, `value_weights` `(N,)`,
//!
//! all float32, materialised like `replay.py::materialize_record` (planes as in
//! `planes.py`; the terminal value-only samples have an empty legal plane and
//! an all-zero policy). Files are written as `<producer>_<time_ns>_<n>.tmp` and
//! renamed to `.npz` once complete, so the learner never reads half a shard.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array4;
use ndarray_npy::NpzWriter;
use reversi_mcts::GameRecord;
use reversi_mcts::PLANES_LEN;
use reversi_mcts::write_position_planes;

/// Samples waiting for the next shard.
#[derive(Default)]
pub struct Samples {
    planes: Vec<f32>,
    policies: Vec<f32>,
    outcomes: Vec<f32>,
    margins: Vec<f32>,
    policy_weights: Vec<f32>,
    value_weights: Vec<f32>,
    search_values: Vec<f32>,
}

impl Samples {
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Append every position of a finished game; returns how many.
    pub fn add_record(&mut self, record: &GameRecord) -> usize {
        for pos in &record.positions {
            let start = self.planes.len();
            self.planes.resize(start + PLANES_LEN, 0.0);
            write_position_planes(pos.board, pos.is_black, pos.legal, &mut self.planes[start..]);
            self.policies.extend_from_slice(&pos.mcts_policy);
            self.outcomes.push(pos.outcome);
            self.margins.push(pos.margin);
            self.policy_weights.push(pos.policy_weight);
            self.value_weights.push(pos.value_weight);
            self.search_values.push(pos.search_value);
        }
        record.positions.len()
    }

    /// Write the samples as one compressed `.npz` to `path`.
    fn write_npz(self, path: &Path) -> Result<(), String> {
        let n = self.len();
        let shape_err = |e: ndarray::ShapeError| e.to_string();
        let planes = Array4::from_shape_vec((n, 4, 8, 8), self.planes).map_err(shape_err)?;
        let policies = Array2::from_shape_vec((n, 64), self.policies).map_err(shape_err)?;
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut npz = NpzWriter::new_compressed(BufWriter::new(file));
        let npz_err = |e: ndarray_npy::WriteNpzError| e.to_string();
        npz.add_array("planes", &planes).map_err(npz_err)?;
        npz.add_array("policies", &policies).map_err(npz_err)?;
        for (name, values) in [
            ("outcomes", self.outcomes),
            ("margins", self.margins),
            ("policy_weights", self.policy_weights),
            ("value_weights", self.value_weights),
            ("search_values", self.search_values),
        ] {
            npz.add_array(name, &Array1::from_vec(values)).map_err(npz_err)?;
        }
        let writer = npz.finish().map_err(npz_err)?;
        let file = writer.into_inner().map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Accumulates games and writes a shard every `positions_per_shard`
/// positions (and on [`ShardWriter::flush`]).
pub struct ShardWriter {
    spool_dir: PathBuf,
    producer_id: String,
    positions_per_shard: usize,
    pending: Samples,
}

impl ShardWriter {
    pub fn new(spool_dir: PathBuf, producer_id: String, positions_per_shard: usize) -> Self {
        ShardWriter { spool_dir, producer_id, positions_per_shard, pending: Samples::default() }
    }

    /// Add a game; returns its number of positions.
    pub fn add_record(&mut self, record: &GameRecord) -> Result<usize, String> {
        let count = self.pending.add_record(record);
        if self.pending.len() >= self.positions_per_shard {
            self.flush()?;
        }
        Ok(count)
    }

    /// Write the pending samples, if any, as one shard.
    pub fn flush(&mut self) -> Result<Option<PathBuf>, String> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        let samples = std::mem::take(&mut self.pending);
        let count = samples.len();
        std::fs::create_dir_all(&self.spool_dir)
            .map_err(|e| format!("{}: {e}", self.spool_dir.display()))?;
        let now_ns = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let basename = format!("{}_{now_ns}_{count}", self.producer_id);
        let tmp = self.spool_dir.join(format!("{basename}.tmp"));
        let path = self.spool_dir.join(format!("{basename}.npz"));
        if let Err(e) = samples.write_npz(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        crate::log_event!(
            "replay",
            "shard-written",
            path = name,
            positions = count,
            producer = self.producer_id,
        );
        Ok(Some(path))
    }
}

#[cfg(test)]
mod tests {
    use ndarray_npy::NpzReader;
    use reversi_mcts::SelfPlayPool;

    use super::*;

    #[test]
    fn writes_games_with_terminal_samples() {
        let dir = std::env::temp_dir().join(format!("rs-shard-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut writer = ShardWriter::new(dir.clone(), "rs0_test".into(), 1_000_000);
        // Uniform priors: a few complete games without a network.
        let mut pool = SelfPlayPool::new(4, 4);
        let mut games = 0;
        let mut positions = 0;
        while games < 4 {
            let n = pool.collect().len();
            pool.apply(&vec![1.0 / 64.0; n * 64], &vec![0.0; n]);
            for record in pool.take_finished() {
                positions += writer.add_record(&record).unwrap();
                games += 1;
            }
        }
        let path = writer.flush().unwrap().expect("a shard");
        assert!(path.extension().is_some_and(|e| e == "npz"));
        let mut npz = NpzReader::new(File::open(&path).unwrap()).unwrap();
        let mut names = npz.names().unwrap();
        names.sort();
        assert_eq!(
            names,
            [
                "margins",
                "outcomes",
                "planes",
                "policies",
                "policy_weights",
                "search_values",
                "value_weights"
            ]
        );
        let planes: Array4<f32> = npz.by_name("planes").unwrap();
        let policies: Array2<f32> = npz.by_name("policies").unwrap();
        let policy_weights: Array1<f32> = npz.by_name("policy_weights").unwrap();
        let value_weights: Array1<f32> = npz.by_name("value_weights").unwrap();
        assert_eq!(planes.shape(), [positions, 4, 8, 8]);
        // Terminal value-only samples (kept with some probability): no legal
        // moves, no policy.
        let terminal: Vec<usize> = (0..positions)
            .filter(|&i| {
                policy_weights[i] == 0.0 && planes.slice(ndarray::s![i, 2, .., ..]).sum() == 0.0
            })
            .collect();
        for &i in &terminal {
            assert_eq!(value_weights[i], 2.0);
            assert_eq!(policies.row(i).sum(), 0.0);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
