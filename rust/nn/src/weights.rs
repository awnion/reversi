use std::collections::HashMap;

/// Magic bytes "REVS" as a little-endian u32.
const MAGIC: u32 = 0x5356_4552;
const VERSION: u16 = 1;

/// Stores named f32 tensors loaded from the binary weight format.
pub struct WeightStore {
    tensors: HashMap<String, Vec<f32>>,
    shapes: HashMap<String, Vec<usize>>,
}

impl WeightStore {
    /// Parse the binary weight format.
    ///
    /// ```text
    /// [magic: u32 = 0x52455653]  ("REVS", little-endian)
    /// [version: u16 = 1]
    /// [n_tensors: u16]
    /// For each tensor:
    ///   [name_len: u8]
    ///   [name: name_len bytes UTF-8]
    ///   [ndim: u8]
    ///   [shape: ndim × u32 little-endian]
    ///   [data: product(shape) × f32 little-endian]
    /// ```
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let mut pos = 0usize;

        macro_rules! read_bytes {
            ($n:expr) => {{
                let end = pos + $n;
                if end > bytes.len() {
                    return Err(format!(
                        "unexpected end of data: need {} bytes at offset {}",
                        $n, pos
                    ));
                }
                let slice = &bytes[pos..end];
                pos = end;
                slice
            }};
        }

        macro_rules! read_u8 {
            () => {{ read_bytes!(1)[0] }};
        }

        macro_rules! read_u16_le {
            () => {{ u16::from_le_bytes(read_bytes!(2).try_into().unwrap()) }};
        }

        macro_rules! read_u32_le {
            () => {{ u32::from_le_bytes(read_bytes!(4).try_into().unwrap()) }};
        }

        // Header
        let magic = read_u32_le!();
        if magic != MAGIC {
            return Err(format!("bad magic: expected 0x{MAGIC:08X}, got 0x{magic:08X}"));
        }

        let version = read_u16_le!();
        if version != VERSION {
            return Err(format!("unsupported version: expected {VERSION}, got {version}"));
        }

        let n_tensors = read_u16_le!() as usize;
        let mut tensors = HashMap::with_capacity(n_tensors);
        let mut shapes = HashMap::with_capacity(n_tensors);

        for _ in 0..n_tensors {
            // Name
            let name_len = read_u8!() as usize;
            let name_bytes = read_bytes!(name_len);
            let name = std::str::from_utf8(name_bytes)
                .map_err(|e| format!("tensor name is not valid UTF-8: {e}"))?
                .to_owned();

            // Shape
            let ndim = read_u8!() as usize;
            let mut shape = Vec::with_capacity(ndim);
            let mut numel = 1usize;
            for _ in 0..ndim {
                let dim = read_u32_le!() as usize;
                shape.push(dim);
                numel = numel
                    .checked_mul(dim)
                    .ok_or_else(|| format!("tensor '{name}' shape overflows usize"))?;
            }

            // Data: numel × 4 bytes
            let data_bytes = read_bytes!(numel * 4);
            let mut data = Vec::with_capacity(numel);
            for chunk in data_bytes.chunks_exact(4) {
                data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }

            tensors.insert(name.clone(), data);
            shapes.insert(name, shape);
        }

        Ok(Self { tensors, shapes })
    }

    /// Look up a tensor by name. Returns `(data_slice, shape_slice)` if found.
    pub fn get(&self, name: &str) -> Option<(&[f32], &[usize])> {
        let data = self.tensors.get(name)?;
        let shape = self.shapes.get(name)?;
        Some((data.as_slice(), shape.as_slice()))
    }

    /// Write the binary weight format into a `Vec<u8>` (for testing / export).
    pub fn write(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        let mut out = Vec::new();

        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u16).to_le_bytes());

        for (name, shape, data) in tensors {
            let name_bytes = name.as_bytes();
            out.push(name_bytes.len() as u8);
            out.extend_from_slice(name_bytes);

            out.push(shape.len() as u8);
            for &dim in *shape {
                out.extend_from_slice(&(dim as u32).to_le_bytes());
            }

            for &val in *data {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }

        out
    }
}
