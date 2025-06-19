/// helper function to convert an array of LE bytes to `f64`
pub(crate) fn bytes_to_float(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks(8)
        .into_iter()
        .map(|x| {
            let mut arr = [0; 8];

            if x.len() != 8 {
                panic!("missing information");
            }

            x.into_iter()
                .enumerate()
                .for_each(|(idx, val)| arr[idx] = *val);

            f64::from_le_bytes(arr)
        })
        .collect()
}
