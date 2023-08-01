use pyo3::prelude::*;
use std::collections::HashSet;

// dedup_wordlist takes a list in and returns a set of the list.
#[pyfunction]
fn dedup_wordlist(l: Vec<String>) -> PyResult<HashSet<String>> {
    let unique_strings = l.into_iter().collect();


    Ok(unique_strings)
}

/// A Python module implemented in Rust.
#[pymodule]
fn mlwl(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(dedup_wordlist, m)?)?;

    Ok(())
}