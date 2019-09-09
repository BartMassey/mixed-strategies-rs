// Copyright © 2019 Bart Massey
// [This program is licensed under the "MIT License"]
// Please see the file LICENSE in the source
// distribution of this software for license terms.

use std::io::{self, stdin, BufRead, BufReader, Read};
use std::process::exit;

use mixed_strategies::*;

/// Read a payoff matrix in textual space-separated form.
pub fn read_matrix<T: Read>(r: T) -> io::Result<Vec<Vec<f64>>> {
    // This is tedious and awkward and error-prone but I
    // don't have a better idea. Suggestions welcome.
    let mut rows = Vec::new();
    let r = BufReader::new(r);
    for line in r.lines() {
        let cols: Vec<f64> = line?
            .split_whitespace()
            .map(|f| {
                f.trim().parse().map_err(|e| {
                    let ek = io::ErrorKind::InvalidData;
                    io::Error::new(ek, e)
                })
            })
            .collect::<io::Result<Vec<f64>>>()?;
        if cols.is_empty() {
            continue;
        }
        rows.push(cols);
    }
    assert!(!rows[0].is_empty());
    if rows.is_empty() {
        let ek = io::ErrorKind::InvalidData;
        return Err(io::Error::new(ek, "empty matrix"));
    }
    let ncols = rows[0].len();
    for r in &rows[1..] {
        if r.len() != ncols {
            let ek = io::ErrorKind::InvalidData;
            return Err(io::Error::new(ek, "ragged matrix"));
        }
    }
    Ok(rows)
}

#[test]
fn test_read_matrix() {
    let f = io::Cursor::new(b"  1 2 \n\n3 4");
    let m = read_matrix(f).unwrap();
    assert_eq!(m, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}

fn main() {
    let m = match read_matrix(stdin()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("could not read payoff matrix: {}", e);
            exit(1);
        },
    };
    let mut s = Schema::from_matrix(m);
    let soln = s.solve();
    print!("{}", soln);
}
