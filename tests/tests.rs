use std::io;

use ndarray::prelude::*;

use mixed_strategies::*;

#[test]
fn test_read_matrix() {
    let f = io::Cursor::new(b"  1 2 \n\n3 4");
    let m = read_matrix(f).unwrap();
    assert_eq!(m, vec![vec![1.0,2.0],vec![3.0,4.0]]);
}

#[cfg(test)]
fn eg_schema() -> Schema {
    let m = vec![
        vec![6.0,  0.0, 3.0],
        vec![8.0, -2.0, 3.0],
        vec![4.0,  6.0, 5.0],
    ];
    Schema::from_matrix(m)
}


#[test]
fn test_from_matrix() {
    // *Compleat Strategyst* p. 220
    let s = eg_schema();

    assert_eq!(s.d, 1.0);
    assert_eq!(s.offset, -2.0);

    let names = Labels([
        vec![Name(Some(0)), Name(Some(1)), Name(Some(2))],
        vec![Name(Some(0)), Name(Some(1)), Name(Some(2))],
        vec![Name(None), Name(None), Name(None)],
        vec![Name(None), Name(None), Name(None)],
    ]);
    assert_eq!(s.names, names);

    let ms = vec![
        vec![ 8.0,  2.0,  5.0, 1.0],
        vec![10.0,  0.0,  5.0, 1.0],
        vec![ 6.0,  8.0,  7.0, 1.0],
        vec![-1.0, -1.0, -1.0, 0.0],
    ];
    let a = Array2::from_shape_fn(
        (ms.len(), ms[0].len()),
        |(i, j)| ms[i][j],
    );
    assert_eq!(s.payoffs, a);
}

#[test]
fn test_pivot() {
    // *Compleat Strategyst* p. 221
    let s = eg_schema();
    let p = s.find_pivot();
    assert_eq!(p, Some((2, 2)));
}

#[test]
fn test_reduce() {
    // *Compleat Strategyst* p. 226
    let mut s = eg_schema();
    let p = s.find_pivot().unwrap();
    s.reduce(p);

    assert_eq!(s.d, 7.0);

    let names = Labels([
        vec![Name(Some(0)), Name(Some(1)), Name(None)],
        vec![Name(Some(0)), Name(Some(1)), Name(None)],
        vec![Name(None), Name(None), Name(Some(2))],
        vec![Name(None), Name(None), Name(Some(2))],
    ]);
    assert_eq!(s.names, names);

    let ms = vec![
        vec![26.0, -26.0, -5.0, 2.0],
        vec![40.0, -40.0, -5.0, 2.0],
        vec![ 6.0,   8.0,  1.0, 1.0],
        vec![-1.0,   1.0,  1.0, 1.0],
    ];
    let a = Array2::from_shape_fn(
        (ms.len(), ms[0].len()),
        |(i, j)| ms[i][j],
    );
    assert_eq!(s.payoffs, a);
}

#[test]
fn test_soln_schema() {
    // *Compleat Strategyst* p. 229
    let mut s = eg_schema();
    while let Some(p) = s.find_pivot() {
        s.reduce(p);
    }

    assert_eq!(s.d, 40.0);

    let names = Labels([
        vec![Name(Some(0)), Name(None), Name(None)],
        vec![Name(None), Name(Some(1)), Name(None)],
        vec![Name(None), Name(Some(0)), Name(Some(2))],
        vec![Name(Some(1)), Name(None), Name(Some(2))],
    ]);
    assert_eq!(s.names, names);

    let ms = vec![
        vec![-26.0,   0.0, -10.0, 4.0],
        vec![  7.0, -40.0,  -5.0, 2.0],
        vec![ -6.0,  80.0,  10.0, 4.0],
        vec![  1.0,   0.0,   5.0, 6.0],
    ];
    let a = Array2::from_shape_fn(
        (ms.len(), ms[0].len()),
        |(i, j)| ms[i][j],
    );
    assert_eq!(s.payoffs, a);
}

#[test]
fn test_solve() {
    // *Compleat Strategyst* p. 229
    let mut s = eg_schema();
    let soln = s.solve();
    fn eqish(v1: f64, v2: f64) -> bool {
        (v1 - v2).abs() < 0.00001
    }
    assert!(eqish(soln.value, 14.0/3.0));

    let left = vec![1.0/3.0, 0.0, 2.0/3.0];
    for (&s, &e) in soln.left_strategy.iter().zip(left.iter()) {
        assert!(eqish(s, e));
    }

    let top = vec![0.0, 1.0/6.0, 5.0/6.0];
    for (&s, &e) in soln.top_strategy.iter().zip(top.iter()) {
        assert!(eqish(s, e));
    }
}
