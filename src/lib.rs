// Copyright © 2019 Bart Massey
// [This program is licensed under the "MIT License"]
// Please see the file LICENSE in the source
// distribution of this software for license terms.

//! Calculate an optimal mixed strategy given the payoff
//! matrix of a two-player zero-sum single-round iterated
//! simultaneous game.  Follows the method of Chapter 6 of
//! J.D. Williams' [*The Compleat Strategyst*](https://www.rand.org/content/dam/rand/pubs/commercial_books/2007/RAND_CB113-1.pdf)
//! (McGraw-Hill 1954). Throughout, the player playing the
//! rows (strategies on the left) is assumed to be the
//! "maximizer" and the player playing the columns
//! (strategies on top) is assumed to be the "minimizer".

use std::fmt::{self, Display, Formatter};
use std::io::{self, Write};
use std::ops::{Index, IndexMut};

use ndarray::{prelude::*, s};
use ordered_float::OrderedFloat;
use tabwriter::*;

/// The name of a row or column. These can appear on the
/// left or right of the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Name(pub Option<usize>);

impl Display for Name {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(name) => write!(f, "{}", name),
            None => Ok(()),
        }
    }
}

/// An `Edge` of a schema contains annotations of the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Left,
    Top,
    Right,
    Bottom,
}
use Edge::*;

/// `Labels` are names to be associated with an
#[derive(Debug, PartialEq, Eq)]
pub struct Labels(pub [Vec<Name>; 4]);

impl Index<Edge> for Labels {
    type Output = Vec<Name>;

    fn index(&self, edge: Edge) -> &Self::Output {
        &self.0[edge as usize]
    }
}

impl IndexMut<Edge> for Labels {
    fn index_mut(&mut self, edge: Edge) -> &mut Self::Output {
        &mut self.0[edge as usize]
    }
}

/// Schema describing a two-player hidden information game.
#[derive(Debug)]
pub struct Schema {
    /// Score offset to make the game fair.
    pub offset: f64,
    /// Current "divisor".
    pub d: f64,
    /// Strategy "names", given as usize labels along the
    /// left and top edge of the payoff matrix.
    pub names: Labels,
    /// Payoff matrix. Note that the dimensions include
    /// the margins.
    pub payoffs: Array2<f64>,
}

/// Display a `Schema` in tabular format.
impl Display for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "O = {:.2}, D = {:.2}", self.offset, self.d)?;

        let write_table = || -> io::Result<Vec<u8>> {
            let table = Vec::new();
            let mut tf = TabWriter::new(table).padding(1);

            for v in self.names[Top].iter() {
                write!(tf, "\t{}", v)?;
            }
            writeln!(tf)?;

            for (r, row) in self.payoffs.outer_iter().enumerate() {
                write!(tf, "{}", self.names[Left][r])?;
                for v in row {
                    write!(tf, "\t{:.2}", v)?;
                }
                write!(tf, "\t{}", self.names[Right][r])?;
                writeln!(tf)?;
            }

            for v in self.names[Bottom].iter() {
                write!(tf, "\t{}", v)?;
            }
            writeln!(tf)?;

            tf.flush().unwrap();
            Ok(tf.into_inner().unwrap())
        };

        let table = write_table().map_err(|_| fmt::Error)?;
        write!(f, "{}", std::str::from_utf8(&table).unwrap())
    }
}

/// A game solution, given as the value of the game and
/// an optimal mixed strategy for each player.
#[derive(Debug)]
pub struct Solution {
    /// Value of game.
    pub value: f64,
    /// Strategy for left player.
    pub left_strategy: Vec<f64>,
    /// Strategy for top player.
    pub top_strategy: Vec<f64>,
}

impl Schema {
    /// Take a nested-`Vec` payoff matrix and make it a `Schema`.
    pub fn from_matrix(mut rows: Vec<Vec<f64>>) -> Self {
        assert!(!rows.is_empty() && !rows[0].is_empty());
        let ncols = rows[0].len();
        for r in &rows[1..] {
            assert!(r.len() == ncols);
        }
        for r in rows.iter_mut() {
            r.push(1.0);
        }
        let ncols = ncols + 1;
        rows.push((0..ncols).map(|_| -1.0).collect());
        let nrows = rows.len();
        let mut payoffs =
            Array2::from_shape_fn((nrows, ncols), |(i, j)| rows[i][j]);
        payoffs[(nrows - 1, ncols - 1)] = 0.0;

        let mut ps = payoffs.slice_mut(s![..-1, ..-1]);
        let offset = OrderedFloat::into_inner(
            ps.iter().cloned().map(OrderedFloat).min().unwrap(),
        );
        for p in ps.iter_mut() {
            *p -= offset;
        }

        let names = Labels([
            (0..nrows - 1).map(|i| Name(Some(i))).collect(),
            (0..ncols - 1).map(|i| Name(Some(i))).collect(),
            (0..nrows - 1).map(|_| Name(None)).collect(),
            (0..ncols - 1).map(|_| Name(None)).collect(),
        ]);

        Schema {
            offset,
            d: 1.0,
            names,
            payoffs,
        }
    }

    /// Find the next pivot. Return `None` iff the
    /// schema is fully reduced.
    pub fn find_pivot(&self) -> Option<(usize, usize)> {
        // *Compleat Strategyst* p. 221
        // Step 3
        let ps = self.payoffs.slice(s![..-1, ..-1]);
        let (bot, right) = ps.dim();
        ps.axis_iter(Axis(1))
            .enumerate()
            .filter_map(|(c, col)| {
                if self.payoffs[(bot, c)] >= 0.0 {
                    return None;
                }
                col.iter()
                    .enumerate()
                    .filter(|&(_, &p)| p > 0.0)
                    .map(|(r, &p)| {
                        // pivot criterion
                        let rp = self.payoffs[(r, right)];
                        let cp = self.payoffs[(bot, c)];
                        ((r, c), -rp * cp / p)
                    })
                    .min_by_key(|&(_, p)| OrderedFloat(p))
            })
            .max_by_key(|&(_, p)| OrderedFloat(p))
            .map(|(p, _)| p)
    }

    /// Reduce using the given pivot. Assumes the pivot is
    /// valid.
    pub fn reduce(&mut self, (pr, pc): (usize, usize)) {
        // *Compleat Strategyst* p. 222—

        // Step 4
        let (nr, nc) = self.payoffs.dim();
        let p = self.payoffs[(pr, pc)];
        let d = self.d;
        assert!(d != 0.0);
        for r in 0..nr {
            for c in 0..nc {
                if r == pr || c == pc {
                    continue;
                }
                let n = self.payoffs[(r, c)];
                let nr = self.payoffs[(r, pc)];
                let nc = self.payoffs[(pr, c)];
                self.payoffs[(r, c)] = (n * p - nr * nc) / d;
            }
        }
        for r in 0..nr {
            self.payoffs[(r, pc)] = -self.payoffs[(r, pc)];
        }
        self.payoffs[(pr, pc)] = d;
        self.d = p;

        // Step 5
        let mut xchg = |e1, e2| {
            let n1 = self.names[e1][pr];
            let n2 = self.names[e2][pc];
            self.names[e2][pc] = n1;
            self.names[e1][pr] = n2;
        };
        xchg(Left, Bottom);
        xchg(Right, Top);
    }

    pub fn solve(&mut self) -> Solution {
        // Step 6
        while let Some(p) = self.find_pivot() {
            self.reduce(p);
        }

        let nr = self.names[Left].len();
        let nc = self.names[Top].len();

        let mut tr = 0.0;
        let mut left_strategy = vec![0.0; nr];
        for (r, &n) in self.names[Right].iter().enumerate() {
            if let Name(Some(sr)) = n {
                let p = self.payoffs[(r, nc)];
                assert!(p > 0.0);
                left_strategy[sr] = p;
                tr += p;
            }
        }
        for s in &mut left_strategy {
            *s /= tr;
        }

        let mut tc = 0.0;
        let mut top_strategy = vec![0.0; nc];
        for (c, &n) in self.names[Bottom].iter().enumerate() {
            if let Name(Some(sc)) = n {
                let p = self.payoffs[(nr, c)];
                assert!(p > 0.0);
                top_strategy[sc] = p;
                tc += p;
            }
        }
        for s in &mut top_strategy {
            *s /= tc;
        }

        let v = self.payoffs[(nr, nc)];
        assert!(v > 0.0);
        let value = self.d / v + self.offset;

        Solution {
            left_strategy,
            top_strategy,
            value,
        }
    }
}
