// Copyright © 2019 Bart Massey
// [This program is licensed under the "MIT License"]
// Please see the file LICENSE in the source
// distribution of this software for license terms.

//! Calculate an optimal mixed strategy given the payoff
//! matrix of a two-player zero-sum single-round iterated
//! simultaneous game.  Follows the method of Chapter 6 of
//! J.D. Williams' [*The Compleat
//! Strategyst*](https://www.rand.org/content/dam/rand/pubs/commercial_books/2007/RAND_CB113-1.pdf)
//! (McGraw-Hill 1954). Throughout, the player playing the
//! rows (strategies on the left, "Blue" in the text) is
//! assumed to be the "maximizer" and the player playing the
//! columns (strategies on top, "Red" in the text) is
//! assumed to be the "minimizer".
//! 
//! The easiest way to use this code is to just call
//! `Schema::from_matrix()` to set up a new schema and
//! then `Schema::solve()` to get a solution.
//!
//! # Examples
//!
//! The first edition of the board game DungeonQuest has a
//! combat system that is very like rock-scissors-paper. The
//! player playing the Hero secretly chooses one of "Slash",
//! "Leap Aside" or "Mighty Blow", the player playing the
//! Monster does so also. The selections are compared on a
//! literal payoff matrix to find out how much damage each
//! side does to the other. (The game is not actually
//! zero-sum — both players can be damaged on the same turn
//! — but we will pretend it is here.) The catch is that of
//! course the Hero should have an edge: examining the
//! payoff matrix you find that the Hero's mighty blow does
//! double damage!  (That's patently unfair to the monster;
//! so sad.)  The overall payoff matrix looks about like this
//!
//! ```text,no_run
//!        M  S  L
//!     M  0  2 -1
//!     S -1  0  1
//!     L  1 -1  0
//! ```
//!
//! The Hero (maximizer) plays the left edge and the
//! Monster (minimizer) plays the top.
//!
//! How much advantage does the Hero have? How should the
//! Hero and the Monster pick? Let's find out:
//!
//! ```
//! use mixed_strategies::*;
//!
//! let payoffs = vec![
//!     vec![ 0.0,  2.0, -1.0],
//!     vec![-1.0,  0.0,  1.0],
//!     vec![ 1.0, -1.0,  0.0],
//! ];
//! let mut schema = Schema::from_matrix(payoffs);
//! let soln = schema.solve();
//! assert!((soln.value - 1.0/12.0).abs() < 0.0001);
//! print!("{}", soln);
//! ```
//!
//! The output should look something like this:
//!
//! ```text,no_run
//!     value 0.083
//!     max 0:0.333 1:0.250 2:0.417
//!     min 0:0.250 1:0.333 2:0.417
//! ```
//!
//! That says that the Hero should choose Mighty Blow 1/3 of
//! the time, Slash 1/4 of the time, and Leap Aside the
//! other 5/12 of the time.  The Monster's best strategy is
//! the same, except Mighty Blow and Slash are
//! exchanged. This makes at least superficial sense. If the
//! players play this way, the expected advantage of the
//! hero is 1/12: that is, for every 12 rounds played the
//! Hero will end up about 1 point ahead.

use std::fmt::{self, Display, Formatter};
use std::io::{self, Write};
use std::ops::{Index, IndexMut};

pub use ndarray;
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
    /// Score offset to make the payoffs start all positive during
    /// internal calculations. This does not affect the calculated
    /// value of the game.
    pub offset: f64,
    /// Current "divisor" for pivot values.
    pub d: f64,
    /// Strategy "names", given as optional indices along
    /// the edges of the payoff matrix.
    pub names: Labels,
    /// Payoff matrix. Note that the dimensions include the
    /// margins.
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
    /// Strategy for left player (maximizer, "Blue").
    pub left_strategy: Vec<f64>,
    /// Strategy for top player (minimizer, "Red").
    pub top_strategy: Vec<f64>,
}

impl Display for Solution {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "value {:.3}", self.value)?;
        let mut show = |name, vals: &[f64]| {
            write!(f, "{}", name)?;
            for (i, v) in vals.iter().enumerate() {
                write!(f, " {}:{:.3}", i, v)?;
            }
            writeln!(f)
        };
        show("max", &self.left_strategy)?;
        show("min", &self.top_strategy)?;
        Ok(())
    }
}

impl Schema {
    /// Take a nested-`Vec` payoff matrix and make it a `Schema`.
    /// # Panics
    /// This code will panic if the input matrix is empty
    /// or the rows are of differing length.
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

    /// Derive a solution from the schema. Assumes
    /// that the schema is in fully-reduced form.
    ///
    /// # Panics
    /// Will panic if passed a non-reduced-form
    /// schema if basic assumptions about the
    /// schema are violated.
    pub fn solution(&self) -> Solution {
        // *Compleat Strategyst* p. 226
        // Step 6
        let nr = self.names[Left].len();
        let nc = self.names[Top].len();

        let mut tr = 0.0;
        let mut left_strategy = vec![0.0; nr];
        for (r, &n) in self.names[Right].iter().enumerate() {
            if let Name(Some(sr)) = n {
                let p = self.payoffs[(r, nc)];
                assert!(p >= 0.0);
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

    /// Find optimal strategies and game value for the given
    /// schema. This is a convenience function that proceeds
    /// by calling `find_pivot()` and `reduce()` iteratively
    /// until the schema is fully reduced, then calling
    /// `solution()` to get the solution.
    pub fn solve(&mut self) -> Solution {
        // *Compleat Strategyst* p. 226
        // Step 6
        while let Some(p) = self.find_pivot() {
            self.reduce(p);
        }

        self.solution()
    }
}
