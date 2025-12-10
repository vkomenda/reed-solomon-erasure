#![allow(dead_code)]
extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::Field;
use smallvec::SmallVec;

const ROW_ARRAY_STACK_LENGTH: usize = 64;
/// Maximum length of the matrix data array in stack, exceeding which leads to heap allocation.
const DATA_ARRAY_STACK_LENGTH: usize = 1024;

#[derive(Debug)]
pub enum Error {
    SingularMatrix,
}

macro_rules! acc {
    (
        $m:ident, $r:expr, $c:expr
    ) => {
        $m.data[$r * $m.col_count + $c]
    };
}

pub fn flatten<T>(m: Vec<Vec<T>>) -> Vec<T> {
    let mut result: Vec<T> = Vec::with_capacity(m.len() * m[0].len());
    for row in m {
        for v in row {
            result.push(v);
        }
    }
    result
}

#[derive(PartialEq, Debug, Clone)]
pub struct Matrix<F: Field> {
    row_count: usize,
    col_count: usize,
    /// Row-major flattened matrix cells.
    data: SmallVec<[F::Elem; DATA_ARRAY_STACK_LENGTH]>,
}

fn calc_matrix_row_start_end(col_count: usize, row: usize) -> (usize, usize) {
    let start = row * col_count;
    let end = start + col_count;

    (start, end)
}

pub type RowRef<'a, T> = &'a [T];
pub type RowRefArr<'a, T> = SmallVec<[RowRef<'a, T>; ROW_ARRAY_STACK_LENGTH]>;

pub type RowMut<'a, T> = &'a mut [T];
pub type RowMutArr<'a, T> = SmallVec<[RowMut<'a, T>; ROW_ARRAY_STACK_LENGTH]>;

pub struct SubmatrixMut<'a, F: Field> {
    row_count: usize,
    col_count: usize,
    rows: RowMutArr<'a, F::Elem>,
}

impl<'a, F: Field> SubmatrixMut<'a, F> {
    pub fn new(row_count: usize, col_count: usize, rows: RowMutArr<'a, F::Elem>) -> Self {
        debug_assert_eq!(rows.len(), row_count);
        debug_assert!(rows.iter().all(|row| row.len() == col_count));

        Self {
            row_count,
            col_count,
            rows,
        }
    }

    /// Write the identity matrix into the rows.
    pub fn make_identity(&mut self) {
        for (i, row) in self.rows.iter_mut().enumerate() {
            for (j, a) in row.iter_mut().enumerate() {
                *a = if i == j { F::one() } else { F::zero() }
            }
        }
    }

    /// Write the Vandermonde matrix into the rows.
    pub fn make_vandermonde(&mut self) {
        for i in 0..self.row_count {
            let row_gen = F::exp(F::generator(), i + 1);
            for j in 0..self.col_count {
                let a = F::exp(row_gen, j);
                self.rows[i][j] = a;
            }
        }
    }

    /// In-place Gaussian elimination.
    pub fn gaussian_elim(&mut self) -> Result<(), Error> {
        let n = self.row_count;

        for i in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for r in i..n {
                if self.rows[r][i] != F::zero() {
                    pivot_row = Some(r);
                    break;
                }
            }
            let pivot_row = pivot_row.ok_or(Error::SingularMatrix)?;

            // Swap to top
            if pivot_row != i {
                self.rows.swap(i, pivot_row);
            }

            // Scale pivot to 1
            let inv_pivot = F::inv(self.rows[i][i]);
            for a in self.rows[i].iter_mut() {
                *a = F::mul(*a, inv_pivot);
            }

            // Eliminate other rows
            for r in 0..n {
                if r != i && self.rows[r][i] != F::zero() {
                    let factor = self.rows[r][i];
                    for j in 0..2 * n {
                        let a = self.rows[r][j];
                        self.rows[r][j] = F::add(a, F::mul(factor, self.rows[i][j]));
                    }
                }
            }
        }

        Ok(())
    }
}

impl<F: Field> Matrix<F> {
    fn calc_row_start_end(&self, row: usize) -> (usize, usize) {
        calc_matrix_row_start_end(self.col_count, row)
    }

    // TODO: rename to `zero`
    pub fn new(rows: usize, cols: usize) -> Matrix<F> {
        let data = SmallVec::from_vec(vec![F::zero(); rows * cols]);

        Matrix {
            row_count: rows,
            col_count: cols,
            data,
        }
    }

    pub fn new_with_data(init_data: Vec<Vec<F::Elem>>) -> Matrix<F> {
        let rows = init_data.len();
        let cols = init_data[0].len();

        for r in init_data.iter() {
            if r.len() != cols {
                panic!("Inconsistent row sizes")
            }
        }

        let data = SmallVec::from_vec(flatten(init_data));

        Matrix {
            row_count: rows,
            col_count: cols,
            data,
        }
    }

    pub fn from_rows<'a>(rows: RowRefArr<'a, F::Elem>) -> Self {
        let row_count = rows.len();
        let col_count = rows[0].len();

        let mut data: SmallVec<[F::Elem; DATA_ARRAY_STACK_LENGTH]> =
            SmallVec::with_capacity(row_count * col_count);

        for row in rows.iter() {
            debug_assert_eq!(row.len(), col_count);
            data.extend_from_slice(row);
        }

        Self {
            row_count,
            col_count,
            data,
        }
    }

    #[cfg(test)]
    pub fn make_random(size: usize) -> Matrix<F>
    where
        rand::distr::StandardUniform: rand::distr::Distribution<F::Elem>,
    {
        let mut vec: Vec<Vec<F::Elem>> = vec![vec![Default::default(); size]; size];
        for v in vec.iter_mut() {
            crate::tests::fill_random(v);
        }

        Matrix::new_with_data(vec)
    }

    pub fn identity(size: usize) -> Matrix<F> {
        let mut result = Self::new(size, size);
        for i in 0..size {
            acc!(result, i, i) = F::one();
        }
        result
    }

    pub fn col_count(&self) -> usize {
        self.col_count
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn get(&self, r: usize, c: usize) -> F::Elem {
        acc!(self, r, c).clone()
    }

    pub fn set(&mut self, r: usize, c: usize, val: F::Elem) {
        acc!(self, r, c) = val;
    }

    pub fn multiply(&self, rhs: &Matrix<F>) -> Matrix<F> {
        if self.col_count != rhs.row_count {
            panic!(
                "Colomn count on left is different from row count on right, lhs: {}, rhs: {}",
                self.col_count, rhs.row_count
            )
        }
        let mut result = Self::new(self.row_count, rhs.col_count);
        for r in 0..self.row_count {
            for c in 0..rhs.col_count {
                let mut val = F::zero();
                for i in 0..self.col_count {
                    let mul = F::mul(acc!(self, r, i).clone(), acc!(rhs, i, c).clone());

                    val = F::add(val, mul);
                }
                acc!(result, r, c) = val;
            }
        }
        result
    }

    pub fn augment_with_identity(&self) -> Matrix<F> {
        let n = self.row_count;
        let mut aug = Matrix::new(n, 2 * n);

        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, F::one()); // identity
        }

        aug
    }

    pub fn submatrix<R, C>(&self, row_range: R, col_range: C) -> Self
    where
        R: std::ops::RangeBounds<usize>,
        C: std::ops::RangeBounds<usize>,
    {
        use std::ops::Bound::*;

        let row_start = match row_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let row_end = match row_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.row_count,
        };

        let col_start = match col_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let col_end = match col_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.col_count,
        };

        let row_count = row_end - row_start;
        let col_count = col_end - col_start;

        let mut m = Matrix::new(row_count, col_count);
        for (r, i) in (row_start..row_end).zip(0..) {
            for (c, j) in (col_start..col_end).zip(0..) {
                m.set(i, j, self.get(r, c));
            }
        }

        m
    }

    pub fn submatrix_mut<'a, R, C>(&'a mut self, row_range: R, col_range: C) -> SubmatrixMut<'a, F>
    where
        R: std::ops::RangeBounds<usize>,
        C: std::ops::RangeBounds<usize>,
    {
        use std::ops::Bound::*;

        let row_start = match row_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let row_end = match row_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.row_count,
        };

        let col_start = match col_range.start_bound() {
            Included(&x) => x,
            Excluded(&x) => x + 1,
            Unbounded => 0,
        };
        let col_end = match col_range.end_bound() {
            Included(&x) => x + 1,
            Excluded(&x) => x,
            Unbounded => self.col_count,
        };

        let row_count = row_end - row_start;
        let col_count = col_end - col_start;
        let base_ptr = self.data.as_mut_ptr();

        let rows: RowMutArr<'a, F::Elem> = (row_start..row_end)
            .map(|i| unsafe {
                let row_ptr = base_ptr.add(i * self.col_count + col_start);
                std::slice::from_raw_parts_mut(row_ptr, col_count)
            })
            .collect();

        SubmatrixMut::new(row_count, col_count, rows)
    }

    pub fn rows<'a>(&'a self) -> RowRefArr<'a, F::Elem> {
        self.data.chunks(self.col_count).collect()
    }

    pub fn rows_mut<'a>(&'a mut self) -> RowMutArr<'a, F::Elem> {
        self.data.chunks_mut(self.col_count).collect()
    }

    pub fn get_row(&self, row: usize) -> &[F::Elem] {
        let (start, end) = self.calc_row_start_end(row);

        &self.data[start..end]
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        let (r1_s, _) = self.calc_row_start_end(r1);
        let (r2_s, _) = self.calc_row_start_end(r2);

        if r1 == r2 {
            return;
        } else {
            for i in 0..self.col_count {
                self.data.swap(r1_s + i, r2_s + i);
            }
        }
    }

    pub fn is_square(&self) -> bool {
        self.row_count == self.col_count
    }

    pub fn invert<'a>(&'a self) -> Result<Matrix<F>, Error> {
        if !self.is_square() {
            panic!("Trying to invert a non-square matrix")
        }

        let mut aug = self.augment_with_identity();
        {
            let mut aug_rows = aug.submatrix_mut(.., ..);
            aug_rows.gaussian_elim()?;
        }

        Ok(aug.submatrix(0..aug.row_count, aug.row_count..aug.col_count))
    }

    pub fn encode_coeffs(data_shards: usize, total_shards: usize) -> Matrix<F> {
        let mut mat = Self::new(total_shards, data_shards);
        {
            let mut top_square = mat.submatrix_mut(0..data_shards, 0..data_shards);
            top_square.make_identity();
        }
        {
            let mut bottom_mat = mat.submatrix_mut(data_shards..total_shards, 0..data_shards);
            bottom_mat.make_vandermonde();
        }
        mat
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;

    use super::Matrix;
    use crate::galois_8;

    macro_rules! matrix {
        (
            $(
                [ $( $x:expr ),+ ]
            ),*
        ) => (
            Matrix::<galois_8::Field>::new_with_data(vec![ $( vec![$( $x ),*] ),* ])
        );
        ($rows:expr, $cols:expr) => (Matrix::new($rows, $cols));
    }

    #[test]
    fn test_matrix_col_count() {
        let m1 = matrix!([1, 0, 0]);
        let m2 = matrix!([0, 0, 0], [0, 0, 0]);
        let m3: Matrix<galois_8::Field> = Matrix::new(1, 4);

        assert_eq!(3, m1.col_count());
        assert_eq!(3, m2.col_count());
        assert_eq!(4, m3.col_count());
    }

    #[test]
    fn test_matrix_row_count() {
        let m1 = matrix!([1, 0, 0]);
        let m2 = matrix!([0, 0, 0], [0, 0, 0]);
        let m3: Matrix<galois_8::Field> = Matrix::new(1, 4);

        assert_eq!(1, m1.row_count());
        assert_eq!(2, m2.row_count());
        assert_eq!(1, m3.row_count());
    }

    #[test]
    fn test_matrix_swap_rows() {
        {
            let mut m1 = matrix!([1, 2, 3], [4, 5, 6], [7, 8, 9]);
            let expect = matrix!([7, 8, 9], [4, 5, 6], [1, 2, 3]);
            m1.swap_rows(0, 2);
            assert_eq!(expect, m1);
        }
        {
            let mut m1 = matrix!([1, 2, 3], [4, 5, 6], [7, 8, 9]);
            let expect = m1.clone();
            m1.swap_rows(0, 0);
            assert_eq!(expect, m1);
            m1.swap_rows(1, 1);
            assert_eq!(expect, m1);
            m1.swap_rows(2, 2);
            assert_eq!(expect, m1);
        }
    }

    #[test]
    #[should_panic]
    fn test_inconsistent_row_sizes() {
        matrix!([1, 0, 0], [0, 1], [0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn test_incompatible_multiply() {
        let m1 = matrix!([0, 1], [0, 1], [0, 1]);
        let m2 = matrix!([0, 1, 2]);

        m1.multiply(&m2);
    }

    #[test]
    fn test_augment_right_identity() {
        let m1 = matrix!([0, 1], [2, 3]);

        let m2 = m1.augment_with_identity();
        let m2_right = m2.sub_matrix(0, 2, 4, 4);

        assert_eq!(m2_right, Matrix::identity(2));
    }

    #[test]
    fn test_matrix_identity() {
        let m1 = Matrix::identity(3);
        let m2 = matrix!([1, 0, 0], [0, 1, 0], [0, 0, 1]);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_matrix_multiply() {
        let m1 = matrix!([1, 2], [3, 4]);
        let m2 = matrix!([5, 6], [7, 8]);
        let actual = m1.multiply(&m2);
        let expect = matrix!([11, 22], [19, 42]);
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_matrix_inverse_pass_cases() {
        {
            // Test case validating inverse of the input Matrix.
            let m = matrix!([56, 23, 98], [3, 100, 200], [45, 201, 123])
                .invert()
                .unwrap();
            let expect = matrix!([175, 133, 33], [130, 13, 245], [112, 35, 126]);
            assert_eq!(m, expect);
        }
        {
            // Test case validating inverse of the input Matrix.
            let m = matrix!(
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [7, 7, 6, 6, 1]
            )
            .invert()
            .unwrap();
            let expect = matrix!(
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [123, 123, 1, 122, 122],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0]
            );
            assert_eq!(m, expect);
        }
    }

    #[test]
    #[should_panic]
    fn test_matrix_inverse_non_square() {
        // Test case with a non-square matrix.
        matrix!([56, 23], [3, 100], [45, 201]).invert().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_matrix_inverse_singular() {
        matrix!([4, 2], [12, 6]).invert().unwrap();
    }
}
