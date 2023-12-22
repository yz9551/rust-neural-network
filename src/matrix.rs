use crate::Data;
use rand;
use rand::Rng;
use std::{ops, iter, convert};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Transpose<T>(T);
#[derive(Debug, Serialize, Deserialize)]
pub struct Matrix(Vec<Vec<Data>>);

impl Matrix {
    pub fn rows(&self) -> usize { self.len() }
    pub fn columns(&self) -> usize { self[0].len() }
    pub fn new(rows: usize, columns: usize) -> Self {
        let mut rng = rand::thread_rng();
        Matrix( 
            (0..rows)
                .map(|_| (0..columns)
                    .map(|_| rng.gen())
                    .collect()
                ).collect()
        )
    }
    pub fn zero(rows: usize, columns: usize) -> Self {
        Matrix(iter::repeat(iter::repeat(0.0).take(columns).collect()).take(rows).collect())
    }
    pub fn from_data(data: Vec<Vec<Data>>) -> Self {
        Matrix(data)
    }
    /*
    pub fn transpose(self) -> Matrix {
        let column = self.columns();
        let mut iters: Vec<_> = self.0.into_iter().map(|n| n.into_iter()).collect();
        // produce a vector of iterators for each of the rows
        Matrix::from_data( 
            // constructing row by row
            (0..column)  // for each original column ...
                .map(|_| {
                    iters
                        .iter_mut()
                        .map(|n| n.next().unwrap())  // take one value out
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
            )
    }
    */
    pub fn transpose(&self) -> Transpose<&Self> {
        Transpose(&self)
    }
}

impl ops::Deref for Matrix {
    type Target = Vec<Vec<Data>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl ops::DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vector(Vec<Data>);

impl Vector {
    pub fn size(&self) -> usize { self.len() }
    pub fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        Vector( 
            (0..size)
                .map(|_| rng.gen())
                .collect()
        )
    }
    pub fn zero(size: usize) -> Self {
        Vector(iter::repeat(0.0).take(size).collect())
    }
    pub fn from_data(data: Vec<Data>) -> Self {
        Vector(data)
    }
    pub fn transpose(&self) -> Transpose<&Self> {
        Transpose(&self)
    }
}

impl ops::Deref for Vector {
    type Target = Vec<Data>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl ops::DerefMut for Vector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl convert::From<Vec<Data>> for Vector {
    fn from(value: Vec<Data>) -> Vector {
        Vector::from_data(value)
    }
}

/*
impl convert::From<Vec<Vec<Data>>> for Vec<Vector> {
    fn from(value: Vec<Vec<Data>>) -> Vec<Vector> {
        value.into_iter().map(|i| i.to_vec().into()).collect()
    }
}
*/

impl ops::Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        // naive method:
        let r = self.rows();
        let s = self.columns();
        assert_eq!(self.columns(), rhs.rows());
        let c = rhs.columns();
        let mut matrix_ans: Self::Output = Matrix::zero(r, c);
        for i in 0..r {
            for j in 0..c {
                for k in 0..s {
                    matrix_ans[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }
        matrix_ans
    }
}

impl ops::Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        let r = self.rows();
        let c = self.columns();
        assert_eq!(self.rows(), rhs.rows());
        assert_eq!(self.columns(), rhs.columns());
        let mut matrix_ans: Self::Output = Matrix::zero(r, c);
        for i in 0..r {
            for j in 0..c {
                matrix_ans[i][j] = self[i][j] + rhs[i][j];
            }
        }
        matrix_ans
    }
}

impl ops::Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        let r = self.rows();
        let c = self.columns();
        assert_eq!(self.rows(), rhs.rows());
        assert_eq!(self.columns(), rhs.columns());
        let mut matrix_ans: Self::Output = Matrix::zero(r, c);
        for i in 0..r {
            for j in 0..c {
                matrix_ans[i][j] = self[i][j] - rhs[i][j];
            }
        }
        matrix_ans
    }
}

impl ops::SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Matrix) {
        let r = self.rows();
        let c = self.columns();
        //println!("{:?} -= {:?}", self, rhs);
        assert!(self.rows() == rhs.rows() && self.columns() == rhs.columns(), "Tried to do {}x{} -= {}x{}", self.rows(), self.columns(), rhs.rows(), rhs.columns());
        for i in 0..r {
            for j in 0..c {
                self[i][j] -= rhs[i][j];
            }
        }
    }
}

impl ops::Mul<Data> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Data) -> Self::Output {
        let r = self.rows();
        let c = self.columns();
        let mut matrix_ans: Self::Output = Matrix::zero(r, c);
        for i in 0..r {
            for j in 0..c {
                matrix_ans[i][j] = rhs * self[i][j];
            }
        }
        matrix_ans
    }
}

impl ops::Mul<Data> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Data) -> Self::Output {
        let r = self.rows();
        let c = self.columns();
        let mut matrix_ans: Self::Output = Matrix::zero(r, c);
        for i in 0..r {
            for j in 0..c {
                matrix_ans[i][j] = rhs * self[i][j];
            }
        }
        matrix_ans
    }
}

impl ops::Mul<&Vector> for &Matrix {
    type Output = Vector;
    fn mul(self, rhs: &Vector) -> Self::Output {
        assert_eq!(self.columns(), rhs.size(), "Tried to do {:?} * {:?}", self, rhs);
        Vector::from_data(
            self.iter()
                .map(|row| 
                    iter::zip(row, rhs.iter())
                        .map( |(a, b)| a*b )
                        .sum()
                )
                .collect()
        )
    }
}

impl ops::Add for &Vector {
    type Output = Vector;
    fn add(self, rhs: &Vector) -> Self::Output {
        assert_eq!(self.size(), rhs.size());
        Vector::from_data(
            iter::zip(self.iter(), rhs.iter())
                .map(|(a, b)| a+b)
                .collect()
        )   
    }   
}

impl ops::Mul<Data> for &Vector {
    type Output = Vector;
    fn mul(self, rhs: Data) -> Self::Output {
        Vector::from_data(
            self.iter().map(|value| (value*rhs)).collect()
        )   
    }   
}

impl ops::SubAssign<&Vector> for Vector {
    fn sub_assign(&mut self, rhs: &Vector) {
        (0..self.size())
            .for_each(|i| (self[i] -= rhs[i]));
    }
}

impl ops::Mul<Transpose<&Vector>> for &Vector {
    type Output = Matrix;
    fn mul(self, rhs: Transpose<&Vector>) -> Self::Output {
        //println!("{:?} * {:?}", self, rhs);
        Matrix::from_data(
            self.iter()
                .map(|value| (rhs.0 * (*value)).0)
                .collect()
        )
    }
}

impl ops::Mul<&Vector> for Transpose<&Matrix> {
    type Output = Vector;
    fn mul(self, rhs: &Vector) -> Self::Output {
        Vector::from_data(
            iter::zip(self.0.iter(), rhs.iter())
                .map(|(row, element)| row.iter().map(|e| e*element).sum())
                .collect()
        )   
    }
}
