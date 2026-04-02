//! Serialize/Deserialize for `Array<T, D>` behind the `serde` feature flag.
//!
//! Arrays serialize as `{ "shape": [d0, d1, ...], "data": [v0, v1, ...] }`.
//! Data is emitted in row-major (C) order via `to_vec_flat()`.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::dimension::Dimension;
use crate::dtype::Element;

use super::Array;

impl<T, D> Serialize for Array<T, D>
where
    T: Element + Serialize,
    D: Dimension,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Array", 2)?;
        state.serialize_field("shape", self.shape())?;
        state.serialize_field("data", &self.to_vec_flat())?;
        state.end()
    }
}

impl<'de, T, D> Deserialize<'de> for Array<T, D>
where
    T: Element + Deserialize<'de>,
    D: Dimension,
{
    fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Shape,
            Data,
        }

        struct ArrayVisitor<T, D>(std::marker::PhantomData<(T, D)>);

        impl<'de, T, D> Visitor<'de> for ArrayVisitor<T, D>
        where
            T: Element + Deserialize<'de>,
            D: Dimension,
        {
            type Value = Array<T, D>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a struct with 'shape' and 'data' fields")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let shape: Vec<usize> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("shape"))?;
                let data: Vec<T> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("data"))?;
                build_array(shape, data)
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut shape: Option<Vec<usize>> = None;
                let mut data: Option<Vec<T>> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Shape => {
                            if shape.is_some() {
                                return Err(de::Error::duplicate_field("shape"));
                            }
                            shape = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    }
                }
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                build_array(shape, data)
            }
        }

        deserializer.deserialize_struct(
            "Array",
            &["shape", "data"],
            ArrayVisitor::<T, D>(std::marker::PhantomData),
        )
    }
}

/// Build an `Array<T, D>` from a shape vec and flat data.
fn build_array<T, D, E>(shape: Vec<usize>, data: Vec<T>) -> Result<Array<T, D>, E>
where
    T: Element,
    D: Dimension,
    E: de::Error,
{
    // Validate rank for fixed-rank dimension types
    if let Some(expected) = D::NDIM {
        if shape.len() != expected {
            return Err(de::Error::custom(format!(
                "expected {expected}D shape, got {}D ({shape:?})",
                shape.len()
            )));
        }
    }

    // Construct the dimension from the shape slice:
    // 1. Create a zero-initialized ndarray dim of the correct type
    // 2. Write the shape values into it
    // 3. Convert to ferray's Dimension via from_ndarray_dim
    use ndarray::Dimension as NdDimension;
    let mut nd_dim = D::NdarrayDim::zeros(shape.len());
    for (dst, &src) in nd_dim.as_array_view_mut().iter_mut().zip(shape.iter()) {
        *dst = src;
    }
    let dim = D::from_ndarray_dim(&nd_dim);

    Array::from_vec(dim, data).map_err(|e| de::Error::custom(e.to_string()))
}

#[cfg(test)]
mod tests {
    use crate::dimension::{Ix1, Ix2, Ix3, IxDyn};

    use super::*;

    #[test]
    fn round_trip_1d() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let json = serde_json::to_string(&arr).unwrap();
        let restored: Array<f64, Ix1> = serde_json::from_str(&json).unwrap();
        assert_eq!(arr, restored);
    }

    #[test]
    fn round_trip_2d() {
        let arr =
            Array::<f32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let json = serde_json::to_string(&arr).unwrap();
        let restored: Array<f32, Ix2> = serde_json::from_str(&json).unwrap();
        assert_eq!(arr, restored);
        assert_eq!(restored.shape(), &[2, 3]);
    }

    #[test]
    fn round_trip_3d() {
        let arr = Array::<i32, Ix3>::from_vec(Ix3::new([2, 1, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let json = serde_json::to_string(&arr).unwrap();
        let restored: Array<i32, Ix3> = serde_json::from_str(&json).unwrap();
        assert_eq!(arr, restored);
    }

    #[test]
    fn round_trip_dynamic() {
        let arr =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let json = serde_json::to_string(&arr).unwrap();
        let restored: Array<f64, IxDyn> = serde_json::from_str(&json).unwrap();
        assert_eq!(arr, restored);
    }

    #[test]
    fn rank_mismatch_error() {
        // Try to deserialize a 3D shape into a 2D array
        let json = r#"{"shape":[2,3,4],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#;
        let result = serde_json::from_str::<Array<i32, Ix2>>(json);
        assert!(result.is_err());
    }

    #[test]
    fn size_mismatch_error() {
        // Shape says 6 elements, data has 4
        let json = r#"{"shape":[2,3],"data":[1,2,3,4]}"#;
        let result = serde_json::from_str::<Array<i32, Ix2>>(json);
        assert!(result.is_err());
    }

    #[test]
    fn empty_array() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([0, 3]), vec![]).unwrap();
        let json = serde_json::to_string(&arr).unwrap();
        let restored: Array<f64, Ix2> = serde_json::from_str(&json).unwrap();
        assert_eq!(arr.shape(), restored.shape());
        assert!(restored.is_empty());
    }
}
