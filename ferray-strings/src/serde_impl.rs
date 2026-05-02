//! Serialize/Deserialize for `StringArray<D>` (#279).
//!
//! Mirrors `ferray_core::array::serde_impl`: emits `{ "shape": [...],
//! "data": [...] }` with strings in row-major order. The single
//! deserialize path validates the rank against `D::NDIM` (for fixed
//! ranks) and rebuilds the dimension via the same ndarray-based
//! construction used for the numeric `Array`.

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use ferray_core::dimension::Dimension;

use crate::string_array::StringArray;

impl<D: Dimension> Serialize for StringArray<D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("StringArray", 2)?;
        state.serialize_field("shape", self.shape())?;
        state.serialize_field("data", self.as_slice())?;
        state.end()
    }
}

impl<'de, D: Dimension> Deserialize<'de> for StringArray<D> {
    fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Shape,
            Data,
        }

        struct StringArrayVisitor<D>(std::marker::PhantomData<D>);

        impl<'de, D: Dimension> Visitor<'de> for StringArrayVisitor<D> {
            type Value = StringArray<D>;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a struct with 'shape' and 'data' fields")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let shape: Vec<usize> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("shape"))?;
                let data: Vec<String> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("data"))?;
                build_string_array(shape, data)
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut shape: Option<Vec<usize>> = None;
                let mut data: Option<Vec<String>> = None;
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
                build_string_array(shape, data)
            }
        }

        deserializer.deserialize_struct(
            "StringArray",
            &["shape", "data"],
            StringArrayVisitor::<D>(std::marker::PhantomData),
        )
    }
}

fn build_string_array<D, E>(shape: Vec<usize>, data: Vec<String>) -> Result<StringArray<D>, E>
where
    D: Dimension,
    E: de::Error,
{
    if let Some(expected) = D::NDIM {
        if shape.len() != expected {
            return Err(de::Error::custom(format!(
                "expected {expected}D shape, got {}D ({shape:?})",
                shape.len()
            )));
        }
    }
    let dim = D::from_dim_slice(&shape).ok_or_else(|| {
        de::Error::custom(format!(
            "shape {shape:?} is not valid for the dimension type"
        ))
    })?;
    StringArray::from_vec(dim, data).map_err(|e| de::Error::custom(e.to_string()))
}

#[cfg(test)]
mod tests {
    use ferray_core::dimension::{Ix1, Ix2, IxDyn};

    use crate::string_array::{StringArray, array};

    #[test]
    fn round_trip_1d() {
        let a = array(&["foo", "bar", "baz"]).unwrap();
        let json = serde_json::to_string(&a).unwrap();
        let restored: StringArray<Ix1> = serde_json::from_str(&json).unwrap();
        assert_eq!(a.shape(), restored.shape());
        assert_eq!(a.as_slice(), restored.as_slice());
    }

    #[test]
    fn round_trip_2d() {
        let a = StringArray::<Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec!["a", "b", "c", "d", "e", "f"]
                .into_iter()
                .map(String::from)
                .collect(),
        )
        .unwrap();
        let json = serde_json::to_string(&a).unwrap();
        let restored: StringArray<Ix2> = serde_json::from_str(&json).unwrap();
        assert_eq!(a.shape(), restored.shape());
        assert_eq!(a.as_slice(), restored.as_slice());
    }

    #[test]
    fn round_trip_dynamic() {
        let a = StringArray::<IxDyn>::from_vec_dyn(
            &[2, 2],
            vec!["one".into(), "two".into(), "three".into(), "four".into()],
        )
        .unwrap();
        let json = serde_json::to_string(&a).unwrap();
        let restored: StringArray<IxDyn> = serde_json::from_str(&json).unwrap();
        assert_eq!(a.shape(), restored.shape());
        assert_eq!(a.as_slice(), restored.as_slice());
    }

    #[test]
    fn unicode_round_trip() {
        // Multi-byte UTF-8 strings must round-trip exactly.
        let a = array(&["こんにちは", "Здравствуйте", "🎉"]).unwrap();
        let json = serde_json::to_string(&a).unwrap();
        let restored: StringArray<Ix1> = serde_json::from_str(&json).unwrap();
        assert_eq!(a.as_slice(), restored.as_slice());
    }

    #[test]
    fn rank_mismatch_error() {
        let json = r#"{"shape":[2,2,2],"data":["a","b","c","d","e","f","g","h"]}"#;
        let result = serde_json::from_str::<StringArray<Ix2>>(json);
        assert!(result.is_err());
    }

    #[test]
    fn size_mismatch_error() {
        let json = r#"{"shape":[2,3],"data":["a","b","c"]}"#;
        let result = serde_json::from_str::<StringArray<Ix2>>(json);
        assert!(result.is_err());
    }
}
