#[cfg(all(feature = "avx512-gfni", target_arch = "x86_64"))]
use core::arch::x86_64;

include!(concat!(env!("OUT_DIR"), "/table_aes.rs"));

/// The field GF(2^8).
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct Field;

impl crate::Field for Field {
    const ORDER: usize = 256;
    type Elem = u8;

    fn add(a: u8, b: u8) -> u8 {
        add(a, b)
    }

    fn mul(a: u8, b: u8) -> u8 {
        mul(a, b)
    }

    fn div(a: u8, b: u8) -> u8 {
        div(a, b)
    }

    fn exp(elem: u8, n: usize) -> u8 {
        exp(elem, n)
    }

    fn zero() -> u8 {
        0
    }

    fn one() -> u8 {
        1
    }

    fn nth_internal(n: usize) -> u8 {
        n as u8
    }

    fn mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
        #[cfg(not(feature = "avx512-gfni"))]
        {
            mul_slice(c, input, out)
        }

        #[cfg(feature = "avx512-gfni")]
        unsafe {
            mul_slice(c, input, out)
        }
    }

    fn mul_slice_add(c: u8, input: &[u8], out: &mut [u8]) {
        #[cfg(not(feature = "avx512-gfni"))]
        {
            mul_slice_xor(c, input, out)
        }

        #[cfg(feature = "avx512-gfni")]
        unsafe {
            mul_slice_xor(c, input, out)
        }
    }
}

/// Type alias of ReedSolomon over GF(2^8).
pub type ReedSolomon = crate::ReedSolomon<Field>;

/// Type alias of ShardByShard over GF(2^8).
pub type ShardByShard<'a> = crate::ShardByShard<'a, Field>;

/// Add two elements.
pub fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Subtract `b` from `a`.
#[cfg(test)]
pub fn sub(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Multiply two elements.
pub fn mul(a: u8, b: u8) -> u8 {
    MUL_TABLE[a as usize][b as usize]
}

/// Divide one element by another. `b`, the divisor, may not be 0.
pub fn div(a: u8, b: u8) -> u8 {
    if a == 0 {
        0
    } else if b == 0 {
        panic!("Divisor is 0")
    } else {
        let log_a = LOG_TABLE[a as usize];
        let log_b = LOG_TABLE[b as usize];
        let mut log_result = log_a as isize - log_b as isize;
        if log_result < 0 {
            log_result += 255;
        }
        EXP_TABLE[log_result as usize]
    }
}

/// Compute a^n.
pub fn exp(a: u8, n: usize) -> u8 {
    if n == 0 {
        1
    } else if a == 0 {
        0
    } else {
        let log_a = LOG_TABLE[a as usize];
        let mut log_result = log_a as usize * n;
        while 255 <= log_result {
            log_result -= 255;
        }
        EXP_TABLE[log_result]
    }
}

const PURE_RUST_UNROLL: isize = 4;

macro_rules! return_if_empty {
    (
        $len:expr
    ) => {
        if $len == 0 {
            return;
        }
    };
}

#[cfg(not(all(
    any(feature = "simd-accel", feature = "avx512-gfni"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_env = "msvc"),
    not(any(target_os = "android", target_os = "ios"))
)))]
pub fn mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
    mul_slice_pure_rust(c, input, out);
}

#[cfg(not(all(
    any(feature = "simd-accel", feature = "avx512-gfni"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_env = "msvc"),
    not(any(target_os = "android", target_os = "ios"))
)))]
pub fn mul_slice_xor(c: u8, input: &[u8], out: &mut [u8]) {
    mul_slice_xor_pure_rust(c, input, out);
}

fn mul_slice_pure_rust(c: u8, input: &[u8], out: &mut [u8]) {
    let mt = &MUL_TABLE[c as usize];
    let mt_ptr: *const u8 = &mt[0];

    assert_eq!(input.len(), out.len());

    let len: isize = input.len() as isize;
    return_if_empty!(len);

    let mut input_ptr: *const u8 = &input[0];
    let mut out_ptr: *mut u8 = &mut out[0];

    let mut n: isize = 0;
    unsafe {
        assert_eq!(4, PURE_RUST_UNROLL);
        if len > PURE_RUST_UNROLL {
            let len_minus_unroll = len - PURE_RUST_UNROLL;
            while n < len_minus_unroll {
                *out_ptr = *mt_ptr.offset(*input_ptr as isize);
                *out_ptr.offset(1) = *mt_ptr.offset(*input_ptr.offset(1) as isize);
                *out_ptr.offset(2) = *mt_ptr.offset(*input_ptr.offset(2) as isize);
                *out_ptr.offset(3) = *mt_ptr.offset(*input_ptr.offset(3) as isize);

                input_ptr = input_ptr.offset(PURE_RUST_UNROLL);
                out_ptr = out_ptr.offset(PURE_RUST_UNROLL);
                n += PURE_RUST_UNROLL;
            }
        }
        while n < len {
            *out_ptr = *mt_ptr.offset(*input_ptr as isize);

            input_ptr = input_ptr.offset(1);
            out_ptr = out_ptr.offset(1);
            n += 1;
        }
    }
    /* for n in 0..input.len() {
     *   out[n] = mt[input[n] as usize]
     * }
     */
}

fn mul_slice_xor_pure_rust(c: u8, input: &[u8], out: &mut [u8]) {
    let mt = &MUL_TABLE[c as usize];
    let mt_ptr: *const u8 = &mt[0];

    assert_eq!(input.len(), out.len());

    let len: isize = input.len() as isize;
    return_if_empty!(len);

    let mut input_ptr: *const u8 = &input[0];
    let mut out_ptr: *mut u8 = &mut out[0];

    let mut n: isize = 0;
    unsafe {
        assert_eq!(4, PURE_RUST_UNROLL);
        if len > PURE_RUST_UNROLL {
            let len_minus_unroll = len - PURE_RUST_UNROLL;
            while n < len_minus_unroll {
                *out_ptr ^= *mt_ptr.offset(*input_ptr as isize);
                *out_ptr.offset(1) ^= *mt_ptr.offset(*input_ptr.offset(1) as isize);
                *out_ptr.offset(2) ^= *mt_ptr.offset(*input_ptr.offset(2) as isize);
                *out_ptr.offset(3) ^= *mt_ptr.offset(*input_ptr.offset(3) as isize);

                input_ptr = input_ptr.offset(PURE_RUST_UNROLL);
                out_ptr = out_ptr.offset(PURE_RUST_UNROLL);
                n += PURE_RUST_UNROLL;
            }
        }
        while n < len {
            *out_ptr ^= *mt_ptr.offset(*input_ptr as isize);

            input_ptr = input_ptr.offset(1);
            out_ptr = out_ptr.offset(1);
            n += 1;
        }
    }
    /* for n in 0..input.len() {
     *   out[n] ^= mt[input[n] as usize];
     * }
     */
}

#[cfg(test)]
fn slice_xor(input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());

    let len: isize = input.len() as isize;
    return_if_empty!(len);

    let mut input_ptr: *const u8 = &input[0];
    let mut out_ptr: *mut u8 = &mut out[0];

    let mut n: isize = 0;
    unsafe {
        assert_eq!(4, PURE_RUST_UNROLL);
        if len > PURE_RUST_UNROLL {
            let len_minus_unroll = len - PURE_RUST_UNROLL;
            while n < len_minus_unroll {
                *out_ptr ^= *input_ptr;
                *out_ptr.offset(1) ^= *input_ptr.offset(1);
                *out_ptr.offset(2) ^= *input_ptr.offset(2);
                *out_ptr.offset(3) ^= *input_ptr.offset(3);

                input_ptr = input_ptr.offset(PURE_RUST_UNROLL);
                out_ptr = out_ptr.offset(PURE_RUST_UNROLL);
                n += PURE_RUST_UNROLL;
            }
        }
        while n < len {
            *out_ptr ^= *input_ptr;

            input_ptr = input_ptr.offset(1);
            out_ptr = out_ptr.offset(1);
            n += 1;
        }
    }
    /* for n in 0..input.len() {
     *   out[n] ^= input[n]
     * }
     */
}

#[cfg(all(
    feature = "simd-accel",
    not(feature = "avx512-gfni"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_env = "msvc"),
    not(any(target_os = "android", target_os = "ios")),
))]
unsafe extern "C" {
    fn reedsolomon_gal_mul(
        low: *const u8,
        high: *const u8,
        input: *const u8,
        out: *mut u8,
        len: libc::size_t,
    ) -> libc::size_t;

    fn reedsolomon_gal_mul_xor(
        low: *const u8,
        high: *const u8,
        input: *const u8,
        out: *mut u8,
        len: libc::size_t,
    ) -> libc::size_t;
}

#[cfg(all(
    feature = "simd-accel",
    not(feature = "avx512-gfni"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_env = "msvc"),
    not(any(target_os = "android", target_os = "ios")),
))]
pub fn mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
    let low: *const u8 = &MUL_TABLE_LOW[c as usize][0];
    let high: *const u8 = &MUL_TABLE_HIGH[c as usize][0];

    assert_eq!(input.len(), out.len());

    let input_ptr: *const u8 = &input[0];
    let out_ptr: *mut u8 = &mut out[0];
    let size: libc::size_t = input.len();

    let bytes_done: usize =
        unsafe { reedsolomon_gal_mul(low, high, input_ptr, out_ptr, size) as usize };

    mul_slice_pure_rust(c, &input[bytes_done..], &mut out[bytes_done..]);
}

#[cfg(all(
    feature = "simd-accel",
    not(feature = "avx512-gfni"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_env = "msvc"),
    not(any(target_os = "android", target_os = "ios")),
))]
pub fn mul_slice_xor(c: u8, input: &[u8], out: &mut [u8]) {
    let low: *const u8 = &MUL_TABLE_LOW[c as usize][0];
    let high: *const u8 = &MUL_TABLE_HIGH[c as usize][0];

    assert_eq!(input.len(), out.len());

    let input_ptr: *const u8 = &input[0];
    let out_ptr: *mut u8 = &mut out[0];
    let size: libc::size_t = input.len();

    let bytes_done: usize =
        unsafe { reedsolomon_gal_mul_xor(low, high, input_ptr, out_ptr, size) as usize };

    mul_slice_xor_pure_rust(c, &input[bytes_done..], &mut out[bytes_done..]);
}

#[cfg(all(feature = "avx512-gfni", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,gfni")]
pub fn mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
    let shard_len = input.len();
    let num_chunks = shard_len / 64;
    // don't handle any tail for performance and assume multiples of 64 bytes
    assert_eq!(
        shard_len % 64,
        0,
        "Shard length must be a multiple of 64 bytes"
    );
    assert_eq!(shard_len, out.len());

    let vcoeff = x86_64::_mm512_set1_epi8(c as i8);
    for chunk in 0..num_chunks {
        let offset = chunk * 64;

        // load 64 bytes of data shard once for all parity rows to improve temporal locality
        let src = unsafe { x86_64::_mm512_loadu_si512(input.as_ptr().add(offset) as *const _) };

        // multiply GF(2^8) using GFNI affine table
        let prod = x86_64::_mm512_gf2p8mul_epi8(src, vcoeff);

        // store back
        unsafe { x86_64::_mm512_storeu_si512(out.as_mut_ptr().add(offset) as *mut _, prod) };
    }
}

#[cfg(all(feature = "avx512-gfni", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,gfni")]
pub fn mul_slice_xor(c: u8, input: &[u8], out: &mut [u8]) {
    let shard_len = input.len();
    let num_chunks = shard_len / 64;
    // don't handle any tail for performance and assume multiples of 64 bytes
    assert_eq!(
        shard_len % 64,
        0,
        "Shard length must be a multiple of 64 bytes"
    );
    assert_eq!(shard_len, out.len());

    let vcoeff = x86_64::_mm512_set1_epi8(c as i8);
    for chunk in 0..num_chunks {
        let offset = chunk * 64;

        // load 64 bytes of data shard once for all parity rows to improve temporal locality
        let src = unsafe { x86_64::_mm512_loadu_si512(input.as_ptr().add(offset) as *const _) };

        // load current parity
        let dst = unsafe { x86_64::_mm512_loadu_si512(out.as_ptr().add(offset) as *const _) };

        // multiply GF(2^8) using GFNI affine table
        let prod = x86_64::_mm512_gf2p8mul_epi8(src, vcoeff);

        // accumulate into parity
        let sum = x86_64::_mm512_xor_si512(dst, prod);

        // store back
        unsafe { x86_64::_mm512_storeu_si512(out.as_mut_ptr().add(offset) as *mut _, sum) };
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;

    use super::*;
    use crate::tests::fill_random;
    use rand;

    #[test]
    fn test_associativity() {
        for a in 0..256 {
            let a = a as u8;
            for b in 0..256 {
                let b = b as u8;
                for c in 0..256 {
                    let c = c as u8;
                    let x = add(a, add(b, c));
                    let y = add(add(a, b), c);
                    assert_eq!(x, y);
                    let x = mul(a, mul(b, c));
                    let y = mul(mul(a, b), c);
                    assert_eq!(x, y);
                }
            }
        }
    }

    quickcheck! {
        fn qc_add_associativity(a: u8, b: u8, c: u8) -> bool {
            add(a, add(b, c)) == add(add(a, b), c)
        }

        fn qc_mul_associativity(a: u8, b: u8, c: u8) -> bool {
            mul(a, mul(b, c)) == mul(mul(a, b), c)
        }
    }

    #[test]
    fn test_identity() {
        for a in 0..256 {
            let a = a as u8;
            let b = sub(0, a);
            let c = sub(a, b);
            assert_eq!(c, 0);
            if a != 0 {
                let b = div(1, a);
                let c = mul(a, b);
                assert_eq!(c, 1);
            }
        }
    }

    quickcheck! {
        fn qc_additive_identity(a: u8) -> bool {
            sub(a, sub(0, a)) == 0
        }

        fn qc_multiplicative_identity(a: u8) -> bool {
            if a == 0 { true }
            else      { mul(a, div(1, a)) == 1 }
        }
    }

    #[test]
    fn test_commutativity() {
        for a in 0..256 {
            let a = a as u8;
            for b in 0..256 {
                let b = b as u8;
                let x = add(a, b);
                let y = add(b, a);
                assert_eq!(x, y);
                let x = mul(a, b);
                let y = mul(b, a);
                assert_eq!(x, y);
            }
        }
    }

    quickcheck! {
        fn qc_add_commutativity(a: u8, b: u8) -> bool {
            add(a, b) == add(b, a)
        }

        fn qc_mul_commutativity(a: u8, b: u8) -> bool {
            mul(a, b) == mul(b, a)
        }
    }

    #[test]
    fn test_distributivity() {
        for a in 0..256 {
            let a = a as u8;
            for b in 0..256 {
                let b = b as u8;
                for c in 0..256 {
                    let c = c as u8;
                    let x = mul(a, add(b, c));
                    let y = add(mul(a, b), mul(a, c));
                    assert_eq!(x, y);
                }
            }
        }
    }

    quickcheck! {
        fn qc_add_distributivity(a: u8, b: u8, c: u8) -> bool {
            mul(a, add(b, c)) == add(mul(a, b), mul(a, c))
        }
    }

    #[test]
    fn test_exp() {
        for a in 0..256 {
            let a = a as u8;
            let mut power = 1u8;
            for j in 0..256 {
                let x = exp(a, j);
                assert_eq!(x, power);
                power = mul(power, a);
            }
        }
    }

    #[test]
    fn test_slice_add() {
        let length_list = [16, 32, 34];
        for len in length_list.iter() {
            let mut input = vec![0; *len];
            fill_random(&mut input);
            let mut output = vec![0; *len];
            fill_random(&mut output);
            let mut expect = vec![0; *len];
            for i in 0..expect.len() {
                expect[i] = input[i] ^ output[i];
            }
            slice_xor(&input, &mut output);
            for i in 0..expect.len() {
                assert_eq!(expect[i], output[i]);
            }
            fill_random(&mut output);
            for i in 0..expect.len() {
                expect[i] = input[i] ^ output[i];
            }
            slice_xor(&input, &mut output);
            for i in 0..expect.len() {
                assert_eq!(expect[i], output[i]);
            }
        }
    }

    #[test]
    fn test_div_a_is_0() {
        assert_eq!(0, div(0, 100));
    }

    #[test]
    #[should_panic]
    fn test_div_b_is_0() {
        div(1, 0);
    }

    #[test]
    fn test_same_as_maybe_ffi() {
        let len = 10_003;
        for _ in 0..100 {
            let c = rand::random::<u8>();
            let mut input = vec![0; len];
            fill_random(&mut input);
            {
                let mut output = vec![0; len];
                fill_random(&mut output);
                let mut output_copy = output.clone();

                mul_slice(c, &input, &mut output);
                mul_slice(c, &input, &mut output_copy);

                assert_eq!(output, output_copy);
            }
            {
                let mut output = vec![0; len];
                fill_random(&mut output);
                let mut output_copy = output.clone();

                mul_slice_xor(c, &input, &mut output);
                mul_slice_xor(c, &input, &mut output_copy);

                assert_eq!(output, output_copy);
            }
        }
    }
}
