use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[cfg(feature = "simd-accel")]
extern crate cc;

const FIELD_SIZE: usize = 256;
const EXP_TABLE_SIZE: usize = FIELD_SIZE * 2 - 2;

const GENERATING_POLYNOMIAL: u8 = 0x1d;
const GENERATING_POLYNOMIAL_AES: u8 = 0x1b;

fn gf_mul(mut a: u8, mut b: u8, polynomial: u8) -> u8 {
    let mut r = 0u8;
    while b != 0 {
        if (b & 1) != 0 {
            r ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= polynomial;
        }
        b >>= 1;
    }
    r
}

fn gen_log_exp_tables(polynomial: u8, prim_elem: u8) -> ([u8; FIELD_SIZE], [u8; EXP_TABLE_SIZE]) {
    let mut log = [0u8; FIELD_SIZE];
    let mut exp = [0u8; EXP_TABLE_SIZE];

    let mut x: u8 = 1;
    // build exp[0..254], log for non-zero
    for (i, e) in exp.iter_mut().take(FIELD_SIZE - 1).enumerate() {
        *e = x;
        log[x as usize] = i as u8;
        x = gf_mul(x, prim_elem, polynomial);
    }

    // copy for overflow-friendly indexing
    for i in 0..255 {
        exp[255 + i] = exp[i];
    }
    (log, exp)
}

fn multiply(log_table: &[u8; FIELD_SIZE], exp_table: &[u8; EXP_TABLE_SIZE], a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        0
    } else {
        let log_a = log_table[a as usize];
        let log_b = log_table[b as usize];
        let log_result = log_a as usize + log_b as usize;
        exp_table[log_result]
    }
}

fn gen_mul_table(
    log_table: &[u8; FIELD_SIZE],
    exp_table: &[u8; EXP_TABLE_SIZE],
) -> [[u8; FIELD_SIZE]; FIELD_SIZE] {
    let mut result: [[u8; FIELD_SIZE]; FIELD_SIZE] = [[0; 256]; 256];

    for a in 0..FIELD_SIZE {
        for b in 0..FIELD_SIZE {
            result[a][b] = multiply(log_table, exp_table, a as u8, b as u8);
        }
    }

    result
}

fn gen_mul_table_half(
    log_table: &[u8; FIELD_SIZE],
    exp_table: &[u8; EXP_TABLE_SIZE],
) -> ([[u8; 16]; FIELD_SIZE], [[u8; 16]; FIELD_SIZE]) {
    let mut low: [[u8; 16]; FIELD_SIZE] = [[0; 16]; FIELD_SIZE];
    let mut high: [[u8; 16]; FIELD_SIZE] = [[0; 16]; FIELD_SIZE];

    for a in 0..low.len() {
        for b in 0..low.len() {
            let mut result = 0;
            if !(a == 0 || b == 0) {
                let log_a = log_table[a];
                let log_b = log_table[b];
                result = exp_table[log_a as usize + log_b as usize];
            }
            if (b & 0x0F) == b {
                low[a][b] = result;
            }
            if (b & 0xF0) == b {
                high[a][b >> 4] = result;
            }
        }
    }
    (low, high)
}

macro_rules! write_table {
    (1D => $file:ident, $table:ident, $name:expr, $type:expr) => {{
        let len = $table.len();
        let mut table_str = String::from(format!("pub static {}: [{}; {}] = [", $name, $type, len));

        for v in $table.iter() {
            let str = format!("{}, ", v);
            table_str.push_str(&str);
        }

        table_str.push_str("];\n");

        $file.write_all(table_str.as_bytes()).unwrap();
    }};
    (2D => $file:ident, $table:ident, $name:expr, $type:expr) => {{
        let rows = $table.len();
        let cols = $table[0].len();
        let mut table_str = String::from(format!(
            "pub static {}: [[{}; {}]; {}] = [",
            $name, $type, cols, rows
        ));

        for a in $table.iter() {
            table_str.push_str("[");
            for b in a.iter() {
                let str = format!("{}, ", b);
                table_str.push_str(&str);
            }
            table_str.push_str("],\n");
        }

        table_str.push_str("];\n");

        $file.write_all(table_str.as_bytes()).unwrap();
    }};
}

fn write_tables_gen(filename: &str, gen_poly: u8, prim_elem: u8) {
    let (log_table, exp_table) = gen_log_exp_tables(gen_poly, prim_elem);
    let mul_table = gen_mul_table(&log_table, &exp_table);

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join(filename);
    let mut f = File::create(&dest_path).unwrap();

    write_table!(1D => f, log_table,      "LOG_TABLE",      "u8");
    write_table!(1D => f, exp_table,      "EXP_TABLE",      "u8");
    write_table!(2D => f, mul_table,      "MUL_TABLE",      "u8");

    if cfg!(feature = "simd-accel") {
        let (mul_table_low, mul_table_high) = gen_mul_table_half(&log_table, &exp_table);

        write_table!(2D => f, mul_table_low,  "MUL_TABLE_LOW",  "u8");
        write_table!(2D => f, mul_table_high, "MUL_TABLE_HIGH", "u8");
    }
}

fn write_tables() {
    write_tables_gen("table.rs", GENERATING_POLYNOMIAL, 2);
    write_tables_gen("table_aes.rs", GENERATING_POLYNOMIAL_AES, 3);
}

#[cfg(all(
    feature = "simd-accel",
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_env = "msvc"),
    not(any(target_os = "android", target_os = "ios"))
))]
fn compile_simd_c() {
    let mut build = cc::Build::new();
    build.opt_level(3);

    match env::var("RUST_REED_SOLOMON_ERASURE_ARCH") {
        Ok(arch) => {
            // Use explicitly specified environment variable as architecture.
            build.flag(&format!("-march={}", arch));
        }
        Err(_error) => {
            // On x86-64 enabling Haswell architecture unlocks useful instructions and improves performance
            // dramatically while allowing it to run ony modern CPU.
            match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
                "x86_64" => {
                    build.flag(&"-march=haswell");
                }
                _ => (),
            }
        }
    }

    build
        .flag("-std=c11")
        .file("simd_c/reedsolomon.c")
        .compile("reedsolomon");
}

#[cfg(not(all(
    feature = "simd-accel",
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(target_env = "msvc"),
    not(any(target_os = "android", target_os = "ios"))
)))]
fn compile_simd_c() {}

fn main() {
    compile_simd_c();
    write_tables();
}
