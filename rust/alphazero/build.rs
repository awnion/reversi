fn main() {
    let date = std::env::var("BUILD_DATE").unwrap_or_else(|_| "dev".to_string());
    println!("cargo:rustc-env=BUILD_DATE={date}");
    if let Ok(score) = std::env::var("TOURNAMENT_SCORE") {
        println!("cargo:rustc-env=TOURNAMENT_SCORE={score}");
    }
}
