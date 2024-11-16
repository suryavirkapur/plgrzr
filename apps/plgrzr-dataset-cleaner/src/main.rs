use rand::{thread_rng, Rng};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

fn generate_cuid() -> String {
    let mut rng = thread_rng();
    let random_chars: String = (0..8)
        .map(|_| {
            let chars = "abcdefghijklmnopqrstuvwxyz0123456789";
            chars.chars().nth(rng.gen_range(0..chars.len())).unwrap()
        })
        .collect();

    format!("prz_{}", random_chars)
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_dir = "/Users/suryavirkapur/Desktop/Plgrzr Dataset";
    let target_dir = "../../data/plgrzr_cleaned_dataset";

    fs::create_dir_all(target_dir)?;

    let mut processed_files = 0;

    for entry in WalkDir::new(source_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        // Check if the entry is a file and has a .pdf extension
        if entry.path().is_file()
            && entry
                .path()
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("pdf"))
        {
            let new_filename = format!("{}.pdf", generate_cuid());

            let target_path = PathBuf::from(target_dir).join(&new_filename);

            println!("Processed file {}", target_path.display());
            fs::copy(entry.path(), target_path)?;

            processed_files += 1;
        }
    }

    println!(
        "File consolidation complete! All PDFs have been moved to '{}'",
        target_dir
    );
    println!("Total files processed: {}", processed_files);

    Ok(())
}
