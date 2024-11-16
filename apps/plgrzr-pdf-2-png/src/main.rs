use anyhow::{Context, Result};
use pdf2image::{RenderOptionsBuilder, PDF};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

fn process_pdf(entry_path: &Path, target_dir: &str) -> Result<()> {
    let filename = entry_path
        .file_name()
        .and_then(|n| n.to_str())
        .context("Invalid filename")?;

    let foldername = filename
        .split('.')
        .next()
        .context("Invalid filename format")?;
    let output_folder = format!("{}/{}", target_dir, foldername);

    fs::create_dir_all(&output_folder).context("Failed to create output directory")?;

    println!("Processing: {:?}", entry_path);

    let pdf = PDF::from_file(entry_path).context("Failed to open PDF file")?;

    let render_options = RenderOptionsBuilder::default()
        .pdftocairo(true)
        .build()
        .context("Failed to build render options")?;

    let pages = pdf
        .render(pdf2image::Pages::All, render_options)
        .context("Failed to render PDF pages")?;

    println!("{} has {} pages", filename, pages.len());

    for (i, page) in pages.iter().enumerate() {
        let output_path = format!("{}/{}_{}.jpg", output_folder, foldername, i + 1);
        page.save(&output_path)
            .with_context(|| format!("Failed to save page {} to {}", i, output_path))?;
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("Current directory: {:?}", std::env::current_dir().unwrap());
    let source_dir = "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_cleaned_dataset";
    let target_dir = "/Users/suryavirkapur/Projekts/plgrzr/data/plgrzr_output";

    // Ensure source directory exists
    if !Path::new(source_dir).exists() {
        anyhow::bail!("Source directory does not exist: {}", source_dir);
    }

    // Create target directory if it doesn't exist
    fs::create_dir_all(target_dir).context("Failed to create target directory")?;

    let mut processed = 0;
    let mut errors = 0;

    for entry in WalkDir::new(source_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.path().is_file()
                && e.path()
                    .extension()
                    .map_or(false, |ext| ext.eq_ignore_ascii_case("pdf"))
        })
    {
        println!("File exists check: {}", entry.path().exists());
        println!("Is file check: {}", entry.path().is_file());

        // Try to read the file metadata
        match fs::metadata(entry.path()) {
            Ok(metadata) => println!("File size: {} bytes", metadata.len()),
            Err(e) => println!("Error reading metadata: {}", e),
        }

        // Try to open the file directly
        match fs::File::open(entry.path()) {
            Ok(_) => println!("Successfully opened file"),
            Err(e) => println!("Error opening file: {}", e),
        }

        // Print the canonical path if possible
        if let Ok(canonical_path) = fs::canonicalize(entry.path()) {
            println!("Canonical path: {:?}", canonical_path);
        }

        match process_pdf(entry.path(), target_dir) {
            Ok(_) => {
                processed += 1;
                println!("Successfully processed: {:?}", entry.path());
            }
            Err(e) => {
                errors += 1;
                eprintln!("Error processing {:?}: {:#}", entry.path(), e);
                // Continue processing other files even if one fails
                continue;
            }
        }
    }

    println!("\nProcessing complete:");
    println!("Successfully processed: {} files", processed);
    println!("Errors encountered: {} files", errors);

    Ok(())
}
