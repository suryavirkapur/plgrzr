use anyhow::{Context, Result};
use pdf2image::{RenderOptionsBuilder, PDF};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use walkdir::WalkDir;

fn process_pdf(entry_path: PathBuf, target_dir: &str) -> Result<()> {
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

    let pdf = PDF::from_file(&entry_path).context("Failed to open PDF file")?;

    let render_options = RenderOptionsBuilder::default()
        .pdftocairo(true)
        .build()
        .context("Failed to build render options")?;

    let pages = pdf
        .render(pdf2image::Pages::All, render_options)
        .context("Failed to render PDF pages")?;

    println!("{} has {} pages", filename, pages.len());

    pages.par_iter().enumerate().try_for_each(|(i, page)| {
        let output_path = format!("{}/{}_{}.jpg", output_folder, foldername, i + 1);
        page.save(&output_path)
            .with_context(|| format!("Failed to save page {} to {}", i, output_path))
    })?;

    Ok(())
}

fn main() -> Result<()> {
    println!("Current directory: {:?}", std::env::current_dir().unwrap());
    let source_dir = "../../data/plgrzr_cleaned_dataset";
    let target_dir = "../../data/plgrzr_output";

    if !Path::new(source_dir).exists() {
        anyhow::bail!("Source directory does not exist: {}", source_dir);
    }

    fs::create_dir_all(target_dir).context("Failed to create target directory")?;

    let processed = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    let pdf_paths: Vec<PathBuf> = WalkDir::new(source_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.path().is_file()
                && e.path()
                    .extension()
                    .map_or(false, |ext| ext.eq_ignore_ascii_case("pdf"))
        })
        .map(|e| e.path().to_owned())
        .collect();

    let total_files = pdf_paths.len();
    println!("Found {} PDF files to process", total_files);

    pdf_paths.par_iter().for_each(|path| {
        let file_exists = path.exists();
        let is_file = path.is_file();

        println!("File exists check: {}", file_exists);
        println!("Is file check: {}", is_file);

        if let Ok(metadata) = fs::metadata(path) {
            println!("File size: {} bytes", metadata.len());
        }

        match fs::File::open(path) {
            Ok(_) => println!("Successfully opened file"),
            Err(e) => println!("Error opening file: {}", e),
        }

        if let Ok(canonical_path) = fs::canonicalize(path) {
            println!("Canonical path: {:?}", canonical_path);
        }

        match process_pdf(path.to_owned(), target_dir) {
            Ok(_) => {
                processed.fetch_add(1, Ordering::SeqCst);
                println!("Successfully processed: {:?}", path);
            }
            Err(e) => {
                errors.fetch_add(1, Ordering::SeqCst);
                eprintln!("Error processing {:?}: {:#}", path, e);
            }
        }
    });

    println!("\nProcessing complete:");
    println!(
        "Successfully processed: {} files",
        processed.load(Ordering::SeqCst)
    );
    println!(
        "Errors encountered: {} files",
        errors.load(Ordering::SeqCst)
    );

    Ok(())
}
