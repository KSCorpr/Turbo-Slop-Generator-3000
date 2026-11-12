#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod catalog;
mod download;
mod engine;
mod studio;

use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use eframe::egui;

fn main() {
    if let Err(error) = run() {
        let message = format!("TurboSlop · {error:#}");
        eprintln!("{message}");
        #[cfg(windows)]
        rfd::MessageDialog::new()
            .set_title("TurboSlop")
            .set_description(&message)
            .set_level(rfd::MessageLevel::Error)
            .show();
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let first = args.next();
    if first.as_deref().is_some_and(|arg| arg == "--self-check") {
        let path = PathBuf::from(args.next().context("Indique le chemin du rapport")?);
        let catalog = catalog::Catalog::load()?;
        let models: Vec<_> = catalog.base_models.iter().map(|model| &model.id).collect();
        fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"), "arch": env::consts::ARCH,
                "platform": env::consts::OS, "models": models,
            }))?,
        )?;
        return Ok(());
    }
    let root = match first.as_deref() {
        None => None,
        Some(arg) if arg == "--root" => Some(PathBuf::from(
            args.next().context("Indique le dossier TurboSlop")?,
        )),
        _ => bail!("Usage : TurboSlop.exe [--root DOSSIER]"),
    };
    let app = studio::Studio::load(root)?;
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("TurboSlop · Rust")
            .with_inner_size([1320.0, 880.0])
            .with_min_inner_size([880.0, 640.0]),
        vsync: true,
        ..Default::default()
    };
    eframe::run_native(
        "TurboSlop Rust",
        native,
        Box::new(move |cc| {
            studio::style(&cc.egui_ctx);
            Ok(Box::new(app))
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(test)]
mod tests;
