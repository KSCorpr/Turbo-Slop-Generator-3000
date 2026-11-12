use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use sha2::{Digest, Sha256};

use crate::catalog;

pub type Progress<'a> = &'a mut dyn FnMut(String, Option<f32>);

pub fn check(cancel: &AtomicBool) -> Result<()> {
    ensure!(!cancel.load(Ordering::Relaxed), "Opération arrêtée.");
    Ok(())
}

pub fn install(root: &Path, cancel: &AtomicBool, progress: Progress) -> Result<()> {
    let client = ureq::AgentBuilder::new()
        .user_agent(concat!("TurboSlop-Native/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(45))
        .build();
    let files: Vec<_> = catalog::WEIGHTS
        .into_iter()
        .filter(|file| !file.installed(root))
        .collect();
    let total: u64 = files.iter().map(|file| file.size).sum();
    let mut completed = 0;
    for file in files {
        check(cancel)?;
        let mut report = |message, fraction: Option<f32>| {
            progress(
                format!("{} · {message}", file.label),
                fraction.map(|f| (completed as f32 + file.size as f32 * f) / total as f32),
            );
        };
        download(
            &client,
            &file.url(),
            &file.path(root),
            file.size,
            Some(file.sha256),
            file.sha256,
            cancel,
            &mut report,
        )?;
        completed += file.size;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn download(
    client: &ureq::Agent,
    url: &str,
    destination: &Path,
    size: u64,
    digest: Option<&str>,
    identity: &str,
    cancel: &AtomicBool,
    progress: Progress,
) -> Result<()> {
    check(cancel)?;
    ensure!(size > 0, "Taille de téléchargement invalide");
    ensure!(
        identity.len() >= 8 && identity.bytes().all(|b| b.is_ascii_hexdigit()),
        "Identifiant de téléchargement invalide"
    );
    if let Some(hash) = digest {
        ensure!(
            hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "SHA-256 invalide"
        );
    }
    let folder = destination
        .parent()
        .context("Dossier de destination absent")?;
    fs::create_dir_all(folder)?;
    let filename = destination
        .file_name()
        .context("Nom de destination absent")?
        .to_string_lossy();
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(folder.join(format!(".{filename}.download-lock")))?;
    FileExt::try_lock_exclusive(&lock)
        .context("Ce fichier est en cours d’installation dans une autre fenêtre")?;
    if destination.exists() {
        ensure!(
            destination.metadata()?.len() == size,
            "Le fichier existant {filename} a une taille différente. Déplace-le avant de relancer."
        );
        return Ok(());
    }
    let partial = folder.join(format!(
        ".{filename}.{}.part",
        &identity[..12.min(identity.len())]
    ));
    let mut offset = partial.metadata().map_or(0, |m| m.len());
    if offset > size {
        fs::remove_file(&partial)?;
        offset = 0;
    }
    if offset < size {
        progress(
            format!("{filename} · {:.2} / {:.2} Go", gb(offset), gb(size)),
            Some(offset as f32 / size as f32),
        );
        let mut request = client.get(url).set("Accept-Encoding", "identity");
        if offset > 0 {
            request = request.set("Range", &format!("bytes={offset}-"));
        }
        let response = request
            .call()
            .with_context(|| format!("Téléchargement de {filename} interrompu"))?;
        match response.status() {
            206 => {
                let expected = format!("bytes {offset}-{}/{size}", size - 1);
                ensure!(
                    response.header("Content-Range") == Some(expected.as_str()),
                    "Reprise HTTP invalide pour {filename}"
                );
            }
            200 => offset = 0,
            status => bail!("Réponse HTTP inattendue ({status})"),
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&partial)?;
        if offset == 0 {
            file.set_len(0)?;
        }
        let mut reader = response.into_reader();
        let mut written = offset;
        let mut buffer = vec![0; 1024 * 1024];
        let mut reported = Instant::now();
        loop {
            check(cancel)?;
            let count = reader
                .read(&mut buffer)
                .context("Connexion interrompue. Relance pour reprendre.")?;
            if count == 0 {
                break;
            }
            ensure!(
                written + count as u64 <= size,
                "Le serveur a envoyé trop de données pour {filename}"
            );
            file.write_all(&buffer[..count])?;
            written += count as u64;
            if reported.elapsed() >= Duration::from_millis(250) {
                progress(
                    format!("{filename} · {:.2} / {:.2} Go", gb(written), gb(size)),
                    Some(written as f32 / size as f32),
                );
                reported = Instant::now();
            }
        }
        ensure!(
            written == size,
            "Transfert incomplet. Relance pour reprendre."
        );
        file.sync_all()?;
    }
    check(cancel)?;
    if let Some(expected) = digest {
        progress(format!("Vérification SHA-256 de {filename}"), Some(1.0));
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(&partial)?);
        let mut buffer = vec![0; 1024 * 1024];
        let mut hash = Sha256::new();
        loop {
            check(cancel)?;
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
        }
        if !format!("{:x}", hash.finalize()).eq_ignore_ascii_case(expected) {
            fs::remove_file(&partial)?;
            bail!("SHA-256 incorrect pour {filename}. Relance le téléchargement.");
        }
    }
    check(cancel)?;
    fs::rename(&partial, destination)?;
    Ok(())
}

pub fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}
