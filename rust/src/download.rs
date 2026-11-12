use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use url::Url;

use crate::catalog::{self, Model, RemoteWeight, Selection, Weight};

pub type Progress<'a> = &'a mut dyn FnMut(String, Option<f32>);

#[derive(Deserialize)]
struct Repo {
    sha: String,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    sha256: String,
}

pub fn client() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .user_agent(concat!("TurboSlop-Rust/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(45))
        .build()
}

pub fn check(cancel: &AtomicBool) -> Result<()> {
    ensure!(!cancel.load(Ordering::Relaxed), "Opération arrêtée.");
    Ok(())
}

pub fn catalog_weights(
    model: &Model,
    cancel: &AtomicBool,
) -> Result<BTreeMap<String, Vec<RemoteWeight>>> {
    let client = client();
    let mut repos = HashMap::<String, Repo>::new();
    let mut result = BTreeMap::new();
    for (role, source) in &model.sources {
        let mut entries = Vec::new();
        for repo in source.repositories() {
            check(cancel)?;
            if !repos.contains_key(repo) {
                let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
                let data: Repo = client
                    .get(&url)
                    .call()
                    .with_context(|| format!("Catalogue Hugging Face indisponible pour {repo}"))?
                    .into_json()?;
                ensure!(
                    data.sha.len() == 40 && data.sha.bytes().all(|b| b.is_ascii_hexdigit()),
                    "Révision Hugging Face invalide pour {repo}"
                );
                repos.insert(repo.to_owned(), data);
            }
            let info = &repos[repo];
            entries.extend(info.siblings.iter().filter_map(|file| {
                let weight = Weight {
                    repo: repo.into(),
                    name: file.rfilename.clone(),
                };
                let size = file.size.filter(|size| *size > 0)?;
                source.accepts(&weight).then(|| RemoteWeight {
                    weight,
                    revision: info.sha.clone(),
                    size,
                    sha256: file.lfs.as_ref().map(|lfs| lfs.sha256.clone()),
                })
            }));
        }
        entries.sort_by_key(|entry| source.priority(&entry.weight));
        entries.dedup_by(|a, b| a.weight == b.weight);
        result.insert(role.clone(), entries);
    }
    Ok(result)
}

pub fn install_model(
    model: &Model,
    root: &Path,
    remote: &BTreeMap<String, Vec<RemoteWeight>>,
    selection: &Selection,
    install_vision: bool,
    cancel: &AtomicBool,
    progress: Progress,
) -> Result<()> {
    let local = catalog::installed(model, root);
    let client = client();
    let mut downloads = Vec::new();
    for (role, _) in catalog::ROLES {
        if !model.needs(role, false, install_vision) {
            continue;
        }
        if let Some(existing) = catalog::chosen(role, selection, &local)
            && catalog::present(&existing.path(root))
        {
            continue;
        }
        let candidates = remote
            .get(*role)
            .with_context(|| format!("Choix de poids absent pour {role}"))?;
        let chosen = match selection.get(*role) {
            Some(selected) => candidates.iter().find(|file| &file.weight == selected),
            None => candidates.first(),
        }
        .with_context(|| format!("Fichier indisponible pour {role}"))?;
        downloads.push(chosen);
    }
    let total = downloads.len();
    for (index, file) in downloads.into_iter().enumerate() {
        let mut report = |message, fraction: Option<f32>| {
            progress(
                format!("{}/{} · {message}", index + 1, total),
                fraction.map(|n| (index as f32 + n) / total as f32),
            );
        };
        install_weight(&client, file, &file.weight.path(root), cancel, &mut report)?;
    }
    Ok(())
}

pub fn install_weight(
    client: &ureq::Agent,
    file: &RemoteWeight,
    destination: &Path,
    cancel: &AtomicBool,
    progress: Progress,
) -> Result<()> {
    let mut url = Url::parse("https://huggingface.co")?;
    url.path_segments_mut()
        .map_err(|_| anyhow::anyhow!("URL Hugging Face invalide"))?
        .pop_if_empty()
        .extend(file.weight.repo.split('/'))
        .push("resolve")
        .push(&file.revision)
        .extend(file.weight.name.split('/'));
    let identity = file.sha256.as_deref().unwrap_or(&file.revision);
    download(
        client,
        url.as_str(),
        destination,
        file.size,
        file.sha256.as_deref(),
        identity,
        cancel,
        progress,
    )
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
