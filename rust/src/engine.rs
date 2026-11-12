use std::{
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{atomic::AtomicBool, mpsc},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use eframe::egui::ColorImage;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tempfile::Builder;
use zip::ZipArchive;

use crate::{
    catalog::{self, Model, Preset, Selection},
    download::{self, Progress},
};

const RELEASES: &str =
    "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=15";
const MINIMUM_BUILD: u32 = 1088;
const CUDA_DLLS: &[&str] = &["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"];

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Backend {
    #[default]
    Cuda,
    Vulkan,
    Cpu,
}

impl Backend {
    pub const ALL: [Self; 3] = [Self::Cuda, Self::Vulkan, Self::Cpu];

    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "NVIDIA CUDA 12",
            Self::Vulkan => "Vulkan",
            Self::Cpu => "CPU",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Cuda => "cuda12",
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    id: u64,
    name: String,
    size: u64,
    digest: Option<String>,
    browser_download_url: String,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    tag: String,
    archive: String,
}

pub fn build_number(tag: &str) -> Option<u32> {
    tag.strip_prefix("master-")?.split('-').next()?.parse().ok()
}

pub fn command(path: impl AsRef<OsStr>) -> Command {
    let cmd = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut cmd = cmd;
        cmd.creation_flags(0x0800_0000);
        cmd
    }
    #[cfg(not(windows))]
    cmd
}

pub fn find(root: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return path.is_file().then(|| path.to_owned());
    }
    let name = if cfg!(windows) {
        "sd-cli.exe"
    } else {
        "sd-cli"
    };
    [
        root.join("bin").join(name),
        root.join(name),
        root.join("bin/bin").join(name),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn exe_in(folder: &Path) -> Option<PathBuf> {
    find(folder, None)
        .filter(|p| p.extension().is_some_and(|e| e == "exe"))
        .or_else(|| {
            ["sd-cli.exe", "bin/sd-cli.exe", "bin/bin/sd-cli.exe"]
                .iter()
                .map(|name| folder.join(name))
                .find(|path| path.is_file())
        })
}

pub fn install(
    root: &Path,
    backend: Backend,
    cancel: &AtomicBool,
    progress: Progress,
) -> Result<PathBuf> {
    download::check(cancel)?;
    let client = download::client();
    progress("Recherche du moteur officiel sd.cpp".into(), None);
    let releases: Vec<Release> = client.get(RELEASES).call()?.into_json()?;
    let suffix = format!("-bin-win-{}-x64.zip", backend.key());
    let (release, archive) = releases
        .iter()
        .filter_map(|release| {
            let asset = release.assets.iter().find(|a| a.name.ends_with(&suffix))?;
            if backend == Backend::Cuda
                && !release
                    .assets
                    .iter()
                    .any(|a| a.name == "cudart-sd-bin-win-cu12-x64.zip")
            {
                return None;
            }
            let build = build_number(&release.tag_name)?;
            (build >= MINIMUM_BUILD).then_some((release, asset, build))
        })
        .max_by_key(|(_, _, build)| *build)
        .map(|(release, asset, _)| (release, asset))
        .context("Aucun moteur Windows x64 récent disponible pour ce mode")?;
    ensure!(
        catalog::safe_name(&release.tag_name) && !release.tag_name.contains('/'),
        "Tag de moteur invalide"
    );
    let engines = root
        .join("userdata-rust")
        .join("engines")
        .join(backend.key());
    fs::create_dir_all(&engines)?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(engines.join(".install-lock"))?;
    FileExt::try_lock_exclusive(&lock).context("Une autre fenêtre installe déjà ce moteur")?;
    let target = engines.join(&release.tag_name);
    if let Some(path) = exe_in(&target)
        && (backend != Backend::Cuda || cuda_present(&path))
    {
        return Ok(path);
    }
    ensure!(
        !target.exists(),
        "Installation du moteur incomplète dans {}. Déplace ce dossier et relance.",
        target.display()
    );
    let stage = Builder::new().prefix(".stage-").tempdir_in(&engines)?;
    let unpacked = stage.path().join("engine");
    fs::create_dir(&unpacked)?;
    let cache = engines.join("archives");
    let main_zip = download_asset(&client, archive, &cache, cancel, progress)?;
    unpack(&main_zip, &unpacked, cancel)?;
    let exe = exe_in(&unpacked).context("sd-cli.exe absent de l’archive officielle")?;
    if backend == Backend::Cuda && !cuda_present(&exe) {
        let asset = release
            .assets
            .iter()
            .find(|a| a.name == "cudart-sd-bin-win-cu12-x64.zip")
            .context("Archive des DLL CUDA 12 introuvable")?;
        let cuda_zip = download_asset(&client, asset, &cache, cancel, progress)?;
        let cuda = stage.path().join("cuda");
        fs::create_dir(&cuda)?;
        unpack(&cuda_zip, &cuda, cancel)?;
        let destination = exe.parent().context("Dossier du moteur absent")?;
        for name in CUDA_DLLS {
            let source = find_filename(&cuda, name, 0)
                .with_context(|| format!("{name} absent de l’archive CUDA"))?;
            if !destination.join(name).is_file() {
                fs::copy(source, destination.join(name))?;
            }
        }
    }
    download::check(cancel)?;
    let manifest = Manifest {
        tag: release.tag_name.clone(),
        archive: archive.name.clone(),
    };
    fs::write(
        exe.parent()
            .context("Dossier du moteur absent")?
            .join("engine-manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    if cfg!(windows) {
        help(&exe).context(
            "Le moteur installé ne démarre pas. Vérifie le pilote du GPU ou essaie Vulkan.",
        )?;
    }
    let relative = exe.strip_prefix(&unpacked)?.to_owned();
    fs::rename(unpacked, &target).context("Impossible d’activer le moteur")?;
    if let Ok(entries) = fs::read_dir(&cache) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == "zip") {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    Ok(target.join(relative))
}

fn download_asset(
    client: &ureq::Agent,
    asset: &Asset,
    cache: &Path,
    cancel: &AtomicBool,
    progress: Progress,
) -> Result<PathBuf> {
    ensure!(
        catalog::safe_name(&asset.name)
            && !asset.name.contains('/')
            && asset
                .browser_download_url
                .starts_with("https://github.com/leejet/stable-diffusion.cpp/releases/download/"),
        "Archive moteur non reconnue"
    );
    let dest = cache.join(format!("{}-{}", asset.id, asset.name));
    let hash = asset
        .digest
        .as_deref()
        .and_then(|digest| digest.strip_prefix("sha256:"));
    download::download(
        client,
        &asset.browser_download_url,
        &dest,
        asset.size,
        hash,
        &format!("{:016x}", asset.id),
        cancel,
        progress,
    )?;
    Ok(dest)
}

fn cuda_present(exe: &Path) -> bool {
    exe.parent()
        .is_some_and(|folder| CUDA_DLLS.iter().all(|name| folder.join(name).is_file()))
}

fn find_filename(folder: &Path, filename: &str, depth: u8) -> Option<PathBuf> {
    if depth > 5 {
        return None;
    }
    for entry in fs::read_dir(folder).ok()?.flatten() {
        let kind = entry.file_type().ok()?;
        if kind.is_file() && entry.file_name().eq_ignore_ascii_case(filename) {
            return Some(entry.path());
        }
        if kind.is_dir()
            && let Some(path) = find_filename(&entry.path(), filename, depth + 1)
        {
            return Some(path);
        }
    }
    None
}

fn unpack(path: &Path, folder: &Path, cancel: &AtomicBool) -> Result<()> {
    let mut archive = ZipArchive::new(File::open(path)?)?;
    for index in 0..archive.len() {
        download::check(cancel)?;
        let mut entry = archive.by_index(index)?;
        let relative = entry
            .enclosed_name()
            .context("Chemin invalide dans une archive sd.cpp")?;
        ensure!(
            !entry.is_symlink(),
            "Lien symbolique interdit dans une archive sd.cpp"
        );
        let destination = folder.join(relative);
        if entry.is_dir() {
            fs::create_dir_all(destination)?;
        } else {
            fs::create_dir_all(destination.parent().context("Dossier ZIP absent")?)?;
            io::copy(&mut entry, &mut File::create(destination)?)?;
        }
    }
    Ok(())
}

pub fn gpu_list() -> Vec<String> {
    let result = command("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ])
        .output();
    match result {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn help(exe: &Path) -> Result<String> {
    let out = command(exe)
        .arg("--help")
        .output()
        .context("Impossible de lancer sd-cli")?;
    let help =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    ensure!(
        help.contains("--diffusion-model"),
        "sd-cli ne répond pas à --help ({})",
        out.status
    );
    Ok(help)
}

#[derive(Clone)]
pub struct Request {
    pub root: PathBuf,
    pub exe: PathBuf,
    pub model: Model,
    pub selection: Selection,
    pub preset: Preset,
    pub prompt: String,
    pub negative: String,
    pub references: Vec<PathBuf>,
    pub width: u32,
    pub height: u32,
    pub seed: i64,
    pub count: u32,
    pub strength: f32,
    pub loras: Vec<(String, f32)>,
    pub backend: Backend,
    pub gpu: Option<u32>,
    pub auto_fit: bool,
    pub flash_attention: bool,
}

pub enum Update {
    Log(String),
    Progress(String, Option<f32>),
    Preview(ColorImage),
}

#[derive(Serialize)]
struct Record<'a> {
    model: &'a str,
    prompt: &'a str,
    negative: &'a str,
    seed: i64,
    width: u32,
    height: u32,
    steps: u32,
    cfg: f32,
    sampler: &'a str,
    scheduler: &'a str,
    preset: &'a str,
    weights: &'a Selection,
    references: Vec<String>,
}

fn model_build(exe: &Path) -> Option<u32> {
    let dir = exe.parent()?;
    [dir, dir.parent().unwrap_or(dir)]
        .iter()
        .filter_map(|folder| fs::read(folder.join("engine-manifest.json")).ok())
        .filter_map(|raw| serde_json::from_slice::<Manifest>(&raw).ok())
        .find_map(|manifest| build_number(&manifest.tag))
}

fn build_command(
    req: &Request,
    resolved: &Selection,
    reference_paths: &[PathBuf],
    output: &Path,
    preview: &Path,
    help: &str,
) -> Result<Command> {
    let mut cmd = command(&req.exe);
    cmd.args(["--mode", "img_gen"]);
    for (role, flag) in [
        ("diffusion", "--diffusion-model"),
        ("vae", "--vae"),
        ("text_encoder", "--llm"),
        ("tokenizer", "--tokenizer"),
        ("text_encoder_vision", "--llm_vision"),
    ] {
        if let Some(file) = resolved.get(role) {
            ensure!(
                help.contains(flag),
                "Moteur incompatible avec {flag}. Mets-le à jour."
            );
            cmd.arg(flag).arg(file.path(&req.root));
        }
    }
    for flag in req
        .model
        .defaults
        .extra_flags
        .iter()
        .filter(|flag| flag.starts_with("--"))
    {
        ensure!(
            help.contains(flag),
            "Moteur incompatible avec {flag}. Mets-le à jour."
        );
    }
    cmd.args(&req.model.defaults.extra_flags);
    let mut prompt = req.prompt.clone();
    let mut loras = req.loras.clone();
    if let Some(adapter) = req.preset.adapter() {
        ensure!(
            catalog::present(&adapter.path(&req.root))
                && adapter.path(&req.root).metadata()?.len() == adapter.size,
            "Télécharge le LoRA du preset dans Modèles."
        );
        ensure!(
            help.contains("--lora-apply-mode"),
            "Moteur incompatible avec les adaptateurs de distillation"
        );
        loras.retain(|(name, _)| name != adapter.tag);
        loras.push((adapter.tag.into(), adapter.strength));
        cmd.args(["--lora-apply-mode", "at_runtime"]);
        if req.preset.krea_distill {
            ensure!(
                reference_paths.is_empty() && help.contains("--sigmas"),
                "Le preset Krea 4 étapes utilise un prompt seul et un moteur avec --sigmas."
            );
            cmd.args(["--sigmas", "1.0,0.90453,0.75951,0.51284,0.0"]);
        }
    }
    if !loras.is_empty() {
        for (name, strength) in loras {
            ensure!(
                catalog::safe_name(&name)
                    && strength.is_finite()
                    && (-2.0..=2.0).contains(&strength)
                    && ["safetensors", "gguf"]
                        .iter()
                        .any(|extension| catalog::present(
                            &req.root.join("loras").join(format!("{name}.{extension}"))
                        )),
                "LoRA indisponible ou invalide : {name}"
            );
            prompt.push_str(&format!(" <lora:{name}:{strength}>"));
        }
        cmd.arg("--lora-model-dir").arg(req.root.join("loras"));
    }
    cmd.args([
        "-p",
        &prompt,
        "--steps",
        &req.preset.steps.to_string(),
        "--cfg-scale",
        &req.preset.cfg_scale.to_string(),
        "--sampling-method",
        &req.preset.sampler,
        "-W",
        &req.width.to_string(),
        "-H",
        &req.height.to_string(),
        "-s",
        &req.seed.to_string(),
        "-b",
        &req.count.to_string(),
    ]);
    if req.preset.cfg_scale > 1.0
        && req.model.defaults.supports_negative
        && !req.negative.is_empty()
    {
        cmd.args(["-n", &req.negative]);
    }
    if req.preset.scheduler != "auto" {
        cmd.args(["--scheduler", &req.preset.scheduler]);
    }
    if req.model.defaults.flow_shift > 0.0 {
        cmd.args(["--flow-shift", &req.model.defaults.flow_shift.to_string()]);
    }
    if req.model.native_references() {
        for reference in reference_paths {
            cmd.arg("-r").arg(reference);
        }
    } else if let Some(reference) = reference_paths.first() {
        cmd.arg("-i")
            .arg(reference)
            .args(["--strength", &req.strength.to_string()]);
    }
    if req.backend != Backend::Cpu {
        if req.auto_fit && help.contains("--auto-fit") {
            cmd.arg("--auto-fit");
            if help
                .lines()
                .any(|line| line.contains("--auto-fit") && line.contains("on|off"))
                || help.contains("'on' or 'off'")
            {
                cmd.arg("on");
            }
            if help.contains("--max-vram") {
                cmd.args(["--max-vram", "-1"]);
            }
        } else if req.auto_fit {
            cmd.arg("--offload-to-cpu");
        }
        if req.flash_attention && help.contains("--diffusion-fa") {
            cmd.arg("--diffusion-fa");
        }
    }
    if help.contains("--vae-tiling") {
        cmd.arg("--vae-tiling");
    }
    if help.contains("--preview-path") {
        cmd.args(["--preview", "proj", "--preview-interval", "1"])
            .arg("--preview-path")
            .arg(preview);
    }
    if req.backend == Backend::Cuda
        && let Some(gpu) = req.gpu
    {
        cmd.env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            .env("CUDA_VISIBLE_DEVICES", gpu.to_string());
    }
    cmd.arg("-o").arg(output).arg("-v");
    Ok(cmd)
}

pub fn generate(
    req: Request,
    cancel: &AtomicBool,
    mut event: impl FnMut(Update),
) -> Result<Vec<PathBuf>> {
    download::check(cancel)?;
    ensure!(!req.prompt.trim().is_empty(), "Saisis un prompt.");
    let model = &req.model;
    let has_refs = !req.references.is_empty();
    ensure!(
        !model.defaults.requires_reference || has_refs,
        "Ce modèle demande au moins une référence."
    );
    ensure!(
        req.references.len() <= model.max_references(),
        "Ce modèle accepte au maximum {} référence(s).",
        model.max_references()
    );
    ensure!(
        model.family != "minimax_h3_image" || model.native_references(),
        "Mode d’édition H3 invalide"
    );
    ensure!(
        (256..=3072).contains(&req.width)
            && (256..=3072).contains(&req.height)
            && req.width.is_multiple_of(model.dimension_multiple())
            && req.height.is_multiple_of(model.dimension_multiple()),
        "Les dimensions doivent être des multiples de {}",
        model.dimension_multiple()
    );
    ensure!(
        (1..=4).contains(&req.count)
            && (1..=100).contains(&req.preset.steps)
            && (0.0..=20.0).contains(&req.preset.cfg_scale)
            && (0.0..=1.0).contains(&req.strength)
            && req.seed >= 0,
        "Paramètres de génération hors limites."
    );
    let available = catalog::installed(model, &req.root);
    let mut resolved = Selection::new();
    for (role, label) in catalog::ROLES {
        if model.needs(role, has_refs, false) {
            let file = catalog::chosen(role, &req.selection, &available)
                .with_context(|| format!("{label} manquant. Télécharge les poids dans Modèles."))?;
            ensure!(
                model.sources[*role].accepts(file) && catalog::present(&file.path(&req.root)),
                "Variante {label} absente. Installe-la ou choisis un fichier local."
            );
            resolved.insert((*role).into(), file.clone());
        }
    }
    if let Some(build) = model_build(&req.exe) {
        ensure!(
            build >= model.defaults.min_engine_build,
            "Mets sd.cpp à jour. {} demande le build {}.",
            model.name,
            model.defaults.min_engine_build
        );
    }
    let help = help(&req.exe)?;
    if model.family == "minimax_h3_image" {
        ensure!(
            help.contains("minimax-h3-image"),
            "MiniMax H3 Image demande sd.cpp build 1088 ou plus récent."
        );
    }
    let temp = req.root.join("tmp").join("rust");
    fs::create_dir_all(&temp)?;
    let job = Builder::new().prefix("generation-").tempdir_in(temp)?;
    let mut refs = Vec::new();
    for (index, source) in req.references.iter().enumerate() {
        download::check(cancel)?;
        let path = job.path().join(format!("reference-{index:02}.png"));
        let mut decoder = image::ImageReader::open(source)?
            .with_guessed_format()?
            .into_decoder()?;
        use image::ImageDecoder;
        let orientation = decoder.orientation()?;
        let mut image = image::DynamicImage::from_decoder(decoder)?;
        image.apply_orientation(orientation);
        image.save(&path)?;
        refs.push(path);
    }
    let preview_dir = job.path().join("preview");
    let results = job.path().join("results");
    fs::create_dir(&preview_dir)?;
    fs::create_dir(&results)?;
    let mut cmd = build_command(
        &req,
        &resolved,
        &refs,
        &results.join("image.png"),
        &preview_dir.join("step_%03d.png"),
        &help,
    )?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    event(Update::Progress("Chargement du modèle".into(), None));
    let mut child = cmd.spawn().context("Démarrage du moteur impossible")?;
    let (tx, rx) = mpsc::channel();
    let stdout = child.stdout.take().context("Sortie moteur indisponible")?;
    let stderr = child.stderr.take().context("Journal moteur indisponible")?;
    let out = read_logs(stdout, tx.clone());
    let err = read_logs(stderr, tx);
    let mut last = None;
    let mut tail = std::collections::VecDeque::with_capacity(15);
    let mut read_events = |event: &mut dyn FnMut(Update)| {
        for line in rx.try_iter().take(256) {
            if let Some((step, total)) = step_progress(&line) {
                event(Update::Progress(
                    format!("Échantillonnage · {step}/{total}"),
                    Some(step as f32 / total as f32),
                ));
            } else if !line.is_empty() {
                if tail.len() == 15 {
                    tail.pop_front();
                }
                tail.push_back(line.clone());
                event(Update::Log(line));
            }
        }
    };
    let status = loop {
        if download::check(cancel).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = out.join();
            let _ = err.join();
            bail!("Génération arrêtée.");
        }
        read_events(&mut event);
        if let Some(path) = pngs(&preview_dir).pop()
            && last.as_ref() != Some(&path)
            && let Ok(frame) = preview_image(&path)
        {
            last = Some(path);
            event(Update::Preview(frame));
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.into());
            }
        }
    };
    let _ = out.join();
    let _ = err.join();
    read_events(&mut event);
    ensure!(
        status.success(),
        "sd-cli a quitté avec {status}.\n{}",
        tail.into_iter().collect::<Vec<_>>().join("\n")
    );
    download::check(cancel)?;
    let files = pngs(&results);
    ensure!(
        files.len() == req.count as usize,
        "Le moteur a produit {} image(s) sur {}.",
        files.len(),
        req.count
    );
    let outputs = req.root.join("outputs");
    fs::create_dir_all(&outputs)?;
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let token = job
        .path()
        .file_name()
        .context("Identifiant de génération absent")?
        .to_string_lossy();
    let mut final_paths = Vec::new();
    for (index, path) in files.iter().enumerate() {
        let frame = preview_image(path).context("PNG de sortie invalide")?;
        let [width, height] = frame.size;
        let final_path = outputs.join(format!("{}-{timestamp}-{token}-{index:02}.png", model.id));
        let seed = req.seed.saturating_add(index as i64);
        let record = Record {
            model: &model.id,
            prompt: &req.prompt,
            negative: &req.negative,
            seed,
            width: width as u32,
            height: height as u32,
            steps: req.preset.steps,
            cfg: req.preset.cfg_scale,
            sampler: &req.preset.sampler,
            scheduler: &req.preset.scheduler,
            preset: &req.preset.name,
            weights: &resolved,
            references: req
                .references
                .iter()
                .filter_map(|p| p.file_name())
                .map(|p| p.to_string_lossy().into_owned())
                .collect(),
        };
        fs::write(
            final_path.with_extension("rust.json"),
            serde_json::to_vec_pretty(&record)?,
        )?;
        fs::write(
            final_path.with_extension("txt"),
            format!(
                "{}\nModel: {} ({})\nSteps: {}, Seed: {}\n",
                req.prompt, model.name, model.id, req.preset.steps, seed
            ),
        )?;
        fs::rename(path, &final_path)?;
        event(Update::Preview(frame));
        final_paths.push(final_path);
    }
    Ok(final_paths)
}

fn read_logs(
    reader: impl io::Read + Send + 'static,
    tx: mpsc::Sender<String>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut buffer = Vec::new();
        for byte in BufReader::new(reader).bytes().map_while(Result::ok) {
            if byte == b'\r' || byte == b'\n' || buffer.len() >= 8192 {
                if !buffer.is_empty() {
                    let line = String::from_utf8_lossy(&buffer).trim().to_owned();
                    if tx.send(line).is_err() {
                        break;
                    }
                    buffer.clear();
                }
            } else {
                buffer.push(byte);
            }
        }
        if !buffer.is_empty() {
            let _ = tx.send(String::from_utf8_lossy(&buffer).trim().to_owned());
        }
    })
}

fn step_progress(line: &str) -> Option<(u32, u32)> {
    let mut parts = line.split('|');
    parts.next()?;
    parts.next()?;
    let fraction = parts.next()?.split_whitespace().next()?;
    let (step, total) = fraction.split_once('/')?;
    let (step, total) = (step.parse().ok()?, total.parse().ok()?);
    (total > 0 && step <= total).then_some((step, total))
}

fn pngs(folder: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) && p.is_file())
        .collect();
    paths.sort();
    paths
}

pub fn preview_image(path: &Path) -> Result<ColorImage> {
    let image = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .to_rgba8();
    let size = [image.width() as usize, image.height() as usize];
    Ok(ColorImage::from_rgba_unmultiplied(size, image.as_raw()))
}
