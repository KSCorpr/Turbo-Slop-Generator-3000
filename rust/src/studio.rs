use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use eframe::egui::{self, Color32, ColorImage, Frame, Margin, RichText, TextureHandle, Vec2};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::{
    catalog, download,
    engine::{self, Device, Request, Update},
};

const ACCENT: Color32 = Color32::from_rgb(250, 174, 116);
const PANEL: Color32 = Color32::from_rgb(29, 32, 38);
const MUTED: Color32 = Color32::from_rgb(166, 172, 184);

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Preferences {
    device: Device,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Images,
    Models,
    Gallery,
    Settings,
}

enum Outcome {
    Generated(Vec<PathBuf>),
    Installed,
    Device(String),
}

enum Event {
    Update(Update),
    Done(Result<Outcome>),
}

struct Job {
    cancel: Arc<AtomicBool>,
    events: mpsc::Receiver<Event>,
}

pub struct Studio {
    root: PathBuf,
    root_edit: String,
    preferences: Preferences,
    gpu: Option<String>,
    page: Page,
    prompt: String,
    width: u32,
    height: u32,
    seed: i64,
    ready: bool,
    job: Option<Job>,
    closing: bool,
    status: String,
    fraction: Option<f32>,
    error: Option<String>,
    preview: Option<TextureHandle>,
    output: Option<PathBuf>,
    gallery: Vec<PathBuf>,
    gallery_selected: Option<PathBuf>,
    gallery_preview: Option<TextureHandle>,
}

pub fn style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(10.0, 8.0);
    style.spacing.button_padding = Vec2::new(12.0, 8.0);
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = Color32::from_rgb(20, 22, 27);
    style.visuals.window_fill = PANEL;
    style.visuals.extreme_bg_color = Color32::from_rgb(16, 18, 23);
    style.visuals.selection.bg_fill = Color32::from_rgb(118, 70, 50);
    style.visuals.selection.stroke.color = ACCENT;
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(40, 44, 53);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(55, 60, 71);
    ctx.set_style(style);
}

impl Studio {
    pub fn load(root: Option<PathBuf>) -> Result<Self> {
        let root = std::path::absolute(root.unwrap_or_else(default_root))?;
        fs::create_dir_all(&root).context("Dossier TurboSlop inaccessible")?;
        let mut app = Self {
            preferences: read_preferences(&root),
            root_edit: root.display().to_string(),
            root,
            gpu: None,
            page: Page::Images,
            prompt: String::new(),
            width: 512,
            height: 512,
            seed: -1,
            ready: false,
            job: None,
            closing: false,
            status: String::new(),
            fraction: None,
            error: None,
            preview: None,
            output: None,
            gallery: Vec::new(),
            gallery_selected: None,
            gallery_preview: None,
        };
        app.refresh_weights();
        app.refresh_gallery();
        Ok(app)
    }

    fn refresh_weights(&mut self) {
        self.ready = catalog::WEIGHTS
            .iter()
            .all(|file| file.installed(&self.root));
    }

    fn refresh_gallery(&mut self) {
        let mut paths: Vec<_> = fs::read_dir(self.root.join("outputs"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let extension = path.extension()?.to_str()?.to_ascii_lowercase();
                if !matches!(extension.as_str(), "png" | "jpg" | "jpeg" | "webp") {
                    return None;
                }
                let metadata = entry.metadata().ok()?;
                metadata
                    .is_file()
                    .then_some((metadata.modified().ok()?, path))
            })
            .collect();
        paths.sort_unstable_by(|a, b| b.cmp(a));
        self.gallery = paths.into_iter().take(250).map(|(_, path)| path).collect();
    }

    fn start(
        &mut self,
        ctx: &egui::Context,
        label: &str,
        worker: impl FnOnce(&AtomicBool, &mut dyn FnMut(Update)) -> Result<Outcome> + Send + 'static,
    ) {
        if self.job.is_some() {
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let ctx = ctx.clone();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut update = |event| {
                let _ = tx.send(Event::Update(event));
                ctx.request_repaint();
            };
            let result = worker(&flag, &mut update);
            let _ = tx.send(Event::Done(result));
            ctx.request_repaint();
        });
        self.status = label.into();
        self.fraction = None;
        self.error = None;
        self.job = Some(Job { cancel, events: rx });
    }

    fn drain(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else {
            return;
        };
        let finished = loop {
            match job.events.try_recv() {
                Ok(Event::Update(Update::Progress(message, fraction))) => {
                    self.status = message;
                    self.fraction = fraction;
                }
                Ok(Event::Update(Update::Preview(frame))) => {
                    put_texture(ctx, &mut self.preview, "generation", frame);
                }
                Ok(Event::Done(result)) => break result,
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => {
                    break Err(anyhow::anyhow!("La tâche s’est interrompue."));
                }
            }
        };
        self.job = None;
        self.fraction = None;
        match finished {
            Ok(Outcome::Generated(paths)) => {
                self.output = paths.into_iter().next();
                self.status = "Image enregistrée".into();
                self.refresh_gallery();
            }
            Ok(Outcome::Installed) => {
                self.refresh_weights();
                self.status = "Poids installés".into();
            }
            Ok(Outcome::Device(name)) => {
                self.status = format!("Calcul Vulkan validé · {name}");
                self.gpu = Some(name);
            }
            Err(error) => {
                self.status.clear();
                self.error = Some(format!("{error:#}"));
            }
        }
    }

    fn generate(&mut self, ctx: &egui::Context) {
        if self.form_invalid() {
            self.error = Some("Saisis un prompt et choisis des dimensions multiples de 16.".into());
            return;
        }
        self.refresh_weights();
        if !self.ready {
            self.page = Page::Models;
            self.error = Some("Télécharge les poids du modèle dans Modèles.".into());
            return;
        }
        let seed = if self.seed < 0 {
            (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                % i32::MAX as u128) as u64
        } else {
            self.seed as u64
        };
        let request = Request {
            root: self.root.clone(),
            prompt: self.prompt.trim().into(),
            width: self.width,
            height: self.height,
            seed,
            device: self.preferences.device,
        };
        self.output = None;
        self.preview = None;
        self.start(
            ctx,
            "Préparation de la génération",
            move |cancel, event| engine::generate(request, cancel, event).map(Outcome::Generated),
        );
    }

    fn form_invalid(&self) -> bool {
        self.prompt.trim().is_empty()
            || !engine::valid_edge(self.width)
            || !engine::valid_edge(self.height)
    }

    fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("studio_nav")
            .frame(
                Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(18, 11)),
            )
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("TurboSlop").size(22.0).strong().color(ACCENT));
                    ui.separator();
                    for (page, title) in [
                        (Page::Images, "Images"),
                        (Page::Models, "Modèles"),
                        (Page::Gallery, "Galerie"),
                        (Page::Settings, "Réglages"),
                    ] {
                        if ui.selectable_label(self.page == page, title).clicked() {
                            self.page = page;
                            if page == Page::Gallery {
                                self.refresh_gallery();
                            }
                            if page == Page::Models {
                                self.refresh_weights();
                            }
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("Rust natif · {}", env!("CARGO_PKG_VERSION")))
                                .small()
                                .color(MUTED),
                        );
                    });
                });
            });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        if self.status.is_empty() && self.error.is_none() {
            return;
        }
        egui::TopBottomPanel::bottom("activity")
            .frame(
                Frame::new()
                    .fill(PANEL)
                    .inner_margin(Margin::symmetric(18, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(job) = &self.job {
                        let stopped = job.cancel.load(Ordering::Relaxed);
                        if ui
                            .add_enabled(!stopped, egui::Button::new("Arrêter"))
                            .clicked()
                        {
                            job.cancel.store(true, Ordering::Relaxed);
                        }
                        if stopped {
                            ui.label("Arrêt demandé…");
                        } else if let Some(fraction) = self.fraction {
                            ui.add(
                                egui::ProgressBar::new(fraction)
                                    .desired_width(128.0)
                                    .show_percentage(),
                            );
                        } else {
                            ui.spinner();
                        }
                    }
                    if let Some(error) = &self.error {
                        if ui.button("Effacer").clicked() {
                            self.error = None;
                        } else {
                            ui.label(
                                RichText::new(error.lines().next().unwrap_or(error))
                                    .color(Color32::LIGHT_RED),
                            );
                        }
                    } else {
                        ui.label(&self.status);
                    }
                });
            });
    }

    fn images(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let left = (ui.available_width() * 0.42).clamp(350.0, 560.0);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                Vec2::new(left, ui.available_height()),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_width(left);
                    egui::ScrollArea::vertical()
                        .id_salt("controls")
                        .show(ui, |ui| {
                            ui.add_enabled_ui(self.job.is_none(), |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.heading(catalog::MODEL_NAME);
                                    if ui.button("Poids").clicked() {
                                        self.page = Page::Models;
                                    }
                                });
                                ui.label(RichText::new("Prompt").strong());
                                ui.add(
                                    egui::TextEdit::multiline(&mut self.prompt)
                                        .desired_rows(5)
                                        .desired_width(f32::INFINITY)
                                        .hint_text("Décris l’image à générer…"),
                                );
                                ui.label(
                                    RichText::new("Flow Match Euler · 4 étapes")
                                        .small()
                                        .color(MUTED),
                                );
                                ui.horizontal_wrapped(|ui| {
                                    for (label, w, h) in [
                                        ("256", 256, 256),
                                        ("384", 384, 384),
                                        ("512", 512, 512),
                                        ("Paysage", 640, 384),
                                        ("Portrait", 384, 640),
                                    ] {
                                        if ui
                                            .selectable_label(
                                                (self.width, self.height) == (w, h),
                                                label,
                                            )
                                            .clicked()
                                        {
                                            (self.width, self.height) = (w, h);
                                        }
                                    }
                                });
                                ui.horizontal_wrapped(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut self.width)
                                            .speed(16)
                                            .range(engine::MIN_EDGE..=engine::MAX_EDGE)
                                            .prefix("L "),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut self.height)
                                            .speed(16)
                                            .range(engine::MIN_EDGE..=engine::MAX_EDGE)
                                            .prefix("H "),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut self.seed)
                                            .range(-1..=i64::MAX)
                                            .prefix("Seed "),
                                    );
                                    ui.label("-1 = aléatoire");
                                });
                                if !self.ready {
                                    ui.label(
                                        RichText::new("Poids à télécharger dans Modèles.")
                                            .color(ACCENT),
                                    );
                                }
                                let run = egui::Button::new(
                                    RichText::new("Générer")
                                        .strong()
                                        .color(Color32::from_rgb(31, 22, 18)),
                                )
                                .fill(ACCENT)
                                .min_size(Vec2::new(180.0, 38.0));
                                let shortcut = ctx.input(|input| {
                                    input.modifiers.ctrl && input.key_pressed(egui::Key::Enter)
                                });
                                if ui.add(run).clicked() || shortcut {
                                    self.generate(ctx);
                                }
                            });
                        });
                },
            );
            ui.separator();
            ui.vertical(|ui| {
                ui.heading("Aperçu");
                let area = Vec2::new(
                    ui.available_width(),
                    (ui.available_height() - 106.0).max(230.0),
                );
                if let Some(texture) = &self.preview {
                    show_texture(ui, texture, area);
                } else {
                    placeholder(ui, area, "L’image apparaît après le décodage");
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Ouvrir le dossier des images").clicked() {
                        reveal(self.root.join("outputs"), &mut self.error);
                    }
                    if let Some(path) = &self.output
                        && ui.button("Ouvrir l’image").clicked()
                        && let Err(error) = open::that(path)
                    {
                        self.error = Some(error.to_string());
                    }
                });
                if let Some(error) = &self.error {
                    ui.label(RichText::new(error).color(Color32::LIGHT_RED));
                }
            });
        });
    }

    fn models(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.heading("Modèles");
        ui.label(RichText::new(catalog::MODEL_NAME).strong());
        ui.label("Génération à partir d’un prompt. Le transformeur est quantifié en mémoire pendant le chargement.");
        for file in catalog::WEIGHTS {
            ui.horizontal_wrapped(|ui| {
                ui.label(file.label);
                ui.label(format!("{:.2} Go", download::gb(file.size)));
                if file.installed(&self.root) {
                    ui.label(RichText::new("Installé").color(ACCENT));
                } else {
                    ui.label(RichText::new("À télécharger").color(MUTED));
                }
            });
        }
        ui.horizontal_wrapped(|ui| {
            let total: u64 = catalog::WEIGHTS.iter().map(|file| file.size).sum();
            if ui
                .add_enabled(
                    self.job.is_none() && !self.ready,
                    egui::Button::new(format!(
                        "Télécharger les poids · {:.2} Go au total",
                        download::gb(total)
                    )),
                )
                .clicked()
            {
                let root = self.root.clone();
                self.start(
                    ctx,
                    "Préparation du téléchargement",
                    move |cancel, event| {
                        download::install(&root, cancel, &mut |msg, p| {
                            event(Update::Progress(msg, p))
                        })?;
                        Ok(Outcome::Installed)
                    },
                );
            }
            if ui.button("Actualiser").clicked() {
                self.refresh_weights();
            }
            if ui.button("Ouvrir le dossier").clicked() {
                reveal(self.root.join("models"), &mut self.error);
            }
        });
        ui.label(
            RichText::new("Poids publics avec reprise des transferts et vérification SHA-256.")
                .color(MUTED),
        );
    }

    fn settings(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.heading("Réglages");
        ui.add_enabled_ui(self.job.is_none(), |ui| {
            ui.label(RichText::new("Dossier TurboSlop").strong());
            ui.label("Tu peux choisir un dossier existant pour partager la galerie et les fichiers de modèles compatibles.");
            ui.horizontal_wrapped(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.root_edit).desired_width(480.0));
                if ui.button("Parcourir").clicked() && let Some(path) = pick_folder() {
                    self.root_edit = path.display().to_string();
                }
                if ui.button("Utiliser ce dossier").clicked() { self.change_root(); }
            });
            ui.separator();
            ui.label(RichText::new("Moteur Rust · Vulkan").strong());
            let previous = self.preferences.device;
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_salt("gpu").selected_text(self.preferences.device.label())
                    .show_ui(ui, |ui| {
                        for device in Device::CHOICES {
                            ui.selectable_value(&mut self.preferences.device, device, device.label());
                        }
                    });
                if ui.button("Tester le GPU").clicked() {
                    let device = self.preferences.device;
                    self.start(ctx, "Test de Vulkan", move |_, _| engine::probe(device).map(Outcome::Device));
                }
                if let Some(gpu) = &self.gpu { ui.label(gpu); }
            });
            if previous != self.preferences.device {
                self.gpu = None;
                if let Err(error) = save_json(&prefs_path(&self.root), &self.preferences) {
                    self.error = Some(format!("Préférences non enregistrées. {error:#}"));
                }
            }
        });
    }

    fn change_root(&mut self) {
        let path = PathBuf::from(self.root_edit.trim());
        let next = (!path.as_os_str().is_empty())
            .then(|| std::path::absolute(path))
            .transpose();
        let Ok(Some(path)) = next else {
            self.error = Some("Sélectionne un dossier existant.".into());
            return;
        };
        if !path.is_dir() {
            self.error = Some("Sélectionne un dossier existant.".into());
            return;
        }
        self.root = path;
        self.preferences = read_preferences(&self.root);
        self.root_edit = self.root.display().to_string();
        self.gpu = None;
        self.preview = None;
        self.output = None;
        self.gallery_selected = None;
        self.gallery_preview = None;
        self.refresh_weights();
        self.refresh_gallery();
        if let Some(session) = session_path()
            && let Err(error) = save_json(&session, &self.root)
        {
            self.error = Some(format!("Dossier non mémorisé. {error:#}"));
        }
    }

    fn gallery(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Galerie");
            if ui.button("Actualiser").clicked() {
                self.refresh_gallery();
            }
            if ui.button("Ouvrir le dossier").clicked() {
                reveal(self.root.join("outputs"), &mut self.error);
            }
        });
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                Vec2::new(326.0, ui.available_height()),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_width(326.0);
                    egui::ScrollArea::vertical()
                        .id_salt("gallery-files")
                        .show(ui, |ui| {
                            if self.gallery.is_empty() {
                                ui.label("Aucune image enregistrée.");
                            }
                            let mut next = None;
                            for path in &self.gallery {
                                let label = path.file_name().unwrap_or_default().to_string_lossy();
                                if ui
                                    .selectable_label(
                                        self.gallery_selected.as_ref() == Some(path),
                                        label,
                                    )
                                    .clicked()
                                {
                                    next = Some(path.clone());
                                }
                            }
                            if let Some(path) = next {
                                match engine::preview_image(&path) {
                                    Ok(image) => {
                                        put_texture(
                                            ctx,
                                            &mut self.gallery_preview,
                                            "gallery",
                                            image,
                                        );
                                        self.gallery_selected = Some(path);
                                    }
                                    Err(error) => {
                                        self.error = Some(format!("Image illisible. {error:#}"))
                                    }
                                }
                            }
                        });
                },
            );
            ui.separator();
            ui.vertical(|ui| {
                if let Some(path) = &self.gallery_selected {
                    ui.horizontal(|ui| {
                        if ui.button("Ouvrir l’image").clicked()
                            && let Err(error) = open::that(path)
                        {
                            self.error = Some(error.to_string());
                        }
                        if ui
                            .add_enabled(
                                self.job.is_none(),
                                egui::Button::new("Réutiliser le prompt"),
                            )
                            .clicked()
                        {
                            match gallery_prompt(path) {
                                Some(prompt) => {
                                    self.prompt = prompt;
                                    self.page = Page::Images;
                                }
                                None => {
                                    self.error =
                                        Some("Aucun prompt enregistré pour cette image.".into())
                                }
                            }
                        }
                    });
                }
                let area = Vec2::new(
                    ui.available_width(),
                    (ui.available_height() - 50.0).max(200.0),
                );
                if let Some(texture) = &self.gallery_preview {
                    show_texture(ui, texture, area);
                } else {
                    placeholder(ui, area, "Sélectionne une image");
                }
            });
        });
    }
}

impl eframe::App for Studio {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain(ctx);
        if self.closing && self.job.is_none() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if let Some(job) = &self.job {
            ctx.request_repaint_after(Duration::from_millis(100));
            if ctx.input(|i| i.viewport().close_requested()) {
                self.closing = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                job.cancel.store(true, Ordering::Relaxed);
            }
        }
        self.header(ctx);
        self.status_bar(ctx);
        egui::CentralPanel::default()
            .frame(Frame::new().inner_margin(Margin::same(18)))
            .show(ctx, |ui| match self.page {
                Page::Images => self.images(ctx, ui),
                Page::Gallery => self.gallery(ctx, ui),
                Page::Models => {
                    egui::ScrollArea::vertical().show(ui, |ui| self.models(ctx, ui));
                }
                Page::Settings => {
                    egui::ScrollArea::vertical().show(ui, |ui| self.settings(ctx, ui));
                }
            });
    }
}

fn put_texture(
    ctx: &egui::Context,
    texture: &mut Option<TextureHandle>,
    name: &str,
    image: ColorImage,
) {
    if let Some(texture) = texture {
        texture.set(image, egui::TextureOptions::LINEAR);
    } else {
        *texture = Some(ctx.load_texture(name, image, egui::TextureOptions::LINEAR));
    }
}

fn placeholder(ui: &mut egui::Ui, size: Vec2, label: &str) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().rect_filled(rect, 8, PANEL);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(16.0),
        MUTED,
    );
}

fn show_texture(ui: &mut egui::Ui, texture: &TextureHandle, area: Vec2) {
    let original = texture.size_vec2();
    let scale = (area.x / original.x).min(area.y / original.y);
    ui.image((texture.id(), original * scale));
}

fn reveal(path: PathBuf, error: &mut Option<String>) {
    if let Err(e) = fs::create_dir_all(&path).and_then(|()| open::that(path)) {
        *error = Some(e.to_string());
    }
}

fn session_path() -> Option<PathBuf> {
    env::var_os("LOCALAPPDATA")
        .or_else(|| env::var_os("XDG_STATE_HOME"))
        .map(PathBuf::from)
        .map(|root| root.join("TurboSlop-Rust-Native").join("workspace.json"))
}

fn default_root() -> PathBuf {
    if let Some(path) = session_path()
        .and_then(|path| fs::read(path).ok())
        .and_then(|data| serde_json::from_slice::<PathBuf>(&data).ok())
        && path.is_dir()
    {
        return path;
    }
    let exe = env::current_exe().ok();
    let base = exe
        .as_deref()
        .and_then(Path::parent)
        .unwrap_or(Path::new("."));
    base.ancestors()
        .find(|path| path.join("config/models.yaml").is_file())
        .unwrap_or(base)
        .to_owned()
}

fn prefs_path(root: &Path) -> PathBuf {
    root.join("userdata-rust-native").join("preferences.json")
}

fn read_preferences(root: &Path) -> Preferences {
    fs::read(prefs_path(root))
        .ok()
        .and_then(|raw| serde_json::from_slice(&raw).ok())
        .unwrap_or_default()
}

fn save_json(path: &Path, value: &impl Serialize) -> Result<()> {
    use std::io::Write;
    let parent = path.parent().context("Dossier de préférences absent")?;
    fs::create_dir_all(parent)?;
    let mut temp = NamedTempFile::new_in(parent)?;
    temp.write_all(&serde_json::to_vec_pretty(value)?)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    Ok(())
}

fn gallery_prompt(path: &Path) -> Option<String> {
    if let Ok(raw) = fs::read(path.with_extension("rust.json"))
        && let Ok(data) = serde_json::from_slice::<serde_json::Value>(&raw)
        && let Some(prompt) = data["prompt"].as_str()
    {
        return Some(prompt.to_owned());
    }
    fs::read_to_string(path.with_extension("txt"))
        .ok()
        .map(|text| {
            text.rsplit_once("\nModel: ")
                .map_or(text.as_str(), |(prompt, _)| prompt)
                .trim()
                .to_owned()
        })
        .filter(|text| !text.is_empty())
}

#[cfg(windows)]
fn pick_folder() -> Option<PathBuf> {
    rfd::FileDialog::new().pick_folder()
}
#[cfg(not(windows))]
fn pick_folder() -> Option<PathBuf> {
    None
}
