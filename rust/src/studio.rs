use std::{
    collections::{BTreeMap, VecDeque},
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
    catalog::{self, Catalog, Model, Preset, RemoteWeight, Selection, Weight},
    download,
    engine::{self, Backend, Request, Update},
};

const ACCENT: Color32 = Color32::from_rgb(250, 174, 116);
const PANEL: Color32 = Color32::from_rgb(29, 32, 38);
const MUTED: Color32 = Color32::from_rgb(166, 172, 184);

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Preferences {
    engine: Option<PathBuf>,
    backend: Backend,
    gpu: Option<u32>,
    disable_auto_fit: bool,
    disable_flash: bool,
    choices: BTreeMap<String, Selection>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Images,
    Models,
    Gallery,
    Settings,
}

type Remote = BTreeMap<String, Vec<RemoteWeight>>;

enum Outcome {
    Generated(Vec<PathBuf>),
    Catalog(String, Remote),
    Engine(PathBuf, Backend),
    Installed,
    Gpus(Vec<String>),
}

enum Event {
    Update(Update),
    Done(Result<Outcome>),
}

struct Job {
    cancel: Arc<AtomicBool>,
    events: mpsc::Receiver<Event>,
}

struct Form {
    prompt: String,
    negative: String,
    references: Vec<PathBuf>,
    #[cfg(not(windows))]
    reference_text: String,
    preset: usize,
    sampling: Preset,
    width: u32,
    height: u32,
    seed: i64,
    count: u32,
    strength: f32,
    lora: Option<String>,
    lora_weight: f32,
}

impl Form {
    fn new(model: &Model) -> Self {
        let d = &model.defaults;
        Self {
            prompt: String::new(),
            negative: String::new(),
            references: Vec::new(),
            #[cfg(not(windows))]
            reference_text: String::new(),
            preset: 0,
            sampling: model.presets.first().cloned().unwrap_or_else(|| Preset {
                name: "Par défaut".into(),
                sampler: d.sampler.clone(),
                scheduler: d.scheduler.clone(),
                steps: d.steps,
                cfg_scale: d.cfg_scale,
                note: String::new(),
                moss_dmd: false,
                kromoss_dmd2: false,
                krea_distill: false,
            }),
            width: d.width,
            height: d.height,
            seed: -1,
            count: 1,
            strength: 0.6,
            lora: None,
            lora_weight: 0.8,
        }
    }
}

pub struct Studio {
    catalog: Catalog,
    model: usize,
    form: Form,
    root: PathBuf,
    preferences: Preferences,
    root_edit: String,
    engine_edit: String,
    install_backend: Backend,
    gpus: Vec<String>,
    page: Page,
    local: BTreeMap<String, Vec<Weight>>,
    remote: Option<(String, Remote)>,
    install_vision: bool,
    loras: Vec<String>,
    job: Option<Job>,
    closing: bool,
    status: String,
    fraction: Option<f32>,
    error: Option<String>,
    log: VecDeque<String>,
    preview: Option<TextureHandle>,
    gallery_preview: Option<TextureHandle>,
    gallery: Vec<PathBuf>,
    gallery_selected: Option<PathBuf>,
    outputs: Vec<PathBuf>,
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
        let catalog = Catalog::load()?;
        let root = std::path::absolute(root.unwrap_or_else(default_root))?;
        fs::create_dir_all(&root).context("Dossier TurboSlop inaccessible")?;
        let preferences = read_preferences(&root);
        let form = Form::new(&catalog.base_models[0]);
        let mut studio = Self {
            catalog,
            model: 0,
            form,
            root_edit: root.display().to_string(),
            engine_edit: preferences
                .engine
                .as_ref()
                .map_or_else(String::new, |p| p.display().to_string()),
            install_backend: preferences.backend,
            root,
            preferences,
            gpus: Vec::new(),
            page: Page::Images,
            local: BTreeMap::new(),
            remote: None,
            install_vision: true,
            loras: Vec::new(),
            job: None,
            closing: false,
            status: String::new(),
            fraction: None,
            error: None,
            log: VecDeque::new(),
            preview: None,
            gallery_preview: None,
            gallery: Vec::new(),
            gallery_selected: None,
            outputs: Vec::new(),
        };
        studio.refresh_local();
        studio.refresh_gallery();
        Ok(studio)
    }

    fn active_model(&self) -> &Model {
        &self.catalog.base_models[self.model]
    }

    fn refresh_local(&mut self) {
        self.local = catalog::installed(self.active_model(), &self.root);
        self.loras = local_loras(&self.root);
    }

    fn refresh_gallery(&mut self) {
        self.gallery = fs::read_dir(self.root.join("outputs"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let extension = path.extension()?.to_str()?.to_ascii_lowercase();
                if !matches!(extension.as_str(), "png" | "jpg" | "jpeg" | "webp") {
                    return None;
                }
                let modified = entry.metadata().ok()?.modified().ok()?;
                Some((modified, path))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|(_, path)| path)
            .collect();
        self.gallery.sort_by_cached_key(|path| {
            std::cmp::Reverse(path.metadata().and_then(|m| m.modified()).ok())
        });
        self.gallery.truncate(250);
    }

    fn save(&mut self) {
        if let Err(error) = save_json(&prefs_path(&self.root), &self.preferences) {
            self.error = Some(format!("Préférences non enregistrées. {error:#}"));
        }
    }

    fn select_model(&mut self, index: usize) {
        if self.model == index {
            return;
        }
        self.model = index;
        let prompt = std::mem::take(&mut self.form.prompt);
        self.form = Form::new(self.active_model());
        self.form.prompt = prompt;
        self.remote = None;
        self.refresh_local();
    }

    fn selection(&self) -> Selection {
        self.preferences
            .choices
            .get(&self.active_model().id)
            .cloned()
            .unwrap_or_default()
    }

    fn set_weight(&mut self, role: &str, weight: Option<Weight>) {
        let id = self.active_model().id.clone();
        let choice = self.preferences.choices.entry(id).or_default();
        if let Some(weight) = weight {
            choice.insert(role.into(), weight);
        } else {
            choice.remove(role);
        }
        self.save();
    }

    fn current_engine(&self) -> Option<PathBuf> {
        engine::find(&self.root, self.preferences.engine.as_deref())
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
        self.log.clear();
        self.job = Some(Job { cancel, events: rx });
    }

    fn drain(&mut self, ctx: &egui::Context) {
        let mut finished = None;
        if let Some(job) = &self.job {
            for _ in 0..256 {
                match job.events.try_recv() {
                    Ok(Event::Update(Update::Log(line))) => {
                        if self.log.len() == 250 {
                            self.log.pop_front();
                        }
                        self.log.push_back(line);
                    }
                    Ok(Event::Update(Update::Progress(message, fraction))) => {
                        self.status = message;
                        self.fraction = fraction;
                    }
                    Ok(Event::Update(Update::Preview(frame))) => {
                        put_texture(ctx, &mut self.preview, "generation", frame);
                    }
                    Ok(Event::Done(result)) => {
                        finished = Some(result);
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        finished = Some(Err(anyhow::anyhow!("La tâche s’est interrompue.")));
                        break;
                    }
                }
            }
        }
        let Some(result) = finished else {
            return;
        };
        self.job = None;
        self.fraction = None;
        match result {
            Ok(Outcome::Generated(paths)) => {
                self.status = format!("{} image(s) enregistrée(s)", paths.len());
                self.outputs = paths;
                self.refresh_gallery();
            }
            Ok(Outcome::Catalog(id, files)) => {
                if self.active_model().id == id {
                    self.remote = Some((id, files));
                }
                self.status = "Catalogue chargé".into();
            }
            Ok(Outcome::Engine(exe, backend)) => {
                self.engine_edit = exe.display().to_string();
                self.preferences.engine = Some(exe);
                self.preferences.backend = backend;
                self.save();
                self.status = "Moteur installé".into();
            }
            Ok(Outcome::Installed) => {
                self.refresh_local();
                self.status = "Poids prêts".into();
            }
            Ok(Outcome::Gpus(gpus)) => {
                self.status = format!("{} GPU NVIDIA détecté(s)", gpus.len());
                self.gpus = gpus;
            }
            Err(error) => {
                self.status.clear();
                self.error = Some(format!("{error:#}"));
            }
        }
    }

    fn generate(&mut self, ctx: &egui::Context) {
        let Some(exe) = self.current_engine() else {
            self.page = Page::Settings;
            self.error = Some(
                "Indique le chemin de sd-cli.exe ou installe le moteur depuis Réglages.".into(),
            );
            return;
        };
        let model = self.active_model().clone();
        if self.form.prompt.trim().is_empty() {
            self.error = Some("Saisis un prompt.".into());
            return;
        }
        let align = |value: u32| {
            let multiple = model.dimension_multiple();
            value.clamp(256, 3072).div_ceil(multiple) * multiple
        };
        self.form.width = align(self.form.width);
        self.form.height = align(self.form.height);
        let seed = if self.form.seed < 0 {
            (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                % i32::MAX as u128) as i64
        } else {
            self.form.seed
        };
        let request = Request {
            root: self.root.clone(),
            exe,
            model,
            selection: self.selection(),
            preset: self.form.sampling.clone(),
            prompt: self.form.prompt.trim().into(),
            negative: self.form.negative.trim().into(),
            references: self.form.references.clone(),
            width: self.form.width,
            height: self.form.height,
            seed,
            count: self.form.count,
            strength: self.form.strength,
            loras: self
                .form
                .lora
                .iter()
                .map(|name| (name.clone(), self.form.lora_weight))
                .collect(),
            backend: self.preferences.backend,
            gpu: self.preferences.gpu,
            auto_fit: !self.preferences.disable_auto_fit,
            flash_attention: !self.preferences.disable_flash,
        };
        self.outputs.clear();
        self.preview = None;
        self.start(
            ctx,
            "Préparation de la génération",
            move |cancel, event| engine::generate(request, cancel, event).map(Outcome::Generated),
        );
    }

    fn add_refs(&mut self, files: impl IntoIterator<Item = PathBuf>) {
        let max = self.active_model().max_references();
        let mut overflow = false;
        for file in files {
            if !file.is_file()
                || !file
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "png" | "jpg" | "jpeg" | "webp"
                        )
                    })
            {
                self.error = Some(format!(
                    "Image PNG, JPEG ou WebP introuvable : {}",
                    file.display()
                ));
            } else if !self.form.references.contains(&file) {
                if self.form.references.len() >= max {
                    overflow = true;
                } else {
                    self.form.references.push(file);
                }
            }
        }
        if overflow {
            self.error = Some(format!("Ce modèle accepte jusqu’à {max} référence(s)."));
        }
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
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("Rust · {}", env!("CARGO_PKG_VERSION")))
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

    fn model_picker(&mut self, ui: &mut egui::Ui) {
        let mut index = self.model;
        egui::ComboBox::from_id_salt("model")
            .width(264.0)
            .selected_text(&self.active_model().name)
            .show_ui(ui, |ui| {
                for (i, model) in self.catalog.base_models.iter().enumerate() {
                    ui.selectable_value(&mut index, i, &model.name);
                }
            });
        self.select_model(index);
    }

    fn weight_picker(
        &mut self,
        ui: &mut egui::Ui,
        role: &str,
        label: &str,
        choices: Vec<(Weight, String)>,
    ) {
        let mut selected = self.selection().remove(role);
        let before = selected.clone();
        let selected_text = selected
            .as_ref()
            .map(|file| file.name.as_str())
            .unwrap_or("Automatique");
        egui::ComboBox::from_id_salt(("weight", role))
            .width(300.0)
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut selected,
                    None,
                    "Automatique · priorité aux poids installés",
                );
                for (file, text) in choices {
                    ui.selectable_value(&mut selected, Some(file), text);
                }
            });
        ui.label(RichText::new(label).small().color(MUTED));
        if before != selected {
            self.set_weight(role, selected);
        }
    }

    fn images(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let dropped = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect::<Vec<_>>()
        });
        if self.job.is_none() {
            self.add_refs(dropped);
        }
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
                                self.image_controls(ctx, ui)
                            });
                        });
                },
            );
            ui.separator();
            ui.vertical(|ui| {
                ui.heading("Aperçu");
                let area = Vec2::new(
                    ui.available_width(),
                    (ui.available_height() - 180.0).max(230.0),
                );
                if let Some(texture) = &self.preview {
                    show_texture(ui, texture, area);
                } else {
                    placeholder(ui, area, "L’aperçu apparaît pendant la génération");
                }
                if !self.outputs.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        for (index, path) in self.outputs.iter().enumerate() {
                            if ui.button(format!("Image {}", index + 1)).clicked()
                                && let Ok(frame) = engine::preview_image(path)
                            {
                                put_texture(ctx, &mut self.preview, "generation", frame);
                            }
                        }
                    });
                }
                if ui.button("Ouvrir le dossier des images").clicked() {
                    reveal(self.root.join("outputs"), &mut self.error);
                }
                if let Some(error) = &self.error {
                    ui.label(RichText::new(error).color(Color32::LIGHT_RED));
                }
                ui.collapsing("Journal du moteur", |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(150.0)
                        .stick_to_bottom(true)
                        .id_salt("log")
                        .show(ui, |ui| {
                            for line in &self.log {
                                ui.monospace(line);
                            }
                        });
                });
            });
        });
    }

    fn image_controls(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            self.model_picker(ui);
            if ui.button("Poids").clicked() {
                self.page = Page::Models;
            }
        });
        ui.label(RichText::new("Prompt").strong());
        ui.add(
            egui::TextEdit::multiline(&mut self.form.prompt)
                .desired_rows(4)
                .desired_width(f32::INFINITY)
                .hint_text("Décris l’image ou la retouche…"),
        );
        let model = self.active_model().clone();
        frame(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Images de référence").strong());
                ui.label(
                    RichText::new(format!(
                        "{}/{}",
                        self.form.references.len(),
                        model.max_references()
                    ))
                    .small()
                    .color(MUTED),
                );
                if ui.button("Ajouter").clicked() {
                    self.add_refs(pick_images());
                }
            });
            ui.label(RichText::new(model.reference_hint()).small().color(MUTED));
            if self.form.references.is_empty() {
                ui.label(
                    RichText::new("Glisse tes fichiers ici")
                        .small()
                        .color(MUTED),
                );
            }
            let mut remove = None;
            let mut swap = None;
            for (index, path) in self.form.references.iter().enumerate() {
                ui.push_id(index, |ui| {
                    ui.horizontal(|ui| {
                        let filename = path.file_name().unwrap_or_default().to_string_lossy();
                        ui.label(format!("{}. {filename}", index + 1))
                            .on_hover_text(path.display().to_string());
                        if ui
                            .add_enabled(index > 0, egui::Button::new("↑").small())
                            .clicked()
                        {
                            swap = Some((index, index - 1));
                        }
                        if ui.button("Retirer").clicked() {
                            remove = Some(index);
                        }
                    });
                });
            }
            if let Some((a, b)) = swap {
                self.form.references.swap(a, b);
            }
            if let Some(index) = remove {
                self.form.references.remove(index);
            }
            #[cfg(not(windows))]
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.form.reference_text);
                if ui.button("Ajouter ce chemin").clicked() {
                    let path = PathBuf::from(std::mem::take(&mut self.form.reference_text));
                    self.add_refs([path]);
                }
            });
        });
        if model.defaults.supports_negative {
            ui.label("Prompt négatif");
            ui.add(
                egui::TextEdit::singleline(&mut self.form.negative).desired_width(f32::INFINITY),
            );
        }
        ui.horizontal(|ui| {
            let before = self.form.preset;
            egui::ComboBox::from_id_salt("preset")
                .width(246.0)
                .selected_text(&self.form.sampling.name)
                .show_ui(ui, |ui| {
                    for (index, preset) in model.presets.iter().enumerate() {
                        ui.selectable_value(&mut self.form.preset, index, &preset.name);
                    }
                });
            if before != self.form.preset {
                self.form.sampling = model.presets[self.form.preset].clone();
            }
            ui.label("Preset");
        });
        if !self.form.sampling.note.is_empty() {
            ui.label(RichText::new(&self.form.sampling.note).small().color(MUTED));
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Carré").clicked() {
                (self.form.width, self.form.height) = (1024, 1024);
            }
            if ui.button("Paysage").clicked() {
                (self.form.width, self.form.height) = (1280, 768);
            }
            if ui.button("Portrait").clicked() {
                (self.form.width, self.form.height) = (768, 1280);
            }
            ui.label(format!("{} × {}", self.form.width, self.form.height));
        });
        let ready = catalog::ROLES.iter().all(|(role, _)| {
            !model.needs(role, !self.form.references.is_empty(), false)
                || catalog::chosen(role, &self.selection(), &self.local)
                    .is_some_and(|file| catalog::present(&file.path(&self.root)))
        });
        if !ready {
            ui.label(
                RichText::new("Poids manquants. Ouvre Modèles pour les télécharger.").color(ACCENT),
            );
        }
        let choices = self
            .local
            .get("diffusion")
            .into_iter()
            .flatten()
            .map(|file| (file.clone(), file.name.clone()))
            .collect();
        ui.horizontal_wrapped(|ui| self.weight_picker(ui, "diffusion", "Diffusion", choices));
        let run = egui::Button::new(
            RichText::new("Générer")
                .strong()
                .color(Color32::from_rgb(31, 22, 18)),
        )
        .fill(ACCENT)
        .min_size(Vec2::new(180.0, 38.0));
        let shortcut =
            ctx.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::Enter));
        if ui.add(run).clicked() || shortcut {
            self.generate(ctx);
        }
        ui.collapsing("Variantes installées", |ui| {
            for (role, label) in catalog::ROLES.iter().skip(1).take(2) {
                let choices = self
                    .local
                    .get(*role)
                    .into_iter()
                    .flatten()
                    .map(|file| (file.clone(), file.name.clone()))
                    .collect();
                self.weight_picker(ui, role, label, choices);
            }
        });
        ui.collapsing("Résolution et échantillonnage", |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.form.width)
                        .range(256..=3072)
                        .prefix("L "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.form.height)
                        .range(256..=3072)
                        .prefix("H "),
                );
            });
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.form.seed)
                        .range(-1..=i64::MAX)
                        .prefix("Seed "),
                );
                ui.label("-1 = aléatoire");
                ui.add(
                    egui::DragValue::new(&mut self.form.count)
                        .range(1..=4)
                        .prefix("Images "),
                );
            });
            let fixed = model.fixed_sampling() || self.form.sampling.adapter().is_some();
            ui.add_enabled_ui(!fixed, |ui| {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.form.sampling.steps)
                            .range(1..=100)
                            .suffix(" étapes"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.form.sampling.cfg_scale)
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .prefix("CFG "),
                    );
                });
            });
            ui.label(format!(
                "{} / {}",
                self.form.sampling.sampler, self.form.sampling.scheduler
            ));
            if !model.native_references() && !self.form.references.is_empty() {
                ui.add(
                    egui::Slider::new(&mut self.form.strength, 0.05..=1.0).text("Force img2img"),
                );
            }
        });
        ui.collapsing("LoRA local", |ui| {
            egui::ComboBox::from_id_salt("lora")
                .width(246.0)
                .selected_text(self.form.lora.as_deref().unwrap_or("Aucun"))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.form.lora, None, "Aucun");
                    for name in &self.loras {
                        ui.selectable_value(&mut self.form.lora, Some(name.clone()), name);
                    }
                });
            if self.form.lora.is_some() {
                ui.add(egui::Slider::new(&mut self.form.lora_weight, -2.0..=2.0).text("Intensité"));
            }
            if ui.button("Rafraîchir").clicked() {
                self.refresh_local();
            }
        });
    }

    fn models(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.heading("Modèles");
        ui.label(RichText::new("Télécharge les variantes du catalogue ou réutilise les poids du dossier TurboSlop.").color(MUTED));
        ui.add_enabled_ui(self.job.is_none(), |ui| {
            self.model_picker(ui);
            let model = self.active_model().clone();
            ui.label(&model.description);
            ui.horizontal_wrapped(|ui| {
                if ui.button("Voir les variantes disponibles").clicked() {
                    let model = model.clone();
                    self.start(ctx, "Lecture du catalogue", move |cancel, _| {
                        download::catalog_weights(&model, cancel)
                            .map(|files| Outcome::Catalog(model.id, files))
                    });
                }
                if ui.button("Détecter les poids locaux").clicked() {
                    self.refresh_local();
                }
                if ui.button("Ouvrir le dossier des modèles").clicked() {
                    reveal(self.root.join("models"), &mut self.error);
                }
            });
            ui.add_space(8.0);
            for (role, label) in catalog::ROLES {
                if !model.sources.contains_key(*role) {
                    continue;
                }
                let mut choices: Vec<(Weight, String)> = self
                    .local
                    .get(*role)
                    .into_iter()
                    .flatten()
                    .map(|file| (file.clone(), format!("{} · installé", file.name)))
                    .collect();
                if let Some((_, remote)) = &self.remote
                    && let Some(files) = remote.get(*role)
                {
                    for file in files {
                        if !choices.iter().any(|(weight, _)| weight == &file.weight) {
                            choices.push((
                                file.weight.clone(),
                                format!("{} · {:.2} Go", file.weight.name, download::gb(file.size)),
                            ));
                        }
                    }
                }
                ui.horizontal_wrapped(|ui| self.weight_picker(ui, role, label, choices));
            }
            if model
                .sources
                .get("text_encoder_vision")
                .is_some_and(|s| s.optional)
            {
                ui.checkbox(
                    &mut self.install_vision,
                    "Inclure le projecteur vision pour l’édition",
                );
            }
            if ui.button("Télécharger les poids sélectionnés").clicked() {
                let root = self.root.clone();
                let selection = self.selection();
                let install_vision = self.install_vision;
                let remote = self.remote.as_ref().map(|(_, files)| files.clone());
                let model = model.clone();
                self.start(
                    ctx,
                    "Préparation du téléchargement",
                    move |cancel, event| {
                        let local = catalog::installed(&model, &root);
                        let missing = catalog::ROLES.iter().any(|(role, _)| {
                            model.needs(role, false, install_vision)
                                && catalog::chosen(role, &selection, &local)
                                    .is_none_or(|file| !catalog::present(&file.path(&root)))
                        });
                        if missing {
                            let remote = match remote {
                                Some(files) => files,
                                None => download::catalog_weights(&model, cancel)?,
                            };
                            download::install_model(
                                &model,
                                &root,
                                &remote,
                                &selection,
                                install_vision,
                                cancel,
                                &mut |label, fraction| event(Update::Progress(label, fraction)),
                            )?;
                        }
                        Ok(Outcome::Installed)
                    },
                );
            }
            for preset in &model.presets {
                let Some(adapter) = preset.adapter() else {
                    continue;
                };
                let installed = adapter
                    .path(&self.root)
                    .metadata()
                    .is_ok_and(|m| m.len() == adapter.size);
                let label = if installed {
                    format!("{} · LoRA installé", preset.name)
                } else {
                    format!(
                        "Installer le LoRA · {} · {:.2} Go",
                        preset.name,
                        download::gb(adapter.size)
                    )
                };
                if ui
                    .add_enabled(!installed, egui::Button::new(label))
                    .clicked()
                {
                    let destination = adapter.path(&self.root);
                    self.start(
                        ctx,
                        "Installation de l’adaptateur",
                        move |cancel, event| {
                            download::install_weight(
                                &download::client(),
                                &adapter.remote(),
                                &destination,
                                cancel,
                                &mut |label, fraction| event(Update::Progress(label, fraction)),
                            )?;
                            Ok(Outcome::Installed)
                        },
                    );
                }
            }
        });
    }

    fn settings(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.heading("Réglages");
        ui.add_enabled_ui(self.job.is_none(), |ui| {
            ui.label(RichText::new("Dossier TurboSlop").strong());
            ui.label("Choisis le dossier principal d’une installation existante pour retrouver tes modèles, LoRA et images.");
            ui.horizontal_wrapped(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.root_edit).desired_width(480.0));
                if ui.button("Parcourir").clicked() && let Some(path) = pick_folder() {
                    self.root_edit = path.display().to_string();
                }
                if ui.button("Utiliser ce dossier").clicked() {
                    let path = PathBuf::from(self.root_edit.trim());
                    match (!path.as_os_str().is_empty()).then(|| std::path::absolute(path)).transpose() {
                        Ok(Some(path)) if path.is_dir() => {
                            self.root = path;
                            self.preferences = read_preferences(&self.root);
                            self.root_edit = self.root.display().to_string();
                            self.engine_edit = self.preferences.engine.as_ref().map_or_else(String::new, |p| p.display().to_string());
                            self.install_backend = self.preferences.backend;
                            self.remote = None;
                            self.preview = None;
                            self.outputs.clear();
                            self.gallery_selected = None;
                            self.gallery_preview = None;
                            self.refresh_local();
                            self.refresh_gallery();
                            if let Some(session) = session_path() && let Err(error) = save_json(&session, &self.root) {
                                self.error = Some(format!("Dossier non mémorisé. {error:#}"));
                            }
                        }
                        _ => self.error = Some("Sélectionne un dossier existant.".into()),
                    }
                }
            });
            ui.separator();
            ui.label(RichText::new("Moteur sd.cpp").strong());
            if let Some(exe) = self.current_engine() {
                ui.label(format!("Actif · {}", exe.display()));
            } else {
                ui.label(RichText::new("Moteur à installer ou à sélectionner").color(ACCENT));
            }
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_salt("backend").selected_text(self.install_backend.label()).show_ui(ui, |ui| {
                    for backend in Backend::ALL { ui.selectable_value(&mut self.install_backend, backend, backend.label()); }
                });
                if ui.button("Installer / mettre à jour").clicked() {
                    let root = self.root.clone();
                    let backend = self.install_backend;
                    self.start(ctx, "Installation du moteur", move |cancel, event| {
                        let path = engine::install(&root, backend, cancel,
                            &mut |label, fraction| event(Update::Progress(label, fraction)))?;
                        Ok(Outcome::Engine(path, backend))
                    });
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.engine_edit).desired_width(480.0).hint_text("Chemin vers sd-cli.exe"));
                if ui.button("Parcourir").clicked() && let Some(path) = pick_engine() {
                    self.engine_edit = path.display().to_string();
                }
                if ui.button("Utiliser ce moteur").clicked() {
                    let path = PathBuf::from(self.engine_edit.trim());
                    if path.is_file() {
                        match std::path::absolute(path) {
                            Ok(path) => {
                                self.preferences.engine = Some(path);
                                self.preferences.backend = self.install_backend;
                                self.save();
                            }
                            Err(error) => self.error = Some(error.to_string()),
                        }
                    } else if path.as_os_str().is_empty() {
                        self.preferences.engine = None;
                        self.preferences.backend = self.install_backend;
                        self.save();
                    } else {
                        self.error = Some("Fichier moteur introuvable.".into());
                    }
                }
            });
            ui.separator();
            ui.label(RichText::new("GPU et mémoire").strong());
            let old_gpu = self.preferences.gpu;
            ui.horizontal_wrapped(|ui| {
                let selected = self.preferences.gpu.map(|n| format!("GPU {n}")).unwrap_or_else(|| "Automatique".into());
                egui::ComboBox::from_id_salt("gpu").selected_text(selected).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.preferences.gpu, None, "Automatique");
                    for line in &self.gpus {
                        if let Some(index) = line.split(',').next().and_then(|s| s.trim().parse().ok()) {
                            ui.selectable_value(&mut self.preferences.gpu, Some(index), line);
                        }
                    }
                });
                if ui.button("Détecter les GPU NVIDIA").clicked() {
                    self.start(ctx, "Détection des GPU", |_, _| Ok(Outcome::Gpus(engine::gpu_list())));
                }
            });
            let mut auto = !self.preferences.disable_auto_fit;
            let mut flash = !self.preferences.disable_flash;
            let changed = ui.checkbox(&mut auto, "Placement automatique RAM / VRAM").changed()
                | ui.checkbox(&mut flash, "Flash Attention").changed();
            self.preferences.disable_auto_fit = !auto;
            self.preferences.disable_flash = !flash;
            if old_gpu != self.preferences.gpu || changed { self.save(); }
            ui.label(RichText::new("Le choix du GPU NVIDIA s’applique au moteur CUDA.").color(MUTED));
        });
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
                                    self.form.prompt = prompt;
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
        if self.job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
            if ctx.input(|i| i.viewport().close_requested()) {
                self.closing = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                if let Some(job) = &self.job {
                    job.cancel.store(true, Ordering::Relaxed);
                }
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

fn frame(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    Frame::new()
        .fill(PANEL)
        .inner_margin(Margin::same(12))
        .corner_radius(8)
        .show(ui, add);
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
        .map(|root| root.join("TurboSlop-Rust").join("workspace.json"))
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
    root.join("userdata-rust").join("preferences.json")
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

fn local_loras(root: &Path) -> Vec<String> {
    fn collect(root: &Path, folder: &Path, depth: u8, names: &mut Vec<String>) {
        for entry in fs::read_dir(folder).into_iter().flatten().flatten() {
            let path = entry.path();
            if depth < 2 && entry.file_type().is_ok_and(|t| t.is_dir()) {
                collect(root, &path, depth + 1, names);
            } else if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|e| matches!(e, "safetensors" | "gguf"))
                && let Ok(relative) = path.strip_prefix(root)
            {
                let name = relative
                    .with_extension("")
                    .to_string_lossy()
                    .replace('\\', "/");
                if catalog::safe_name(&name) {
                    names.push(name);
                }
            }
        }
    }
    let dir = root.join("loras");
    let mut names = Vec::new();
    collect(&dir, &dir, 0, &mut names);
    names.sort();
    names.dedup();
    names
}

#[cfg(windows)]
fn pick_images() -> Vec<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
        .pick_files()
        .unwrap_or_default()
}
#[cfg(not(windows))]
fn pick_images() -> Vec<PathBuf> {
    Vec::new()
}

#[cfg(windows)]
fn pick_folder() -> Option<PathBuf> {
    rfd::FileDialog::new().pick_folder()
}
#[cfg(not(windows))]
fn pick_folder() -> Option<PathBuf> {
    None
}

#[cfg(windows)]
fn pick_engine() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Exécutable Windows", &["exe"])
        .pick_file()
}
#[cfg(not(windows))]
fn pick_engine() -> Option<PathBuf> {
    None
}
