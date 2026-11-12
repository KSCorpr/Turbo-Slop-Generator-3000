use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use glob::Pattern;
use serde::{Deserialize, Serialize};

const CATALOG: &str = include_str!("../../config/models.yaml");
const DIFFUSION_QUANT: &str = "Q4_K_M";
const ENCODER_QUANT: &str = "Q4_K_M";

#[derive(Deserialize)]
pub struct Catalog {
    pub base_models: Vec<Model>,
}

#[derive(Clone, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub family: String,
    pub description: String,
    pub sources: BTreeMap<String, Source>,
    pub defaults: Defaults,
    pub presets: Vec<Preset>,
}

#[derive(Clone, Deserialize)]
pub struct Source {
    pub repo: String,
    #[serde(rename = "match")]
    pattern: String,
    #[serde(default)]
    selectable: Vec<String>,
    #[serde(default)]
    alternatives: Vec<Alternative>,
    auto_quant: Option<String>,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Clone, Deserialize)]
struct Alternative {
    repo: String,
    #[serde(rename = "match")]
    pattern: String,
}

#[derive(Clone, Deserialize)]
pub struct Defaults {
    pub steps: u32,
    pub cfg_scale: f32,
    pub sampler: String,
    pub scheduler: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub flow_shift: f32,
    #[serde(default)]
    pub supports_negative: bool,
    #[serde(default)]
    pub extra_flags: Vec<String>,
    #[serde(default)]
    pub edit: String,
    #[serde(default)]
    pub requires_reference: bool,
    #[serde(default)]
    pub fixed_sampling: bool,
    #[serde(default)]
    pub min_engine_build: u32,
    max_references: Option<usize>,
    size_multiple: Option<u32>,
}

#[derive(Clone, Deserialize)]
pub struct Preset {
    pub name: String,
    pub sampler: String,
    pub scheduler: String,
    pub steps: u32,
    pub cfg_scale: f32,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub moss_dmd: bool,
    #[serde(default)]
    pub kromoss_dmd2: bool,
    #[serde(default)]
    pub krea_distill: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Weight {
    pub repo: String,
    pub name: String,
}

pub type Selection = BTreeMap<String, Weight>;

#[derive(Clone)]
pub struct RemoteWeight {
    pub weight: Weight,
    pub revision: String,
    pub size: u64,
    pub sha256: Option<String>,
}

pub struct Adapter {
    pub repo: &'static str,
    pub revision: &'static str,
    pub name: &'static str,
    pub tag: &'static str,
    pub strength: f32,
    pub size: u64,
    pub sha256: &'static str,
}

pub const ROLES: &[(&str, &str)] = &[
    ("diffusion", "Diffusion"),
    ("text_encoder", "Encodeur texte"),
    ("vae", "VAE"),
    ("text_encoder_vision", "Vision"),
    ("tokenizer", "Tokenizer"),
];

impl Catalog {
    pub fn load() -> Result<Self> {
        let catalog: Self =
            serde_yaml::from_str(CATALOG).context("Catalogue de modèles invalide")?;
        ensure!(!catalog.base_models.is_empty(), "Catalogue vide");
        Ok(catalog)
    }
}

impl Model {
    pub fn native_references(&self) -> bool {
        self.defaults.edit == "full"
    }

    pub fn max_references(&self) -> usize {
        if self.native_references() {
            self.defaults.max_references.unwrap_or(3)
        } else {
            1
        }
    }

    pub fn dimension_multiple(&self) -> u32 {
        self.defaults
            .size_multiple
            .unwrap_or(if self.family == "qwen21" { 32 } else { 16 })
    }

    pub fn fixed_sampling(&self) -> bool {
        self.defaults.fixed_sampling || self.family == "qwen21"
    }

    pub fn needs(&self, role: &str, references: bool, install_vision: bool) -> bool {
        self.sources.get(role).is_some_and(|source| {
            if role == "text_encoder_vision" {
                (references && self.native_references()) || install_vision || !source.optional
            } else {
                !source.optional
            }
        })
    }

    pub fn reference_hint(&self) -> &'static str {
        match self.family.as_str() {
            "minimax_h3_image" => {
                "Référence les images par leur ordre avec <Picture 1>, <Picture 2>, etc."
            }
            "fibo2" | "fibo2_turbo" => "Référence les images avec <image_1>, <image_2>, etc.",
            _ if self.native_references() => {
                "Décris la modification et l’ordre des images dans le prompt."
            }
            _ => "Image de départ facultative pour une génération image-to-image.",
        }
    }
}

impl Preset {
    pub fn adapter(&self) -> Option<Adapter> {
        if self.krea_distill {
            Some(Adapter {
                repo: "lvladikov/Krea2-Turbo-Distill-4step-LoRA",
                revision: "d4d6c8e9edb60587f9f6531b89995e668d5982dc",
                name: "krea2_turbo_4step_rank_64_lora_comfyui.safetensors",
                tag: "krea2-turbo-distill/krea2_turbo_4step_rank_64_lora_comfyui",
                strength: 1.0,
                size: 438_141_176,
                sha256: "27a3d1644fa1a8c137b8549591e50acc4746f8e9bc736fccff80e506a8ee8ac7",
            })
        } else if self.kromoss_dmd2 {
            Some(Adapter {
                repo: "lodestones/Kromoss",
                revision: "0c5417ec89301cb82da9ee164f8ed696523910a9",
                name: "8-steps-dmd2-lora.safetensors",
                tag: "kromoss-dmd2/8-steps-dmd2-lora",
                strength: 1.0,
                size: 1_189_176_800,
                sha256: "7683fe1fb51f73f5d690f24dd6b7f46f31ab9a9873729a854ac5c9a420ffe6f2",
            })
        } else if self.moss_dmd {
            Some(Adapter {
                repo: "lodestones/MOSS-Image-SFT-8-steps-dmd",
                revision: "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f",
                name: "lora_latest.safetensors",
                tag: "moss-image-sft-dmd/lora_latest",
                strength: 0.7,
                size: 1_189_176_800,
                sha256: "be82ab2bef78194651815bb6de443cec67553e01c14559f2ad365449445f0452",
            })
        } else {
            None
        }
    }
}

impl Adapter {
    pub fn path(&self, root: &Path) -> PathBuf {
        root.join("loras").join(format!("{}.safetensors", self.tag))
    }

    pub fn remote(&self) -> RemoteWeight {
        RemoteWeight {
            weight: Weight {
                repo: self.repo.into(),
                name: self.name.into(),
            },
            revision: self.revision.into(),
            size: self.size,
            sha256: Some(self.sha256.into()),
        }
    }
}

fn matches(pattern: &str, name: &str) -> bool {
    Pattern::new(pattern).is_ok_and(|p| p.matches(name))
}

pub fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.split('/').all(|part| {
            !part.is_empty()
                && !part.starts_with('.')
                && !part.ends_with(['.', ' '])
                && !part
                    .chars()
                    .any(|c| c.is_control() || "\\:\"<>|?*".contains(c))
                && ![
                    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6",
                    "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7",
                    "LPT8", "LPT9",
                ]
                .contains(
                    &part
                        .split('.')
                        .next()
                        .unwrap_or_default()
                        .to_ascii_uppercase()
                        .as_str(),
                )
        })
}

impl Source {
    pub fn repositories(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.repo.as_str()).chain(self.alternatives.iter().map(|a| a.repo.as_str()))
    }

    pub fn accepts(&self, weight: &Weight) -> bool {
        safe_name(&weight.name)
            && ((self.repo == weight.repo
                && if self.selectable.is_empty() {
                    matches(
                        &self
                            .pattern
                            .replace("{quant}", "*")
                            .replace("{enc_quant}", "*"),
                        &weight.name,
                    )
                } else {
                    self.selectable.contains(&weight.name)
                })
                || self
                    .alternatives
                    .iter()
                    .any(|a| a.repo == weight.repo && matches(&a.pattern, &weight.name)))
    }

    pub fn priority(&self, weight: &Weight) -> (u8, usize) {
        let preferred = self.pattern.replace("{quant}", DIFFUSION_QUANT).replace(
            "{enc_quant}",
            self.auto_quant.as_deref().unwrap_or(ENCODER_QUANT),
        );
        if self.repo == weight.repo && matches(&preferred, &weight.name) {
            return (0, 0);
        }
        if self.pattern.contains("{") && self.repo == weight.repo {
            let filename = weight.name.to_ascii_uppercase();
            let ladder = [
                "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q4_1", "Q3_K_M", "Q3_K_S", "Q3_K", "Q2_K_M",
                "Q2_K", "Q5_K_M", "Q5_K_S", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16", "F32",
            ];
            return (
                1,
                ladder
                    .iter()
                    .position(|q| filename.contains(&format!("-{q}.gguf").to_ascii_uppercase()))
                    .unwrap_or(ladder.len()),
            );
        }
        (
            2,
            self.selectable
                .iter()
                .position(|n| n == &weight.name)
                .unwrap_or(0),
        )
    }
}

impl Weight {
    pub fn path(&self, root: &Path) -> PathBuf {
        root.join("models")
            .join(self.repo.replace('/', "__"))
            .join(&self.name)
    }
}

pub fn installed(model: &Model, root: &Path) -> BTreeMap<String, Vec<Weight>> {
    let mut repositories = HashMap::new();
    model
        .sources
        .iter()
        .map(|(role, source)| {
            let mut weights = Vec::new();
            for repo in source.repositories() {
                let files = repositories.entry(repo).or_insert_with(|| {
                    let folder = root.join("models").join(repo.replace('/', "__"));
                    let mut files = Vec::new();
                    scan(&folder, &folder, 0, &mut files);
                    files
                });
                weights.extend(files.iter().filter_map(|name| {
                    let weight = Weight {
                        repo: repo.into(),
                        name: name.clone(),
                    };
                    source.accepts(&weight).then_some(weight)
                }));
            }
            weights.sort();
            weights.dedup();
            weights.sort_by_key(|w| source.priority(w));
            (role.clone(), weights)
        })
        .collect()
}

fn scan(root: &Path, folder: &Path, depth: usize, files: &mut Vec<String>) {
    if depth > 8 {
        return;
    }
    for entry in fs::read_dir(folder).into_iter().flatten().flatten() {
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let path = entry.path();
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            scan(root, &path, depth + 1, files);
        } else if present(&path)
            && let Ok(relative) = path.strip_prefix(root)
        {
            files.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

pub fn present(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|meta| meta.is_file() && meta.len() > 0)
}

pub fn chosen<'a>(
    role: &str,
    selection: &'a Selection,
    installed: &'a BTreeMap<String, Vec<Weight>>,
) -> Option<&'a Weight> {
    selection.get(role).or_else(|| installed.get(role)?.first())
}
