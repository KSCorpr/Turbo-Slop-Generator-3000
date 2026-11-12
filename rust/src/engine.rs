//! FLUX.2 Klein runs here, inside TurboSlop. Burn owns GPU buffers and
//! dispatches the Rust tensor operations through Vulkan.
use std::{
    collections::BTreeMap,
    fs,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, atomic::AtomicBool},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, ensure};
use burn::{
    backend::wgpu::{RuntimeOptions, Wgpu, WgpuDevice, WgpuRuntime, graphics::Vulkan, init_setup},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use combs_diffusion::{
    NoiseSource,
    flux2::{
        FlowMatchEuler, Flux2Config, Flux2LatentStats, Flux2Transformer, QuantizingSource,
        image_ids, text_ids, unpack_latents, unpatchify_latents,
    },
    vae::VAEDecoder,
};
use combs_formats::{
    ModelMetadata, ModelSource, QuantFormat, QuantTensor, SafetensorsSource, TensorDtype,
    TensorReader, TokenizerSpec, open_model_source, quants::dequantize_q6_k,
};
use combs_models::{CacheConfig, GenerativeModel, LlamaModel};
use cubecl::prelude::Runtime;
use eframe::egui::ColorImage;
use image::{DynamicImage, ImageFormat, RgbImage};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::{catalog, download};

type Gpu = Wgpu;
const PROMPT_TOKENS: usize = 512;
const TAPS: [usize; 3] = [9, 18, 27];
pub const MIN_EDGE: u32 = 256;
pub const MAX_EDGE: u32 = 768;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Device {
    #[default]
    Auto,
    FirstGpu,
    SecondGpu,
}

impl Device {
    pub const CHOICES: [Self; 3] = [Self::Auto, Self::FirstGpu, Self::SecondGpu];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "GPU automatique",
            Self::FirstGpu => "GPU dédié 1",
            Self::SecondGpu => "GPU dédié 2",
        }
    }

    fn wgpu(self) -> WgpuDevice {
        match self {
            Self::Auto => WgpuDevice::DefaultDevice,
            Self::FirstGpu => WgpuDevice::DiscreteGpu(0),
            Self::SecondGpu => WgpuDevice::DiscreteGpu(1),
        }
    }
}

pub struct Request {
    pub root: PathBuf,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub device: Device,
}

pub enum Update {
    Progress(String, Option<f32>),
    Preview(ColorImage),
}

#[derive(Serialize)]
struct Record<'a> {
    model: &'static str,
    engine: &'static str,
    gpu: &'a str,
    prompt: &'a str,
    seed: u64,
    width: u32,
    height: u32,
    steps: usize,
    sampler: &'static str,
}

fn select_device(choice: Device) -> Result<(WgpuDevice, String)> {
    static DEVICES: OnceLock<Mutex<BTreeMap<Device, String>>> = OnceLock::new();
    let cache = DEVICES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut entries = cache.lock().unwrap_or_else(|p| p.into_inner());
    let device = choice.wgpu();
    if let Some(name) = entries.get(&choice) {
        return Ok((device, name.clone()));
    }
    let setup = catch_unwind(|| init_setup::<Vulkan>(&device, RuntimeOptions::default())).map_err(
        |error| {
            anyhow!(
                "Initialisation de Vulkan impossible. {}",
                panic_text(&*error)
            )
        },
    )?;
    let name = setup.adapter.get_info().name;
    entries.insert(choice, name.clone());
    Ok((device, name))
}

pub fn probe(choice: Device) -> Result<String> {
    let (device, name) = select_device(choice)?;
    catch_unwind(AssertUnwindSafe(|| {
        let two = Tensor::<Gpu, 1>::from_data([2.0], &device);
        let result = (two * 3.0)
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| anyhow!("Lecture du résultat GPU impossible : {error:?}"))?;
        ensure!(result == [6.0], "Résultat GPU invalide.");
        Ok(name)
    }))
    .unwrap_or_else(|error| Err(anyhow!("Test Vulkan impossible. {}", panic_text(&*error))))
}

fn panic_text(error: &(dyn std::any::Any + Send)) -> &str {
    error
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| error.downcast_ref::<&str>().copied())
        .unwrap_or("Erreur du moteur de calcul.")
}

fn weights_dir(path: &Path) -> Result<&Path> {
    path.parent().context("Dossier des poids absent")
}

pub fn valid_edge(n: u32) -> bool {
    (MIN_EDGE..=MAX_EDGE).contains(&n) && n.is_multiple_of(16)
}

fn validate(req: &Request) -> Result<()> {
    ensure!(!req.prompt.trim().is_empty(), "Saisis un prompt.");
    ensure!(
        valid_edge(req.width) && valid_edge(req.height),
        "Utilise des dimensions multiples de 16, entre {MIN_EDGE} et {MAX_EDGE} pixels."
    );
    for weight in catalog::WEIGHTS {
        ensure!(
            weight.installed(&req.root),
            "Fichier manquant ou incomplet. Télécharge {} dans Modèles.",
            weight.label
        );
    }
    Ok(())
}

fn render_prompt(template: Option<&str>, prompt: &str) -> Result<String> {
    if let Some(template) = template {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        return env
            .render_str(
                template,
                context! {
                    messages => vec![serde_json::json!({"role":"user","content":prompt})],
                    tools => serde_json::Value::Null,
                    add_generation_prompt => true,
                    enable_thinking => false,
                },
            )
            .context("Le modèle de conversation Qwen3 n’a pas pu être appliqué");
    }
    Ok(format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ))
}

// Klein only consumes three residual streams from the encoder. Use a prompt-sized
// embedding table and stop the decoder stack as soon as the last tap is available.
struct PromptEncoderSource<'a> {
    source: &'a dyn ModelSource,
    metadata: ModelMetadata,
    embedding_name: &'static str,
    embedding_bytes: Vec<u8>,
}

impl<'a> PromptEncoderSource<'a> {
    fn new(source: &'a dyn ModelSource, ids: &mut [u32]) -> Result<Self> {
        let mut metadata = source.metadata().clone();
        ensure!(
            metadata.architecture == "qwen3"
                && metadata.num_hidden_layers >= TAPS[2]
                && metadata.hidden_size * TAPS.len() == Flux2Config::klein_4b().joint_attention_dim,
            "Cet encodeur ne correspond pas à FLUX.2 Klein 4B."
        );
        let embedding_name = if source
            .tensor_names()
            .iter()
            .any(|name| name == "model.embed_tokens.weight")
        {
            "model.embed_tokens.weight"
        } else {
            "embed_tokens.weight"
        };
        let table = source
            .open_tensor_quant(embedding_name)?
            .context("Table de vocabulaire quantifiée absente")?;
        ensure!(
            table.format == QuantFormat::Q6K
                && table.shape == [metadata.vocab_size, metadata.hidden_size]
                && metadata.hidden_size.is_multiple_of(256),
            "Table de vocabulaire Qwen3 inattendue."
        );
        let row_bytes = metadata.hidden_size / 256 * 210;
        let mut rows = BTreeMap::new();
        let mut selected = Vec::new();
        for id in ids {
            let next = rows.len() as u32;
            let row = rows.entry(*id).or_insert_with(|| {
                selected.push(*id as usize);
                next
            });
            *id = *row;
        }
        let mut packed = Vec::with_capacity(selected.len() * row_bytes);
        for index in selected {
            let offset = index * row_bytes;
            packed.extend_from_slice(
                table
                    .data
                    .get(offset..offset + row_bytes)
                    .context("Token Qwen3 hors vocabulaire")?,
            );
        }
        let values = dequantize_q6_k(&packed, rows.len() * metadata.hidden_size)?;
        let embedding_bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
        metadata.vocab_size = rows.len();
        metadata.num_hidden_layers = TAPS[2];
        metadata.max_position_embeddings = PROMPT_TOKENS;
        metadata.tie_word_embeddings = true;
        Ok(Self {
            source,
            metadata,
            embedding_name,
            embedding_bytes,
        })
    }
}

impl ModelSource for PromptEncoderSource<'_> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensor_names(&self) -> Vec<String> {
        self.source.tensor_names()
    }

    fn sampler_defaults(&self) -> Option<combs_formats::SamplerConfig> {
        self.source.sampler_defaults()
    }

    fn tokenizer(&self) -> combs_formats::Result<TokenizerSpec> {
        self.source.tokenizer()
    }

    fn open_tensor(&self, name: &str) -> combs_formats::Result<TensorReader<'_>> {
        if name == self.embedding_name {
            return Ok(TensorReader::new(
                name.to_string(),
                vec![self.metadata.vocab_size, self.metadata.hidden_size],
                TensorDtype::F32,
                &self.embedding_bytes,
            ));
        }
        self.source.open_tensor(name)
    }

    fn open_tensor_quant(&self, name: &str) -> combs_formats::Result<Option<QuantTensor<'_>>> {
        if name == self.embedding_name {
            return Ok(None);
        }
        self.source.open_tensor_quant(name)
    }
}

fn encode_prompt(path: &Path, prompt: &str, device: &WgpuDevice) -> Result<TensorData> {
    let source = open_model_source(path).context("Lecture de l’encodeur GGUF impossible")?;
    let spec = source.tokenizer()?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(spec.json_bytes()?)
        .map_err(|e| anyhow!("Tokenizer GGUF invalide : {e}"))?;
    let formatted = render_prompt(spec.chat_template.as_deref(), prompt)?;
    let encoding = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow!("Encodage du prompt impossible : {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    ensure!(
        ids.len() <= PROMPT_TOKENS,
        "Le prompt formaté dépasse {PROMPT_TOKENS} tokens."
    );
    let pad = tokenizer
        .token_to_id("<|endoftext|>")
        .context("Jeton de remplissage Qwen3 absent")?;
    ids.resize(PROMPT_TOKENS, pad);

    let encoder = PromptEncoderSource::new(source.as_ref(), &mut ids)?;
    let mut model =
        LlamaModel::<Gpu>::load(&encoder, device).context("Chargement de Qwen3 impossible")?;
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(PROMPT_TOKENS));
    let ids: Tensor<Gpu, 2, Int> = Tensor::from_data(
        TensorData::new(
            ids.into_iter().map(|id| id as i32).collect::<Vec<_>>(),
            [1, PROMPT_TOKENS],
        ),
        device,
    );
    let embeddings = model.embed(ids);
    let taps = model.prefill_taps(embeddings, cache.as_mut(), 0..PROMPT_TOKENS as u32, &TAPS)?;
    Ok(taps.into_data())
}

fn denoise(
    path: &Path,
    text: TensorData,
    req: &Request,
    device: &WgpuDevice,
    cancel: &AtomicBool,
    report: &mut impl FnMut(Update),
) -> Result<TensorData> {
    let source = SafetensorsSource::load_weights_only(weights_dir(path)?, "flux2-transformer")
        .context("Lecture du transformeur FLUX.2 impossible")?;
    let source = QuantizingSource::new(&source);
    let config = Flux2Config::klein_4b();
    let channels = config.in_channels;
    let model = Flux2Transformer::<Gpu>::load(&source, config, device)
        .context("Chargement du transformeur FLUX.2 impossible")?;
    download::check(cancel)?;
    let (gh, gw) = (req.height as usize / 16, req.width as usize / 16);
    let image_positions = image_ids(gh, gw);
    let text_positions = text_ids(PROMPT_TOKENS);
    let text: Tensor<Gpu, 3> = Tensor::from_data(text, device);
    let sampler = FlowMatchEuler::new(catalog::STEPS, gh * gw);
    let mut noise = NoiseSource::new(Some(req.seed));
    let mut latent = noise
        .normal_tensor::<Gpu>([1, channels, gh, gw], device)
        .reshape([1, channels, gh * gw])
        .permute([0, 2, 1]);
    for step in 0..sampler.num_steps() {
        download::check(cancel)?;
        let velocity = model.forward(
            latent.clone(),
            text.clone(),
            sampler.timestep(step),
            &image_positions,
            &text_positions,
        );
        latent = sampler.step(latent, velocity, step);
        Gpu::sync(device)?;
        report(Update::Progress(
            format!("Échantillonnage · {}/{}", step + 1, sampler.num_steps()),
            Some(0.35 + 0.50 * (step + 1) as f32 / sampler.num_steps() as f32),
        ));
    }
    Ok(latent.into_data())
}

fn decode(path: &Path, latent: TensorData, req: &Request, device: &WgpuDevice) -> Result<RgbImage> {
    let source = SafetensorsSource::load_weights_only(weights_dir(path)?, "flux2-vae")?;
    let vae = VAEDecoder::<Gpu>::load_with_latent_channels(&source, 32, device)?;
    let stats = Flux2LatentStats::<Gpu>::load(&source, device)?;
    let (gh, gw) = (req.height as usize / 16, req.width as usize / 16);
    let latent: Tensor<Gpu, 3> = Tensor::from_data(latent, device);
    let decoded = vae.forward(unpatchify_latents(
        stats.denormalize(unpack_latents(latent, gh, gw)),
    ));
    ensure!(
        decoded.dims() == [1, 3, req.height as usize, req.width as usize],
        "Dimensions de l’image décodée inattendues."
    );
    let pixels = decoded
        .permute([0, 2, 3, 1])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow!("Conversion des pixels impossible : {e:?}"))?;
    rgb_image(req.width, req.height, &pixels)
}

pub(crate) fn rgb_image(width: u32, height: u32, pixels: &[f32]) -> Result<RgbImage> {
    ensure!(
        pixels.iter().all(|value| value.is_finite()),
        "Le moteur a produit des pixels non finis."
    );
    let bytes = pixels
        .iter()
        .map(|value| (((*value * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round()) as u8)
        .collect();
    RgbImage::from_raw(width, height, bytes).context("Taille du tampon d’image invalide")
}

pub fn generate(
    req: Request,
    cancel: &AtomicBool,
    mut report: impl FnMut(Update),
) -> Result<Vec<PathBuf>> {
    validate(&req)?;
    download::check(cancel)?;
    report(Update::Progress(
        "Initialisation du moteur Rust · Vulkan".into(),
        None,
    ));
    let (device, gpu) = select_device(req.device)?;
    let client = WgpuRuntime::client(&device);
    let result = catch_unwind(AssertUnwindSafe(|| {
        report(Update::Progress(
            format!("Encodage du prompt · {gpu}"),
            Some(0.10),
        ));
        let text = encode_prompt(&catalog::ENCODER.path(&req.root), &req.prompt, &device)?;
        client.memory_cleanup();
        download::check(cancel)?;
        report(Update::Progress(
            "Chargement du transformeur · quantification Q8 en mémoire".into(),
            Some(0.20),
        ));
        let latent = denoise(
            &catalog::DIFFUSION.path(&req.root),
            text,
            &req,
            &device,
            cancel,
            &mut report,
        )?;
        client.memory_cleanup();
        download::check(cancel)?;
        report(Update::Progress("Décodage de l’image".into(), Some(0.92)));
        let image = decode(&catalog::VAE.path(&req.root), latent, &req, &device)?;
        download::check(cancel)?;
        let rgba = DynamicImage::ImageRgb8(image.clone()).to_rgba8();
        let output = publish(&req, &gpu, image)?;
        report(Update::Preview(ColorImage::from_rgba_unmultiplied(
            [req.width as usize, req.height as usize],
            rgba.as_raw(),
        )));
        Ok(vec![output])
    }));
    client.memory_cleanup();
    result.unwrap_or_else(|error| {
        Err(anyhow!(
            "Le calcul Vulkan a échoué. {}",
            panic_text(&*error)
        ))
    })
}

fn publish(req: &Request, gpu: &str, image: RgbImage) -> Result<PathBuf> {
    let folder = req.root.join("outputs");
    fs::create_dir_all(&folder)?;
    let mut temp = NamedTempFile::new_in(&folder)?;
    image.write_to(&mut temp, ImageFormat::Png)?;
    let ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let token = temp
        .path()
        .file_name()
        .context("Nom de sortie absent")?
        .to_string_lossy();
    let path = folder.join(format!(
        "{}-{ms}-{}.png",
        catalog::MODEL_ID,
        token.trim_start_matches('.')
    ));
    let record = Record {
        model: catalog::MODEL_ID,
        engine: "burn-vulkan",
        gpu,
        prompt: &req.prompt,
        seed: req.seed,
        width: req.width,
        height: req.height,
        steps: catalog::STEPS,
        sampler: "flow-match-euler",
    };
    fs::write(
        path.with_extension("rust.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    fs::write(
        path.with_extension("txt"),
        format!(
            "{}\nModel: {} ({})\nSteps: {}, Seed: {}\n",
            req.prompt,
            catalog::MODEL_NAME,
            catalog::MODEL_ID,
            catalog::STEPS,
            req.seed
        ),
    )?;
    temp.persist_noclobber(&path)?;
    Ok(path)
}

pub fn preview_image(path: &Path) -> Result<ColorImage> {
    let rgba = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .to_rgba8();
    Ok(ColorImage::from_rgba_unmultiplied(
        [rgba.width() as usize, rgba.height() as usize],
        rgba.as_raw(),
    ))
}
