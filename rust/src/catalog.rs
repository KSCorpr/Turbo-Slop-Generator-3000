use std::path::{Path, PathBuf};

pub const MODEL_ID: &str = "flux2-klein-4b";
pub const MODEL_NAME: &str = "FLUX.2 Klein 4B";
pub const STEPS: usize = 4;

pub struct Weight {
    pub label: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub filename: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const DIFFUSION: Weight = Weight {
    label: "Transformeur FLUX.2 · BF16 sur disque",
    repo: "black-forest-labs/FLUX.2-klein-4B",
    revision: "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    filename: "transformer/diffusion_pytorch_model.safetensors",
    size: 7_751_109_744,
    sha256: "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e",
};
pub const ENCODER: Weight = Weight {
    label: "Encodeur Qwen3 4B · Q4_K_M",
    repo: "Qwen/Qwen3-4B-GGUF",
    revision: "bc640142c66e1fdd12af0bd68f40445458f3869b",
    filename: "Qwen3-4B-Q4_K_M.gguf",
    size: 2_497_280_256,
    sha256: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
};
pub const VAE: Weight = Weight {
    label: "Autoencodeur FLUX.2",
    repo: "black-forest-labs/FLUX.2-klein-4B",
    revision: "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    filename: "vae/diffusion_pytorch_model.safetensors",
    size: 168_120_878,
    sha256: "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
};
pub const WEIGHTS: [&Weight; 3] = [&DIFFUSION, &ENCODER, &VAE];

impl Weight {
    pub fn path(&self, root: &Path) -> PathBuf {
        root.join("models")
            .join(self.repo.replace('/', "__"))
            .join(self.filename)
    }

    pub fn installed(&self, root: &Path) -> bool {
        self.path(root)
            .metadata()
            .is_ok_and(|m| m.is_file() && m.len() == self.size)
    }

    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.filename
        )
    }
}
