use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    sync::atomic::AtomicBool,
    thread,
    time::Duration,
};

use sha2::{Digest, Sha256};
use tempfile::tempdir;

use crate::{
    catalog::{self, Catalog, Weight},
    download, engine,
};

#[test]
fn embedded_catalog_covers_the_nine_image_models_and_preserves_presets() {
    let catalog = Catalog::load().unwrap();
    assert_eq!(catalog.base_models.len(), 9);
    for model in &catalog.base_models {
        assert_eq!(model.presets[0].steps, model.defaults.steps);
        assert_eq!(model.presets[0].cfg_scale, model.defaults.cfg_scale);
        assert!(model.sources.contains_key("diffusion"));
        assert!(model.sources.contains_key("text_encoder"));
        assert!(model.sources.contains_key("vae"));
    }
    let h3 = catalog
        .base_models
        .iter()
        .find(|model| model.id == "minimax-h3-image")
        .unwrap();
    assert_eq!(h3.max_references(), 16);
    assert_eq!(h3.dimension_multiple(), 32);
    assert_eq!(
        h3.defaults.extra_flags,
        ["--sigmas", "minimax-h3-image", "--rng", "cpu"]
    );
    assert!(h3.fixed_sampling());
    assert_eq!(engine::build_number("master-1088-32d516c"), Some(1088));
    assert_eq!(engine::build_number("invalid"), None);
    assert!(
        catalog
            .base_models
            .iter()
            .any(|model| model.presets.iter().any(|p| p.adapter().is_some()))
    );
}

#[test]
fn local_weights_reuse_python_layout_and_do_not_select_unapproved_paths() {
    let root = tempdir().unwrap();
    let model = Catalog::load()
        .unwrap()
        .base_models
        .into_iter()
        .find(|m| m.id == "minimax-h3-image")
        .unwrap();
    let repo = "leejet/MiniMax-H3-GGUF";
    let q4 = Weight {
        repo: repo.into(),
        name: "qwen3vl_32b_minimax_h3-Q4_K_M.gguf".into(),
    };
    let q2 = Weight {
        repo: repo.into(),
        name: "qwen3vl_32b_minimax_h3-Q2_K_M.gguf".into(),
    };
    let path = q4.path(root.path());
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"q4").unwrap();
    assert_eq!(
        catalog::installed(&model, root.path())["text_encoder"].as_slice(),
        std::slice::from_ref(&q4)
    );
    fs::write(q2.path(root.path()), b"q2").unwrap();
    assert_eq!(
        catalog::installed(&model, root.path())["text_encoder"],
        [q2, q4]
    );
    for invalid in [
        "../../a.gguf",
        "/root.gguf",
        "a\\b.gguf",
        "C:/x.gguf",
        "CON.txt",
        "bad. ",
        "a//b",
    ] {
        assert!(!catalog::safe_name(invalid), "{invalid}");
    }
    assert!(catalog::safe_name("vae/a.safetensors"));
    let w4 = Weight {
        repo: "Comfy-Org/MiniMax-H3-Image".into(),
        name: "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors".into(),
    };
    assert!(!model.sources["diffusion"].accepts(&w4));
}

#[test]
fn flux_variant_matching_accepts_the_default_diffusion_weight() {
    let root = tempdir().unwrap();
    let model = Catalog::load()
        .unwrap()
        .base_models
        .into_iter()
        .find(|m| m.id == "flux2-klein-9b")
        .unwrap();
    let weight = Weight {
        repo: "unsloth/FLUX.2-klein-9B-GGUF".into(),
        name: "flux-2-klein-9b-Q4_K_M.gguf".into(),
    };
    let path = weight.path(root.path());
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, b"flux checkpoint").unwrap();
    assert_eq!(
        catalog::installed(&model, root.path())["diffusion"],
        [weight]
    );
}

#[test]
fn interrupted_download_resumes_and_publishes_only_a_verified_file() {
    let directory = tempdir().unwrap();
    let bytes = b"TurboSlop";
    let sha = format!("{:x}", Sha256::digest(bytes));
    let destination = directory.path().join("weight.bin");
    let partial = directory
        .path()
        .join(format!(".weight.bin.{}.part", &sha[..12]));
    fs::write(&partial, &bytes[..3]).unwrap();
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let url = format!("http://{}/model", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut range_seen = false;
        for line in reader.by_ref().lines() {
            let line = line.unwrap();
            if line.is_empty() {
                break;
            }
            range_seen |= line.eq_ignore_ascii_case("range: bytes=3-");
        }
        assert!(range_seen);
        write!(stream, "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-8/9\r\nContent-Length: 6\r\nConnection: close\r\n\r\n").unwrap();
        stream.write_all(&bytes[3..]).unwrap();
    });
    let client = ureq::AgentBuilder::new()
        .try_proxy_from_env(false)
        .timeout(Duration::from_secs(8))
        .build();
    let cancel = AtomicBool::new(false);
    download::download(
        &client,
        &url,
        &destination,
        9,
        Some(&sha),
        &sha,
        &cancel,
        &mut |_, _| {},
    )
    .unwrap();
    server.join().unwrap();
    assert_eq!(fs::read(destination).unwrap(), bytes);
    assert!(!partial.exists());
}

#[test]
fn corrupt_download_never_appears_as_an_installed_weight() {
    let directory = tempdir().unwrap();
    let destination = directory.path().join("weight.bin");
    let sha = format!("{:x}", Sha256::digest(b"expected"));
    let partial = directory
        .path()
        .join(format!(".weight.bin.{}.part", &sha[..12]));
    fs::write(&partial, b"mismatch").unwrap();
    let client = ureq::AgentBuilder::new().try_proxy_from_env(false).build();
    let error = download::download(
        &client,
        "http://127.0.0.1:9/",
        &destination,
        8,
        Some(&sha),
        &sha,
        &AtomicBool::new(false),
        &mut |_, _| {},
    )
    .unwrap_err();
    assert!(error.to_string().contains("SHA-256 incorrect"));
    assert!(!destination.exists());
    assert!(!partial.exists());
}

#[cfg(unix)]
#[test]
fn simulated_engine_generates_preview_and_ordered_edit_with_the_real_job_runner() {
    use crate::{
        catalog::Selection,
        engine::{Backend, Request, Update},
    };
    use std::os::unix::fs::PermissionsExt;

    let directory = tempdir().unwrap();
    let root = directory.path();
    let model = Catalog::load()
        .unwrap()
        .base_models
        .into_iter()
        .find(|m| m.id == "minimax-h3-image")
        .unwrap();
    for weight in [
        Weight {
            repo: "Comfy-Org/MiniMax-H3-Image".into(),
            name: "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors".into(),
        },
        Weight {
            repo: "Comfy-Org/MiniMax-H3-Image".into(),
            name: "vae/minimax_h3_image_vae_fp16.safetensors".into(),
        },
        Weight {
            repo: "leejet/MiniMax-H3-GGUF".into(),
            name: "qwen3vl_32b_minimax_h3-Q2_K_M.gguf".into(),
        },
    ] {
        let path = weight.path(root);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"test weight").unwrap();
    }
    let exe = root.join("sd-cli");
    let fixture = r#"#!/bin/sh
if [ "$1" = "--help" ]; then
    echo '--diffusion-model --vae --llm --sigmas minimax-h3-image --rng --vae-tiling --preview-path'
    exit 0
fi
cd "$(dirname "$0")"
printf '%s\n' "$@" > args.txt
while [ "$#" -gt 0 ]; do
    case "$1" in
      -o) output="$2"; shift ;;
      -r) if [ -z "$ref" ]; then ref="$2"; fi; shift ;;
      --preview-path) preview="$2"; shift ;;
    esac
    shift
done
cp "$ref" "$(printf "$preview" 1)"
printf '|##| 1/6\r'
cp "$ref" "$output"
"#;
    fs::write(&exe, fixture).unwrap();
    fs::set_permissions(&exe, fs::Permissions::from_mode(0o755)).unwrap();
    let refs = [root.join("α.png"), root.join("β.png")];
    for (path, color) in refs.iter().zip([[90, 0, 0], [0, 90, 0]]) {
        image::RgbImage::from_pixel(256, 256, image::Rgb(color))
            .save(path)
            .unwrap();
    }
    let preset = model.presets[0].clone();
    let request = Request {
        root: root.into(),
        exe,
        model,
        selection: Selection::new(),
        preset,
        prompt: "Edit <Picture 1> using <Picture 2>.".into(),
        negative: String::new(),
        references: refs.to_vec(),
        width: 256,
        height: 256,
        seed: 42,
        count: 1,
        strength: 0.6,
        loras: Vec::new(),
        backend: Backend::Cpu,
        gpu: None,
        auto_fit: true,
        flash_attention: true,
    };
    let mut preview = false;
    let mut progress = false;
    let outputs = engine::generate(request, &AtomicBool::new(false), |update| match update {
        Update::Preview(_) => preview = true,
        Update::Progress(_, Some(_)) => progress = true,
        _ => {}
    })
    .unwrap();
    let log = fs::read_to_string(root.join("args.txt")).unwrap();
    assert!(log.contains("--sigmas\nminimax-h3-image\n--rng\ncpu\n"));
    assert!(log.contains("-p\nEdit <Picture 1> using <Picture 2>.\n"));
    let args: Vec<_> = log.lines().collect();
    let native_refs: Vec<_> = args
        .windows(2)
        .filter(|w| w[0] == "-r")
        .map(|w| w[1])
        .collect();
    assert_eq!(native_refs.len(), 2);
    assert!(native_refs[0].ends_with("reference-00.png"));
    assert!(native_refs[1].ends_with("reference-01.png"));
    assert!(preview && progress);
    let rendered = image::open(&outputs[0]).unwrap().to_rgb8();
    assert_eq!(*rendered.get_pixel(0, 0), image::Rgb([90, 0, 0]));
    assert!(outputs[0].with_extension("rust.json").is_file());
    assert!(
        fs::read_dir(root.join("tmp/rust"))
            .unwrap()
            .next()
            .is_none()
    );
}

#[test]
#[ignore = "réseau réel requis"]
fn public_hugging_face_catalog_and_windows_engine_install() {
    let root = tempdir().unwrap();
    let model = Catalog::load()
        .unwrap()
        .base_models
        .into_iter()
        .find(|m| m.id == "moss-image-sft")
        .unwrap();
    let cancel = AtomicBool::new(false);
    let files = download::catalog_weights(&model, &cancel).unwrap();
    let tokenizer = &files["tokenizer"][0];
    let local = tokenizer.weight.path(root.path());
    download::install_weight(
        &download::client(),
        tokenizer,
        &local,
        &cancel,
        &mut |msg, _| println!("{msg}"),
    )
    .unwrap();
    assert!(local.metadata().unwrap().len() > 1_000_000);
    let exe = engine::install(root.path(), engine::Backend::Cpu, &cancel, &mut |msg, _| {
        println!("{msg}")
    })
    .unwrap();
    assert_eq!(&fs::read(exe).unwrap()[..2], b"MZ");
}
