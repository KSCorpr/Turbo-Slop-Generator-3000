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

use crate::{catalog, download, engine};

#[test]
fn weights_use_the_shared_models_directory_and_require_the_exact_size() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let vae = catalog::VAE.path(root);
    assert!(vae.ends_with(
        "models/black-forest-labs__FLUX.2-klein-4B/vae/diffusion_pytorch_model.safetensors"
    ));
    assert!(!catalog::VAE.installed(root));
    fs::create_dir_all(vae.parent().unwrap()).unwrap();
    let file = fs::File::create(&vae).unwrap();
    file.set_len(catalog::VAE.size - 1).unwrap();
    assert!(!catalog::VAE.installed(root));
    file.set_len(catalog::VAE.size).unwrap();
    assert!(catalog::VAE.installed(root));
}

#[test]
fn output_pixel_conversion_checks_non_finite_values_and_clamps_extremes() {
    let image = engine::rgb_image(1, 1, &[-2.0, 0.0, 2.0]).unwrap();
    assert_eq!(image.into_raw(), [0, 128, 255]);
    assert!(engine::rgb_image(1, 1, &[f32::NAN, 0.0, 1.0]).is_err());
    assert!(engine::rgb_image(1, 1, &[1.0]).is_err());
}

#[test]
fn sizes_are_multiples_of_sixteen_in_the_supported_range() {
    for size in [256, 512, 768] {
        assert!(engine::valid_edge(size));
    }
    for size in [240, 255, 257, 767, 784] {
        assert!(!engine::valid_edge(size));
    }
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
