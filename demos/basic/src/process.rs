use crate::options;

use anyhow::Result;

use log::info;

use clap::CommandFactory;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn make_positions<F: fidget::eval::Function>(
    shape: fidget::shape::Shape<F>,
    num_samples: u32,
    num_steps: u32,
) -> Vec<nalgebra::Vector3<f32>> {
    let mut positions = vec![];

    // simple advection
    let mut rng = StdRng::seed_from_u64(42);
    let tape = shape.point_tape(Default::default());
    let mut eval = fidget::shape::Shape::<F>::new_point_eval();

    let eps = 1e-3;

    for _ in 0..num_samples {
        let mut pos = nalgebra::Vector3::new(
            rng.random_range(-1.0..=1.0),
            rng.random_range(-1.0..=1.0),
            rng.random_range(-1.0..=1.0),
        );
        for _ in 0..num_steps {
            let value = eval.eval(&tape, pos[0], pos[1], pos[2]).unwrap().0;
            let value_dx =
                eval.eval(&tape, pos[0] + eps, pos[1], pos[2]).unwrap().0;
            let value_dy =
                eval.eval(&tape, pos[0], pos[1] + eps, pos[2]).unwrap().0;
            let value_dz =
                eval.eval(&tape, pos[0], pos[1], pos[2] + eps).unwrap().0;
            let grad = nalgebra::Vector3::new(
                (value_dx - value) / eps,
                (value_dy - value) / eps,
                (value_dz - value) / eps,
            );
            pos -= 0.5 * value * grad;
        }
        // warn!("final pos {:?} norm {}", pos, pos.norm());
        positions.push(pos);
    }

    positions
}

fn run_sample<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    num_samples: u32,
    num_steps: u32,
    output: &Option<PathBuf>,
) {
    let positions = make_positions(shape, num_samples, num_steps);

    if let Some(path) = output {
        let mut text = format!(
            "ply
format ascii 1.0
comment Created in Blender version 4.0.2
element vertex {}
property float x
property float y
property float z
end_header
",
            positions.len(),
        );

        for pos in positions {
            text = format!("{}{} {} {}\n", text, pos[0], pos[1], pos[2]);
        }

        info!("Writing PLY to {:?}", path);
        let mut output = File::create(path).unwrap();
        write!(output, "{}", text).ok();
    }
}

fn run_render_3d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &options::ImageSettings,
    mode: &options::RenderMode3D,
    camera: &options::CameraSettings,
    model_transform: &options::TransformSettings,
    num_repeats: usize,
    num_threads: usize,
) -> Vec<u8> {
    let mut mat = nalgebra::Transform3::identity();
    for ii in 0..3 {
        *mat.matrix_mut().get_mut((ii, ii)).unwrap() =
            1.0 / model_transform.scale;
    }

    if !camera.no_default_position {
        let mat_aa = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            std::f32::consts::PI / -4.0,
        );
        let mat_bb = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Vector3::x_axis(),
            std::f32::consts::PI / -6.0,
        );
        mat = mat_aa * mat_bb * mat;
    }

    {
        // apply model elevation
        let mat_elev =
            nalgebra::Translation3::new(0.0, model_transform.elevation, 0.0);
        // apply model rotation
        let mat_rot = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            std::f32::consts::PI / 180.0 * model_transform.angle,
        );
        mat = mat_elev * mat_rot * mat;
    }

    if !camera.is_isometric {
        *mat.matrix_mut().get_mut((3, 2)).unwrap() = 0.3;
    }

    let threads = match num_threads {
        0 => Some(fidget::render::ThreadPool::Global),
        1 => None,
        nn => Some(fidget::render::ThreadPool::Custom(
            rayon::ThreadPoolBuilder::new()
                .num_threads(nn)
                .build()
                .unwrap(),
        )),
    };
    let threads = threads.as_ref();

    let view = fidget::render::View3::from_center_and_scale(
        nalgebra::Vector3::new(0.0, 0.0, 0.0),
        1.0,
    );
    let cfg = fidget::render::VoxelRenderConfig {
        image_size: fidget::render::VoxelSize::from(settings.size),
        tile_sizes: F::tile_sizes_3d(),
        view,
        threads,
        ..Default::default()
    };

    let shape_ = shape.clone().apply_transform(mat.into());

    let mut image = Default::default();
    for _ in 0..num_repeats {
        image = cfg.run(shape_.clone()).unwrap();
    }

    let out = match mode {
        options::RenderMode3D::HeightMap => {
            let z_min = image.iter().map(|p| p.depth).min().unwrap_or(0);
            let z_max = image.iter().map(|p| p.depth).max().unwrap_or(1);
            info!("Depth min {} max {}", z_min, z_max);
            image
                .into_iter()
                .flat_map(|p| {
                    if p.depth > 0 {
                        let z = (p.depth * 255 / z_max) as u8;
                        [z, z, z, 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        options::RenderMode3D::NormalMap { denoise } => {
            let image = if *denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            image
                .into_iter()
                .flat_map(|p| {
                    if p.depth > 0 {
                        let c = p.to_color();
                        [c[0], c[1], c[2], 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        options::RenderMode3D::Shaded { ssao, denoise } => {
            let image = if *denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            let color =
                fidget::render::effects::apply_shading(&image, *ssao, threads);
            image
                .into_iter()
                .zip(color)
                .flat_map(|(p, c)| {
                    if p.depth > 0 {
                        [c[0], c[1], c[2], 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        options::RenderMode3D::RawOcclusion { denoise } => {
            let image = if *denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            let ssao = fidget::render::effects::compute_ssao(&image, threads);
            ssao.into_iter()
                .flat_map(|p| {
                    if p.is_nan() {
                        [255; 4]
                    } else {
                        let v = (p * 255.0).min(255.0) as u8;
                        [v, v, v, 255]
                    }
                })
                .collect()
        }
        options::RenderMode3D::BlurredOcclusion { denoise } => {
            let image = if *denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            let ssao = fidget::render::effects::compute_ssao(&image, threads);
            let blurred = fidget::render::effects::blur_ssao(&ssao, threads);
            blurred
                .into_iter()
                .flat_map(|p| {
                    if p.is_nan() {
                        [255; 4]
                    } else {
                        let v = (p * 255.0).min(255.0) as u8;
                        [v, v, v, 255]
                    }
                })
                .collect()
        }
        options::RenderMode3D::ModelPosition => {
            let img_size = settings.size;
            let world_to_model: nalgebra::Matrix4<f32> = mat.into();
            let screen_to_world: nalgebra::Matrix4<f32> = cfg.mat();
            let screen_to_model = world_to_model * screen_to_world;
            image
                .into_iter()
                .enumerate()
                .flat_map(|(xy_, p)| {
                    if p.depth > 0 {
                        let xy = xy_ as u32;
                        let x_ = (xy % img_size) as f32;
                        let y_ = (xy / img_size) as f32;
                        let z_ = p.depth as f32;
                        let p_ = nalgebra::Vector4::new(x_, y_, z_, 1.0);
                        let p = screen_to_model * p_;
                        let red =
                            if p[0] > 0.0 { (p[0] * 255.0) as u8 } else { 0 };
                        let green =
                            if p[1] > 0.0 { (p[1] * 255.0) as u8 } else { 0 };
                        let blue =
                            if p[2] > 0.0 { (p[2] * 255.0) as u8 } else { 0 };
                        [red, green, blue, 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        options::RenderMode3D::NearestSite { ssao, denoise } => {
            let sites = make_positions(shape.clone(), 128, 16);

            let img_size = settings.size;
            let world_to_model: nalgebra::Matrix4<f32> = mat.into();
            let screen_to_world: nalgebra::Matrix4<f32> = cfg.mat();
            let screen_to_model = world_to_model * screen_to_world;
            let mut site_id_to_colors: HashMap<usize, [f32; 3]> =
                HashMap::new();

            // let mut rng = StdRng::seed_from_u64(42);
            let mut rng = rand::rng();

            let image = if *denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            let shaded_color =
                fidget::render::effects::apply_shading(&image, *ssao, threads);

            let image = image
                .into_iter()
                .zip(shaded_color)
                .enumerate()
                .flat_map(|(xy_, (pixel, shaded_color))| -> [u8; 4] {
                    if pixel.depth > 0 {
                        let xy = xy_ as u32;
                        let x_ = (xy % img_size) as f32;
                        let y_ = (xy / img_size) as f32;
                        let z_ = pixel.depth as f32;
                        let p_ = nalgebra::Vector4::new(x_, y_, z_, 1.0);
                        let p = screen_to_model * p_;

                        // FIXME use kdtree
                        let mut min_data: Option<(f32, usize)> = None;
                        for (site_id, site_pos) in sites.iter().enumerate() {
                            let pos = p.xyz();
                            let dist = (site_pos - pos).norm();
                            match min_data {
                                None => {
                                    min_data = Some((dist, site_id));
                                }
                                Some((dist_, _)) => {
                                    if dist < dist_ {
                                        min_data = Some((dist, site_id));
                                    }
                                }
                            }
                        }

                        let mut site_color: [f32; 3] = [
                            rng.random_range(0.0..=1.0),
                            rng.random_range(0.0..=1.0),
                            rng.random_range(0.0..=1.0),
                        ];
                        if let Some((_, site_id)) = min_data {
                            site_color = *site_id_to_colors
                                .entry(site_id)
                                .or_insert(site_color);
                        }

                        [
                            (shaded_color[0] as f32 * site_color[0]) as u8,
                            (shaded_color[1] as f32 * site_color[1]) as u8,
                            (shaded_color[2] as f32 * site_color[2]) as u8,
                            255,
                        ]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect();
            info!(
                "Contibuting sites {}/{}",
                site_id_to_colors.len(),
                sites.len()
            );
            image
        }
    };

    out
}

fn run_render_2d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &options::ImageSettings,
    mode: &options::RenderMode2D,
    num_repeats: usize,
    num_threads: usize,
) -> Vec<u8> {
    use options::RenderMode2D;
    if matches!(mode, RenderMode2D::Brute) {
        let tape = shape.float_slice_tape(Default::default());
        let mut eval = fidget::shape::Shape::<F>::new_float_slice_eval();
        let mut out: Vec<bool> = vec![];
        for _ in 0..num_repeats {
            let mut xs = vec![];
            let mut ys = vec![];
            let div = (settings.size - 1) as f64;
            for i in 0..settings.size {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..settings.size {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    xs.push(x as f32);
                    ys.push(y as f32);
                }
            }
            let zs = vec![0.0; xs.len()];
            let values = eval.eval(&tape, &xs, &ys, &zs).unwrap();
            out = values.iter().map(|v| *v <= 0.0).collect();
        }
        // Convert from Vec<bool> to an image
        out.into_iter()
            .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
            .flat_map(|i| i.into_iter())
            .collect()
    } else {
        let threads = match num_threads {
            0 => Some(fidget::render::ThreadPool::Global),
            1 => None,
            nn => Some(fidget::render::ThreadPool::Custom(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(nn)
                    .build()
                    .unwrap(),
            )),
        };
        let threads = threads.as_ref();

        let cfg = fidget::render::ImageRenderConfig {
            image_size: fidget::render::ImageSize::from(settings.size),
            tile_sizes: F::tile_sizes_2d(),
            threads,
            ..Default::default()
        };

        match mode {
            RenderMode2D::Mono => {
                let mut image = fidget::render::Image::default();
                for _ in 0..num_repeats {
                    image = cfg
                        .run::<_, fidget::render::BitRenderMode>(shape.clone())
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|a| if a { [255; 4] } else { [0, 0, 0, 255] })
                    .collect()
            }
            RenderMode2D::SdfExact => {
                let mut image = fidget::render::Image::default();
                for _ in 0..num_repeats {
                    image = cfg
                        .run::<_, fidget::render::SdfPixelRenderMode>(
                            shape.clone(),
                        )
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                    .collect()
            }
            RenderMode2D::SdfApprox => {
                let mut image = fidget::render::Image::default();
                for _ in 0..num_repeats {
                    image = cfg
                        .run::<_, fidget::render::SdfRenderMode>(shape.clone())
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                    .collect()
            }
            RenderMode2D::Debug => {
                let mut image = fidget::render::Image::default();
                for _ in 0..num_repeats {
                    image = cfg
                        .run::<_, fidget::render::DebugRenderMode>(
                            shape.clone(),
                        )
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|p| p.as_debug_color().into_iter())
                    .collect()
            }
            RenderMode2D::Brute => unreachable!(),
        }
    }
}

fn run_mesh<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &options::MeshSettings,
    model_transform: &options::TransformSettings,
    num_repeats: usize,
    num_threads: usize,
) -> fidget::mesh::Mesh {
    let mut mesh = fidget::mesh::Mesh::new();

    // Transform the shape based on our render settings
    let s = 1.0 / model_transform.scale;
    let scale = nalgebra::Scale3::new(s, s, s);
    let center =
        nalgebra::Translation3::new(0.0, model_transform.elevation, 0.0);
    let rotation = nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Vector3::y_axis(),
        std::f32::consts::PI / 180.0 * model_transform.angle,
    );
    let t = rotation.to_homogeneous()
        * center.to_homogeneous()
        * scale.to_homogeneous();
    let shape = shape.apply_transform(t);

    let threads = match num_threads {
        0 => Some(fidget::render::ThreadPool::Global),
        1 => None,
        nn => Some(fidget::render::ThreadPool::Custom(
            rayon::ThreadPoolBuilder::new()
                .num_threads(nn)
                .build()
                .unwrap(),
        )),
    };
    let threads = threads.as_ref();

    let mut octree_time = std::time::Duration::ZERO;
    let mut mesh_time = std::time::Duration::ZERO;
    for _ in 0..num_repeats {
        let settings = fidget::mesh::Settings {
            depth: settings.depth,
            threads,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let octree = fidget::mesh::Octree::build(&shape, settings);
        octree_time += start.elapsed();

        let start = std::time::Instant::now();
        mesh = octree.walk_dual(settings);
        mesh_time += start.elapsed();
    }

    // (mesh, octree_time, mesh_time)
    mesh
}

pub fn run_action(
    ctx: fidget::context::Context,
    root: fidget::context::Node,
    args: &options::Options,
) -> Result<()> {
    use options::{ActionCommand, EvalMode};
    let mut top = Instant::now();
    match &args.action {
        ActionCommand::Sample {
            num_samples,
            num_steps,
            output,
        } => {
            match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    run_sample(shape, *num_samples, *num_steps, output)
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (VM)", top.elapsed());
                    run_sample(shape, *num_samples, *num_steps, output)
                }
            };
        }
        ActionCommand::Render3d {
            settings,
            mode,
            camera,
            model_transform,
        } => {
            use options::Options;
            use options::RenderMode3D;
            use options::RenderMode3DArg;
            if camera.ssao && !matches!(mode, RenderMode3DArg::Shaded) {
                let mut cmd = Options::command();
                let sub = cmd.find_subcommand_mut("render3d").unwrap();
                let error = sub.error(
                    clap::error::ErrorKind::ArgumentConflict,
                    "`--ssao` is only allowed when `--mode=shaded`",
                );
                error.exit();
            }
            if camera.no_denoise && matches!(mode, RenderMode3DArg::HeightMap) {
                let mut cmd = Options::command();
                let sub = cmd.find_subcommand_mut("render3d").unwrap();
                let error = sub.error(
                    clap::error::ErrorKind::ArgumentConflict,
                    "`--no-denoise` is not allowed when `--mode=height-map`",
                );
                error.exit();
            }
            let denoise = !camera.no_denoise;
            let mode = match mode {
                RenderMode3DArg::Shaded => RenderMode3D::Shaded {
                    ssao: camera.ssao,
                    denoise,
                },
                RenderMode3DArg::HeightMap => RenderMode3D::HeightMap,
                RenderMode3DArg::BlurredOcclusion => {
                    RenderMode3D::BlurredOcclusion { denoise }
                }
                RenderMode3DArg::RawOcclusion => {
                    RenderMode3D::RawOcclusion { denoise }
                }
                RenderMode3DArg::NormalMap => {
                    RenderMode3D::NormalMap { denoise }
                }
                RenderMode3DArg::ModelPosition => RenderMode3D::ModelPosition,
                RenderMode3DArg::NearestSite => RenderMode3D::NearestSite {
                    ssao: camera.ssao,
                    denoise,
                },
            };
            let buffer = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_render_3d(
                        shape,
                        settings,
                        &mode,
                        &camera,
                        model_transform,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (VM)", top.elapsed());
                    top = Instant::now();
                    run_render_3d(
                        shape,
                        settings,
                        &mode,
                        &camera,
                        model_transform,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
            };
            info!(
                "Rendered {}x at {:?} ms/frame",
                args.num_repeats,
                top.elapsed().as_micros() as f64
                    / 1000.0
                    / (args.num_repeats as f64)
            );
            if let Some(path) = &settings.output {
                info!("Writing PNG to {:?}", path);
                image::save_buffer(
                    path,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        ActionCommand::Render2d { settings, mode } => {
            let buffer = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_render_2d(
                        shape,
                        settings,
                        mode,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (VM)", top.elapsed());
                    top = Instant::now();
                    run_render_2d(
                        shape,
                        settings,
                        mode,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
            };
            info!(
                "Rendered {}x at {:?} ms/frame",
                args.num_repeats,
                top.elapsed().as_micros() as f64
                    / 1000.0
                    / (args.num_repeats as f64)
            );
            if let Some(path) = &settings.output {
                info!("Writing PNG to {:?}", path);
                image::save_buffer(
                    path,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        ActionCommand::Mesh {
            settings,
            model_transform,
        } => {
            let mesh = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_mesh(
                        shape,
                        settings,
                        model_transform,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (VM)", top.elapsed());
                    top = Instant::now();
                    run_mesh(
                        shape,
                        settings,
                        model_transform,
                        args.num_repeats,
                        args.num_threads,
                    )
                }
            };
            info!(
                "Rendered {}x at {:?} ms/iter",
                args.num_repeats,
                top.elapsed().as_micros() as f64
                    / 1000.0
                    / (args.num_repeats as f64)
            );
            info!(
                "Mesh has {} vertices {} triangles",
                mesh.vertices.len(),
                mesh.triangles.len()
            );
            if let Some(path) = &settings.output {
                info!("Writing STL to {:?}", path);
                let mut handle = std::fs::File::create(path).unwrap();
                mesh.write_stl(&mut handle)?;
            }
        }
    }

    Ok(())
}
