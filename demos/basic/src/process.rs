use crate::options;

use anyhow::Result;
use log::info;
use log::warn;
use nalgebra::clamp;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::num::NonZero;
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
) -> () {
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
    color_mode: &options::ColorMode,
    isometric: bool,
    use_default_camera: bool,
    model_angle: f32,
    model_scale: f32,
    num_repeats: usize,
    num_threads: usize,
) -> Vec<u8> {
    let mut mat = nalgebra::Transform3::identity();
    for ii in 0..3 {
        *mat.matrix_mut().get_mut((ii, ii)).unwrap() = 1.0 / model_scale;
    }

    if use_default_camera {
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
        // apply model rotation
        let mat_rot = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Vector3::y_axis(),
            std::f32::consts::PI / 180.0 * model_angle,
        );
        mat = mat_rot * mat;
    }

    if !isometric {
        *mat.matrix_mut().get_mut((3, 2)).unwrap() = 0.3;
    }

    let pool: Option<rayon::ThreadPool>;
    let threads = match num_threads {
        0 => Some(fidget::render::ThreadPool::Global),
        1 => None,
        nn => {
            pool = Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(nn)
                    .build()
                    .unwrap(),
            );
            pool.as_ref().map(fidget::render::ThreadPool::Custom)
        }
    };
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

    let mut depth = vec![];
    let mut color = vec![];
    for _ in 0..num_repeats {
        (depth, color) = cfg.run(shape_.clone()).unwrap();
    }

    let out = match color_mode {
        options::ColorMode::NearestSite => {
            let sites = make_positions(shape.clone(), 128, 16);
            let img_size = settings.size;
            let world_to_model: nalgebra::Matrix4<f32> = mat.into();
            let screen_to_world: nalgebra::Matrix4<f32> = cfg.mat();
            let screen_to_model = world_to_model * screen_to_world;
            let mut site_id_to_colors: HashMap<usize, [f32; 3]> =
                HashMap::new();
            // let mut rng = StdRng::seed_from_u64(42);
            let mut rng = rand::rng();
            let foo = depth
                .into_iter()
                .zip(color)
                .enumerate()
                .flat_map(|(xy_, (d, c))| -> [u8; 4] {
                    if d > 0 {
                        let xy = xy_ as u32;
                        let x_ = (xy % img_size) as f32;
                        let y_ = (xy / img_size) as f32;
                        let z_ = d as f32;
                        let p_ = nalgebra::Vector4::new(x_, y_, z_, 1.0);
                        let p = screen_to_model * p_;

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

                        let mut color: [f32; 3] = [
                            rng.random_range(0.0..=1.0),
                            rng.random_range(0.0..=1.0),
                            rng.random_range(0.0..=1.0),
                        ];
                        if let Some((_, site_id)) = min_data {
                            if !site_id_to_colors.contains_key(&site_id) {
                                site_id_to_colors.insert(site_id, color);
                            }
                            color = site_id_to_colors
                                .get(&site_id)
                                .unwrap()
                                .clone();
                        }

                        let gx_ = c[0] as f32 / 255.0 - 0.5;
                        let gy_ = c[1] as f32 / 255.0 - 0.5;
                        let gz_ = c[2] as f32 / 255.0 - 0.5;
                        let g_ = nalgebra::Vector4::new(gx_, gy_, gz_, 0.0);

                        let dir = nalgebra::Vector4::new(1.0, -1.0, 1.0, 0.0);
                        let mut aa = dir.normalize().dot(&g_);
                        aa = clamp(aa, 0.0, 1.0);
                        aa = 64.0 + (255.0 - 64.0) * aa;

                        [
                            (aa * color[0]) as u8,
                            (aa * color[1]) as u8,
                            (aa * color[2]) as u8,
                            255,
                        ]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect();
            warn!(
                "Contibuting sites {}/{}",
                site_id_to_colors.len(),
                sites.len()
            );
            foo
        }
        options::ColorMode::CameraNormalMap => depth
            .into_iter()
            .zip(color)
            .flat_map(|(d, c)| {
                if d > 0 {
                    [c[0], c[1], c[2], 255]
                } else {
                    [0, 0, 0, 0]
                }
            })
            .collect(),
        options::ColorMode::Depth => {
            let z_min = depth.iter().min().cloned().unwrap_or(0);
            let z_max = depth.iter().max().cloned().unwrap_or(1);
            info!("Depth min {} max {}", z_min, z_max);
            depth
                .into_iter()
                .flat_map(|d| {
                    if d > 0 {
                        let z = (d * 255 / z_max) as u8;
                        [z, z, z, 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        options::ColorMode::ModelPosition => {
            let img_size = settings.size;
            let world_to_model: nalgebra::Matrix4<f32> = mat.into();
            let screen_to_world: nalgebra::Matrix4<f32> = cfg.mat();
            let screen_to_model = world_to_model * screen_to_world;
            info!("Model position");
            depth
                .into_iter()
                .enumerate()
                .flat_map(|(xy_, d)| {
                    if d > 0 {
                        let xy = xy_ as u32;
                        let x_ = (xy % img_size) as f32;
                        let y_ = (xy / img_size) as f32;
                        let z_ = d as f32;
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
    };

    out
}

fn run_render_2d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &options::ImageSettings,
    brute: bool,
    sdf: bool,
    num_repeats: usize,
    num_threads: usize,
) -> Vec<u8> {
    if brute {
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
        let pool: Option<rayon::ThreadPool>;
        let threads = match num_threads {
            0 => Some(fidget::render::ThreadPool::Global),
            1 => None,
            nn => {
                pool = Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(nn)
                        .build()
                        .unwrap(),
                );
                pool.as_ref().map(fidget::render::ThreadPool::Custom)
            }
        };
        let cfg = fidget::render::ImageRenderConfig {
            image_size: fidget::render::ImageSize::from(settings.size),
            tile_sizes: F::tile_sizes_2d(),
            threads,
            ..Default::default()
        };
        if sdf {
            let mut image = vec![];
            for _ in 0..num_repeats {
                image = cfg
                    .run::<_, fidget::render::SdfRenderMode>(shape.clone())
                    .unwrap();
            }
            image
                .into_iter()
                .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                .collect()
        } else {
            let mut image = vec![];
            for _ in 0..num_repeats {
                image = cfg
                    .run::<_, fidget::render::DebugRenderMode>(shape.clone())
                    .unwrap();
            }
            image
                .into_iter()
                .flat_map(|p| p.as_debug_color().into_iter())
                .collect()
        }
    }
}

fn run_mesh<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &options::MeshSettings,
    num_repeats: usize,
    num_threads: usize,
) -> fidget::mesh::Mesh {
    use fidget::mesh::ThreadCount;

    let mut mesh = fidget::mesh::Mesh::new();

    let threads = match num_threads {
        0 => ThreadCount::Many(NonZero::new(8).unwrap()),
        1 => ThreadCount::One,
        nn => ThreadCount::Many(NonZero::new(nn).unwrap()),
    };
    for _ in 0..num_repeats {
        let settings = fidget::mesh::Settings {
            threads,
            depth: settings.depth,
            ..Default::default()
        };
        let octree = fidget::mesh::Octree::build(&shape, settings);
        mesh = octree.walk_dual(settings);
    }

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
            color_mode,
            isometric,
            use_default_camera,
            model_angle,
            model_scale,
        } => {
            let buffer = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_render_3d(
                        shape,
                        settings,
                        color_mode,
                        *isometric,
                        *use_default_camera,
                        *model_angle,
                        *model_scale,
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
                        color_mode,
                        *isometric,
                        *use_default_camera,
                        *model_angle,
                        *model_scale,
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
        ActionCommand::Render2d {
            settings,
            brute,
            sdf,
        } => {
            let buffer = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_render_2d(
                        shape,
                        settings,
                        *brute,
                        *sdf,
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
                        *brute,
                        *sdf,
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
        ActionCommand::Mesh { settings } => {
            let mesh = match args.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?} (JIT)", top.elapsed());
                    top = Instant::now();
                    run_mesh(
                        shape,
                        settings,
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
