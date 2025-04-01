use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
pub struct Options {
    /// Main action
    #[clap(subcommand)]
    pub action: ActionCommand,

    /// Input file
    #[clap(short, long)]
    pub input: Option<PathBuf>,

    /// Input file
    #[clap(short = 's', long, value_enum, default_value_t = HardcodedShape::SphereAsm)]
    pub hardcoded_shape: HardcodedShape,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t = EvalMode::Jit)]
    pub eval: EvalMode,

    /// Number of threads to use
    #[clap(long, default_value_t = 0)]
    pub num_threads: usize,

    /// Number of times to render (for benchmarking)
    #[clap(long, default_value_t = 1)]
    pub num_repeats: usize,
}

#[derive(ValueEnum, Clone)]
pub enum EvalMode {
    #[cfg(feature = "jit")]
    Jit,
    Vm,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum HardcodedShape {
    SphereAsm,
    SphereTree,
}

#[derive(strum::EnumDiscriminants, Clone)]
#[strum_discriminants(name(RenderMode3DArg), derive(ValueEnum))]
pub enum RenderMode3D {
    /// Pixels are colored based on height
    HeightMap,
    /// Pixels are colored based on normals
    NormalMap { denoise: bool },
    // /// Pixels are shaded
    // Shaded { denoise: bool, ssao: bool },
    // /// Raw (unblurred) SSAO occlusion, for debugging
    // RawOcclusion { denoise: bool },
    // /// Blurred SSAO occlusion, for debugging
    // BlurredOcclusion { denoise: bool },
}

impl Default for RenderMode3DArg {
    fn default() -> Self {
        Self::HeightMap
    }
}

#[derive(ValueEnum, Default, Clone)]
pub enum RenderMode2D {
    /// Pixels are colored based on interval results
    #[default]
    Debug,
    /// Monochrome rendering (white-on-black)
    Mono,
    /// Approximate signed distance field visualization
    SdfApprox,
    /// Exact signed distance field visualization (more expensive)
    SdfExact,
    /// Brute-force (pixel-by-pixel) evaluation
    Brute,
}

#[derive(Subcommand)]
pub enum ActionCommand {
    Sample {
        #[clap(short, long)]
        output: Option<PathBuf>,

        #[clap(short, long, default_value_t = 16 * 1024)]
        num_samples: u32,

        #[clap(short, long, default_value_t = 64)]
        num_steps: u32,
    },

    Render2d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Render mode
        #[clap(long, value_enum, default_value_t)]
        mode: RenderMode2D,
    },

    Render3d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Render mode
        #[clap(long, value_enum, default_value_t)]
        mode: RenderMode3DArg,

        /// Render using an isometric perspective
        #[clap(long)]
        isometric: bool,

        /// Rotate camera
        #[clap(long, default_value_t = true)]
        use_default_camera: bool,

        /// Rotate model
        #[clap(short = 'a', long, default_value_t = 0.0)]
        model_angle: f32,

        /// Rotate camera
        #[clap(short = 's', long, default_value_t = 1.0)]
        model_scale: f32,

        /// Apply SSAO to a shaded image
        ///
        /// Only compatible with `--mode=shaded`
        #[clap(long)]
        ssao: bool,

        /// Skip denoising of normals
        ///
        /// Incompatible with `--mode=heightmap`
        #[clap(long)]
        no_denoise: bool,
    },

    Mesh {
        #[clap(flatten)]
        settings: MeshSettings,
    },
}

#[derive(Parser)]
pub struct ImageSettings {
    /// Image size
    #[clap(long = "image-size", default_value_t = 1024)]
    pub size: u32,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct MeshSettings {
    /// Octree depth
    #[clap(short, long, default_value_t = 6)]
    pub depth: u8,

    /// Name of a `.stl` file to write
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

pub fn parse_options() -> Options {
    Options::parse()
}
