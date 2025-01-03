import {
  ImageResponse,
  RenderMode,
  RequestKind,
  ScriptRequest,
  ScriptResponse,
  ShapeRequest,
  StartedResponse,
  WorkerRequest,
} from "./message";

import { RENDER_SIZE } from "./constants";

var fidget: any = null;

class Worker {
  render(s: ShapeRequest) {
    const shape = fidget.deserialize_tape(s.tape);
    let out: Uint8Array;
    switch (s.mode) {
      case RenderMode.Bitmap: {
        out = fidget.render_region_2d(shape, RENDER_SIZE);
        break;
      }
      case RenderMode.Heightmap: {
        out = fidget.render_region_heightmap(shape, RENDER_SIZE);
        break;
      }
      case RenderMode.Normals: {
        out = fidget.render_region_normals(shape, RENDER_SIZE);
        break;
      }
    }
    postMessage(new ImageResponse(out), { transfer: [out.buffer] });
  }

  run(s: ScriptRequest) {
    let shape = null;
    let result = "Ok(..)";
    try {
      shape = fidget.eval_script(s.script);
    } catch (error) {
      // Do some string formatting to make errors cleaner
      result = error
        .toString()
        .replace("Rhai evaluation error: ", "Rhai evaluation error:\n")
        .replace(" (line ", "\n(line ")
        .replace(" (expecting ", "\n(expecting ");
    }

    let tape = null;
    if (shape) {
      tape = fidget.serialize_into_tape(shape);
      postMessage(new ScriptResponse(result, tape), {
        transfer: [tape.buffer],
      });
    } else {
      postMessage(new ScriptResponse(result, tape));
    }
  }
}

async function run() {
  fidget = await import("../../crate/pkg/fidget_wasm_demo.js")!;
  await fidget.default();
  await fidget.initThreadPool(navigator.hardwareConcurrency);

  let worker = new Worker();
  onmessage = function (e: any) {
    let req = e.data as WorkerRequest;
    switch (req.kind) {
      case RequestKind.Shape: {
        worker!.render(req as ShapeRequest);
        break;
      }
      case RequestKind.Script: {
        worker!.run(req as ScriptRequest);
        break;
      }
      default:
        console.error(`unknown worker request ${req}`);
    }
  };
  postMessage(new StartedResponse());
}
run();
