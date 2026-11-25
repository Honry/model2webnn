// A WebNN stand-in for tests in Node: builds "graphs" that only know their input / output interface (shapes inferred from the operators), and a context that
// VALIDATES every createTensor / writeTensor / dispatch / readTensor call the way a real implementation would reject them: unknown or missing names,
// dtype / shape mismatches, byte lengths, an output tensor that is also an input of the same dispatch. It computes nothing: logits are a pattern the test controls.
/* eslint-disable @typescript-eslint/no-explicit-any */

const BYTES: Record<string, number> = { float32: 4, float16: 2, int32: 4, int64: 8, int8: 1, uint8: 1, int4: 0.5, uint4: 0.5 };
const count = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

function broadcast(a: number[], b: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i <= Math.max(a.length, b.length); i++) {
    const x = a[a.length - i] ?? 1, y = b[b.length - i] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw new Error(`cannot broadcast [${a}] with [${b}]`);
    out.unshift(x === 1 ? y : x);
  }
  return out;
}

export interface MockOperand { dataType: string; shape: number[] }

export class MockBuilder {
  inputs: Record<string, MockOperand> = {};
  constructor(public context: MockContext) {}
  private op(dataType: string, shape: number[]): MockOperand { return { dataType, shape: [...shape] }; }
  input(name: string, d: MockOperand) { return (this.inputs[name] = this.op(d.dataType, d.shape)); }
  constant(d: MockOperand, data: ArrayBufferView) {
    if (data.byteLength !== 0 && data.byteLength !== Math.ceil(count(d.shape) * BYTES[d.dataType])) throw new Error(`constant [${d.shape}] ${d.dataType}: ${data.byteLength} bytes`);
    return this.op(d.dataType, d.shape);
  }
  private binary = (a: MockOperand, b: MockOperand) => {
    if (a.dataType !== b.dataType) throw new Error(`operand types differ: ${a.dataType} vs ${b.dataType}`);
    return this.op(a.dataType, broadcast(a.shape, b.shape));
  };
  add = this.binary; sub = this.binary; mul = this.binary; div = this.binary; pow = this.binary;
  sqrt = (x: MockOperand) => this.op(x.dataType, x.shape);
  sigmoid = (x: MockOperand) => this.op(x.dataType, x.shape);
  cast = (x: MockOperand, to: string) => this.op(to, x.shape);
  reshape = (x: MockOperand, shape: number[]) => { if (count(shape) !== count(x.shape)) throw new Error(`reshape [${x.shape}] -> [${shape}]`); return this.op(x.dataType, shape); };
  transpose = (x: MockOperand, o: { permutation: number[] }) => this.op(x.dataType, o.permutation.map((p) => x.shape[p]));
  slice = (x: MockOperand, starts: number[], sizes: number[]) => {
    sizes.forEach((s, i) => { if (starts[i] + s > x.shape[i]) throw new Error(`slice out of range on axis ${i}`); });
    return this.op(x.dataType, sizes);
  };
  concat = (xs: MockOperand[], axis: number) => this.op(xs[0].dataType, xs[0].shape.map((d, i) => (i === axis ? xs.reduce((s, x) => s + x.shape[axis], 0) : d)));
  expand = (x: MockOperand, shape: number[]) => this.op(x.dataType, shape);
  private reduce = (x: MockOperand, o: { axes: number[]; keepDimensions?: boolean }) => this.op(x.dataType, o.keepDimensions ? x.shape.map((d, i) => (o.axes.includes(i) ? 1 : d)) : x.shape.filter((_, i) => !o.axes.includes(i)));
  reduceMean = this.reduce; reduceMax = this.reduce; reduceSum = this.reduce;
  softmax = (x: MockOperand) => this.op(x.dataType, x.shape);
  matmul = (a: MockOperand, b: MockOperand) => {
    if (a.shape[a.shape.length - 1] !== b.shape[b.shape.length - 2]) throw new Error(`matmul [${a.shape}] x [${b.shape}]`);
    return this.op(a.dataType, [...broadcast(a.shape.slice(0, -2), b.shape.slice(0, -2)), a.shape[a.shape.length - 2], b.shape[b.shape.length - 1]]);
  };
  gemm = (a: MockOperand, b: MockOperand, o: { aTranspose?: boolean; bTranspose?: boolean } = {}) => {
    const [m, k] = o.aTranspose ? [a.shape[1], a.shape[0]] : a.shape, [k2, n] = o.bTranspose ? [b.shape[1], b.shape[0]] : b.shape;
    if (k !== k2) throw new Error(`gemm [${a.shape}] x [${b.shape}]`);
    return this.op(a.dataType, [m, n]);
  };
  dequantizeLinear = (x: MockOperand, scale: MockOperand) => this.op(scale.dataType, x.shape);
  tensorScatter = (x: MockOperand) => this.op(x.dataType, x.shape);
  scatterND = (x: MockOperand) => this.op(x.dataType, x.shape);
  async build(outputs: Record<string, MockOperand>) {
    for (const [name, o] of Object.entries(outputs)) if (o === undefined) throw new Error(`output ${name} is undefined`);
    return { inputs: this.inputs, outputs, destroyed: false, destroy() { this.destroyed = true; } };
  }
}

export interface MockTensor { id: number; dataType: string; shape: number[]; writable: boolean; readable: boolean; destroyed: boolean; destroy(): void }
export interface DispatchRecord { graph: any; inputs: Record<string, MockTensor>; outputs: Record<string, MockTensor> }

export class MockContext {
  tensors: MockTensor[] = [];
  dispatches: DispatchRecord[] = [];
  /** kvInPlace: the same MLTensor as an input and an output of one dispatch (WebNN implementations may reject it) */
  allowAlias = false;
  writes: { tensor: MockTensor; data: ArrayBufferView }[] = [];
  /** called by readTensor: fill the destination (a Float16Array of logits) */
  onRead: (tensor: MockTensor, destination: any) => void = () => {};
  async createTensor(d: { dataType: string; shape: number[]; writable?: boolean; readable?: boolean }): Promise<MockTensor> {
    if (!(d.dataType in BYTES)) throw new Error(`unknown dataType ${d.dataType}`);
    const tensor: MockTensor = { id: this.tensors.length, dataType: d.dataType, shape: [...d.shape], writable: !!d.writable, readable: !!d.readable, destroyed: false, destroy() { this.destroyed = true; } };
    this.tensors.push(tensor);
    return tensor;
  }
  writeTensor(tensor: MockTensor, data: ArrayBufferView) {
    if (!tensor.writable) throw new Error(`tensor ${tensor.id} is not writable`);
    const expected = Math.ceil(count(tensor.shape) * BYTES[tensor.dataType]);
    if (data.byteLength !== expected) throw new Error(`writeTensor: ${data.byteLength} bytes into a [${tensor.shape}] ${tensor.dataType} tensor (${expected} bytes)`);
    this.writes.push({ tensor, data });
  }
  dispatch(graph: any, inputs: Record<string, MockTensor>, outputs: Record<string, MockTensor>) {
    for (const [side, declared, given] of [['input', graph.inputs, inputs], ['output', graph.outputs, outputs]] as const) {
      const wanted = Object.keys(declared).sort(), got = Object.keys(given).sort();
      if (wanted.join() !== got.join()) throw new Error(`dispatch: ${side} names differ.\n  graph: ${wanted}\n  given: ${got}`);
      for (const name of wanted) {
        const tensor = given[name], d = declared[name];
        if (tensor.destroyed) throw new Error(`dispatch: ${side} '${name}' is a destroyed tensor`);
        if (tensor.dataType !== d.dataType || !same(tensor.shape, d.shape)) throw new Error(`dispatch: ${side} '${name}' is ${tensor.dataType}[${tensor.shape}], the graph wants ${d.dataType}[${d.shape}]`);
      }
    }
    const inputIds = new Set(Object.values(inputs).map((t) => t.id));
    if (!this.allowAlias) for (const [name, tensor] of Object.entries(outputs)) if (inputIds.has(tensor.id)) throw new Error(`dispatch: output '${name}' is also an input tensor (tensor ${tensor.id})`);
    this.dispatches.push({ graph, inputs, outputs });
  }
  async readTensor(tensor: MockTensor, destination: any) {
    if (!tensor.readable) throw new Error(`tensor ${tensor.id} is not readable`);
    if (destination.length !== count(tensor.shape)) throw new Error(`readTensor: destination of ${destination.length} elements for a [${tensor.shape}] tensor`);
    this.onRead(tensor, destination);
  }
}

/** installs the mock as the global MLGraphBuilder for the duration of a test */
export function installMockBuilder() {
  (globalThis as any).MLGraphBuilder = class extends MockBuilder { constructor(context: MockContext) { super(context); } };
}
