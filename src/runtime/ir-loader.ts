// Loader for the "m2w-graph" WebNN-level IR written by the Python compiler (python/m2w).
// It only TRANSLATES: every node becomes one MLGraphBuilder call, shapes and types are checked by WebNN itself.
// Spec: spec/graph.schema.json. Operator and option names are the W3C WebNN names.

export const GRAPH_FORMAT = 'm2w-graph';
export const GRAPH_VERSION = 0;

export type IRDataType = 'float32' | 'float16' | 'int32' | 'int64' | 'int8' | 'uint8' | 'int4' | 'uint4';

export interface IRInput { name: string; dataType: IRDataType; shape: number[]; role?: string }
export interface IRConstant {
  name: string; dataType: IRDataType; shape: number[];
  fill?: number; data?: number[]; external?: string;
}
export interface IRNode { op: string; inputs: (string | null)[]; outputs: string[]; options?: Record<string, unknown> }
export interface IROutput { name: string; dataType: IRDataType; shape: number[]; value: string }
export interface GraphDoc {
  format: string; version: number; name: string;
  inputs: IRInput[]; constants: IRConstant[]; nodes: IRNode[]; outputs: IROutput[];
  meta?: Record<string, unknown>;
}

// The subset of MLGraphBuilder the loader uses. Typed loosely on purpose: the loader must also run against a mock builder in Node tests.
/* eslint-disable @typescript-eslint/no-explicit-any */
export type BuilderLike = any;
export type OperandLike = any;

/** Supplies the data of constants that reference the weights package (`external`). */
export type ExternalConstantProvider = (constant: IRConstant) => ArrayBufferView;

export interface LoadedGraph {
  /** graph input operands by name (the runtime binds tensors with the same names) */
  inputs: Record<string, OperandLike>;
  /** graph output operands by output name, ready for `builder.build(outputs)` */
  outputs: Record<string, OperandLike>;
}

export function parseGraphDoc(text: string): GraphDoc {
  return checkGraphDoc(JSON.parse(text));
}

export function checkGraphDoc(doc: GraphDoc): GraphDoc {
  if (doc.format !== GRAPH_FORMAT) throw new Error(`not an ${GRAPH_FORMAT} document`);
  if (doc.version !== GRAPH_VERSION) {
    throw new Error(`unsupported ${GRAPH_FORMAT} version ${doc.version} (this runtime reads version ${GRAPH_VERSION})`);
  }
  return doc;
}

const elementCount = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

// float32 -> IEEE half bits, for runtimes without Float16Array
function toHalfBits(value: number): number {
  const f32 = new Float32Array(1); const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const x = u32[0], sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15, mant = x & 0x7fffff;
  if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // inf / nan
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    const half = mant >> shift, rem = mant & ((1 << shift) - 1), mid = 1 << (shift - 1);
    return sign | (half + (rem > mid || (rem === mid && (half & 1)) ? 1 : 0));
  }
  const half = sign | (exp << 10) | (mant >> 13), rem = mant & 0x1fff;
  return half + (rem > 0x1000 || (rem === 0x1000 && (half & 1)) ? 1 : 0);
}

/** Data of a literal constant: `fill` (every element) or `data` (row-major). int4 / uint4 are nibble-packed, low nibble first. */
export function constantData(c: IRConstant): ArrayBufferView {
  const count = elementCount(c.shape);
  const values = (i: number) => (c.fill !== undefined ? c.fill : (c.data as number[])[i]);
  if (c.fill === undefined && (!c.data || c.data.length !== count)) throw new Error(`constant ${c.name}: needs fill or ${count} data values`);
  switch (c.dataType) {
    case 'float32': return Float32Array.from({ length: count }, (_, i) => values(i));
    case 'float16': {
      const half = Uint16Array.from({ length: count }, (_, i) => toHalfBits(values(i)));
      const F16 = (globalThis as any).Float16Array;
      return F16 ? new F16(half.buffer) : half;   // Uint16Array holds the half bits: the bytes are what WebNN reads
    }
    case 'int32': return Int32Array.from({ length: count }, (_, i) => values(i));
    case 'int64': return BigInt64Array.from({ length: count }, (_, i) => BigInt(values(i)));
    case 'int8': return Int8Array.from({ length: count }, (_, i) => values(i));
    case 'uint8': return Uint8Array.from({ length: count }, (_, i) => values(i));
    case 'int4': case 'uint4': {
      const bytes = new Uint8Array(Math.ceil(count / 2));
      for (let i = 0; i < count; i++) bytes[i >> 1] |= (values(i) & 0xf) << ((i & 1) * 4);
      return bytes;
    }
  }
}

type Ref = (name: string | null) => OperandLike;

const BINARY = new Set(['add', 'sub', 'mul', 'div', 'pow']);
const UNARY = new Set(['sqrt', 'sigmoid', 'tanh', 'exp', 'log', 'softplus']);
const REDUCE = new Set(['reduceMean', 'reduceMax', 'reduceSum']);

function emitNode(b: BuilderLike, n: IRNode, ref: Ref): OperandLike {
  const o = (n.options ?? {}) as Record<string, any>;
  const x = ref(n.inputs[0]);
  if (BINARY.has(n.op)) return b[n.op](x, ref(n.inputs[1]));
  if (UNARY.has(n.op)) return b[n.op](x);
  if (REDUCE.has(n.op)) return b[n.op](x, { axes: o.axes, keepDimensions: o.keepDimensions ?? false });
  switch (n.op) {
    case 'cast': return b.cast(x, o.to);
    case 'reshape': return b.reshape(x, o.newShape);
    case 'transpose': return b.transpose(x, { permutation: o.permutation });
    case 'slice': return b.slice(x, o.starts, o.sizes);
    case 'concat': return b.concat(n.inputs.map(ref), o.axis);
    case 'expand': return b.expand(x, o.newShape);
    case 'softmax': return b.softmax(x, o.axis);
    case 'matmul': return b.matmul(x, ref(n.inputs[1]));
    case 'gemm': {
      const opts: Record<string, boolean> = {};
      if (o.bTranspose) opts.bTranspose = true;
      if (o.aTranspose) opts.aTranspose = true;
      return b.gemm(x, ref(n.inputs[1]), opts);
    }
    case 'conv2d': return b.conv2d(x, ref(n.inputs[1]), { groups: o.groups ?? 1, strides: o.strides, dilations: o.dilations, padding: o.padding, inputLayout: 'nchw', filterLayout: 'oihw' });
    case 'dequantizeLinear': return b.dequantizeLinear(x, ref(n.inputs[1]), n.inputs[2] === null ? null : ref(n.inputs[2]));
    case 'tensorScatter': {
      if (typeof b.tensorScatter !== 'function') throw new Error('this browser has no MLGraphBuilder.tensorScatter');
      return b.tensorScatter(x, ref(n.inputs[1]), ref(n.inputs[2]), { axis: o.axis });
    }
    case 'scatterND': {
      if (typeof b.scatterND !== 'function') throw new Error('this browser has no MLGraphBuilder.scatterND');
      return b.scatterND(x, ref(n.inputs[1]), ref(n.inputs[2]));
    }
    default: throw new Error(`m2w-graph: unsupported operator '${n.op}'`);
  }
}

/** Record every node of `doc` on `builder`; returns the input and output operands (call `await builder.build(outputs)` yourself). */
export function loadGraph(builder: BuilderLike, doc: GraphDoc, externalConstant?: ExternalConstantProvider): LoadedGraph {
  checkGraphDoc(doc);
  const operands = new Map<string, OperandLike>();
  const inputs: Record<string, OperandLike> = {};
  for (const i of doc.inputs) {
    const operand = builder.input(i.name, { dataType: i.dataType, shape: i.shape });
    operands.set(i.name, operand);
    inputs[i.name] = operand;
  }
  for (const c of doc.constants) {
    let data: ArrayBufferView;
    if (c.external !== undefined) {
      if (!externalConstant) throw new Error(`constant ${c.name} references external weights '${c.external}' but no provider was given`);
      data = externalConstant(c);
    } else {
      data = constantData(c);
    }
    operands.set(c.name, builder.constant({ dataType: c.dataType, shape: c.shape }, data));
  }
  const ref: Ref = (name) => {
    if (name === null) return null;
    const operand = operands.get(name);
    if (operand === undefined) throw new Error(`m2w-graph '${doc.name}': operand '${name}' is used before it is defined`);
    return operand;
  };
  for (const n of doc.nodes) operands.set(n.outputs[0], emitNode(builder, n, ref));
  const outputs: Record<string, OperandLike> = {};
  for (const o of doc.outputs) outputs[o.name] = ref(o.value);
  return { inputs, outputs };
}

/** loadGraph + build. */
export async function buildGraph(builder: BuilderLike, doc: GraphDoc, externalConstant?: ExternalConstantProvider) {
  const { inputs, outputs } = loadGraph(builder, doc, externalConstant);
  const graph = await builder.build(outputs);
  return { graph, inputs, outputs };
}
