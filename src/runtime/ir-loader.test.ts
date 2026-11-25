import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { checkGraphDoc, constantData, loadGraph, parseGraphDoc, type GraphDoc } from './ir-loader.js';

const golden = (name: string): GraphDoc => parseGraphDoc(readFileSync(new URL(`../../tests/golden/${name}.json`, import.meta.url), 'utf8'));

// A recording builder: every call returns an operand that remembers how it was made.
function mockBuilder() {
  const calls: { op: string; args: unknown[] }[] = [];
  const make = (op: string) => (...args: unknown[]) => { calls.push({ op, args }); return { op, id: calls.length }; };
  const target: Record<string, unknown> = {};
  const builder = new Proxy(target, {
    get: (_t, prop: string) => (prop === 'calls' ? calls : make(prop)),
  });
  return { builder: builder as any, calls };
}

describe('ir-loader', () => {
  it('translates every node of the golden graph into one builder call, in order', () => {
    const doc = golden('tiny_all_ops');
    const { builder, calls } = mockBuilder();
    const { inputs, outputs } = loadGraph(builder, doc);
    expect(Object.keys(inputs)).toEqual(doc.inputs.map((i) => i.name));
    expect(Object.keys(outputs)).toEqual(['e', 'updated', 'scattered', 'half_const']);
    expect(calls.length).toBe(doc.inputs.length + doc.constants.length + doc.nodes.length);
    const ops = calls.slice(doc.inputs.length + doc.constants.length).map((c) => c.op);
    expect(ops).toEqual(doc.nodes.map((n) => n.op));
    // every operator the IR defines appears in the golden graph
    for (const op of ['add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'sigmoid', 'tanh', 'cast', 'reshape', 'transpose', 'slice', 'concat', 'expand', 'reduceMean',
      'reduceMax', 'reduceSum', 'softmax', 'matmul', 'gemm', 'dequantizeLinear', 'tensorScatter', 'scatterND']) expect(ops).toContain(op);
  });

  it('maps options to WebNN call shapes', () => {
    const doc = golden('tiny_all_ops');
    const { builder, calls } = mockBuilder();
    loadGraph(builder, doc);
    const find = (op: string) => calls.find((c) => c.op === op)!;
    expect(find('gemm').args[2]).toEqual({ bTranspose: true });
    expect(find('transpose').args[1]).toEqual({ permutation: [1, 0, 2] });
    expect(find('reduceMean').args[1]).toEqual({ axes: [2], keepDimensions: true });
    expect(find('tensorScatter').args[3]).toEqual({ axis: 2 });
    expect(find('cast').args[1]).toBe('float32');
    expect(find('slice').args.slice(1)).toEqual([[0, 0], [1, 4]]);
    const dequantize = calls.filter((c) => c.op === 'dequantizeLinear');
    expect(dequantize[0].args[2]).not.toBeNull();   // with the zero-point constant
    expect(dequantize[1].args[2]).toBeNull();       // zero point omitted
  });

  it('builds literal constants with the right bytes', () => {
    const half = (value: number) => new Uint16Array(new Uint8Array(constantData({ name: 'c', dataType: 'float16', shape: [1], fill: value }).buffer.slice(0)).buffer)[0];
    // expected bit patterns from numpy.float16
    expect(half(1)).toBe(0x3c00); expect(half(0.5)).toBe(0x3800); expect(half(-2)).toBe(0xc000); expect(half(65504)).toBe(0x7bff);
    expect(half(1e-6)).toBe(0x11); expect(half(1e-5)).toBe(0xa8); expect(half(0.08838834764831843)).toBe(0x2da8); expect(half(0.1)).toBe(0x2e66);
    expect(Array.from(new Uint8Array(constantData({ name: 'z', dataType: 'int4', shape: [2, 3, 1], fill: 0 }).buffer))).toEqual([0, 0, 0]);
    expect(Array.from(new Uint8Array(constantData({ name: 'n', dataType: 'int4', shape: [4], data: [1, -1, 7, -8] }).buffer))).toEqual([0xf1, 0x87]);
    expect(Array.from(constantData({ name: 'i', dataType: 'int64', shape: [1], fill: 3 }) as BigInt64Array)).toEqual([3n]);
  });

  it('rejects documents of another format or version and unresolved external weights', () => {
    const doc = golden('tiny_all_ops');
    expect(() => checkGraphDoc({ ...doc, version: 99 })).toThrow(/version 99/);
    expect(() => checkGraphDoc({ ...doc, format: 'onnx' })).toThrow(/not an m2w-graph/);
    const external: GraphDoc = { ...doc, constants: [{ name: 'w', dataType: 'uint8', shape: [2], external: 'lm_head_weight' }], nodes: [], outputs: [] };
    expect(() => loadGraph(mockBuilder().builder, external)).toThrow(/no provider/);
  });
});
