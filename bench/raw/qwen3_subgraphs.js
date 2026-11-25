// =====================================================================================================================
// Qwen3-4B as a handful of small, static-shape WebNN subgraphs -- the same split that OpenVINO GenAI (NPUW) feeds the NPU.
//
// Reference: the NPUW dump in OV_GenAI_Qwen3/OVGenAI dump (under bench/data), summarised in docs/npu-local/05-ovgenai-npuw-dump.md.
//
// A decoder layer has two halves:
//     attention half : input_layernorm -> query/key/value projections -> query/key RMSNorm -> rotary embedding
//                      -> attention over the KV cache -> (attention_output, new key/value slice)
//     feed-forward half : output projection + residual add -> post_attention_layernorm -> gate/up -> SiLU(gate)*up -> down
//                      + residual add
// NPUW cuts the model so that ONE compiled graph can be re-used for every layer:
//
//   [host]  embedding lookup ......................................... hidden_states  f16 [1,S,2560]
//   HEAD    attention half of layer 0
//   BLOCK   feed-forward half of layer i  +  attention half of layer i+1      (compiled once, run 35 times,
//                                                                               every call binds that layer's weights)
//   TAIL    feed-forward half of layer 35  +  final RMSNorm (+ last row only when S > 1)
//   LM HEAD vocabulary projection ..................................... logits f16 [1,1,151936]
//   (+ KV UPDATE: a tiny helper graph that merges the new key/value rows into the device-side KV cache)
//
// Each graph is built twice: DECODE (S = 1 new token, past length P = KV cache capacity - 1) and PREFILL (S = prefill
// window, P = 0 because the whole prompt sits inside the window).
//
// What is copied 1:1 from the NPUW dump (these are exactly the things that decide what the NPU compiler recognises):
//   * Block interior computes in float32; the block boundary is float16 (cast in, cast out).
//   * Weights, scales and RMSNorm gammas are graph INPUTS (bound once as MLTensors), not constants.
//   * int4 weights are laid out [groups, outputChannels, 128] with scales [groups, outputChannels, 1], symmetric (no
//     zero point). Dequantization: dequantizeLinear -> transpose [1,0,2] -> reshape [outputChannels, inputChannels]
//     -> cast float32 -> gemm(transposeB).   (NPUW: Convert(i4->f16) -> Multiply(scale) -> Transpose -> Reshape -> Convert -> MatMul)
//   * RMSNorm: gamma * (x * (1 / sqrt(mean(x^2) + eps)))    -- "x times reciprocal", not "x divided by sqrt".
//   * Attention: query is scaled by 1/sqrt(headDim) after rotary; the value cache is stored TRANSPOSED [1,kvHeads,headDim,T];
//     grouped-query attention expands key/value with reshape -> expand -> reshape; softmax in float32.
//   * The graphs only emit the NEW key/value slice (present_key / present_value); merging it into the cache happens in the
//     separate KV-update graph (NPUW lets the host do it, but MLTensors cannot be partially written).
// Differences on purpose: cos/sin/mask are host inputs (NPUW computes them inside the head graph from position ids), and the
// embedding lookup runs on the host (the NPUW head graph also receives the already-looked-up f16 embedding).
//
// options (all optional):  matmulStyle  "gemm" (gemm + bTranspose, default) | "matmul" (matmul + explicit transpose)
//                          zeroPoint    "const" (all-zero int4 constant; the optional-zero-point Chromium drops it) | "none"
//                          dequantStyle "dequantizeLinear" (default) | "castMultiply" (cast to f16, then multiply by scale)
//                                       | "scaleAfterMatmul" (NPUW "DQ" form: int4 -> f16 cast only, one MatMul per quantization group,
//                                         the group scale is applied to the small MatMul output and the groups are summed)
//                          computeType  "float32" (default, the NPUW dump form) | "float16" (no f32 inside the graphs; RMSNorm x^2 can overflow where it is not fused)
//                          -- diagnostics for the per-block cost (default = the NPUW form; every one of these is an experiment) --
//                          attentionType "float16"   : the whole attention (concat, matmuls, softmax) in f16, no f16 -> f32 casts of the caches
//                          rmsStyle      "divide"    : RMSNorm as x / sqrt(mean+eps) instead of x * (1 / sqrt(mean+eps))
//                          valueLayout   "natural"   : value cache [1,kvHeads,T,128] like the key cache (no transposes of V)
//                          gqaStyle      "grouped"   : grouped-query attention without materialising the 4x expanded K/V: the 4 query heads of a group
//                                                      become extra query rows ([1,8,4*S,128] @ K^T [1,8,128,T])
//                          blockPart     "ffn" | "attention" (block graph only): build just one half of the block
//                          kvStyle       "scatterND" (needs valueLayout "natural"): the fused KV update is a scatterND of the new rows into the cache
//                                        (extra input kv_indices int32 [1,kvHeads,S,3] = (0, head, cache row)) instead of past * keep + selector @ new
//                                        attention then reads the scatterND result itself (cache length P, mask columns = cache rows), no concat(past, new), like the WebNN EP static GroupQueryAttention
//                          fuseKvUpdate experiment: the attention half also merges its new key/value rows into the KV cache INSIDE the graph
//                                       (extra inputs rows_to_keep / row_selector, extra outputs updated_key / updated_value), so no separate
//                                       KV-update dispatch is needed:  updated = past * rows_to_keep + row_selector @ new rows
//                          constants    experiment only: which weight kinds become graph CONSTANTS instead of inputs, a list of
//                                       "weight" | "scale" | "gamma" (needs weightProvider([name, dataType, shape]) -> typed array)
// =====================================================================================================================
import { CFG } from "./layer.js?v=20261119x";

const {
    E: hiddenSize, NH: numAttentionHeads, KVH: numKeyValueHeads, D: headDim, FF: feedForwardSize,
    BLOCK: quantGroupSize, EPS: rmsNormEpsilon, VOCAB: vocabularySize,
} = CFG;
const attentionWidth = numAttentionHeads * headDim;        // 4096: width of the attention output before the output projection
const keyValueWidth = numKeyValueHeads * headDim;          // 1024
const elementCount = (shape) => shape.reduce((a, b) => a * b, 1);
const BYTES_PER_ELEMENT = { int4: 0.5, uint4: 0.5, int8: 1, uint8: 1, float16: 2, float32: 4 };
export const byteLengthOf = ([, dataType, shape]) => elementCount(shape) * BYTES_PER_ELEMENT[dataType];

// ---------------------------------------------------------------------------------------------------------------------
// Weight tensors of one layer: [name, dataType, shape]. These names are the graph input names AND the keys in the weight
// manifest (tools/extract_ov_weights.py), so they must stay in sync.
// ---------------------------------------------------------------------------------------------------------------------
// weight layout (experiment switch): "groupMajor" = [groups, outputChannels, 128] as in the NPUW dump (default);
// "outputMajor" = [outputChannels, groups, 128] (the original OpenVINO / ONNX layout, no transpose after dequantization)
// "float16"     = plain f16 weights [outputChannels, inputChannels], no scale: what a BUILD-TIME dequantization would produce
let weightLayout = "groupMajor";
export const setWeightLayout = (layout) => { weightLayout = layout; };
function quantizedProjection(prefix, inputChannels, outputChannels) {
    const groups = inputChannels / quantGroupSize;
    const outputMajor = weightLayout === "outputMajor";
    if (weightLayout === "float16") return [[`${prefix}_weight`, "float16", [outputChannels, inputChannels]]];
    return [
        [`${prefix}_weight`, "int4", outputMajor ? [outputChannels, groups, quantGroupSize] : [groups, outputChannels, quantGroupSize]],   // symmetric int4
        [`${prefix}_scale`, "float16", outputMajor ? [outputChannels, groups, 1] : [groups, outputChannels, 1]],                          // one scale per (group, output channel)
    ];
}
// mergedProjections (experiment, like NPUW's "parallel MatMul merge"): q/k/v are ONE projection (output channels 4096 + 1024 + 1024) and gate/up ONE projection
// (2 x 9728); the outputs are sliced. Fewer weight inputs per block (-6) and fewer MatMuls (-3). The weight files would have to be merged along the output channels.
let mergedProjections = false;
export const setMergedProjections = (on) => { mergedProjections = on; };
export const attentionWeightSpecs = () => [
    ["input_layernorm_gamma", "float16", [1, 1, hiddenSize]],
    ...(mergedProjections
        ? [...quantizedProjection("qkv_projection", hiddenSize, attentionWidth + 2 * keyValueWidth), ["query_norm_gamma", "float16", [1, 1, 1, headDim]], ["key_norm_gamma", "float16", [1, 1, 1, headDim]]]
        : [...quantizedProjection("query", hiddenSize, attentionWidth), ["query_norm_gamma", "float16", [1, 1, 1, headDim]],
           ...quantizedProjection("key", hiddenSize, keyValueWidth), ["key_norm_gamma", "float16", [1, 1, 1, headDim]],
           ...quantizedProjection("value", hiddenSize, keyValueWidth)]),
];
export const feedForwardWeightSpecs = () => [
    ...quantizedProjection("output_projection", attentionWidth, hiddenSize), ["post_attention_layernorm_gamma", "float16", [1, 1, hiddenSize]],
    ...(mergedProjections ? quantizedProjection("gate_up_projection", hiddenSize, 2 * feedForwardSize)
        : [...quantizedProjection("gate_projection", hiddenSize, feedForwardSize), ...quantizedProjection("up_projection", hiddenSize, feedForwardSize)]),
    ...quantizedProjection("down_projection", feedForwardSize, hiddenSize),
];
export const tailWeightSpecs = () => [...feedForwardWeightSpecs(), ["final_norm_gamma", "float16", [1, 1, hiddenSize]]];
// lm_head shares the (tied) embedding table: uint8, asymmetric, one scale and one zero point per vocabulary row
// lmHeadFormat (experiment): "uint8Asymmetric" (the OpenVINO model, default) | "int8Symmetric" (int8 weight + scale, no zero point)
let lmHeadFormat = "uint8Asymmetric";
export const setLmHeadFormat = (format) => { lmHeadFormat = format; };
export const lmHeadWeightSpecs = () => (lmHeadFormat === "int8Symmetric"
    ? [["lm_head_weight", "int8", [vocabularySize, hiddenSize]], ["lm_head_scale", "float16", [vocabularySize, 1]]]
    : [["lm_head_weight", "uint8", [vocabularySize, hiddenSize]], ["lm_head_zero_point", "uint8", [vocabularySize, 1]], ["lm_head_scale", "float16", [vocabularySize, 1]]]);

// ---------------------------------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------------------------------
function createGraphContext(builder, options) {
    const context = {
        builder,
        options: { matmulStyle: "gemm", zeroPoint: "const", dequantStyle: "dequantizeLinear", computeType: "float32", ...options },
        inputs: {}, constantCache: new Map(), sequenceLength: 0,
    };
    context.computeType = context.options.computeType;
    context.declareInput = (name, dataType, shape) => (context.inputs[name] = builder.input(name, { dataType, shape }));
    // inputs that all layers of a multi-layer graph share (rope tables, mask, KV selectors): declared once
    context.sharedInput = (name, dataType, shape) => context.inputs[name] ?? context.declareInput(name, dataType, shape);
    const constantKinds = new Set(context.options.constants || []);
    const kindOf = (name) => (name.endsWith("_weight") ? "weight" : name.endsWith("_scale") ? "scale" : "gamma");
    // gammaType "float32" (experiment): the RMSNorm gammas are float32 graph inputs like in the NPUW dump (default: float16 inputs, converted inside the graph)
    const gammaAsFloat32 = (name, dataType) => (context.options.gammaType === "float32" && name.endsWith("_gamma") ? "float32" : dataType);
    context.declareWeights = (specs, prefix = "") => Object.fromEntries(specs.map(([name, dataType, shape]) => [name,
        constantKinds.has(kindOf(name)) ? builder.constant({ dataType, shape }, context.options.weightProvider([name, dataType, shape])) : context.declareInput(prefix + name, gammaAsFloat32(name, dataType), shape)]));
    // scalar-like constant (shape [1,1,..]) in the compute type, e.g. the exponent 2 or epsilon of the RMSNorm
    context.scalar = (value, rank) => {
        const key = `scalar:${value}:${rank}`;
        if (!context.constantCache.has(key)) {
            const data = context.computeType === "float16" ? new Float16Array([value]) : new Float32Array([value]);
            context.constantCache.set(key, builder.constant({ dataType: context.computeType, shape: new Array(rank).fill(1) }, data));
        }
        return context.constantCache.get(key);
    };
    context.zeroPointOfZeros = (shape, dataType = "int4") => {
        const key = `zero_point:${dataType}:${shape}`;
        if (!context.constantCache.has(key)) {
            context.constantCache.set(key, builder.constant({ dataType, shape }, new Uint8Array(Math.ceil(elementCount(shape) * BYTES_PER_ELEMENT[dataType]))));
        }
        return context.constantCache.get(key);
    };
    return context;
}
// cast to the compute type (float32 in the NPUW form); no-op when already that type
const toComputeType = (context, x) => (x.dataType === context.computeType ? x : context.builder.cast(x, context.computeType));
// cast to float16 at the graph boundary
// (an operand that is directly a graph input cannot be a graph output: with computeType=float16 the "cast" is a no-op, so give it a copy)
const toFloat16 = (context, x) => (x.dataType !== "float16" ? context.builder.cast(x, "float16")
    : Object.values(context.inputs).includes(x) ? context.builder.reshape(x, x.shape) : x);

// ---------------------------------------------------------------------------------------------------------------------
// quantizedLinear:  y = x @ W^T   with W stored as symmetric int4 + per-group scale
//   x [1,S,inputChannels] (compute type)  ->  y [1,S,outputChannels]
//   weight  int4 [groups, outputChannels, 128],  scale f16 [groups, outputChannels, 1]
//   steps:  dequantizeLinear           -> f16 [groups, outputChannels, 128]
//           transpose [1,0,2]          -> f16 [outputChannels, groups, 128]
//           reshape                    -> f16 [outputChannels, inputChannels]
//           cast to compute type       -> f32
//           gemm(x, W, transposeB)     -> [S, outputChannels]  (== MatMul(transpose_b=true) in OpenVINO)
// ---------------------------------------------------------------------------------------------------------------------
function quantizedLinear(context, x, weight, scale, inputChannels, outputChannels, sequenceLength) {
    const { builder, options } = context, groups = inputChannels / quantGroupSize;
    if (weightLayout === "float16") {           // experiment: weights were dequantized to f16 beforehand
        const weightMatrix = toComputeType(context, weight);
        if (options.matmulStyle === "matmul") return builder.matmul(x, builder.transpose(weightMatrix, { permutation: [1, 0] }));
        return builder.reshape(builder.gemm(builder.reshape(x, [sequenceLength, inputChannels]), weightMatrix, { bTranspose: true }), [1, sequenceLength, outputChannels]);
    }
    if (options.dequantStyle === "scaleAfterMatmul") {
        // y = sum over groups g of ( x_g @ W_g^T ) * scale_g      x_g [S,128], W_g int4 [N,128] -> f16 only (no scale), scale_g [1,N]
        const weightsByGroup = builder.transpose(toComputeType(context, builder.cast(weight, "float16")), { permutation: [0, 2, 1] });   // [groups,128,N]
        const inputsByGroup = builder.transpose(builder.reshape(x, [sequenceLength, groups, quantGroupSize]), { permutation: [1, 0, 2] });  // [groups,S,128]
        const scalePerOutput = toComputeType(context, builder.reshape(scale, [groups, 1, outputChannels]));                               // [groups,1,N]
        const partial = builder.mul(builder.matmul(inputsByGroup, weightsByGroup), scalePerOutput);                                      // [groups,S,N]
        return builder.reshape(builder.reduceSum(partial, { axes: [0], keepDimensions: false }), [1, sequenceLength, outputChannels]);
    }
    let dequantized;
    if (options.dequantStyle === "castMultiply") {
        dequantized = builder.mul(builder.cast(weight, "float16"), scale);
    } else if (options.zeroPoint === "none") {
        dequantized = builder.dequantizeLinear(weight, scale, null);
    } else {
        dequantized = builder.dequantizeLinear(weight, scale, context.zeroPointOfZeros(weightLayout === "outputMajor" ? [outputChannels, groups, 1] : [groups, outputChannels, 1]));
    }
    const groupMajorFixed = weightLayout === "outputMajor" ? dequantized : builder.transpose(dequantized, { permutation: [1, 0, 2] });   // -> [outputChannels, groups, 128]
    const weightMatrix = toComputeType(context, builder.reshape(groupMajorFixed, [outputChannels, inputChannels]));
    if (options.matmulStyle === "matmul") return builder.matmul(x, builder.transpose(weightMatrix, { permutation: [1, 0] }));
    const rows = builder.reshape(x, [sequenceLength, inputChannels]);
    return builder.reshape(builder.gemm(rows, weightMatrix, { bTranspose: true }), [1, sequenceLength, outputChannels]);
}

// ---------------------------------------------------------------------------------------------------------------------
// rmsNorm:  gamma * ( x * ( 1 / sqrt( mean(x^2) + epsilon ) ) )      reduced over the last axis
//   pow(x,2) -> reduceMean(last axis, keepDimensions) -> add(epsilon) -> sqrt -> div(1, .) -> mul(x, .) -> mul(gamma, .)
// This is the form in the NPUW dump (x times the reciprocal). It is also what the NPU compiler pattern-matches; see 05 notes.
// ---------------------------------------------------------------------------------------------------------------------
function rmsNorm(context, x, gamma, rank) {
    const { builder } = context;
    const meanSquare = builder.reduceMean(builder.pow(x, context.scalar(2, rank)), { axes: [rank - 1], keepDimensions: true });
    const rms = builder.sqrt(builder.add(meanSquare, context.scalar(rmsNormEpsilon, rank)));
    if (context.options.rmsStyle === "divide") return builder.mul(toComputeType(context, gamma), builder.div(x, rms));   // diagnostic form
    return builder.mul(toComputeType(context, gamma), builder.mul(x, builder.div(context.scalar(1, rank), rms)));
}

// rotary position embedding on x [1, heads, S, headDim]:  x * cos + rotateHalf(x) * sin
//   rotateHalf(x) = concat( -x[..., headDim/2:], x[..., :headDim/2] )
function applyRotaryEmbedding(context, x, cos, sin, heads) {
    const { builder } = context, S = context.sequenceLength, half = headDim / 2;
    const firstHalf = builder.slice(x, [0, 0, 0, 0], [1, heads, S, half]);
    const secondHalf = builder.slice(x, [0, 0, 0, half], [1, heads, S, half]);
    const rotated = builder.concat([builder.mul(secondHalf, context.scalar(-1, 4)), firstHalf], 3);
    return builder.add(builder.mul(x, toComputeType(context, cos)), builder.mul(rotated, toComputeType(context, sin)));
}

// grouped-query attention: repeat every key/value head for its group of query heads
//   [1, kvHeads, ...trailing] -> reshape [1,kvHeads,1,...] -> expand [1,kvHeads,group,...] -> reshape [1,queryHeads,...]
function repeatKeyValueHeads(context, x, trailingShape) {
    const { builder } = context, group = numAttentionHeads / numKeyValueHeads;
    const expanded = builder.expand(builder.reshape(x, [1, numKeyValueHeads, 1, ...trailingShape]), [1, numKeyValueHeads, group, ...trailingShape]);
    return builder.reshape(expanded, [1, numAttentionHeads, ...trailingShape]);
}

// ---------------------------------------------------------------------------------------------------------------------
// updateKeyValueCache (fuseKvUpdate): write the new key / value rows into the fixed-size cache, inside the graph. Returns { updated_key, updated_value }.
//   cache [1,8,P,128] (value cache transposed [1,8,128,P] unless valueLayout=natural); new rows [1,8,S,128] / [1,8,128,S]; the host says WHERE through
//   rows_to_keep [1,1,P,1] (0 at the rows that get replaced, 1 elsewhere) and row_selector [1,1,P,S] (one-hot: row r of the cache <- new row s)
//   kvStyle "onehot"    : past * rows_to_keep + row_selector @ new           (mul + matmul with a contraction of S + add)
//          "broadcast"  : past * rows_to_keep + row_selector * new           (S = 1 only: the matmul with contraction 1 is an outer product = a broadcast mul)
//          "select"     : where(row_selector > 0, new, past)                 (S = 1 only: one element-wise select, new rows broadcast)
//          "scatterND"  : scatterND(past, kv_indices, new)                   (kv_indices int32 [1,8,S,3] = (0, head, row); natural value layout)
//          "tensorScatter": tensorScatter(past, new, kv_write_index, {axis: 2})  (kv_write_index int32|int64 [1] = the first cache row that is written; the S new rows go to
//                           rows kv_write_index .. kv_write_index+S-1; natural value layout; attention reads the result like scatterND)
// ---------------------------------------------------------------------------------------------------------------------
function updateKeyValueCache(context, { pastKey, pastValue, key, value, pastLength, sequenceLength, naturalValue, selectorPrefix = "", style }) {
    const { builder } = context, kvStyle = style || context.options.kvStyle || "onehot";
    const presentKey = toFloat16(context, key), presentValue = toFloat16(context, value);
    if (kvStyle === "scatterND") {
        if (!naturalValue) throw new Error("kvStyle=scatterND needs valueLayout=natural (key and value caches with the same layout)");
        const indices = context.sharedInput("kv_indices", "int32", [1, numKeyValueHeads, sequenceLength, 3]);   // (0, kv head, cache row) of every new row
        return { updated_key: builder.scatterND(pastKey, indices, presentKey), updated_value: builder.scatterND(pastValue, indices, presentValue) };
    }
    if (kvStyle === "tensorScatter") {
        if (!naturalValue) throw new Error("kvStyle=tensorScatter needs valueLayout=natural (key and value caches with the same layout)");
        if (typeof builder.tensorScatter !== "function") throw new Error("this browser has no MLGraphBuilder.tensorScatter");
        // one write position for the whole [1, kvHeads, S, 128] update: the sequence axis is 2 in both caches
        const writeIndex = context.sharedInput("kv_write_index", context.options.scatterIndexType || "int32", [1]);
        return { updated_key: builder.tensorScatter(pastKey, presentKey, writeIndex, { axis: 2 }), updated_value: builder.tensorScatter(pastValue, presentValue, writeIndex, { axis: 2 }) };
    }
    const selectOnly = sequenceLength === 1 && kvStyle === "select";   // select needs only row_selector (an unused graph input would be dropped and break the binding)
    const rowSelector = context.sharedInput(selectorPrefix + "row_selector", "float16", [1, 1, pastLength, sequenceLength]);
    // deriveRowsToKeep (experiment, S = 1): the selector is one-hot, so rows_to_keep = 1 - row_selector is computed in the graph (one bound tensor less per dispatch)
    const rowsToKeep = selectOnly ? null : context.options.deriveRowsToKeep && sequenceLength === 1
        ? builder.sub(builder.constant({ dataType: "float16", shape: [1] }, new Float16Array([1])), rowSelector)
        : context.sharedInput(selectorPrefix + "rows_to_keep", "float16", [1, 1, pastLength, 1]);
    const transposeLast2 = (x) => builder.transpose(x, { permutation: [0, 1, 3, 2] });
    if (sequenceLength === 1 && (kvStyle === "broadcast" || kvStyle === "select")) {
        const alongRows = (x) => builder.reshape(x, [1, 1, 1, pastLength]);   // [1,1,P,1] -> [1,1,1,P] (S = 1: only a reshape, no data moves) for the transposed value cache
        if (kvStyle === "broadcast") {
            return {
                updated_key: builder.add(builder.mul(pastKey, rowsToKeep), builder.mul(rowSelector, presentKey)),                                           // [1,8,P,128]
                updated_value: naturalValue ? builder.add(builder.mul(pastValue, rowsToKeep), builder.mul(rowSelector, presentValue))
                                            : builder.add(builder.mul(pastValue, alongRows(rowsToKeep)), builder.mul(presentValue, alongRows(rowSelector))),   // [1,8,128,P]
            };
        }
        const isNewRow = builder.greater(rowSelector, builder.constant({ dataType: "float16", shape: [1] }, new Float16Array([0])));                       // uint8 [1,1,P,1]
        return {
            updated_key: builder.where(isNewRow, presentKey, pastKey),
            updated_value: builder.where(naturalValue ? isNewRow : alongRows(isNewRow), presentValue, pastValue),
        };
    }
    return {
        updated_key: builder.add(builder.mul(pastKey, rowsToKeep), builder.matmul(rowSelector, presentKey)),                                                // [1,8,P,128]
        updated_value: naturalValue ? builder.add(builder.mul(pastValue, rowsToKeep), builder.matmul(rowSelector, presentValue))
                                    : builder.add(builder.mul(pastValue, transposeLast2(rowsToKeep)), builder.matmul(presentValue, transposeLast2(rowSelector))),   // [1,8,128,P]
    };
}

// ---------------------------------------------------------------------------------------------------------------------
// attentionHalf: one layer's attention, from the residual stream to the attention output.
//   residualStream  [1,S,2560] (compute type)
//   inputs it declares itself: rope_cos, rope_sin [1,1,S,128]; attention_mask [1,1,S,P+S] (additive); past_key
//   [1,8,P,128]; past_value [1,8,128,P] (transposed!)   (past_* only when P > 0)
//   returns (all float16): attention_output [1,S,4096], present_key [1,8,S,128], present_value [1,8,128,S]
// ---------------------------------------------------------------------------------------------------------------------
// With fuseKvUpdate the graph updates the cache itself, so the new key / value rows are not an output (each bound tensor costs ~10 us per dispatch)
const presentRowOutputs = (context, attention, keyName = "present_key", valueName = "present_value") =>
    (Object.keys(attention.kvUpdate).length && !context.options.keepPresentRows ? {} : { [keyName]: attention.presentKey, [valueName]: attention.presentValue });   // keepPresentRows: A/B experiment
function attentionHalf(context, residualStream, weights, { sequenceLength, pastLength, maskDataType, prefix = "" }) {
    const { builder } = context;
    // kvStyle=scatterND (like the WebNN EP's static GroupQueryAttention): the new rows are scattered into the fixed-size cache and attention reads the
    // scatterND RESULT directly (cache length P, the new rows already sit at their positions, the mask hides the rest) - no concat(past, new) of length P+S.
    // attendInCache=1 does the same for the one-hot styles: attention reads the updated cache (length P) instead of concat(past, new) (length P+S)
    // scatterAttention "concat" (diagnostic): kvStyle=scatterND only writes the cache (graph output); attention still reads concat(past, new) like the one-hot form
    const attendInCache = pastLength > 0 && context.options.fuseKvUpdate
        && (((context.options.kvStyle === "scatterND" || context.options.kvStyle === "tensorScatter") && context.options.scatterAttention !== "concat") || context.options.attendInCache === true);
    // kvStyle=staged (decode only; use valueLayout=natural): the O(cache) one-hot update is replaced by an update of a SMALL staging buffer [1,8,R,128] that holds the newest R rows;
    // attention reads concat(main cache, updated staging). The main cache is only written by a separate "flush" graph once every R steps (see 05 notes 27).
    const staged = pastLength > 0 && sequenceLength === 1 && context.options.fuseKvUpdate && context.options.kvStyle === "staged";
    const stagingRows = context.options.stagingRows || 64;
    const totalLength = attendInCache ? pastLength : staged ? pastLength + stagingRows : pastLength + sequenceLength;
    context.sequenceLength = sequenceLength;
    // mergeRope (experiment): ONE input "rope" [1,2,S,128] (cos, then sin) instead of rope_cos and rope_sin: one bound tensor less per dispatch, two slices more
    if (context.options.mergeRope && !context.ropeParts) {
        const rope = context.sharedInput("rope", "float16", [1, 2, sequenceLength, headDim]);
        context.ropeParts = { cos: builder.slice(rope, [0, 0, 0, 0], [1, 1, sequenceLength, headDim]), sin: builder.slice(rope, [0, 1, 0, 0], [1, 1, sequenceLength, headDim]) };
    }
    const cos = context.ropeParts ? context.ropeParts.cos : context.sharedInput("rope_cos", "float16", [1, 1, sequenceLength, headDim]);
    const sin = context.ropeParts ? context.ropeParts.sin : context.sharedInput("rope_sin", "float16", [1, 1, sequenceLength, headDim]);
    const attentionMask = context.sharedInput("attention_mask", maskDataType, [1, 1, sequenceLength, totalLength]);
    const pastKey = pastLength > 0 ? context.declareInput(prefix + "past_key", "float16", [1, numKeyValueHeads, pastLength, headDim]) : null;
    const naturalValue = context.options.valueLayout === "natural";   // diagnostic: value cache [1,kvHeads,T,128] (default: transposed [1,kvHeads,128,T])
    const pastValue = pastLength > 0 ? context.declareInput(prefix + "past_value", "float16", naturalValue ? [1, numKeyValueHeads, pastLength, headDim] : [1, numKeyValueHeads, headDim, pastLength]) : null;

    // 1. normalise, then project to query / key / value
    const normalized = rmsNorm(context, residualStream, weights.input_layernorm_gamma, 3);
    const splitHeads = (projected, heads, normGamma) => builder.transpose(   // [1,S,heads*128] -> RMSNorm per head -> [1,heads,S,128]
        rmsNorm(context, builder.reshape(projected, [1, sequenceLength, heads, headDim]), normGamma, 4), { permutation: [0, 2, 1, 3] });
    const merged = !!weights.qkv_projection_weight;   // mergedProjections: one projection, three slices
    const qkv = merged ? quantizedLinear(context, normalized, weights.qkv_projection_weight, weights.qkv_projection_scale, hiddenSize, attentionWidth + 2 * keyValueWidth, sequenceLength) : null;
    const slicePart = (start, width) => builder.slice(qkv, [0, 0, start], [1, sequenceLength, width]);
    const query = applyRotaryEmbedding(context, splitHeads(
        merged ? slicePart(0, attentionWidth) : quantizedLinear(context, normalized, weights.query_weight, weights.query_scale, hiddenSize, attentionWidth, sequenceLength),
        numAttentionHeads, weights.query_norm_gamma), cos, sin, numAttentionHeads);
    const key = applyRotaryEmbedding(context, splitHeads(
        merged ? slicePart(attentionWidth, keyValueWidth) : quantizedLinear(context, normalized, weights.key_weight, weights.key_scale, hiddenSize, keyValueWidth, sequenceLength),
        numKeyValueHeads, weights.key_norm_gamma), cos, sin, numKeyValueHeads);
    const value = builder.transpose(builder.reshape(   // value is kept transposed: [1,kvHeads,128,S]  (natural layout: [1,kvHeads,S,128])
        merged ? slicePart(attentionWidth + keyValueWidth, keyValueWidth) : quantizedLinear(context, normalized, weights.value_weight, weights.value_scale, hiddenSize, keyValueWidth, sequenceLength),
        [1, sequenceLength, numKeyValueHeads, headDim]), { permutation: naturalValue ? [0, 2, 1, 3] : [0, 2, 3, 1] });
    const scaledQuery = builder.mul(query, context.scalar(1 / Math.sqrt(headDim), 4));

    // 2. attention over [past cache + new rows]
    const attentionType = context.options.attentionType === "float16" ? "float16" : context.computeType;
    const asAttentionType = (x) => (x.dataType === attentionType ? x : builder.cast(x, attentionType));
    const queryForAttention = asAttentionType(scaledQuery), keyForAttention = asAttentionType(key), valueForAttention = asAttentionType(value);
    const updateCache = () => updateKeyValueCache(context, { pastKey, pastValue, key, value, pastLength, sequenceLength, naturalValue });
    const updatedCache = attendInCache ? updateCache() : null;
    let stagingUpdate = null;
    if (staged) {
        const stagingKey = context.declareInput(prefix + "staging_key", "float16", [1, numKeyValueHeads, stagingRows, headDim]);
        const stagingValue = context.declareInput(prefix + "staging_value", "float16", naturalValue ? [1, numKeyValueHeads, stagingRows, headDim] : [1, numKeyValueHeads, headDim, stagingRows]);
        // stagingUpdate "none" (timing diagnostic, WRONG results: the new token is not stored): attention reads the staging buffer as it is, to separate the cost of the update
        stagingUpdate = context.options.stagingUpdate === "none"
            ? { updated_key: builder.reshape(stagingKey, stagingKey.shape), updated_value: builder.reshape(stagingValue, stagingValue.shape) }
            : updateKeyValueCache(context, { pastKey: stagingKey, pastValue: stagingValue, key, value, pastLength: stagingRows, sequenceLength, naturalValue, selectorPrefix: "staging_",
                                              style: context.options.stagingUpdate });   // "select" / "broadcast" / default one-hot
    }
    const allKeys = staged ? builder.concat([asAttentionType(pastKey), asAttentionType(stagingUpdate.updated_key)], 2)
        : attendInCache ? asAttentionType(updatedCache.updated_key)
        : pastLength > 0 ? builder.concat([asAttentionType(pastKey), keyForAttention], 2) : keyForAttention;                                              // [1,8,T,128]
    const allValues = staged ? builder.concat([asAttentionType(pastValue), asAttentionType(stagingUpdate.updated_value)], naturalValue ? 2 : 3)
        : attendInCache ? asAttentionType(updatedCache.updated_value)
        : pastLength > 0 ? builder.concat([asAttentionType(pastValue), valueForAttention], naturalValue ? 2 : 3) : valueForAttention;                     // [1,8,128,T] or natural [1,8,T,128]
    const maskForAttention = asAttentionType(attentionMask);
    const transposeLast2 = (x) => builder.transpose(x, { permutation: [0, 1, 3, 2] });
    let keysTransposed, scores, probabilities, valuesTransposed, attentionContext, attentionOutput;
    if (context.options.gqaStyle === "grouped") {
        // the 4 query heads that share a key/value head become 4 extra query ROWS of that head: no 4x expanded K/V is ever built
        const group = numAttentionHeads / numKeyValueHeads, S = sequenceLength;
        const groupedQuery = builder.reshape(builder.reshape(queryForAttention, [1, numKeyValueHeads, group, S, headDim]), [1, numKeyValueHeads, group * S, headDim]);
        keysTransposed = transposeLast2(allKeys);                                                                                    // [1,8,128,T]
        const rawScores = builder.reshape(builder.matmul(groupedQuery, keysTransposed), [1, numKeyValueHeads, group, S, totalLength]);
        scores = builder.add(rawScores, builder.reshape(maskForAttention, [1, 1, 1, S, totalLength]));                              // + mask broadcast over heads
        probabilities = builder.softmax(scores, 4);
        valuesTransposed = naturalValue ? allValues : transposeLast2(allValues);                                                      // [1,8,T,128]
        attentionContext = builder.matmul(builder.reshape(probabilities, [1, numKeyValueHeads, group * S, totalLength]), valuesTransposed);   // [1,8,4S,128]
        attentionOutput = builder.reshape(builder.transpose(builder.reshape(attentionContext, [1, numAttentionHeads, S, headDim]), { permutation: [0, 2, 1, 3] }), [1, S, attentionWidth]);
    } else {
        keysTransposed = transposeLast2(repeatKeyValueHeads(context, allKeys, [totalLength, headDim]));                              // [1,32,128,T]
        scores = builder.add(builder.matmul(queryForAttention, keysTransposed), maskForAttention);                                    // [1,32,S,T]
        probabilities = builder.softmax(scores, 3);
        valuesTransposed = naturalValue ? repeatKeyValueHeads(context, allValues, [totalLength, headDim])
                                        : transposeLast2(repeatKeyValueHeads(context, allValues, [headDim, totalLength]));             // [1,32,T,128]
        attentionContext = builder.matmul(probabilities, valuesTransposed);                                                           // [1,32,S,128]
        attentionOutput = builder.reshape(builder.transpose(attentionContext, { permutation: [0, 2, 1, 3] }), [1, sequenceLength, attentionWidth]);
    }
    // fuseKvUpdate (experiment): merge the new rows into the cache in the same graph (same math as buildKvUpdateGraph)
    const kvUpdate = staged ? stagingUpdate : attendInCache ? updatedCache : context.options.fuseKvUpdate && pastLength > 0 ? updateCache() : {};
    // attentionGuard (experiment): a tiny extra consumer of the scores (max over the key axis, [1,32,S,1]) so that the NPU compiler does not
    // fuse scores -> softmax -> matmul into its attention kernel; that fused path returned wrong results for S > 1 with a past cache
    const guardSource = { scores, probabilities, values: valuesTransposed, context: attentionContext, allValues, allKeys, keysTransposed }[context.options.attentionGuard === true ? "scores" : context.options.attentionGuard];
    const guard = guardSource ? { attention_guard: toFloat16(context, builder.reduceMax(guardSource, { axes: [guardSource.shape.length - 1], keepDimensions: true })) } : {};
    const debug = context.options.debugOutputs ? { debug_scores: toFloat16(context, scores), debug_probabilities: toFloat16(context, probabilities), debug_keys_expanded: toFloat16(context, keysTransposed), debug_all_values: toFloat16(context, allValues), debug_values_expanded: toFloat16(context, valuesTransposed), debug_attention_context: toFloat16(context, attentionContext), debug_query: toFloat16(context, scaledQuery) } : {};
    Object.assign(debug, guard);
    return { attentionOutput: toFloat16(context, attentionOutput), presentKey: toFloat16(context, key), presentValue: toFloat16(context, value), debug, kvUpdate };
}

// ---------------------------------------------------------------------------------------------------------------------
// feedForwardHalf: one layer's output projection + MLP, from (residual, attention output) to the layer output.
//   h = residual + output_projection(attention_output)
//   layer_output = h + down( silu(gate(norm(h))) * up(norm(h)) )
// ---------------------------------------------------------------------------------------------------------------------
function feedForwardHalf(context, residual, attentionOutput, weights, sequenceLength) {
    const { builder } = context;
    const afterAttention = builder.add(residual, quantizedLinear(context, attentionOutput, weights.output_projection_weight, weights.output_projection_scale,
        attentionWidth, hiddenSize, sequenceLength));
    const normalized = rmsNorm(context, afterAttention, weights.post_attention_layernorm_gamma, 3);
    const gateUp = weights.gate_up_projection_weight   // mergedProjections: one projection, two slices
        ? quantizedLinear(context, normalized, weights.gate_up_projection_weight, weights.gate_up_projection_scale, hiddenSize, 2 * feedForwardSize, sequenceLength) : null;
    const gate = gateUp ? builder.slice(gateUp, [0, 0, 0], [1, sequenceLength, feedForwardSize])
        : quantizedLinear(context, normalized, weights.gate_projection_weight, weights.gate_projection_scale, hiddenSize, feedForwardSize, sequenceLength);
    const siluGate = builder.mul(gate, builder.sigmoid(gate));                     // SiLU(x) = x * sigmoid(x)  (OpenVINO: Swish)
    const up = gateUp ? builder.slice(gateUp, [0, 0, feedForwardSize], [1, sequenceLength, feedForwardSize])
        : quantizedLinear(context, normalized, weights.up_projection_weight, weights.up_projection_scale, hiddenSize, feedForwardSize, sequenceLength);
    const down = quantizedLinear(context, builder.mul(siluGate, up), weights.down_projection_weight, weights.down_projection_scale, feedForwardSize, hiddenSize, sequenceLength);
    return builder.add(afterAttention, down);
}

// =====================================================================================================================
// The graphs. Each returns { outputs, inputs }; the caller runs `await builder.build(outputs)`.
// =====================================================================================================================

// HEAD:  hidden_states (the looked-up embedding, f16 [1,S,2560]) -> attention half of layer 0
//   outputs: residual (= hidden_states, f16), attention_output (f16 [1,S,4096]), present_key, present_value
export function buildHeadGraph(builder, { sequenceLength, pastLength, maskDataType = "float16", options }) {
    const context = createGraphContext(builder, options);
    const hiddenStates = context.declareInput("hidden_states", "float16", [1, sequenceLength, hiddenSize]);
    const weights = context.declareWeights(attentionWeightSpecs());
    const residualStream = toComputeType(context, hiddenStates);
    const attention = attentionHalf(context, residualStream, weights, { sequenceLength, pastLength, maskDataType });
    return {
        outputs: { residual: toFloat16(context, residualStream), attention_output: attention.attentionOutput,
                   ...presentRowOutputs(context, attention), ...attention.debug, ...attention.kvUpdate },
        inputs: context.inputs,
    };
}

// BLOCK:  (residual, attention_output) of layer i  ->  feed-forward half of layer i  ->  attention half of layer i+1
//   compiled ONCE and dispatched 35 times; each call binds the weights of layer i (feed-forward) and layer i+1 (attention)
//   outputs: next_residual, next_attention_output (f16), present_key / present_value of layer i+1
export function buildBlockGraph(builder, { sequenceLength, pastLength, maskDataType = "float16", options }) {
    const context = createGraphContext(builder, options);
    const residual = context.declareInput("residual", "float16", [1, sequenceLength, hiddenSize]);
    const attentionOutput = context.declareInput("attention_output", "float16", [1, sequenceLength, attentionWidth]);
    const blockPart = context.options.blockPart;   // diagnostic: "ffn" = feed-forward half only, "attention" = attention half only
    if (blockPart === "ffn") {
        const ffnOutput = feedForwardHalf(context, toComputeType(context, residual), toComputeType(context, attentionOutput), context.declareWeights(feedForwardWeightSpecs()), sequenceLength);
        return { outputs: { next_residual: toFloat16(context, ffnOutput) }, inputs: context.inputs };
    }
    const layerOutput = blockPart === "attention" ? builder.add(toComputeType(context, residual), builder.mul(builder.reduceMean(toComputeType(context, attentionOutput), { axes: [2], keepDimensions: true }), context.scalar(0, 3)))
        : feedForwardHalf(context, toComputeType(context, residual), toComputeType(context, attentionOutput), context.declareWeights(feedForwardWeightSpecs()), sequenceLength);
    const nextAttention = attentionHalf(context, layerOutput, context.declareWeights(attentionWeightSpecs()), { sequenceLength, pastLength, maskDataType });
    return {
        outputs: { next_residual: toFloat16(context, layerOutput), next_attention_output: nextAttention.attentionOutput,
                   ...presentRowOutputs(context, nextAttention), ...nextAttention.debug, ...nextAttention.kvUpdate },
        inputs: context.inputs,
    };
}

// TAIL:  (residual, attention_output) of the last layer -> feed-forward half -> final RMSNorm
//   prefill (S > 1): only the LAST row is kept, because only the last prompt token produces the first output token
//   output: final_hidden_states, f16 [1,1,2560]
export function buildTailGraph(builder, { sequenceLength, options }) {
    const context = createGraphContext(builder, options);
    const residual = context.declareInput("residual", "float16", [1, sequenceLength, hiddenSize]);
    const attentionOutput = context.declareInput("attention_output", "float16", [1, sequenceLength, attentionWidth]);
    const weights = context.declareWeights(tailWeightSpecs());
    const layerOutput = feedForwardHalf(context, toComputeType(context, residual), toComputeType(context, attentionOutput), weights, sequenceLength);
    let finalHiddenStates = rmsNorm(context, layerOutput, weights.final_norm_gamma, 3);
    if (sequenceLength > 1) finalHiddenStates = builder.slice(finalHiddenStates, [0, sequenceLength - 1, 0], [1, 1, hiddenSize]);
    finalHiddenStates = toFloat16(context, finalHiddenStates);
    return { outputs: { final_hidden_states: finalHiddenStates }, inputs: context.inputs };
}

// LM HEAD:  final_hidden_states f16 [1,1,2560] -> logits f16 [1,1,151936]   (the matmul itself runs in the compute type; both boundaries are f16)
//   weight uint8 [151936,2560] (tied with the embedding), asymmetric per-row: dequantizeLinear(weight, scale, zero_point)
//   -> cast f32 -> gemm(transposeB).  weightMode "constant" (as the NPUW dump) | "input"
export function buildLmHeadGraph(builder, { weightMode = "constant", weightProvider, options }) {
    const context = createGraphContext(builder, options);
    const finalHiddenStates = toComputeType(context, context.declareInput("final_hidden_states", "float16", [1, 1, hiddenSize]));
    const weights = {};
    for (const spec of lmHeadWeightSpecs()) {
        const [name, dataType, shape] = spec;
        weights[name] = weightMode === "input" ? context.declareInput(name, dataType, shape) : builder.constant({ dataType, shape }, weightProvider(spec));
    }
    const zeroPoint = weights.lm_head_zero_point ?? context.zeroPointOfZeros([vocabularySize, 1], "int8");   // int8Symmetric: all-zero constant (an optional-zero-point Chromium drops it)
    const dequantized = toComputeType(context, builder.dequantizeLinear(weights.lm_head_weight, weights.lm_head_scale, zeroPoint));  // [vocab,2560]
    const row = builder.reshape(finalHiddenStates, [1, hiddenSize]);
    const logits = context.options.matmulStyle === "matmul"
        ? builder.matmul(row, builder.transpose(dequantized, { permutation: [1, 0] }))
        : builder.gemm(row, dequantized, { bTranspose: true });
    return { outputs: { logits: toFloat16(context, builder.reshape(logits, [1, 1, vocabularySize])) }, inputs: context.inputs };
}

// KV UPDATE: merge the S new key/value rows into the past cache of one layer, entirely on the device
//   updated = past * rows_to_keep + row_selector @ new          (0/1 selection, exact in float16)
//   key cache   [1,8,P,128]   rows_to_keep [1,1,P,1]   (1 = keep the old row, 0 = this row is overwritten)
//   value cache [1,8,128,P]   (stored transposed, so rows_to_keep / row_selector are transposed inside the graph)
//   row_selector [1,1,P,S]    (row_selector[i][j] = 1 means cache row i receives new row j)
// A TensorScatter-style in-place update would replace this graph (someone else owns that).
export function buildKvUpdateGraph(builder, { sequenceLength, pastLength, valueLayout = "transposed" }) {
    const inputs = {}, declare = (name, shape) => (inputs[name] = builder.input(name, { dataType: "float16", shape }));
    const pastKey = declare("past_key", [1, numKeyValueHeads, pastLength, headDim]), newKey = declare("new_key", [1, numKeyValueHeads, sequenceLength, headDim]);
    const natural = valueLayout === "natural";   // value cache [1,8,P,128] like the key cache (default: transposed [1,8,128,P])
    const pastValue = declare("past_value", natural ? [1, numKeyValueHeads, pastLength, headDim] : [1, numKeyValueHeads, headDim, pastLength]);
    const newValue = declare("new_value", natural ? [1, numKeyValueHeads, sequenceLength, headDim] : [1, numKeyValueHeads, headDim, sequenceLength]);
    const rowsToKeep = declare("rows_to_keep", [1, 1, pastLength, 1]), rowSelector = declare("row_selector", [1, 1, pastLength, sequenceLength]);
    const transposeLast2 = (x) => builder.transpose(x, { permutation: [0, 1, 3, 2] });
    const updatedKey = builder.add(builder.mul(pastKey, rowsToKeep), builder.matmul(rowSelector, newKey));                              // [1,8,P,128]
    const updatedValue = natural ? builder.add(builder.mul(pastValue, rowsToKeep), builder.matmul(rowSelector, newValue))                                              // [1,8,P,128]
        : builder.add(builder.mul(pastValue, transposeLast2(rowsToKeep)), builder.matmul(newValue, transposeLast2(rowSelector)));                                    // [1,8,128,P]
    return { outputs: { updated_key: updatedKey, updated_value: updatedValue }, inputs };
}

// DEQUANTIZE WEIGHT (build time, experiment): int4 weight + scale -> f16 weight [outputChannels, inputChannels], the layout the "float16"
// weight layout expects. Run once per projection while loading; the f16 output MLTensor is kept on the device and bound to the block graphs.
export function buildDequantizeWeightGraph(builder, { inputChannels, outputChannels }) {
    const groups = inputChannels / quantGroupSize;
    const weight = builder.input("weight", { dataType: "int4", shape: [groups, outputChannels, quantGroupSize] });
    const scale = builder.input("scale", { dataType: "float16", shape: [groups, outputChannels, 1] });
    const zeroPoint = builder.constant({ dataType: "int4", shape: [groups, outputChannels, 1] }, new Uint8Array(Math.ceil(groups * outputChannels / 2)));
    const dequantized = builder.dequantizeLinear(weight, scale, zeroPoint);                                   // f16 [groups, outputChannels, 128]
    const weightF16 = builder.reshape(builder.transpose(dequantized, { permutation: [1, 0, 2] }), [outputChannels, inputChannels]);
    return { outputs: { weight_f16: weightF16 }, inputs: { weight, scale } };
}

// BLOCK GROUP (experiment): `blockCount` consecutive BLOCKs in ONE graph, to cut the number of dispatches (every dispatch has a
// fixed overhead). Block j of the group is declared with the input/output name prefix "b<j>_" (b0_query_weight, b1_past_key, b2_updated_key, ...);
// rope_cos / rope_sin / attention_mask / rows_to_keep / row_selector are shared. The group takes (residual, attention_output) of its first block and
// returns next_residual / next_attention_output of its last block; intermediate residuals never leave the graph (kept in the compute type).
export function buildBlockGroupGraph(builder, { blockCount, sequenceLength, pastLength, maskDataType = "float16", options }) {
    const context = createGraphContext(builder, options);
    let residual = toComputeType(context, context.declareInput("residual", "float16", [1, sequenceLength, hiddenSize]));
    let attentionOutput = toComputeType(context, context.declareInput("attention_output", "float16", [1, sequenceLength, attentionWidth]));
    const outputs = {};
    let lastAttentionOutput = null;
    for (let block = 0; block < blockCount; block++) {
        const prefix = `b${block}_`;
        const layerOutput = feedForwardHalf(context, residual, attentionOutput, context.declareWeights(feedForwardWeightSpecs(), prefix), sequenceLength);
        const attention = attentionHalf(context, layerOutput, context.declareWeights(attentionWeightSpecs(), prefix), { sequenceLength, pastLength, maskDataType, prefix });
        Object.assign(outputs, presentRowOutputs(context, attention, prefix + "present_key", prefix + "present_value"));
        for (const [name, operand] of Object.entries({ ...attention.debug, ...attention.kvUpdate })) outputs[prefix + name] = operand;
        residual = layerOutput; attentionOutput = toComputeType(context, attention.attentionOutput); lastAttentionOutput = attention.attentionOutput;
        if (block === blockCount - 1) outputs.next_residual = toFloat16(context, layerOutput);
    }
    outputs.next_attention_output = lastAttentionOutput;
    return { outputs, inputs: context.inputs };
}
