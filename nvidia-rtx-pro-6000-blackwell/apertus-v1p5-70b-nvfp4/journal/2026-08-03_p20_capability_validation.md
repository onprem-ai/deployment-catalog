# 2026-08-03 - p20 wheel image capability validation

## Summary

Validated the final local image build:

```text
onpremai/vllm-apertus-1p5:p20
local image: sha256:2406bb7e9b06cbcd8904638b716aab6356cd73f35e57f5d25a28ce1e5d85102c
local registry digest: sha256:b847175940f1606f895f93feda86f1ddf94a5fd660fe27ff7d658afd39e20664
```

The model was served from:

```text
/persistent/models/onprem-ai--Apertus-v1.5-70B-NVFP4--model-apertus-model
```

The directory contained only the canonical Hugging Face tokenizer weight names:

```text
model-vision_tokenizer-model.safetensors
model-wavtokenizer-model.safetensors
```

No legacy aliases or symlinks were present.

Deployment configuration:

```text
model: apertus-v15-70b
max model length: 16384
KV cache: fp8_e4m3
GPU memory utilization: 0.64
image limit: 2
sound limit: 0
```

The 16k profile needed `additional_memory_gi=1`. At utilization 0.63, vLLM
reported 1.96 GiB available for KV cache but required 2.5 GiB. Utilization
0.64 initialized a 2.91 GiB KV cache successfully.

## Runtime validation

The service became ready with zero restarts. The following initialization
stages completed:

- `Apertus1p5ForConditionalGeneration` architecture resolution
- Apertus 1.5 Hugging Face processor construction
- Emu3.5 vision tokenizer loading
- WavTokenizer audio tokenizer loading
- All five safetensors shards loading
- Dummy image encoder profiling
- Torch graph compilation
- FlashInfer warmup
- CUDA graph capture
- FP8 KV cache allocation

Manual API checks passed:

- Text returned exactly `APERTUS_P20_OK`
- Standard tool calling returned `get_weather` with valid JSON arguments
- Vision identified red top-left, green top-right, and blue bottom-right

## Official capability suite results

The application capability suites were run through `POST /api/capability-test`.
Overall result: 9 of 11 tests passed.

### Vision: 4 of 4 passed

| Test | Result | Detail |
|------|--------|--------|
| `color_quadrants` | PASS | Identified red, green, blue, and yellow |
| `ocr_hello_42` | PASS | Returned `HELLO 42` |
| `landscape_scene` | PASS | Recognized the synthetic landscape |
| `animal_scene` | PASS | Recognized the synthetic animal scene |

### Tool calling: 4 of 5 passed

| Test | Result | Detail |
|------|--------|--------|
| `single_tool` | PASS | Selected `get_weather` for Berlin |
| `multi_tool_selection` | PASS | Selected `send_email` with the correct recipient |
| `parallel_tool_calls` | FAIL | Returned one call instead of separate Berlin and London calls |
| `tool_result_roundtrip` | PASS | Produced natural language from the tool result |
| `no_tool_restraint` | PASS | Did not call a tool for `2 + 2` |

### Vision plus tools: 1 of 2 passed

| Test | Result | Detail |
|------|--------|--------|
| `vision_tool_call` | FAIL | Returned the schema property as the function name and a list as arguments |
| `vision_with_tools_restraint` | PASS | Described the image without calling a tool |

## Reproduce with the capability-test API

Set the endpoint and service name:

```bash
API=http://<control-plane-address>:8437
NAME=<service-name>
BASE_URL=http://<inference-service-address>/v1
```

Start a suite. Replace `vision` with `tools` or `vision-tools` for the other
suites:

```bash
curl --fail-with-body -X POST \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  "$API/api/capability-test?name=$NAME&namespace=<namespace>" \
  -d "{
    \"base_url\": \"$BASE_URL\",
    \"model\": \"apertus-v15-70b\",
    \"api_key\": null,
    \"test_type\": \"vision\"
  }"
```

Poll the result:

```bash
curl --fail-with-body \
  -H "X-API-Key: $API_KEY" \
  "$API/api/capability-test?name=$NAME&namespace=<namespace>&test_type=vision"
```

Delete an old test Job before rerunning the same suite:

```bash
curl --fail-with-body -X DELETE \
  -H "X-API-Key: $API_KEY" \
  "$API/api/capability-test?name=$NAME&namespace=<namespace>&test_type=vision"
```

## Reproduce the parallel tool-call failure

The suite sends one weather tool and asks for two cities:

```json
{
  "model": "apertus-v15-70b",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Berlin and London?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "temperature": 0
}
```

Expected: two entries in `message.tool_calls`, one for Berlin and one for
London.

Observed: only one entry in `message.tool_calls`.

## Reproduce the vision-driven tool-call failure

Use a synthetic four-quadrant image with red, green, blue, and yellow
quadrants. The request asks the model to call this tool:

```json
{
  "type": "function",
  "function": {
    "name": "describe_image",
    "description": "Report structured information about an image",
    "parameters": {
      "type": "object",
      "properties": {
        "colors": {
          "type": "array",
          "items": {"type": "string"}
        },
        "layout": {"type": "string"}
      },
      "required": ["colors", "layout"]
    }
  }
}
```

Prompt:

```text
Look at this image and use the describe_image tool to report what colors you
see and how they are arranged.
```

Expected:

```json
{
  "name": "describe_image",
  "arguments": "{\"colors\":[\"red\",\"green\",\"blue\",\"yellow\"],\"layout\":\"quadrants\"}"
}
```

Observed:

```json
{
  "name": "colors",
  "arguments": "[\"red\", \"green\", \"blue\", \"yellow\"]"
}
```

After JSON decoding, `arguments` is a list rather than the required object.
This is a genuine combined vision and structured-tool failure. Standalone
vision and ordinary tool selection both pass.

## Interpretation

The image is suitable for text inference, normal single-tool calling,
multi-tool selection, tool-result round trips, tool restraint, and standalone
vision. Current known behavioral limitations are:

1. A request for multiple parallel calls may produce only one call.
2. An image-driven structured call may confuse a schema property with the
   function name and emit the property's value as the complete arguments.

These are model or parser behavior limitations, not startup, checkpoint, or
container failures. The serving process remained healthy with zero restarts
throughout all capability tests.
