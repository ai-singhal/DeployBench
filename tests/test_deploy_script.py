"""Tests for utils/deploy_script.py — generate_deploy_script() pure function.

Key invariants verified:
  - Output is always valid Python (ast.parse must not raise).
  - All 4 canonical config names produce distinct, correct loading code.
  - Unknown config names fall back gracefully (FP32 baseline).
  - model_name is embedded in the output.
  - Required structural patterns are present in every generated script.
"""
import ast
import pytest
from utils.deploy_script import generate_deploy_script


# ---------------------------------------------------------------------------
# Canonical config names from the spec
# ---------------------------------------------------------------------------
VALID_CONFIGS = [
    "FP32 (Baseline)",
    "FP16 (Half Precision)",
    "INT8 (Quantized)",
    "ONNX + FP16",
]

MODELS = ["yolov8n", "yolov8s", "yolov8m"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(script: str) -> ast.Module:
    """Return the AST for *script*, raising AssertionError with the script text on failure."""
    try:
        return ast.parse(script)
    except SyntaxError as exc:
        pytest.fail(f"generate_deploy_script produced invalid Python:\n{script}\n\nError: {exc}")


# ---------------------------------------------------------------------------
# Return-type and basic structure
# ---------------------------------------------------------------------------

class TestGenerateDeployScriptReturnType:
    def test_returns_string(self):
        result = generate_deploy_script("yolov8s", "FP32 (Baseline)")
        assert isinstance(result, str)

    def test_non_empty_string(self):
        result = generate_deploy_script("yolov8s", "FP32 (Baseline)")
        assert len(result.strip()) > 0


class TestGenerateDeployScriptValidPython:
    """All config × model combinations must produce syntactically valid Python."""

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    @pytest.mark.parametrize("model_name", MODELS)
    def test_valid_python_for_all_configs_and_models(self, model_name, config):
        script = generate_deploy_script(model_name, config)
        _parse(script)  # raises on SyntaxError

    def test_unknown_config_produces_valid_python(self):
        script = generate_deploy_script("yolov8s", "TOTALLY_UNKNOWN_CONFIG")
        _parse(script)

    def test_empty_config_produces_valid_python(self):
        script = generate_deploy_script("yolov8s", "")
        _parse(script)

    def test_special_characters_in_model_name_produce_valid_python(self):
        # Model names should be safe identifiers; verify the generated code is parseable.
        script = generate_deploy_script("yolov8s-custom", "FP32 (Baseline)")
        _parse(script)


class TestGenerateDeployScriptModelName:
    """The model_name must appear in the generated script."""

    @pytest.mark.parametrize("model_name", MODELS)
    def test_model_name_in_output(self, model_name):
        script = generate_deploy_script(model_name, "FP32 (Baseline)")
        assert model_name in script

    @pytest.mark.parametrize("model_name", MODELS)
    def test_model_pt_file_referenced(self, model_name):
        """The .pt checkpoint file should be referenced in the generated code."""
        script = generate_deploy_script(model_name, "FP32 (Baseline)")
        assert f"{model_name}.pt" in script


class TestGenerateDeployScriptRequiredPatterns:
    """Every generated script must contain the required Modal boilerplate."""

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_imports_modal(self, config):
        script = generate_deploy_script("yolov8s", config)
        assert "import modal" in script

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_imports_yolo(self, config):
        script = generate_deploy_script("yolov8s", config)
        assert "YOLO" in script

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_gpu_t4(self, config):
        script = generate_deploy_script("yolov8s", config)
        assert 'gpu="T4"' in script

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_predict_function(self, config):
        script = generate_deploy_script("yolov8s", config)
        assert "def predict" in script

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_web_endpoint(self, config):
        script = generate_deploy_script("yolov8s", config)
        assert "web_endpoint" in script

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_modal_image(self, config):
        script = generate_deploy_script("yolov8s", config)
        assert "modal.Image" in script

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_pip_install(self, config):
        script = generate_deploy_script("yolov8s", config)
        assert "pip_install" in script


class TestGenerateDeployScriptConfigSpecificCode:
    """Each config must produce its specific model-loading snippet."""

    def test_fp32_loads_plain_yolo(self):
        script = generate_deploy_script("yolov8s", "FP32 (Baseline)")
        assert 'YOLO("yolov8s.pt")' in script
        assert ".half()" not in script
        assert "quantize_dynamic" not in script
        assert 'format="onnx"' not in script

    def test_fp16_calls_half(self):
        script = generate_deploy_script("yolov8s", "FP16 (Half Precision)")
        assert ".half()" in script

    def test_int8_calls_quantize_dynamic(self):
        script = generate_deploy_script("yolov8s", "INT8 (Quantized)")
        assert "quantize_dynamic" in script
        assert "qint8" in script

    def test_onnx_exports_and_reloads(self):
        script = generate_deploy_script("yolov8s", "ONNX + FP16")
        assert 'format="onnx"' in script

    def test_unknown_config_falls_back_to_fp32(self):
        """An unrecognised config must use the FP32 baseline loading code."""
        fp32_script = generate_deploy_script("yolov8s", "FP32 (Baseline)")
        unknown_script = generate_deploy_script("yolov8s", "UNKNOWN_CONFIG")
        # Both scripts should load the model the same way (plain YOLO).
        # We verify the unknown script does NOT include half/quantize/onnx.
        assert ".half()" not in unknown_script
        assert "quantize_dynamic" not in unknown_script
        assert 'YOLO("yolov8s.pt")' in unknown_script


class TestGenerateDeployScriptDifferentModels:
    """Changing model_name must update all relevant references in the script."""

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_correct_model_referenced_in_loading_code(self, model_name, config):
        script = generate_deploy_script(model_name, config)
        assert model_name in script

    def test_different_models_produce_different_scripts(self):
        script_n = generate_deploy_script("yolov8n", "FP32 (Baseline)")
        script_m = generate_deploy_script("yolov8m", "FP32 (Baseline)")
        assert script_n != script_m

    def test_different_configs_produce_different_scripts(self):
        fp32_script = generate_deploy_script("yolov8s", "FP32 (Baseline)")
        fp16_script = generate_deploy_script("yolov8s", "FP16 (Half Precision)")
        assert fp32_script != fp16_script


class TestGenerateDeployScriptASTStructure:
    """Use the AST to verify the generated code has the required definitions."""

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_function_named_predict(self, config):
        script = generate_deploy_script("yolov8s", config)
        tree = _parse(script)
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "predict" in func_names, (
            f"No 'predict' function found in generated script for config={config!r}"
        )

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_function_named_api(self, config):
        script = generate_deploy_script("yolov8s", config)
        tree = _parse(script)
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "api" in func_names, (
            f"No 'api' function found in generated script for config={config!r}"
        )

    @pytest.mark.parametrize("config", VALID_CONFIGS)
    def test_has_import_modal_statement(self, config):
        script = generate_deploy_script("yolov8s", config)
        tree = _parse(script)
        import_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.extend(alias.name for alias in node.names)
        assert "modal" in import_names, (
            f"'import modal' not found in AST for config={config!r}"
        )
