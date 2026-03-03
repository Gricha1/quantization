# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = ["wandb", "mlflow", "swanlab", "vemlp_wandb", "tensorboard", "console", "clearml", "comet_ml"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            wandb.init(project=project_name, name=experiment_name, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "comet_ml" in default_backend:
            self.logger["comet_ml"] = CometMLLogger(project_name, experiment_name, config)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()

        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "comet_ml" in self.logger:
            self.logger["comet_ml"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, (int, float, np.floating, np.integer)):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This invocation of ClearML logger\'s function is incorrect so this attribute was dropped. ')

    def finish(self):
        self._task.mark_completed()


class CometMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        import comet_ml

        self.project_name = project_name
        self.experiment_name = experiment_name

        # Initialize Comet ML experiment (will be stored globally)
        self._experiment = comet_ml.Experiment(
            project_name=project_name,
            experiment_name=experiment_name,
            auto_param_logging=False,
            auto_metric_logging=False,
        )

        # Log config as hyperparameters
        if config is not None:
            self._experiment.log_parameters(_flatten_dict(_transform_params_to_json_serializable(config, convert_list_to_dict=True), sep="/"))

    def log(self, data, step):
        import numpy as np

        for k, v in data.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                self._experiment.log_metric(k, v, step=step)
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, np.floating, np.integer)) for x in v):
                # Log list of numbers as metrics
                for i, val in enumerate(v):
                    self._experiment.log_metric(f"{k}_{i}", val, step=step)

    def finish(self):
        self._experiment.end()


class _TensorboardAdapter:
    def __init__(self):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "comet_ml" in loggers:
            self.log_generations_to_comet_ml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

    def log_generations_to_wandb(self, samples, step):
        """Log samples to wandb as a table"""
        import wandb

        # Check if samples have ground truth and correctness (5 elements) or just basic info (3 elements)
        has_gt = len(samples) > 0 and len(samples[0]) >= 5
        
        if has_gt:
            # Create table with 4 columns: Task, LLM Answer, Ground Truth, Correct
            columns = ["Task", "LLM Answer", "Ground Truth", "Correct"]
            table_data = []
            for sample in samples:
                input_text, output_text, score, ground_truth, correct = sample[:5]
                table_data.append([input_text, output_text, str(ground_truth), "✓" if correct else "✗"])
            
            new_table = wandb.Table(columns=columns, data=table_data)
            wandb.log({"val/generations": new_table}, step=step)
        else:
            # Fallback to old format for backward compatibility
            columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

            if not hasattr(self, "validation_table"):
                # Initialize the table on first call
                self.validation_table = wandb.Table(columns=columns)

            # Create a new table with same columns and existing data
            # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
            new_table = wandb.Table(columns=columns, data=self.validation_table.data)

            # Add new row with all data
            row_data = []
            row_data.append(step)
            for sample in samples:
                row_data.extend(sample[:3])  # Only take first 3 elements for backward compatibility

            new_table.add_data(*row_data)

            # Update reference and log
            wandb.log({"val/generations": new_table}, step=step)
            self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        # Check if samples have ground truth and correctness (5 elements) or just basic info (3 elements)
        has_gt = len(samples) > 0 and len(samples[0]) >= 5

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            if has_gt:
                input_text, output_text, score, ground_truth, correct = sample[:5]
                row_text = f"""
                Task: {input_text}
                
                ---
                
                LLM Answer: {output_text}
                
                ---
                
                Ground Truth: {ground_truth}
                
                ---
                
                Correct: {"✓" if correct else "✗"}
                
                ---
                
                Score: {score}
                """
            else:
                row_text = f"""
                input: {sample[0]}
                
                ---
                
                output: {sample[1]}
                
                ---
                
                score: {sample[2]}
                """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        # Check if samples have ground truth and correctness (5 elements) or just basic info (3 elements)
        has_gt = len(samples) > 0 and len(samples[0]) >= 5

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    if has_gt:
                        data = {
                            "Task": sample[0],
                            "LLM Answer": sample[1],
                            "Ground Truth": str(sample[3]),
                            "Correct": "✓" if sample[4] else "✗",
                            "Score": sample[2],
                        }
                    else:
                        data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file, indent=2)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        # Check if samples have ground truth and correctness (5 elements) or just basic info (3 elements)
        has_gt = len(samples) > 0 and len(samples[0]) >= 5
        
        if has_gt:
            table = [
                {
                    "Task": sample[0],
                    "LLM Answer": sample[1],
                    "Ground Truth": str(sample[3]),
                    "Correct": "✓" if sample[4] else "✗",
                    "Score": sample[2],
                }
                for sample in samples
            ]
        else:
            table = [
                {
                    "step": step,
                    "input": sample[0],
                    "output": sample[1],
                    "score": sample[2],
                }
                for sample in samples
            ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_comet_ml(self, samples, step):
        """Log validation generation to comet_ml as table"""
        import comet_ml
        import pandas as pd
        import json
        import tempfile
        import os

        # Get current experiment (should be initialized by CometMLLogger)
        experiment = comet_ml.get_global_experiment()
        if experiment is None:
            print("WARNING: Comet ML experiment not found. Table will not be logged.")
            return

        try:
            # Check if samples have ground truth and correctness (5 elements) or just basic info (3 elements)
            has_gt = len(samples) > 0 and len(samples[0]) >= 5
            
            # Helper function to truncate and clean text (shorter limit for Comet ML)
            def clean_text(text, max_length=2000):
                """Truncate text and ensure it's a valid string"""
                if text is None:
                    return ""
                text_str = str(text)
                # Remove or replace problematic characters
                text_str = text_str.replace('\x00', '')  # Remove null bytes
                text_str = text_str.replace('\r', ' ')  # Replace carriage returns
                # Truncate if too long
                if len(text_str) > max_length:
                    text_str = text_str[:max_length] + "... [truncated]"
                return text_str
            
            if has_gt:
                table_data = []
                for sample in samples:
                    try:
                        task = clean_text(sample[0], max_length=2000)
                        llm_answer = clean_text(sample[1], max_length=2000)
                        ground_truth = clean_text(sample[3], max_length=2000)
                        # Use simple True/False instead of special characters
                        correct = "Yes" if sample[4] else "No"
                        score = float(sample[2]) if isinstance(sample[2], (int, float)) else 0.0
                        
                        table_data.append({
                            "Task": task,
                            "LLM Answer": llm_answer,
                            "Ground Truth": ground_truth,
                            "Correct": correct,
                            "Score": score,
                        })
                    except Exception as e:
                        print(f"WARNING: Error processing sample: {e}")
                        continue
            else:
                table_data = []
                for sample in samples:
                    try:
                        table_data.append({
                            "step": int(step),
                            "input": clean_text(sample[0], max_length=2000),
                            "output": clean_text(sample[1], max_length=2000),
                            "score": float(sample[2]) if isinstance(sample[2], (int, float)) else 0.0,
                        })
                    except Exception as e:
                        print(f"WARNING: Error processing sample: {e}")
                        continue

            if not table_data:
                print("WARNING: No valid samples to log to Comet ML")
                return

            # Try multiple methods to log to Comet ML
            success = False
            
            # Method 1: Try log_table with DataFrame (original method)
            try:
                df = pd.DataFrame.from_records(table_data)
                # Ensure all columns are of appropriate types
                for col in df.columns:
                    if col in ["Score", "score", "step"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    else:
                        df[col] = df[col].astype(str)
                
                # Try logging as table
                experiment.log_table(f"val/generations_step_{step}", tabular_data=df, step=step)
                success = True
                print(f"Successfully logged validation table with {len(table_data)} samples to Comet ML at step {step}")
            except Exception as e1:
                print(f"WARNING: log_table failed: {e1}, trying alternative method...")
                
                # Method 2: Log as CSV file (most reliable format)
                try:
                    df = pd.DataFrame.from_records(table_data)
                    # Ensure all columns are of appropriate types
                    for col in df.columns:
                        if col in ["Score", "score", "step"]:
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                        else:
                            df[col] = df[col].astype(str)
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                        df.to_csv(f, index=False, encoding='utf-8')
                        temp_path = f.name
                    
                    # Log as asset (CSV format)
                    experiment.log_asset(temp_path, file_name=f"val_generations_step_{step}.csv", step=step)
                    os.unlink(temp_path)  # Clean up temp file
                    success = True
                    print(f"Successfully logged validation table as CSV with {len(table_data)} samples to Comet ML at step {step}")
                except Exception as e2:
                    print(f"WARNING: CSV logging failed: {e2}, trying JSON...")
                    # Method 3: Log as JSON file (fallback)
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                            json.dump(table_data, f, indent=2, ensure_ascii=False)
                            temp_path = f.name
                        
                        # Log as asset
                        experiment.log_asset(temp_path, file_name=f"val_generations_step_{step}.json", step=step)
                        os.unlink(temp_path)  # Clean up temp file
                        success = True
                        print(f"Successfully logged validation table as JSON with {len(table_data)} samples to Comet ML at step {step}")
                    except Exception as e3:
                        print(f"ERROR: All file methods failed. log_table: {e1}, CSV: {e2}, JSON: {e3}")
                        import traceback
                        traceback.print_exc()
            
            if not success:
                # Method 4: Log as text summary (final fallback)
                try:
                    text_summary = f"Validation Results at Step {step}\n\n"
                    for i, row in enumerate(table_data[:5]):  # Only first 5 for text
                        text_summary += f"Sample {i+1}:\n"
                        for key, value in row.items():
                            text_summary += f"  {key}: {str(value)[:200]}\n"
                        text_summary += "\n"
                    experiment.log_text(text_summary, step=step)
                    print(f"Logged validation summary as text to Comet ML at step {step}")
                except Exception as e4:
                    print(f"ERROR: All logging methods failed. Last error: {e4}")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"ERROR: Failed to log validation table to Comet ML: {e}")
            import traceback
            traceback.print_exc()

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Check if samples have ground truth and correctness (5 elements) or just basic info (3 elements)
        has_gt = len(samples) > 0 and len(samples[0]) >= 5

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Check if sample has ground truth and correctness
            if has_gt and len(sample) >= 5:
                input_text, output_text, score, ground_truth, correct = sample[:5]
                text_content += f"**Task:** {input_text}\n\n"
                text_content += f"**LLM Answer:** {output_text}\n\n"
                text_content += f"**Ground Truth:** {ground_truth}\n\n"
                text_content += f"**Correct:** {'✓' if correct else '✗'}\n\n"
                text_content += f"**Score:** {score}\n\n"
            elif len(sample) >= 3:
                # Fallback to old format
                input_text, output_text, score = sample[0], sample[1], sample[2]
                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
