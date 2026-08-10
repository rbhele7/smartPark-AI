import os
import json
import time
import asyncio
from typing import List, Tuple, Dict, Any
import numpy as np
import tensorflow as tf

from app.config import MODEL_PATH, FALLBACK_MODEL_PATH, CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, IMG_SIZE


class ModelManager:
    """Singleton Model Manager for loading and executing Keras CNN model asynchronously."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.class_names = ["occupied", "vacant"]
            cls._instance.is_loaded = False
            cls._instance.model_name = "SmartParkingCNN"
            cls._instance.loaded_path = ""
            cls._instance.total_inferences = 0
            cls._instance.total_inference_time_ms = 0.0
            # Auto load model upon instantiation
            cls._instance.load()
        return cls._instance

    def load(self) -> bool:
        """Load the model file and run a warm-up inference."""
        if self.is_loaded and self.model is not None:
            return True

        # Load class names if present
        if os.path.exists(CLASS_NAMES_PATH):
            try:
                with open(CLASS_NAMES_PATH, "r") as f:
                    self.class_names = json.load(f)
            except Exception as e:
                print(f"[ModelManager] Warning loading class names: {e}")

        model_to_load = None
        if os.path.exists(MODEL_PATH):
            model_to_load = MODEL_PATH
        elif os.path.exists(FALLBACK_MODEL_PATH):
            model_to_load = FALLBACK_MODEL_PATH

        if model_to_load:
            try:
                print(f"[ModelManager] Loading model from {model_to_load}...")
                self.model = tf.keras.models.load_model(model_to_load)
                self.loaded_path = model_to_load
                self.is_loaded = True
                print("[ModelManager] Keras model successfully loaded.")
            except Exception as e:
                print(f"[ModelManager] Error loading model file {model_to_load}: {e}")

        if not self.is_loaded or self.model is None:
            # Build lightweight fallback dummy model if model weights are missing/incompatible
            print("[ModelManager] Creating lightweight fallback CNN model...")
            inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
            x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.loaded_path = "in_memory_fallback_cnn"
            self.is_loaded = True

        # Warm-up model inference to optimize execution latency
        self._warmup()
        return self.is_loaded

    def _warmup(self):
        """Perform warm-up prediction on dummy input tensor to compile TF graph/kernels."""
        try:
            start_t = time.perf_counter()
            dummy_batch = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            _ = self.model.predict(dummy_batch, verbose=0)
            elapsed_ms = (time.perf_counter() - start_t) * 1000
            print(f"[ModelManager] Model warm-up complete in {elapsed_ms:.2f} ms.")
        except Exception as e:
            print(f"[ModelManager] Warm-up failed: {e}")

    def _predict_sync_batch(self, batch_tensor: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """Synchronous internal batch inference."""
        if not self.is_loaded or self.model is None:
            self.load()

        start_t = time.perf_counter()
        raw_preds = self.model.predict(batch_tensor, verbose=0)
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        self.total_inferences += len(batch_tensor)
        self.total_inference_time_ms += elapsed_ms

        results = []
        for raw_val in raw_preds:
            # Handle binary output or 2-class softmax output
            if isinstance(raw_val, (list, np.ndarray)) and len(raw_val) > 1:
                prob = float(raw_val[1])
                class_idx = int(np.argmax(raw_val))
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else "unknown"
                confidence = float(raw_val[class_idx])
            else:
                prob = float(raw_val[0]) if isinstance(raw_val, (list, np.ndarray)) else float(raw_val)
                # Binary sigmoid: >= 0.5 -> vacant (index 1), < 0.5 -> occupied (index 0)
                if prob >= CONFIDENCE_THRESHOLD:
                    class_name = self.class_names[1] if len(self.class_names) > 1 else "vacant"
                    confidence = prob
                else:
                    class_name = self.class_names[0] if len(self.class_names) > 0 else "occupied"
                    confidence = 1.0 - prob

            results.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "raw_probability": round(prob, 4)
            })

        return results, elapsed_ms

    async def predict_single(self, image_tensor: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Asynchronously predict for a single image tensor (1, H, W, 3)."""
        results, elapsed_ms = await asyncio.to_thread(self._predict_sync_batch, image_tensor)
        return results[0], elapsed_ms

    async def predict_batch(self, batch_tensor: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """Asynchronously predict for a batch image tensor (N, H, W, 3)."""
        return await asyncio.to_thread(self._predict_sync_batch, batch_tensor)


# Global singleton instance
model_manager = ModelManager()
