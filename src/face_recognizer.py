import logging

import cv2
import faiss
import numpy as np

from src.device_utils import onnxruntime_gpu_available, resolve_device

logger = logging.getLogger("attendance.recognizer")

try:
    from insightface.app import FaceAnalysis
except ImportError as e:
    missing_module = str(e).replace("No module named ", "").strip("'\"")
    if missing_module and missing_module not in ("onnx", "onnxruntime", "onnx_cpp2py_export"):
        # A specific sub-dependency is missing (e.g. matplotlib, scipy...) -
        # this is almost always a PyInstaller build issue (the module got
        # excluded, or wasn't installed in the build venv), NOT the VC++
        # Redistributable issue below. Point at the real cause directly.
        raise ImportError(
            f"\n\nFailed to import InsightFace because '{missing_module}' is missing.\n"
            f"If you're running from source: pip install {missing_module}\n"
            f"If you're running the built .exe: check attendance.spec - make sure "
            f"'{missing_module}' is NOT listed in `excludes=[...]`, and is listed in "
            f"`hiddenimports`, then rebuild with:\n"
            f"  .venv\\Scripts\\pyinstaller attendance.spec --noconfirm --clean\n"
            f"(insightface pulls in matplotlib/scipy/scikit-image as real, load-time "
            f"dependencies even though this project doesn't use their features "
            f"directly - excluding them from the build breaks the import.)\n"
            f"\nOriginal error: {e}"
        ) from e
    raise ImportError(
        "\n\nFailed to import InsightFace/onnx. This is almost always caused "
        "by a missing 'Microsoft Visual C++ Redistributable' on Windows, not "
        "a bug in this project.\n"
        "Fix:\n"
        "  1. Install: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
        "  2. Restart the computer (required for the DLL to load).\n"
        "  3. If it still fails, run:\n"
        "     pip uninstall onnx onnxruntime -y\n"
        "     pip install onnx==1.14.1 onnxruntime==1.16.3 --force-reinstall --no-cache-dir\n"
        "  4. If it still fails, this is often more reliable on Python 3.10/3.11 "
        "than on Python 3.8.\n"
        f"\nOriginal error: {e}"
    ) from e


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class FaceRecognizer:
    def __init__(self, model_name="buffalo_l", detector="scrfd",
                 embedding_dim=512, match_threshold=0.50, review_threshold=0.35,
                 min_face_size_px=20, device="auto", model_root="~/.insightface",
                 face_crop_padding=0.20, face_crop_scale=2.0):
        self.embedding_dim = embedding_dim
        self.match_threshold = match_threshold
        self.review_threshold = review_threshold
        self.min_face_size_px = min_face_size_px
        self.face_crop_padding = max(0.0, float(face_crop_padding))
        self.face_crop_scale = max(1.0, float(face_crop_scale))

        device = resolve_device(device)
        if device == "cuda" and not onnxruntime_gpu_available():
            logger.warning(
                "InsightFace CUDA provider is unavailable; using CPU execution explicitly."
            )
            device = "cpu"
        ctx_id = 0 if device == "cuda" else -1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        self._index = faiss.IndexFlatIP(embedding_dim)
        self._index_employee_ids = [] 

        logger.info(f"Face recognition model loaded: {model_name} on {device}")

    def rebuild_index(self, embeddings_with_ids):
        """
        embeddings_with_ids: [(employee_id, embedding_bytes), ...]
        """
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index_employee_ids = []
        if not embeddings_with_ids:
            logger.warning("Face database is empty - the system will register all cases as 'needs review/naming'.")
            return

        vectors = []
        for emp_id, blob in embeddings_with_ids:
            vec = bytes_to_embedding(blob)
            vectors.append(vec)
            self._index_employee_ids.append(emp_id)

        matrix = np.vstack(vectors).astype(np.float32)
        faiss.normalize_L2(matrix)
        self._index.add(matrix)
        logger.info(f"Face recognition index built with {len(vectors)} reference images for "
                    f"{len(set(self._index_employee_ids))} employees.")

    def extract_best_face(self, frame, bbox=None):
        """
        Detects all faces in the frame (or within the person's bbox if specified), 
        and returns the best face (largest and clearest) with its embedding. Returns None if no valid face is found.
        """
        region = frame
        offset_x, offset_y = 0, 0
        scale = 1.0
        if bbox is not None:    
            x1, y1, x2, y2 = bbox
            box_width, box_height = x2 - x1, y2 - y1
            padding_x = int(box_width * self.face_crop_padding)
            padding_y = int(box_height * self.face_crop_padding)
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(frame.shape[1], x2 + padding_x)
            crop_y2 = min(frame.shape[0], y2 + padding_y)
            region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            offset_x, offset_y = crop_x1, crop_y1
            if region.size == 0:
                return None

            # Person boxes can be large while the visible face is only a few
            # dozen pixels. Upscaling gives SCRFD more pixels to work with.
            if self.face_crop_scale > 1.0:
                scale = self.face_crop_scale
                region = np.ascontiguousarray(
                    cv2.resize(
                        region, None, fx=self.face_crop_scale, fy=self.face_crop_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                )

        faces = self.app.get(region)
        if not faces and bbox is not None:
            # A tracker box can clip the head during a crossing. A full-frame
            # fallback is safer than giving up, but only faces centered inside
            # the tracked person box are allowed through.
            full_frame_faces = self.app.get(frame)
            x1, y1, x2, y2 = bbox
            faces = [
                face for face in full_frame_faces
                if x1 <= (face.bbox[0] + face.bbox[2]) / 2 <= x2
                and y1 <= (face.bbox[1] + face.bbox[3]) / 2 <= y2
            ]
            if faces:
                offset_x, offset_y, scale = 0, 0, 1.0

        candidates = self._face_results(faces, offset_x, offset_y, scale)
        if not candidates:
            return None
        # "Best" = largest bounding box area (closest/clearest face), matching
        # what every caller (pipeline.py, archive_pipeline.py, review_dialog.py)
        # expects: a single face dict, not a list.
        return max(
            candidates,
            key=lambda f: max(0, f["bbox"][2] - f["bbox"][0]) * max(0, f["bbox"][3] - f["bbox"][1]),
        )

    def extract_faces(self, frame):
        """Detect every usable face in a full frame and return face results."""
        return self._face_results(self.app.get(frame), 0, 0, 1.0)

    def _face_results(self, faces, offset_x, offset_y, scale):
        if not faces:
            return []

        results = []
        for face in faces:
            width = (face.bbox[2] - face.bbox[0]) / scale
            height = (face.bbox[3] - face.bbox[1]) / scale
            if width < self.min_face_size_px or height < self.min_face_size_px:
                continue
            results.append({
                "embedding": face.normed_embedding.astype(np.float32),
                "bbox": (
                    int(face.bbox[0] / scale) + offset_x,
                    int(face.bbox[1] / scale) + offset_y,
                    int(face.bbox[2] / scale) + offset_x,
                    int(face.bbox[3] / scale) + offset_y,
                ),
                "det_score": float(face.det_score),
            })
        return results


    def match(self, embedding: np.ndarray):
        """
        Compares the embedding with all employee database records and returns:
        (employee_id or None, confidence, status)
        status: "confirmed" if above match_threshold،
                "needs_review" if between review_threshold and match_threshold،
                "rejected" if below review_threshold (will not be recorded even in review).
        """
        if self._index.ntotal == 0:
            return None, 0.0, "needs_review"

        query = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        similarities, indices = self._index.search(query, k=1)
        best_sim = float(similarities[0][0])
        best_idx = int(indices[0][0])

        if best_sim >= self.match_threshold:
            return self._index_employee_ids[best_idx], best_sim, "confirmed"
        elif best_sim >= self.review_threshold:
            return self._index_employee_ids[best_idx], best_sim, "needs_review"
        else:
            return None, best_sim, "needs_review"