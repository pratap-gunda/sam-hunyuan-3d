from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from api_client import DOWNLOAD_FILES, ApiClient

logger = logging.getLogger(__name__)


class PointPromptCanvas(QWidget):
    prompts_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()
        self.preview_mask = QImage()
        self.points: list[list[int]] = []
        self.labels: list[int] = []
        self.setMinimumSize(520, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def load_image(self, path: str | Path) -> None:
        loaded = QImage(str(path))
        if loaded.isNull():
            raise ValueError(f"Could not load image: {path}")
        self.image = loaded.convertToFormat(QImage.Format.Format_RGBA8888)
        self.preview_mask = QImage()
        self.points.clear()
        self.labels.clear()
        self.prompts_changed.emit()
        self.update()

    def load_preview_mask(self, path: str | Path) -> None:
        loaded = QImage(str(path))
        if loaded.isNull():
            raise ValueError(f"Could not load preview mask: {path}")
        if not self.image.isNull() and loaded.size() != self.image.size():
            loaded = loaded.scaled(
                self.image.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.preview_mask = loaded.convertToFormat(QImage.Format.Format_Grayscale8)
        self.update()

    def clear_points(self) -> None:
        self.points.clear()
        self.labels.clear()
        self.preview_mask = QImage()
        self.prompts_changed.emit()
        self.update()

    def remove_last_point(self) -> None:
        if self.points:
            self.points.pop()
            self.labels.pop()
            self.preview_mask = QImage()
            self.prompts_changed.emit()
            self.update()

    def prompt_payload(self) -> tuple[list[list[int]], list[int]]:
        return [point[:] for point in self.points], self.labels[:]

    def has_points(self) -> bool:
        return bool(self.points)

    def sizeHint(self) -> QSize:
        return QSize(760, 460)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 30, 34))

        if self.image.isNull():
            painter.setPen(QColor(180, 185, 195))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load an image, then left-click foreground and right-click background",
            )
            return

        target = self._image_rect()
        painter.drawImage(target, self.image)

        if not self.preview_mask.isNull():
            overlay = QImage(self.preview_mask.size(), QImage.Format.Format_ARGB32)
            overlay.fill(QColor(40, 170, 255, 115))
            overlay.setAlphaChannel(self.preview_mask)
            painter.drawImage(target, overlay)

        painter.setPen(QColor(75, 78, 86))
        painter.drawRect(target.adjusted(0, 0, -1, -1))
        self._draw_points(painter, target)

    def mousePressEvent(self, event) -> None:
        if self.image.isNull():
            return
        point = self._widget_to_image(event.position().toPoint())
        if point is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._add_point(point, 1)
        elif event.button() == Qt.MouseButton.RightButton:
            self._add_point(point, 0)

    def _add_point(self, point: QPoint, label: int) -> None:
        self.points.append([point.x(), point.y()])
        self.labels.append(label)
        self.preview_mask = QImage()
        self.prompts_changed.emit()
        self.update()

    def _draw_points(self, painter: QPainter, target: QRect) -> None:
        for index, point in enumerate(self.points):
            widget_point = self._image_to_widget(QPoint(point[0], point[1]), target)
            color = QColor(25, 200, 75) if self.labels[index] == 1 else QColor(235, 55, 55)
            painter.setPen(QPen(QColor(0, 0, 0), 3))
            painter.setBrush(color)
            painter.drawEllipse(widget_point, 8, 8)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(widget_point + QPoint(11, -9), str(index + 1))

    def _image_rect(self) -> QRect:
        if self.image.isNull():
            return QRect()
        scaled = self.image.size()
        scaled.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(QPoint(x, y), scaled)

    def _widget_to_image(self, point: QPoint) -> QPoint | None:
        target = self._image_rect()
        if not target.contains(point):
            return None
        x = int((point.x() - target.x()) * self.image.width() / target.width())
        y = int((point.y() - target.y()) * self.image.height() / target.height())
        return QPoint(
            max(0, min(self.image.width() - 1, x)),
            max(0, min(self.image.height() - 1, y)),
        )

    def _image_to_widget(self, point: QPoint, target: QRect) -> QPoint:
        x = target.x() + int(point.x() * target.width() / self.image.width())
        y = target.y() + int(point.y() * target.height() / self.image.height())
        return QPoint(x, y)


class PreviewWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, client: ApiClient, job_id: str, destination: Path) -> None:
        super().__init__()
        self.client = client
        self.job_id = job_id
        self.destination = destination

    def run(self) -> None:
        try:
            path = self.client.preview_mask(self.job_id, self.destination)
            self.finished_ok.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))


class ViewPreviewWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, client: ApiClient, job_id: str, destination_dir: Path) -> None:
        super().__init__()
        self.client = client
        self.job_id = job_id
        self.destination_dir = destination_dir

    def run(self) -> None:
        try:
            paths = self.client.preview_views(self.job_id, self.destination_dir)
            self.finished_ok.emit({name: str(path) for name, path in paths.items()})
        except Exception as exc:
            self.failed.emit(str(exc))


class DownloadWorker(QThread):
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, client: ApiClient, job_id: str, destination_dir: Path) -> None:
        super().__init__()
        self.client = client
        self.job_id = job_id
        self.destination_dir = destination_dir

    def run(self) -> None:
        try:
            paths = [
                str(self.client.download(self.job_id, filename, self.destination_dir))
                for filename in DOWNLOAD_FILES
            ]
            self.finished_ok.emit(paths)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SAM2 + Hunyuan3D Client")
        self.resize(940, 680)

        self.job_id: str | None = None
        self.input_image_path: str | None = None
        self.active_operation: str | None = None
        self.points_dirty = False
        self.preview_worker: PreviewWorker | None = None
        self.view_preview_worker: ViewPreviewWorker | None = None
        self.download_worker: DownloadWorker | None = None

        self.server_url = QLineEdit("http://127.0.0.1:8000")
        self.input_image = QLineEdit()
        self.input_image.setReadOnly(True)

        self.preview_button = QPushButton("Generate Mask Preview")
        self.generate_views_button = QPushButton("Generate Views")
        self.generate_button = QPushButton("Generate Mesh")
        self.clear_points_button = QPushButton("Clear Points")
        self.remove_last_button = QPushButton("Remove Last Point")
        self.download_button = QPushButton("Download")
        self.preview_button.setEnabled(False)
        self.generate_views_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.download_button.setEnabled(False)

        self.canvas = PointPromptCanvas()
        self.front_view = self._view_label("Front View")
        self.side_view = self._view_label("Side View")
        self.back_view = self._view_label("Back View")
        self.view_panel = self._build_view_panel()
        self.viewer_stack = QStackedWidget()
        self.viewer_stack.addWidget(self.canvas)
        self.viewer_stack.addWidget(self.view_panel)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Idle")
        self.job_label = QLabel("No job")
        self.points_label = QLabel("Points: 0")
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.poll_status)

        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        grid = QGridLayout()

        grid.addWidget(QLabel("Server URL"), 0, 0)
        grid.addWidget(self.server_url, 0, 1, 1, 2)

        input_browse = QPushButton("Browse")
        input_browse.clicked.connect(self.select_input_image)
        grid.addWidget(QLabel("Input Image"), 1, 0)
        grid.addWidget(self.input_image, 1, 1)
        grid.addWidget(input_browse, 1, 2)

        buttons = QHBoxLayout()
        buttons.addWidget(self.preview_button)
        buttons.addWidget(self.generate_views_button)
        buttons.addWidget(self.generate_button)
        buttons.addWidget(self.clear_points_button)
        buttons.addWidget(self.remove_last_button)
        buttons.addWidget(self.download_button)
        buttons.addStretch(1)

        root.addLayout(grid)
        root.addWidget(self.viewer_stack, 1)
        root.addWidget(self.points_label)
        root.addLayout(buttons)
        root.addWidget(self.progress)
        root.addWidget(self.status_label)
        root.addWidget(self.job_label)
        root.addWidget(QLabel("Log Output"))
        root.addWidget(self.log_output, 1)
        self.setCentralWidget(central)

    def _connect(self) -> None:
        self.preview_button.clicked.connect(self.preview_mask)
        self.generate_views_button.clicked.connect(self.generate_views)
        self.generate_button.clicked.connect(self.generate_mesh)
        self.clear_points_button.clicked.connect(self.clear_points)
        self.remove_last_button.clicked.connect(self.remove_last_point)
        self.download_button.clicked.connect(self.download)
        self.canvas.prompts_changed.connect(self._sync_prompt_buttons)
        self.canvas.prompts_changed.connect(self._mark_points_dirty)

    def _view_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(220, 220)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setStyleSheet("QLabel { border: 1px solid #555; background: #202226; color: #cfd3dc; }")
        return label

    def _build_view_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.front_view)
        layout.addWidget(self.side_view)
        layout.addWidget(self.back_view)
        return panel

    def client(self) -> ApiClient:
        return ApiClient(self.server_url.text().strip())

    def select_input_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not path:
            return
        try:
            self.input_image_path = path
            self.input_image.setText(path)
            self.canvas.load_image(path)
            self.job_id = None
            self.points_dirty = False
            self.active_operation = None
            self.progress.setValue(0)
            self.job_label.setText("No job")
            self.status_label.setText("Add foreground and background points")
            self.download_button.setEnabled(False)
            self._clear_view_previews()
            self._show_canvas()
            self._sync_prompt_buttons()
            self.append_log(f"Loaded image: {path}")
        except Exception as exc:
            self.show_error(str(exc))

    def clear_points(self) -> None:
        self.canvas.clear_points()
        self._clear_view_previews()
        self._show_canvas()
        self._sync_prompt_buttons()
        self.append_log("Cleared points")

    def remove_last_point(self) -> None:
        self.canvas.remove_last_point()
        self._clear_view_previews()
        self._show_canvas()
        self._sync_prompt_buttons()
        self.append_log("Removed last point")

    def preview_mask(self) -> None:
        if not self._ensure_ready_for_prompts():
            return
        try:
            job_id = self._upload_or_update_points()
            preview_path = Path(__file__).resolve().parents[1] / "downloads" / job_id / "refined_mask.png"
            self.preview_button.setEnabled(False)
            self.status_label.setText("Running SAM2 preview")
            self.append_log("Running mask preview")
            self.preview_worker = PreviewWorker(self.client(), job_id, preview_path)
            self.preview_worker.finished_ok.connect(self.preview_finished)
            self.preview_worker.failed.connect(self.preview_failed)
            self.preview_worker.start()
        except Exception as exc:
            logger.exception("Preview failed")
            self.show_error(f"Preview failed: {exc}")

    def generate_views(self) -> None:
        if not self._ensure_ready_for_prompts():
            return
        try:
            job_id = self._upload_or_update_points()
            response = self.client().generate_views(job_id)
            self.active_operation = "views"
            self.append_log(response.get("message", "View generation started"))
            self.status_label.setText("View generation started")
            self.preview_button.setEnabled(False)
            self.generate_views_button.setEnabled(False)
            self.generate_button.setEnabled(False)
            self.download_button.setEnabled(False)
            self.timer.start()
        except Exception as exc:
            logger.exception("Generate views failed")
            self.show_error(f"Generate views failed: {exc}")

    def preview_finished(self, path: str) -> None:
        try:
            self.canvas.load_preview_mask(path)
            self._show_canvas()
            self.status_label.setText("Mask preview ready")
            self.append_log(f"Preview downloaded: {path}")
        except Exception as exc:
            self.show_error(str(exc))
        finally:
            self._sync_prompt_buttons()

    def preview_failed(self, message: str) -> None:
        logger.error("Preview failed: %s", message)
        self.show_error(f"Preview failed: {message}")
        self._sync_prompt_buttons()

    def generate_mesh(self) -> None:
        if not self._ensure_ready_for_prompts():
            return
        try:
            job_id = self._upload_or_update_points()
            response = self.client().generate(job_id)
            self.active_operation = "mesh"
            self.append_log(response.get("message", "Mesh generation started"))
            self.status_label.setText("Mesh generation started")
            self.generate_button.setEnabled(False)
            self.preview_button.setEnabled(False)
            self.download_button.setEnabled(False)
            self.timer.start()
        except Exception as exc:
            logger.exception("Generate failed")
            self.show_error(f"Generate failed: {exc}")

    def _upload_or_update_points(self) -> str:
        if not self.input_image_path:
            raise ValueError("Choose an input image first")
        points, labels = self.canvas.prompt_payload()
        client = self.client()
        if self.job_id is None:
            self.job_id = client.upload(self.input_image_path, points, labels)
            self.job_label.setText(f"Job ID: {self.job_id}")
            self.points_dirty = False
            self.append_log(f"Uploaded image and points: {self.job_id}")
        elif self.points_dirty:
            client.update_points(self.job_id, points, labels)
            self.points_dirty = False
            self._clear_view_previews()
            self.append_log(f"Updated points for job: {self.job_id}")
        return self.job_id

    def _ensure_ready_for_prompts(self) -> bool:
        if not self.input_image_path:
            self.show_error("Choose an input image first.")
            return False
        points, labels = self.canvas.prompt_payload()
        if not points:
            self.show_error("Add at least one foreground point with left click.")
            return False
        if 1 not in labels:
            self.show_error("Add at least one foreground point with left click.")
            return False
        return True

    def poll_status(self) -> None:
        if not self.job_id:
            return
        try:
            payload = self.client().status(self.job_id)
            progress = int(payload.get("progress", 0))
            status = str(payload.get("status", "unknown"))
            message = str(payload.get("message", ""))
            self.progress.setValue(progress)
            self.status_label.setText(f"{status}: {message}")
            self.append_log(f"{progress}% - {message}")

            if self.active_operation == "views" and status == "queued" and progress >= 80:
                self.timer.stop()
                self.load_view_previews()
            elif status == "completed":
                self.timer.stop()
                self.active_operation = None
                self.download_button.setEnabled(True)
                self._sync_prompt_buttons()
                self.append_log("Generation complete")
            elif status == "failed":
                self.timer.stop()
                self.active_operation = None
                self._sync_prompt_buttons()
                self.show_error(f"Generation failed: {message}")
        except Exception as exc:
            logger.exception("Status polling failed")
            self.timer.stop()
            self.active_operation = None
            self._sync_prompt_buttons()
            self.show_error(f"Status polling failed: {exc}")

    def load_view_previews(self) -> None:
        if not self.job_id:
            return
        destination = Path(__file__).resolve().parents[1] / "downloads" / self.job_id
        self.status_label.setText("Loading view previews")
        self.view_preview_worker = ViewPreviewWorker(self.client(), self.job_id, destination)
        self.view_preview_worker.finished_ok.connect(self.view_previews_finished)
        self.view_preview_worker.failed.connect(self.view_previews_failed)
        self.view_preview_worker.start()

    def view_previews_finished(self, paths: dict) -> None:
        self._set_view_pixmap(self.front_view, paths.get("front"))
        self._set_view_pixmap(self.side_view, paths.get("side"))
        self._set_view_pixmap(self.back_view, paths.get("back"))
        self._show_views()
        self.active_operation = None
        self.status_label.setText("View previews ready")
        self.append_log("Front, side, and back views are ready")
        self._sync_prompt_buttons()

    def view_previews_failed(self, message: str) -> None:
        self.active_operation = None
        self.show_error(f"View preview failed: {message}")
        self._sync_prompt_buttons()

    def download(self) -> None:
        if not self.job_id:
            self.show_error("No completed job is selected.")
            return
        destination = Path(__file__).resolve().parents[1] / "downloads" / self.job_id
        self.append_log(f"Downloading results to {destination}")
        self.download_button.setEnabled(False)
        self.download_worker = DownloadWorker(self.client(), self.job_id, destination)
        self.download_worker.finished_ok.connect(self.download_finished)
        self.download_worker.failed.connect(self.download_failed)
        self.download_worker.start()

    def download_finished(self, paths: list[str]) -> None:
        self.download_button.setEnabled(True)
        for path in paths:
            self.append_log(f"Downloaded {path}")
        self.status_label.setText("Downloads complete")

    def download_failed(self, message: str) -> None:
        self.download_button.setEnabled(True)
        logger.error("Download failed: %s", message)
        self.show_error(f"Download failed: {message}")

    def _sync_prompt_buttons(self) -> None:
        enabled = self.active_operation is None and self.input_image_path is not None and self.canvas.has_points()
        self.preview_button.setEnabled(enabled)
        self.generate_views_button.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)
        self.points_label.setText(f"Points: {len(self.canvas.points)}")

    def _mark_points_dirty(self) -> None:
        self.points_dirty = self.job_id is not None
        self._show_canvas()

    def _clear_view_previews(self) -> None:
        self.front_view.setPixmap(QPixmap())
        self.side_view.setPixmap(QPixmap())
        self.back_view.setPixmap(QPixmap())
        self.front_view.setText("Front View")
        self.side_view.setText("Side View")
        self.back_view.setText("Back View")

    def _show_canvas(self) -> None:
        self.viewer_stack.setCurrentWidget(self.canvas)

    def _show_views(self) -> None:
        self.viewer_stack.setCurrentWidget(self.view_panel)

    def _set_view_pixmap(self, label: QLabel, path: str | None) -> None:
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            label.setText("Preview unavailable")
            return
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        logger.info(message)

    def show_error(self, message: str) -> None:
        self.append_log(message)
        QMessageBox.warning(self, "Error", message)
