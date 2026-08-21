from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox,
)
from PyQt6.QtCore import Qt


class EmployeesDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Employees")
        self.resize(680, 480)
        self._db = db
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        root = QHBoxLayout(self)

        # Left — employee list
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selected)
        root.addWidget(self._list, 1)

        # Right — details + actions
        panel = QGroupBox("Employee")
        panel_layout = QVBoxLayout(panel)

        self._lbl_count = QLabel("Select an employee")
        panel_layout.addWidget(self._lbl_count)

        panel_layout.addWidget(QLabel("Add new employee:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Full name…")
        panel_layout.addWidget(self._name_input)

        btn_add = QPushButton("Add Employee")
        btn_add.clicked.connect(self._add_employee)
        panel_layout.addWidget(btn_add)

        panel_layout.addSpacing(16)

        self._btn_deactivate = QPushButton("Deactivate Selected")
        self._btn_deactivate.setEnabled(False)
        self._btn_deactivate.clicked.connect(self._deactivate_employee)
        panel_layout.addWidget(self._btn_deactivate)

        panel_layout.addStretch()
        root.addWidget(panel, 1)

    def _refresh_list(self):
        self._list.clear()
        for emp in self._db.list_employees():
            count = self._db.count_embeddings_for(emp["employee_id"])
            item = QListWidgetItem(f"{emp['full_name']}  ({count} face sample{'s' if count != 1 else ''})")
            item.setData(Qt.ItemDataRole.UserRole, emp)
            self._list.addItem(item)

    def _on_selected(self, current, _prev):
        if current is None:
            self._lbl_count.setText("Select an employee")
            self._btn_deactivate.setEnabled(False)
            return
        emp = current.data(Qt.ItemDataRole.UserRole)
        count = self._db.count_embeddings_for(emp["employee_id"])
        self._lbl_count.setText(
            f"Name: {emp['full_name']}\n"
            f"ID: {emp['employee_id']}\n"
            f"Face samples: {count}\n"
            f"Added: {emp['created_at'][:10]}"
        )
        self._btn_deactivate.setEnabled(True)

    def _add_employee(self):
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a name first.")
            return
        self._db.add_employee(name)
        self._name_input.clear()
        self._refresh_list()
        QMessageBox.information(self, "Done", f"Employee '{name}' added.")

    def _deactivate_employee(self):
        item = self._list.currentItem()
        if item is None:
            return
        emp = item.data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Confirm",
            f"Deactivate '{emp['full_name']}'?\n"
            "They will no longer appear in the system but their history is preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._db.deactivate_employee(emp["employee_id"])
            self._refresh_list()
