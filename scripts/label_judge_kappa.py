"""Local, sequential annotation of the blind Issue #14 sheet."""

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from validate_judge_kappa_sheet import ANNOTATION_COLUMNS, SHEET_DIR, read_sheet, validate

SCORES = ANNOTATION_COLUMNS[:-1]
UNASSESSABLE = "не могу оценить"
QUESTIONS = {
    "faithful": "Все факты в объяснении подтверждаются источниками?",
    "citations_ok": "Первая цитата — находка, остальные подтверждают вывод?",
    "recommendation_ok": "Правка адресна, исправляет ошибку и сохраняет другие случаи?",
    "verdict_ok": "Верны и вердикт, и категория?",
}


class Sheet:
    def __init__(self, source: Path, path: Path):
        self.source = source
        self.path = path
        validate(source, path, partial=True)
        self.header, self.rows = read_sheet(path)
        self.columns = {name: index for index, name in enumerate(self.header)}

    @property
    def completed(self) -> int:
        return sum(any(row[self.columns[name]] for name in ANNOTATION_COLUMNS) for row in self.rows)

    def record(self, index: int) -> dict[str, str]:
        return dict(zip(self.header, self.rows[index], strict=True))

    def confirm(self, index: int, annotations: dict[str, str]) -> None:
        if index != self.completed or index >= len(self.rows):
            raise ValueError("Можно подтвердить только текущую строку.")
        if annotations.keys() != set(ANNOTATION_COLUMNS) or not any(annotations.values()):
            raise ValueError("Заполните оценки или отметьте, что строку невозможно оценить.")
        self._save(index, annotations)

    def correct_comment(self, index: int, comment: str) -> None:
        if not 0 <= index < self.completed:
            raise ValueError("Укажите ID уже подтверждённой строки.")
        if not any(self.record(index)[name] for name in SCORES):
            raise ValueError("Статус «не могу оценить» зафиксирован.")
        self._save(index, {"comment": comment})

    def _save(self, index: int, changes: dict[str, str]) -> None:
        rows = self.rows.copy()
        rows[index] = rows[index].copy()
        for name, value in changes.items():
            rows[index][self.columns[name]] = value

        pending = self.path.with_suffix(".csv.pending")
        # Exclusive creation serializes saves from multiple instances of this app.
        stream = pending.open("x", encoding="utf-8", newline="")
        try:
            with stream:
                csv.writer(stream).writerows([self.header, *rows])
            if read_sheet(self.path) != (self.header, self.rows):
                raise ValueError("CSV изменился. Перезапустите интерфейс перед сохранением.")
            validate(self.source, pending, partial=True)
            pending.replace(self.path)
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
        self.rows = rows


class App:
    def __init__(self, root: tk.Tk, sheet: Sheet):
        self.root = root
        self.sheet = sheet
        root.title("Issue #14 — ручная разметка")
        root.geometry("1100x780")
        root.minsize(780, 620)

        body = ttk.Frame(root, padding=16)
        body.pack(fill="both", expand=True)
        self.progress = ttk.Label(body, font=("Segoe UI", 15, "bold"))
        self.progress.pack(anchor="w", pady=(0, 8))
        self.source_text = ScrolledText(body, wrap="word", height=12, padx=12, pady=8)
        self.source_text.pack(fill="both", expand=True)
        self.source_text.tag_configure("heading", font=("Segoe UI", 11, "bold"))

        self.form = ttk.Frame(body, padding=(0, 8))
        self.form.pack(fill="x")
        self.form.columnconfigure(2, weight=1)
        self.scores = {}
        self.selectors = {}
        self.reasons = {}
        for index, (name, question) in enumerate(QUESTIONS.items()):
            row = index * 2
            ttk.Label(self.form, text=name).grid(row=row, column=0, sticky="w", padx=(0, 10))
            self.scores[name] = tk.StringVar()
            selector = ttk.Combobox(
                self.form,
                textvariable=self.scores[name],
                values=("0", "1"),
                state="readonly",
                width=5,
            )
            selector.grid(row=row, column=1, padx=(0, 10), pady=2)
            selector.bind("<<ComboboxSelected>>", lambda event: self.update_fields())
            self.selectors[name] = selector
            ttk.Label(self.form, text=question).grid(row=row, column=2, sticky="w")
            reason_row = ttk.Frame(self.form)
            reason_row.grid(row=row + 1, column=2, sticky="ew", pady=(0, 2))
            ttk.Label(reason_row, text="Причина оценки 0").pack(side="left", padx=(0, 8))
            reason = tk.StringVar()
            ttk.Entry(reason_row, textvariable=reason).pack(fill="x", expand=True)
            self.reasons[name] = (reason_row, reason)

        self.skip = tk.BooleanVar()
        self.skip_button = ttk.Checkbutton(
            self.form,
            text="Не могу оценить строку",
            variable=self.skip,
            command=self.update_fields,
        )
        self.skip_button.grid(row=2 * len(SCORES), columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(body, text="comment — дополнительный комментарий").pack(anchor="w")
        self.comment = tk.Text(body, height=3, wrap="word", padx=6, pady=4)
        self.comment.pack(fill="x")

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(10, 0))
        self.save_button = ttk.Button(actions, text="Подтвердить и сохранить", command=self.save)
        self.save_button.pack(side="left")
        self.correct_button = ttk.Button(
            actions,
            text="Исправить опечатку в комментарии",
            command=self.correct,
        )
        self.correct_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Проверить таблицу", command=self.check).pack(side="right")
        self.show_current()

    def show_current(self) -> None:
        self.index = self.sheet.completed
        total = len(self.sheet.rows)
        self.current = self.sheet.record(self.index) if self.index < total else None
        item = self.current["item"] if self.current else "Разметка завершена"
        self.progress.configure(text=f"{item}   ·   Сохранено {self.index}/{total}")
        self.correct_button.configure(state="normal" if self.index else "disabled")
        self.source_text.configure(state="normal")
        self.source_text.delete("1.0", "end")
        if self.current:
            for name in self.sheet.header:
                if name != "item" and name not in ANNOTATION_COLUMNS:
                    self.source_text.insert("end", f"{name.capitalize()}\n", "heading")
                    self.source_text.insert("end", (self.current[name] or "—") + "\n\n")
        else:
            self.source_text.insert("end", "Все строки сохранены. Теперь можно проверить таблицу.")
        self.source_text.configure(state="disabled")
        self.source_text.yview_moveto(0)

        self.skip.set(False)
        for value in self.scores.values():
            value.set("")
        for _, reason in self.reasons.values():
            reason.set("")
        self.comment.configure(state="normal")
        self.comment.delete("1.0", "end")
        self.update_fields()

    def update_fields(self) -> None:
        active = self.current is not None
        skip = self.skip.get()
        for name, selector in self.selectors.items():
            automatic = (
                active and name == "recommendation_ok" and not self.current["рекомендация"].strip()
            )
            if automatic:
                self.scores[name].set("n/a" if not skip else "")
            selector.configure(
                state="readonly" if active and not skip and not automatic else "disabled"
            )
            reason_row, _ = self.reasons[name]
            if active and not skip and self.scores[name].get() == "0":
                reason_row.grid()
            else:
                reason_row.grid_remove()
        self.skip_button.configure(state="normal" if active else "disabled")
        self.comment.configure(state="normal" if active and not skip else "disabled")
        self.save_button.configure(state="normal" if active else "disabled")

    def save(self) -> None:
        if self.skip.get():
            annotations = dict.fromkeys(SCORES, "")
            annotations["comment"] = UNASSESSABLE
        else:
            annotations = {name: value.get() for name, value in self.scores.items()}
            if any(not annotations[name] for name in SCORES):
                messagebox.showerror("Строка не сохранена", "Заполните все четыре оценки.")
                return
            comments = []
            for name in SCORES:
                if annotations[name] == "0":
                    reason = self.reasons[name][1].get().strip()
                    if not reason:
                        messagebox.showerror(
                            "Строка не сохранена", f"Укажите причину оценки 0 для {name}."
                        )
                        return
                    comments.append(f"{name}: {reason}")
            comments.append(self.comment.get("1.0", "end-1c").strip())
            annotations["comment"] = "\n".join(filter(None, comments))
        try:
            self.sheet.confirm(self.index, annotations)
        except (OSError, ValueError, csv.Error) as error:
            messagebox.showerror("Строка не сохранена", str(error))
            return
        self.show_current()

    def correct(self) -> None:
        item = simpledialog.askstring("Исправление опечатки", "ID сохранённой строки, например r01")
        if item is None:
            return
        for index in range(self.sheet.completed):
            record = self.sheet.record(index)
            if record["item"] == item.strip():
                break
        else:
            messagebox.showerror("Строка недоступна", "Укажите ID уже подтверждённой строки.")
            return
        if not any(record[name] for name in SCORES):
            messagebox.showerror("Исправление недоступно", "Статус «не могу оценить» зафиксирован.")
            return
        comment = CommentCorrection(self.root, record["item"], record["comment"]).result
        if comment is None or comment == record["comment"]:
            return
        try:
            self.sheet.correct_comment(index, comment.strip())
        except (OSError, ValueError, csv.Error) as error:
            messagebox.showerror("Исправление не сохранено", str(error))
            return
        messagebox.showinfo("Исправление сохранено", f"Комментарий {record['item']} обновлён.")

    def check(self) -> None:
        try:
            validate(self.sheet.source, self.sheet.path)
        except (OSError, ValueError, csv.Error) as error:
            messagebox.showerror("Проверка таблицы", str(error))
            return
        messagebox.showinfo("Проверка таблицы", "Все 60 строк прошли проверку.")


class CommentCorrection(simpledialog.Dialog):
    def __init__(self, root: tk.Tk, item: str, comment: str):
        self.original = comment
        super().__init__(root, title=f"{item} — исправление опечатки")

    def body(self, master):
        ttk.Label(master, text="Исправьте техническую опечатку, сохранив смысл.").pack(anchor="w")
        self.editor = ScrolledText(master, width=68, height=7, wrap="word")
        self.editor.pack(fill="both", expand=True)
        self.editor.insert("1.0", self.original)
        return self.editor

    def buttonbox(self) -> None:
        buttons = ttk.Frame(self)
        ttk.Button(buttons, text="Сохранить исправление", command=self.ok).pack(side="left", padx=4)
        ttk.Button(buttons, text="Отмена", command=self.cancel).pack(side="left", padx=4)
        buttons.pack(pady=8)
        self.bind("<Escape>", self.cancel)

    def apply(self) -> None:
        self.result = self.editor.get("1.0", "end-1c")


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        sheet = Sheet(SHEET_DIR / "sheet.csv", SHEET_DIR / "sheet-taisia.csv")
    except (OSError, ValueError, csv.Error) as error:
        messagebox.showerror("Не удалось открыть таблицу", str(error), parent=root)
        root.destroy()
        raise SystemExit(1) from error
    App(root, sheet)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
