from typing import Optional, Tuple

from dialog_model.raw_dialogs.dialogs_iterator import DialogsIterator


class PikabuDialogsIterator(DialogsIterator):
    def __init__(self, file_path, min_n_messages_in_dialog=1):
        super().__init__(file_path, min_n_messages_in_dialog)

    def _process_comment(self, text) -> Optional[str]:
        if not text:
            return None

        text = text.strip()
        if text.startswith('Комментарий удален.'):
            return None

        return text

    def _get_context_from_line_data(self, line_data) -> str:
        return (line_data['story']['text'] or '').strip()

    def _get_tags_from_line_data(self, line_data) -> Tuple[str]:
        return tuple(line_data['story']['tags']) or tuple()
