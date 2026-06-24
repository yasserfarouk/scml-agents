from __future__ import annotations


class DebugMixin:
    debug_log_enabled = False
    debug_log_file_path = ""

    def log_message(self, *args, **kwargs) -> None:
        return

    def _debug_log_decision(self, *args, **kwargs) -> None:
        return

    def _debug_world_structure(self) -> dict[str, object]:
        return {}

    def _debug_offer_dict(self, offers: dict) -> dict[str, object]:
        return {}

    def _debug_response_dict(self, responses: dict) -> dict[str, object]:
        return {}

    def _log_bankrupt_partner_once(self, partner) -> None:
        try:
            self._logged_bankrupt_partners.add(str(partner))
        except Exception:
            pass

    def _log_cash_bankruptcy_forecast(self, recorded: int = 0) -> None:
        return
