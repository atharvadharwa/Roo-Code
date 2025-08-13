import { Package } from "../../shared/package"

export const DEFAULT_HEADERS = {
	"User-Agent": "python-requests/2.33.4",
	"Connection": "keep-alive",
	"Accept": "*/*",
	"Accept-Encoding": "gzip, deflate",
	"Content-Type": "application/json",
	"Authorization": "", // This will be set dynamically
}
