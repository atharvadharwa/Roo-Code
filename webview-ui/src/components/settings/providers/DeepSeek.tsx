import { useCallback } from "react"
import { VSCodeTextField } from "@vscode/webview-ui-toolkit/react"

import type { ProviderSettings } from "@roo-code/types"

import { useAppTranslation } from "@src/i18n/TranslationContext"
import { VSCodeButtonLink } from "@src/components/common/VSCodeButtonLink"

import { inputEventTransform } from "../transforms"

type DeepSeekProps = {
	apiConfiguration: ProviderSettings
	setApiConfigurationField: (field: keyof ProviderSettings, value: ProviderSettings[keyof ProviderSettings]) => void
}

export const DeepSeek = ({ apiConfiguration, setApiConfigurationField }: DeepSeekProps) => {
	const { t } = useAppTranslation()

	const handleInputChange = useCallback(
		<K extends keyof ProviderSettings, E>(
			field: K,
			transform: (event: E) => ProviderSettings[K] = inputEventTransform,
		) =>
			(event: E | Event) => {
				setApiConfigurationField(field, transform(event as E))
			},
		[setApiConfigurationField],
	)

	return (
		<>
			<VSCodeTextField
				value={apiConfiguration?.apiModelId || ""}
				onInput={handleInputChange("apiModelId")}
				placeholder="Model name"
				className="w-full">
				<label className="block font-medium mb-1">Model</label>
			</VSCodeTextField>
			<div className="text-sm text-vscode-descriptionForeground -mt-2">
				Enter the model name (e.g., DeepSeek-R1-Distill-Qwen-32B)
			</div>

			<VSCodeTextField
				value={apiConfiguration?.deepSeekApiKey || ""}
				type="password"
				onInput={handleInputChange("deepSeekApiKey")}
				placeholder={t("settings:placeholders.apiKey")}
				className="w-full mt-4">
				<label className="block font-medium mb-1">{t("settings:providers.deepSeekApiKey")}</label>
			</VSCodeTextField>
			<div className="text-sm text-vscode-descriptionForeground -mt-2">
				{t("settings:providers.apiKeyStorageNotice")}
			</div>

			<VSCodeTextField
				value={apiConfiguration?.deepSeekBaseUrl || ""}
				onInput={handleInputChange("deepSeekBaseUrl")}
				placeholder={t("settings:placeholders.endpoint")}
				className="w-full mt-4">
				<label className="block font-medium mb-1">{t("settings:providers.endpoint")}</label>
			</VSCodeTextField>
			<div className="text-sm text-vscode-descriptionForeground -mt-2">
				{t("settings:providers.endpointDescription")}
			</div>
			
			<VSCodeTextField
				value={apiConfiguration?.deepSeekCaBundlePath || ""}
				onInput={handleInputChange("deepSeekCaBundlePath")}
				placeholder={t("settings:placeholders.caBundlePath")}
				className="w-full mt-4">
				<label className="block font-medium mb-1">{t("settings:providers.caBundlePath")}</label>
			</VSCodeTextField>
			<div className="text-sm text-vscode-descriptionForeground -mt-2">
				{t("settings:providers.caBundlePathDescription")}
			</div>

			{!apiConfiguration?.deepSeekApiKey && (
				<VSCodeButtonLink href="https://platform.deepseek.com/" appearance="secondary">
					{t("settings:providers.getDeepSeekApiKey")}
				</VSCodeButtonLink>
			)}
		</>
	)
}
