import { app } from "../../scripts/app.js";

// Adds an upload button to the nodes

app.registerExtension({
	name: "Comfy.UploadFile",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.input?.required?.file?.[1]?.file_upload === true) {
			nodeData.input.required.upload = ["FILEUPLOAD"];
		}
	},
});
