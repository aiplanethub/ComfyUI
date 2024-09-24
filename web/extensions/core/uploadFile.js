// File: extensions/core/UploadFile.js

import { app } from "../../scripts/app";

app.registerExtension({
    name: "Comfy.UploadFile",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const requiredInputs = nodeData?.input?.required;
        if (requiredInputs) {
            for (const [key, value] of Object.entries(requiredInputs)) {
                if (value[1]?.upload_file === true) {
                    // If upload_file is true, add a FILEUPLOAD input
                    nodeData.input.required[`${key}_upload`] = ["FILEUPLOAD"];
                }
            }
        }
    },
});

// Add FILEUPLOAD to ComfyWidgets
app.registerExtension({
    name: "Comfy.UploadFile.Widget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add FILEUPLOAD widget type if it doesn't exist
        if (!app.ui.settings.comfyWidgets.FILEUPLOAD) {
            app.ui.settings.comfyWidgets.FILEUPLOAD = function(node, inputName, inputData) {
                const fileInput = document.createElement("input");
                Object.assign(fileInput, {
                    type: "file",
                    accept: "*/*",  // Accept all file types
                    style: "display: none",
                    onchange: async () => {
                        if (fileInput.files.length) {
                            await uploadFile(fileInput.files[0], true);
                        }
                    },
                });
                document.body.append(fileInput);

                const uploadWidget = node.addWidget("button", inputName, "file", () => {
                    fileInput.click();
                });
                uploadWidget.label = "Choose file to upload";
                uploadWidget.serialize = false;

                async function uploadFile(file, updateNode) {
                    try {
                        const body = new FormData();
                        body.append("file", file);
                        const resp = await api.fetchApi("/upload/file", {
                            method: "POST",
                            body,
                        });

                        if (resp.status === 200) {
                            const data = await resp.json();
                            let path = data.name;
                            if (data.subfolder) path = data.subfolder + "/" + path;

                            if (updateNode) {
                                node.widgets.find(w => w.name === inputName.replace("_upload", "")).value = path;
                            }
                        } else {
                            alert(resp.status + " - " + resp.statusText);
                        }
                    } catch (error) {
                        alert(error);
                    }
                }

                return { widget: uploadWidget };
            };
        }
    },
});