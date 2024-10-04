import { app } from "../../scripts/app.js";

const ext = {
	name: "Comfy.OpenAGI",

	async setup(){

    window.parent.postMessage(
      JSON.stringify({
        event: "iframeLoaded", // Indicate that the iframe (ComfyUI) has been initialized
      }),
      "*" // Use "*" to allow communication regardless of origin
    );

    const handleWorkflowEvents = async function (event) {
      const parsedEventData = JSON.parse(event.data);

      if (parsedEventData.event === "loadWorkflow") {
        await app.loadGraphData(parsedEventData.data);
      } else if (parsedEventData.event === "getWorkflow") {
        const graphData = await app.graphToPrompt();
        event.source.postMessage(
          JSON.stringify({
            event: "getWorkflow",
            data: graphData.workflow,
            isSilent: parsedEventData.isSilent,
          }),
          event.origin
        );
      } else if (parsedEventData.event === "queuePrompt") {
        try {
          const application_id = parsedEventData.application_id;
          const access_token = parsedEventData.accessToken;
          const result = await app.queuePrompt(
            0,
            app.batchCount,
            application_id,
            access_token
          );

          // If it reaches here, that means the queuePrompt finished without throwing an error
          event.source.postMessage(
            JSON.stringify({
              event: "queuePrompt",
              status: "success",
              message: "Queue processing done",
            }),
            event.origin
          );
        } catch (error) {
          // If any error occurs during queuePrompt, catch it here
          event.source.postMessage(
            JSON.stringify({
              event: "queuePrompt",
              status: "error",
              message: "Error during queue processing",
              error: error.message, // Send the error message back to the caller
            }),
            event.origin
          );
        }
      }
    };
		window.addEventListener("message", handleWorkflowEvents);
	}
};

app.registerExtension(ext);