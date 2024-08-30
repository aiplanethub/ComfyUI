import { app } from "../../scripts/app.js";

const ext = {
	name: "Comfy.OpenAGI",

	async setup(){
		const handleWorkflowEvents = async function (event) {
			const parsedEventData = JSON.parse(event.data)

			if(parsedEventData.event === "loadWorkflow"){
				await app.loadGraphData(parsedEventData.data)
			}
			else if(parsedEventData.event === "getWorkflow"){
				const graphData = await app.graphToPrompt();
				event.source.postMessage(
					JSON.stringify({event:"getWorkflow",data:graphData.workflow}), 
					event.origin
				)
				
			}else if(parsedEventData.event === "queuePrompt"){
				await app.queuePrompt(0, app.batchCount)
			}			
		}
		window.addEventListener("message", handleWorkflowEvents);
	}
};

app.registerExtension(ext);
