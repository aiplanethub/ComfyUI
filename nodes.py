import os
import sys
import traceback
import time
import logging
import folder_paths
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=16384
import logging

class InputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {"multiline": False}),
                "name": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("INPUT",)
    FUNCTION = "process"
    CATEGORY = "Input"

    def process(self, instruction, name):
        payload = {
            "instruction": instruction,
            "name": name
        }
        return (payload,)


class LLMNode:
    LLM_PROVIDERS = ['openai', 'azure', 'groq', 'cohere', 'gemini'] 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INPUT",),
                "llm_name": (cls.LLM_PROVIDERS,),
            },
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "select"
    CATEGORY = "LLM"

    def select(self, input, llm_name):
        # Use the llm_config_name directly from the widget values
        payload = {"provider": llm_name}
        # Return the LLM configuration
        return (payload,)
    
class KnowledgeBaseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "knowledgebases": ("STRING", {"multiline": True}),  # Ensure multiline input is handled
            },
        }

    RETURN_TYPES = ("KNOWLEDGE_BASE",)
    FUNCTION = "select"
    CATEGORY = "Knowledge Base"

    def select(self, knowledgebases):
        # Split the knowledgebases string by newline, ensuring all are handled as a list
        knowledge_base_list = knowledgebases.split('\n')  # Split the multiline string into a list

        payload = {
            "knowledgebases": knowledge_base_list  # Ensure the payload has the list of knowledge base names
        }
        # Return the list of knowledge bases
        return (payload,)

class MemoryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "collection_name": ("STRING", {"default": "default_collection"}),
            },
            "optional": {
                "host": ("STRING", {"default": ""}),
                "port": ("INT", {"default": 8000}),
                "persist_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MEMORY",)
    FUNCTION = "generate_config"
    CATEGORY = "Memory"

    def generate_config(self, collection_name, host="", port=8000, persist_path=""):
        payload = {
            "collection_name": collection_name,
            "host": host,
            "port": port,
            "persist_path": persist_path
        }
        return (payload,)


class ToolsNode:
    TOOL_NAMES = [
                "DocumentLoader", "GitHubSearchTool", "SerpSearch", 
                "SerperSearch", "TavilyQASearch", "UnstructuredIO", "SummarizerAction", "FormatterAction", 
                "ReadFileAction", "WriteFileAction", "CreateFileAction",
                "WebLoader", "YouTubeSearchTool", "WebBaseContextTool", "DuckDuckGoNewsSearch"
            ]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                tool: ("BOOLEAN", {"default": False}) for tool in cls.TOOL_NAMES
            }
        }

    RETURN_TYPES = ("TOOL",)
    FUNCTION = "output_tools"
    CATEGORY = "Actions"

    def output_tools(self, **tool_states):
        selected_tools = [tool for tool, state in tool_states.items() if state]
        return (selected_tools,)


class TaskPlannerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_config": ("LLM",),
                "tools": ("TOOL", {"multiple": True}),
                "auto_assign": ("BOOLEAN", {"default": False}),
                "memory": ("MEMORY",)
            },
            "optional": {
                "knowledge_base": ("KNOWLEDGE_BASE",)
            }
        }

    RETURN_TYPES = ("TASK_PLAN",)
    FUNCTION = "plan_tasks"
    CATEGORY = "Task Planning"

    def plan_tasks(self, llm_config, tools, auto_assign, memory, knowledge_base=None):
        if not isinstance(tools, (list, tuple)):
            tools = [tools]

        # Prepare the task plan; include knowledge_base only if provided
        task_plan = {
            "llm_config": llm_config,
            "memory": memory,
            "current_task": 0,
            "completed_tasks": {},
            "auto_assigned": auto_assign
        }

        # Add knowledge_base only if provided
        if knowledge_base:
            task_plan["knowledge_base"] = knowledge_base

        return (task_plan,)



class WorkerNode:
    TOOL_NAMES = [
                "DocumentLoader", "GitHubSearchTool", "SerpSearch", 
                "SerperSearch", "TavilyQASearch", "UnstructuredIO", "SummarizerAction", "FormatterAction", 
                "ReadFileAction", "WriteFileAction", "CreateFileAction",
                "WebLoader", "YouTubeSearchTool", "WebBaseContextTool", "DuckDuckGoNewsSearch"
            ]
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_plan": ("TASK_PLAN",),
                "worker_role": ("STRING", {"default": "Worker Role"}),  
                "worker_instruction" : ("STRING", {"default": "Worker Instruction"}),  
                "llm_config": ("LLM",),
                **{tool: ("BOOLEAN", {"default": False}) for tool in cls.TOOL_NAMES}
            },
            "optional": {
                "previous_output": ("WORKER_OUTPUT",),
            }
        }

    RETURN_TYPES = ("WORKER_OUTPUT",)
    FUNCTION = "execute_task"
    CATEGORY = "Worker"

    def execute_task(self, task_plan, worker_role, worker_instruction, llm_config, previous_output=None, **tools):
        # Extract the tool states
        selected_tools = [tool for tool, state in tools.items() if state]
        
        # Create the payload for the API call
        payload = {
            "task_plan": task_plan,
            "worker_role": worker_role,  
            "worker_instruction": worker_instruction,
            "llm_config": llm_config,
            "selected_tools": selected_tools,
            "previous_output": previous_output
        }
        task_result = {
            "worker_role": worker_role,  
            "tools": {tool: tools.get(tool, False) for tool in self.TOOL_NAMES}
        }

        # Return the task result
        return (task_result,)

class OutputNode:
    SUPPORTED_FORMATS = None  # Set to None to allow all file formats

    @classmethod
    def INPUT_TYPES(cls):
        # You can use any folder, like Downloads or Desktop, based on OS-specific paths.
        input_dir = "C:/Users/sucha/Downloads/"  # Customize this to your desired folder path
        input_dir1 = folder_paths.get_input_directory() 
        logging.info(input_dir1)

        # Check if directory exists and list the files
        if not os.path.exists(input_dir):
            return {"required": {"input": ("WORKER_OUTPUT",)}, "optional": {"files": ("No valid files found", {})}}

        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))  # Remove format check to allow all files
        ]

        if not files:
            files.append("No valid files found")  # Handle case where no files are found

        return {
            "required": {
                "input": ("WORKER_OUTPUT",),  # Keep Worker Output as the main input
            },
            "optional": {
                # Add a dropdown for file selection with enlarged width allocation (longer display for filenames)
                "files": (sorted(files), {"default": "Select a file"})  # Adjust width with widget_width
            }
        }

    CATEGORY = "Output"

    RETURN_TYPES = ("OUTPUT",)
    FUNCTION = "process_output"
    OUTPUT_NODE = True

    def process_output(self, input, files):
        """
        Process the WORKER_OUTPUT and handle the selected file.
        """
        # Ensure a file is selected
        if files == "Select a file" or files == "No valid files found":
            return ({"status": "error", "message": "No valid file selected."},)

        # input_dir = folder_paths.get_input_directory()  # Use the dynamically fetched directory
        # file_path = os.path.join(input_dir, files)  # Customize this to your desired folder path
        
        # Get the full file path from the selected file
        file_path = os.path.join("C:/Users/sucha/Downloads/", files)  # Customize this to your desired folder path

        # Since we are allowing all file types, no format validation is needed
        if not os.path.isfile(file_path):
            return ({"status": "error", "message": "File not found."},)

        # Prepare the file information payload
        file_info = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "message": "File processed successfully."
        }
        logging.info(file_info)
        # Combine the file information and input (Worker Output)
        payload = {
            "input": input,
            "file_info": file_info
        }

        return (payload,)

    def is_valid_file(self, file_path):
        """
        Check if the selected file is valid.
        Since we're allowing all formats, we just check if the file exists.
        """
        return os.path.isfile(file_path)

    @classmethod
    def IS_CHANGED(cls, files):
        """
        Detects if the selected file has changed by calculating its hash.
        """
        if not files or files == "Select a file":
            return ""
        
        # input_dir = folder_paths.get_input_directory()  # Use the dynamically fetched directory
        # file_path = os.path.join(input_dir, files)  # Customize this to your desired folder path

        file_path = os.path.join("C:/Users/sucha/Downloads/", files)  # Customize this to your desired folder path
        m = hashlib.sha256()
        with open(file_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, files):
        """
        Validates the input to ensure the file exists.
        """
        # input_dir = folder_paths.get_input_directory()  # Use the dynamically fetched directory
        # file_path = os.path.join(input_dir, files)  # Customize this to your desired folder path

        file_path = os.path.join("C:/Users/sucha/Downloads/", files)  # Customize this to your desired folder path
        if not os.path.isfile(file_path):
            return "Invalid file path: {}".format(files)
        return True
    

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "InputNode": InputNode,
    "LLMNode": LLMNode,
    "KnowledgeBaseNode": KnowledgeBaseNode,
    "ToolsNode": ToolsNode,
    "TaskPlannerNode": TaskPlannerNode,
    "WorkerNode": WorkerNode,
    "OutputNode": OutputNode,
    "MemoryNode": MemoryNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InputNode": "Input",
    "LLMNode": "LLM Processor",
    "KnowledgeBaseNode": "Knowledge Base",
    "ToolsNode": "Tools",
    "TaskPlannerNode": "Task Planner",
    "WorkerNode": "Worker",
    "OutputNode": "Output",
    "MemoryNode": "Memory"
}


EXTENSION_WEB_DIRS = {}


def get_module_name(module_path: str) -> str:
    """
    Returns the module name based on the given module path.
    Examples:
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__.py") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node/__init__/") -> "my_custom_node"
        get_module_name("C:/Users/username/ComfyUI/custom_nodes/my_custom_node.disabled") -> "custom_nodes
    Args:
        module_path (str): The path of the module.
    Returns:
        str: The module name.
    """
    base_path = os.path.basename(module_path)
    if os.path.isfile(module_path):
        base_path = os.path.splitext(base_path)[0]
    return base_path


def load_custom_node(module_path: str, ignore=set(), module_parent="custom_nodes") -> bool:
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        logging.debug("Trying to load custom node {}".format(module_path))
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if os.path.isdir(web_dir):
                EXTENSION_WEB_DIRS[module_name] = web_dir

        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
                if name not in ignore:
                    NODE_CLASS_MAPPINGS[name] = node_cls
                    node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            return True
        else:
            logging.warning(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")
            return False
    except Exception as e:
        logging.warning(traceback.format_exc())
        logging.warning(f"Cannot import {module_path} module for custom nodes: {e}")
        return False

def init_external_custom_nodes():
    """
    Initializes the external custom nodes.

    This function loads custom nodes from the specified folder paths and imports them into the application.
    It measures the import times for each custom node and logs the results.

    Returns:
        None
    """
    base_node_names = set(NODE_CLASS_MAPPINGS.keys())
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(os.path.realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py": continue
            if module_path.endswith(".disabled"): continue
            time_before = time.perf_counter()
            success = load_custom_node(module_path, base_node_names, module_parent="custom_nodes")
            node_import_times.append((time.perf_counter() - time_before, module_path, success))

    if len(node_import_times) > 0:
        logging.info("\nImport times for custom nodes:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

def init_builtin_extra_nodes():
    """
    Initializes the built-in extra nodes in ComfyUI.

    This function loads the extra node files located in the "comfy_extras" directory and imports them into ComfyUI.
    If any of the extra node files fail to import, a warning message is logged.

    Returns:
        None
    """
    extras_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy_extras")
    extras_files = []

    import_failed = []
    for node_file in extras_files:
        if not load_custom_node(os.path.join(extras_dir, node_file), module_parent="comfy_extras"):
            import_failed.append(node_file)

    return import_failed


def init_extra_nodes(init_custom_nodes=True):
    import_failed = init_builtin_extra_nodes()

    if init_custom_nodes:
        init_external_custom_nodes()
    else:
        logging.info("Skipping loading of custom nodes")

    if len(import_failed) > 0:
        logging.warning("WARNING: some comfy_extras/ nodes did not import correctly. This may be because they are missing some dependencies.\n")
        for node in import_failed:
            logging.warning("IMPORT FAILED: {}".format(node))
        logging.warning("\nThis issue might be caused by new missing dependencies added the last time you updated ComfyUI.")
        if args.windows_standalone_build:
            logging.warning("Please run the update script: update/update_comfyui.bat")
        else:
            logging.warning("Please do a: pip install -r requirements.txt")
        logging.warning("")
