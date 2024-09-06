import asyncio
import requests
import torch
import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from api import api
import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

MAX_RESOLUTION=16384
import asyncio
import logging
from api import api

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

        try:
            response = asyncio.run(api.call_api("InputNode", payload))
            logging.info(f"API response for InputNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)
        
        return ({"instruction": instruction, "name": name},)


class LLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INPUT",),
                "llm_config_name": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("LLM",)
    FUNCTION = "select"
    CATEGORY = "LLM"

    def select(self, input, llm_config_name):
        # Use the llm_config_name directly from the widget values
        payload = {"input": input, "llm_config_name": llm_config_name}

        try:
            response = asyncio.run(api.call_api("LLMNode", payload))
            logging.info(f"API response for LLMNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)

        # Return the LLM configuration
        return ({"provider": llm_config_name},)

class KnowledgeBaseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "knowledgebases": ("STRING", {"multiline": True, "default": ""}),  # Placeholder for dynamic values
            },
        }

    RETURN_TYPES = ("KNOWLEDGE_BASE",)
    FUNCTION = "select"
    CATEGORY = "Knowledge Base"

    def __init__(self, knowledge_bases=None):
        # Initialize with the knowledge bases provided dynamically or default to empty
        self.knowledge_bases = knowledge_bases if knowledge_bases else []

    def select(self, knowledgebases):
        # This method handles the processing of the selected knowledge bases
        payload = {
            "knowledgebases": knowledgebases  # Payload to send the selected bases to the backend or workflow
        }

        try:
            # Call the backend API to process the knowledge bases asynchronously
            response = asyncio.run(api.call_api("KnowledgeBaseNode", payload))
            logging.info(f"API response for KnowledgeBaseNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)

        # Return the selected knowledge base for use in the workflow
        return ({"knowledgebases": knowledgebases},)

    @classmethod
    def update_knowledge_bases(cls, knowledge_bases):
        # Dynamically update INPUT_TYPES to handle all received knowledge bases
        inputs = {
            "required": {
                f"knowledgebase_{idx+1}": ("STRING", {"default": kb, "multiline": False})
                for idx, kb in enumerate(knowledge_bases)
            }
        }
        cls.INPUT_TYPES = lambda: inputs  # Update the class input types dynamically

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

        try:
            response = asyncio.run(api.call_api("MemoryNode", payload))
            logging.info(f"API response for MemoryNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)

        config = {
            "collection_name": collection_name,
            "host": host,
            "port": port,
            "persist_path": persist_path
        }

        return (config,)


class ToolsNode:
    TOOL_NAMES = [
        "DocumentLoader", "GitHubSearchTool", "SerpSearch",
        "SerperSearch", "TavilyQASearch", "UnstructuredIO",
        "WebLoader", "YouTubeSearch", "DuckDuckGoNewsSearch", "WebBaseContextTool"
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

        payload = {"selected_tools": selected_tools}

        try:
            response = asyncio.run(api.call_api("ToolsNode", payload))
            logging.info(f"API response for ToolsNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)

        return (selected_tools,)


class TaskPlannerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_config": ("LLM",),
                "knowledge_base": ("KNOWLEDGE_BASE",),
                "tools": ("TOOL", {"multiple": True}),
                "auto_assign": ("BOOLEAN", {"default": False}),
                "memory": ("MEMORY",)
            }
        }

    RETURN_TYPES = ("TASK_PLAN",)
    FUNCTION = "plan_tasks"
    CATEGORY = "Task Planning"

    def plan_tasks(self, llm_config, knowledge_base, tools, auto_assign, memory):
        if not isinstance(tools, (list, tuple)):
            tools = [tools]

        payload = {
            "llm_config": llm_config,
            "knowledge_base": knowledge_base,
            "tools": tools,
            "auto_assign": auto_assign,
            "memory": memory
        }

        try:
            response = asyncio.run(api.call_api("TaskPlannerNode", payload))
            logging.info(f"API response for TaskPlannerNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)

        task_plan = self.generate_task_plan(llm_config, knowledge_base, tools)

        task_plan["auto_assigned"] = auto_assign

        return (task_plan,)

    def generate_task_plan(self, llm_config, knowledge_base, tools):
        task_plan = {
            "llm_config": llm_config,
            "knowledge_base": knowledge_base,
            "planned_tasks": [
                {
                    "id": 0,
                    "description": "Analyze the objective",
                    "dependencies": [],
                    "required_actions": tools[:2]
                },
                {
                    "id": 1,
                    "description": "Research using knowledge base",
                    "dependencies": [0],
                    "required_actions": tools[2:4]
                },
                {
                    "id": 2,
                    "description": "Formulate response",
                    "dependencies": [1],
                    "required_actions": tools[4:]
                }
            ],
            "current_task": 0,
            "completed_tasks": {}
        }

        return task_plan

class WorkerNode:
    TOOL_NAMES = [
        "DocumentLoader", "GitHubSearchTool", "SerpSearch",
        "SerperSearch", "TavilyQASearch", "UnstructuredIO",
        "WebLoader", "YouTubeSearch", "WebBaseContextTool", "DuckDuckGoNewsSearch"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_plan": ("TASK_PLAN",),
                "worker_role": ("STRING", {"default": "Worker Role"}),  # Worker role comes first
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

    def execute_task(self, task_plan, worker_role, llm_config, previous_output=None, **tools):
        # Extract the tool states
        selected_tools = [tool for tool, state in tools.items() if state]
        
        # Create the payload for the API call
        payload = {
            "task_plan": task_plan,
            "worker_role": worker_role,  # Display worker role instead of worker id
            "llm_config": llm_config,
            "selected_tools": selected_tools,
            "previous_output": previous_output
        }

        try:
            # Call the API using asyncio
            response = asyncio.run(api.call_api("WorkerNode", payload))
            logging.info(f"API response for WorkerNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)

        # Prepare the task result with worker role
        task_result = {
            "worker_role": worker_role,  # Show worker role at the top
            "tools": {tool: tools.get(tool, False) for tool in self.TOOL_NAMES}
        }

        # Return the task result
        return (task_result,)

class OutputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("WORKER_OUTPUT",),
            },
        }

    RETURN_TYPES = ("OUTPUT",)
    FUNCTION = "process_output"
    OUTPUT_NODE = True
    CATEGORY = "Output"

    def process_output(self, input):
        payload = {"input": input}

        try:
            response = asyncio.run(api.call_api("OutputNode", payload))
            logging.info(f"API response for OutputNode: {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,)

        return (input,)


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

# ComfyUI node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "InputNode": "Input",
    "AzureChatConfigModelNode": "Azure Chat Config",
    "GroqConfigModelNode": "Groq Config",
    "GeminiConfigModelNode": "Gemini Config",
    "HuggingFaceConfigModelNode": "HuggingFace Config",
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
    extras_files = [
        "nodes_latent.py",
        "nodes_hypernetwork.py",
        "nodes_upscale_model.py",
        "nodes_post_processing.py",
        "nodes_mask.py",
        "nodes_compositing.py",
        "nodes_rebatch.py",
        "nodes_model_merging.py",
        "nodes_tomesd.py",
        "nodes_clip_sdxl.py",
        "nodes_canny.py",
        "nodes_freelunch.py",
        "nodes_custom_sampler.py",
        "nodes_hypertile.py",
        "nodes_model_advanced.py",
        "nodes_model_downscale.py",
        "nodes_images.py",
        "nodes_video_model.py",
        "nodes_sag.py",
        "nodes_perpneg.py",
        "nodes_stable3d.py",
        "nodes_sdupscale.py",
        "nodes_photomaker.py",
        "nodes_cond.py",
        "nodes_morphology.py",
        "nodes_stable_cascade.py",
        "nodes_differential_diffusion.py",
        "nodes_ip2p.py",
        "nodes_model_merging_model_specific.py",
        "nodes_pag.py",
        "nodes_align_your_steps.py",
        "nodes_attention_multiply.py",
        "nodes_advanced_samplers.py",
        "nodes_webcam.py",
        "nodes_audio.py",
        "nodes_sd3.py",
        "nodes_gits.py",
        "nodes_controlnet.py",
        "nodes_hunyuan.py",
    ]

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
