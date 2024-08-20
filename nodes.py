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
class InputNode:

    @classmethod
    def INPUT_TYPES(s):
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
            logging.info(f"API response for InputNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 
            
        return ({"instruction": instruction, "name": name},)
class LLMNode:
    LLM_PROVIDERS = ["Azure", "Groq", "Claude", "OpenAI"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("INPUT",),
                "llm_provider_name": (s.LLM_PROVIDERS,),
            },
        }
    
    RETURN_TYPES = ("LLM",)
    FUNCTION = "select"
    CATEGORY = "LLM"

    def select(self, input, llm_provider_name):
        # Placeholder - replace with actual knowledge base loading logic
        payload = {"input": input, "llm_provider_name": llm_provider_name}

        try:
            response = asyncio.run(api.call_api("LLMNode", payload))
            logging.info(f"API response for LLMNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 

        return (llm_provider_name,)

class KnowledgeBaseNode:
    KNOWLEDGE_BASES = ["General Knowledge", "Specialized Domain", "Custom KB 1", "Custom KB 2"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "knowledgebase_name": (s.KNOWLEDGE_BASES,),
            },
        }
    
    RETURN_TYPES = ("KNOWLEDGE_BASE",)
    FUNCTION = "select"
    CATEGORY = "Knowledge Base"

    def select(self, knowledgebase_name):
        # Placeholder - replace with actual knowledge base loading logic
        payload = {"knowledgebase_name": knowledgebase_name}

        try:
            response = asyncio.run(api.call_api("KnowledgeBaseNode", payload))
            logging.info(f"API response for KnowledgeBaseNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 

        return (knowledgebase_name,)


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
            logging.info(f"API response for MemoryNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 

        config = {
            "collection_name": collection_name,
        }
        
        if host and port:
            config["host"] = host
            config["port"] = port
        elif persist_path:
            config["persist_path"] = persist_path
        
        return (config,)


class ToolsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tools": (["DocumentLoader", "GitHubSearchTool", "SerpSearch", "SerperSearch", "TavilyQASearch", "UnstructuredIO", "WebLoader", "YouTubeSearch"],),
            },
        }
    
    RETURN_TYPES = ("BASE_ACTIONS",)
    FUNCTION = "select_tools"
    CATEGORY = "Actions"

    def select_tools(self, tools):
        payload = {"tools": tools}

        try:
            response = asyncio.run(api.call_api("ToolsNode", payload))
            logging.info(f"API response for ToolsNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 

        return (tools,)

class TaskPlannerNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_config": ("LLM",),
                "knowledge_base": ("KNOWLEDGE_BASE",),
                "available_actions": ("BASE_ACTIONS",),
                "auto_assign": ("BOOLEAN", {"default": False}),
                "memory": ("MEMORY",)
            },
        }
    
    RETURN_TYPES = ("TASK_PLAN",)
    FUNCTION = "plan_tasks"
    CATEGORY = "Task Planning"

    def plan_tasks(self, llm_config, knowledge_base, available_actions, auto_assign, memory):
        payload = {
        "llm_config": llm_config,
        "knowledge_base": knowledge_base,
        "available_actions": available_actions,
        "auto_assign": auto_assign,
        "memory": memory
        }

        try:
            response = asyncio.run(api.call_api("TaskPlannerNode", payload))
            logging.info(f"API response for TaskPlannerNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 

        task_plan = self.generate_task_plan(llm_config, knowledge_base, available_actions)
    
        if auto_assign:
            task_plan["auto_assigned"] = True
        else:
            task_plan["auto_assigned"] = False
        
        return (task_plan,)

    def generate_task_plan(self, llm_config, knowledge_base, available_actions):
        payload = {
        "llm_config": llm_config,
        "knowledge_base": knowledge_base,
        "available_actions": available_actions,
        }

        try:
            response = asyncio.run(api.call_api("TaskPlannerNode", payload))
            logging.info(f"API response for TaskPlannerNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 

        # Placeholder for actual task planning logic
        task_plan = {
            "llm_config": llm_config,
            "knowledge_base": knowledge_base,
            "planned_tasks": [
                {
                    "id": 0,
                    "description": "Analyze the objective",
                    "dependencies": [],
                    "required_actions": available_actions[:2]  # Using first two actions as an example
                },
                {
                    "id": 1,
                    "description": "Research using knowledge base",
                    "dependencies": [0],
                    "required_actions": available_actions[2:4]  # Using next two actions as an example
                },
                {
                    "id": 2,
                    "description": "Formulate response",
                    "dependencies": [1],
                    "required_actions": available_actions[4:]  # Using remaining actions as an example
                }
            ],
            "current_task": 0,
            "completed_tasks": {}
        }
        
        return task_plan

class WorkerNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_plan": ("TASK_PLAN",),
                "worker_id": ("INT", {"default": 0, "min": 0, "max": 99}),
                "llm_config": ("LLM",),
                "available_actions": ("BASE_ACTIONS",),
            },
            "optional": {
                "previous_output": ("WORKER_OUTPUT",),
            }
        }
    
    RETURN_TYPES = ("WORKER_OUTPUT", "UPDATED_TASK_PLAN")
    FUNCTION = "execute_task"
    CATEGORY = "Worker"

    def execute_task(self, task_plan, worker_id, llm_config, available_actions, previous_output=None):
        payload = {
            "task_plan": task_plan,
            "worker_id": worker_id,
            "llm_config": llm_config,
            "available_actions": available_actions,
            "previous_output": previous_output
        }

        try:
            response = asyncio.run(api.call_api("WorkerNode", payload))
            logging.info(f"API response for WorkerNode : {response}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return (None,) 

        if task_plan["current_task"] >= len(task_plan["planned_tasks"]):
            return ("All tasks completed", task_plan)

        current_task = task_plan["planned_tasks"][task_plan["current_task"]]
        
        if all(dep in task_plan["completed_tasks"] for dep in current_task["dependencies"]):
            task_result = f"Worker {worker_id} executed: {current_task['description']} using actions: {', '.join(current_task['required_actions'])}"
            if previous_output:
                task_result += f" Using previous output: {previous_output}"
            
            task_plan["completed_tasks"][current_task["id"]] = task_result
            task_plan["current_task"] += 1
        else:
            task_result = f"Worker {worker_id} waiting: Dependencies not met for task {current_task['id']}"
        
        return (task_result, task_plan)

class OutputNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("WORKER_OUTPUT",),  # Accept any input type
            },
        }
    
    RETURN_TYPES = ("OUTPUT",)
    FUNCTION = "process_output"
    OUTPUT_NODE = True
    CATEGORY = "Output"

    def process_output(self, input):
        # Process the input here
        payload = {"input": input}

        try:
            response = asyncio.run(api.call_api("OutputNode", payload))
            logging.info(f"API response for OutputNode : {response}")
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
