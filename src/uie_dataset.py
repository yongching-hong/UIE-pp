# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""InstructUIE Dataset."""

from copy import deepcopy
import json
import os
import random
from typing import List
import datasets
from hashlib import md5
import torch
from transformers import AutoTokenizer
from universal_ie.SSIGenerator import DynamicSSIGenerator

from universal_ie.text2spotasoc import Text2SpotAsoc
from universal_ie.uie_convert import convert_graph
from universal_ie.structure_marker import text_start

logger = datasets.logging.get_logger(__name__)
TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
ANSWER_PREFIX = "Answer:"
SINGLE_QUOTES_SUBSTITUTE = "#$%#"
AUX_PROB = 0.3


def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
               data_args.instruction_file + data_args.instruction_strategy + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class UIEConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            instruction_file=None,
            instruction_strategy=None,
            task_config_dir=None,
            prompt_file=None,
            num_examples=None,
            max_num_instances_per_task=None,
            max_num_instances_per_eval_task=None,
            over_sampling=None,
            min_negative_labels=-1,
            min_positive_labels=-1,
            ordered_prompt=True,
            model_name_or_path=None,
            model_cache_dir=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.over_sampling = over_sampling
        self.instructions = self._parse_instruction(instruction_file)
        self.prompt = self._load_prompt(prompt_file)
        self.task_configs = self._parse_task_config(task_config_dir)
        self.instruction_strategy = instruction_strategy
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task
        self.min_negative_labels = min_negative_labels
        self.min_positive_labels = min_positive_labels
        self.ordered_prompt = ordered_prompt
        self.model_name_or_path = model_name_or_path
        self.model_cache_dir = model_cache_dir

    def _parse_instruction(self, instruction_file):
        """
        Instruction example:
        {
          "RE": [
            {"instruction_type": "zero-shot", "instruction": "Given a phrase that describes the relationship between
            two words, extract the words and the lexical relationship between them.
            The output format should be :[(word1, relation, word2)]. \n"},
          ],
          "NER": [
            {"instruction_type": "zero-shot", "instruction": "Please list all entity words in the text that
            fit the category.Output format is [(word1, type1), (word2, type2))]. \n"},
          ],
          "EE": [
            {"instruction_type": "zero-shot", "instruction": "Extract the event information in the text
            and return them in the event list. \n"}
          ]
        }
        """
        if not instruction_file:
            return None
        instructions = {"zero-shot": {}, "few-shot": {}}

        with open(instruction_file, 'r+') as f:
            origin_instructions = json.load(f)

        for task in origin_instructions:
            for task_instruction in origin_instructions[task]:
                instruct_type = task_instruction["instruction_type"]
                if instruct_type == "zero-shot":
                    instructions['zero-shot'][task] = instructions['zero-shot'].get(task, [])
                    instructions['zero-shot'][task].append(task_instruction["instruction"])
                elif instruct_type == "few-shot":
                    instructions['few-shot'][task] = instructions['few-shot'].get(task, [])
                    instructions['few-shot'][task].append(task_instruction["instruction"])
                else:
                    raise ValueError("Invalid instruction type {}, please check your instruction file {}"
                                     .format(instruct_type, instruction_file))
        return instructions

    def _parse_task_config(self, task_config_dir):
        """
        Task config file example:
            {
              "RE": [
                {"sampling strategy": "random", "dataset name": "conll04"}
              ],
              "NER": [
                {"sampling strategy": "random", "dataset name": "ACE05_coarse-grained"},
                {"sampling strategy": "full", "dataset name": "conll2003"}
              ],
              "EE": [
                {"sampling strategy": "random", "dataset name": "GENIA"}
              ]
            }
        """
        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs

    def _load_prompt(self, prompt_file):
        if not prompt_file:
            return None
        
        with open(prompt_file, 'r+') as f:
            prompt = json.loads(f.read())
            return prompt

# TODO, few-shot, 需要 load 的时候就将值存好，放在 "Examples" 里面
class UIEInstructions(datasets.GeneratorBasedBuilder):
    """InstructUIE Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = UIEConfig
    BUILDER_CONFIGS = [
        UIEConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "Task": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "Samples": [{
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "instruction": datasets.Value("string"),
                        "ground_truth": datasets.Value("string"),
                        "answer_prefix": datasets.Value("string")
                    },
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_configs is None:
            logger.error("Please provide right input: data_dir or task_config_dir!")

        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir
        task_configs = self.config.task_configs

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['train'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['dev'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": None,  # default load total test samples to test
                    "subset": "test"
                }),
        ]

    def _load_dataset(self, dataset_path, labels_path = None):
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
        labels = None
        if labels_path:
            with open(labels_path, encoding="utf-8") as labels_f:
                labels = json.load(labels_f)

        return instances, labels

    def _get_instruction(self, task):
        assert self.config.instruction_strategy in INSTRUCTION_STRATEGIES
        if self.config.num_examples is not None and self.config.num_examples > 0:
            task_instructions = self.config.instructions['few-shot'][task]
        else:
            task_instructions = self.config.instructions['zero-shot'][task]
        if self.config.instruction_strategy == "single":
            return task_instructions[0]
        else:
            return random.choice(task_instructions)
        
    def _construct_uie_labels(self, datasets):
        generation_class = Text2SpotAsoc

        prompts, record_schema = convert_graph(
            generation_class,
            datasets=datasets
        )

        return prompts, record_schema

    def _get_prompt(self, instruction, labels_str=None, explain=None):
        if explain:
            prompt = self.config.prompt['prompt_w_explain'].format(instruction=instruction, text='{0}', explain=explain)
        elif labels_str:
            prompt = self.config.prompt['prompt_option'].format(instruction=instruction, options=labels_str, text='{0}')
        else:
            prompt = self.config.prompt['prompt_no_option'].format(instruction=instruction, text='{0}')
        return prompt
        
    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances):
        print("Sampling strategy: {}".format(sampling_strategy))
        print('Total instances: {}'.format(len(instances)))
        if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
            instances = instances[:max_num_instances]
        if (max_num_instances!=None) and self.config.over_sampling and (len(instances) < max_num_instances):
            origin_instances = instances.copy()
            while len(instances) < max_num_instances:
                instances.append(random.choice(origin_instances))
        print('max_num_instances: {}'.format(max_num_instances))
        print('Sampled instances: {}'.format(len(instances)))
        return instances
    
    def sample_negative(self, positive: List[str], candidates: List[str], min_k=5):
        if min_k < 0 or min_k > len(candidates):
            k = len(candidates)
        else:
            k = random.randint(min_k, len(candidates))
    
        negative_set = []
        randperm = torch.randperm(len(candidates))[:k].tolist()
        
        for index in randperm:
            negative = candidates[index]
            if negative not in positive + negative_set:
                negative_set.append(negative)
        
        return negative_set

    def sample_positive(self, positive: List[str], min_k=1):
        if len(positive) == 0:
            return list()
        
        if len(positive) == min_k:
            return positive
        
        if min_k < 0 or min_k > len(positive):
            k = len(positive)
        else:
            k = random.randint(min_k, len(positive))

        chosen_positive = []
        randperm = torch.randperm(len(positive))[:k].tolist()
        for index in randperm:
            pos = positive[index]
            if pos not in chosen_positive:
                chosen_positive.append(pos)
        
        return chosen_positive
    
    def _sampling_labels(self, label_list, positive: List[str]):
        negative_set = self.sample_negative(
            positive,
            label_list,
            self.config.min_negative_labels,
        )
        # Unchosen positive labels will be removed in doc.event_list
        positive_set = self.sample_positive(positive, self.config.min_positive_labels)
        
        labels = negative_set + positive_set
        if self.config.ordered_prompt:
            labels.sort()
        else:
            random.shuffle(labels)
            
        return labels, negative_set, positive_set
    
    def load_NER_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "NER", "Dataset": dataset_name, "Samples": [], "subset": subset}

        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)
        uie_labels, record_schema = self._construct_uie_labels(instances)
        if len(record_schema.type_list) < len(labels):
            record_schema.type_list = labels

        uie_sampler = DynamicSSIGenerator(
            tokenizer=self.tokenizer,
            schema=record_schema,
            positive_rate=1,
            negative=-1,
            ordered_prompt=True
        )

        for (idx, instance), label in zip(enumerate(instances), uie_labels):
            positive = [ ent['type'] for ent in instance['entities'] if ent['type'] not in ['', 'NA']]
            spot_prefix_ids, spot_prefix, positive_spot, negative_spot = uie_sampler.sample_spot(positive)
            example = sample_template.copy()
            
            instruction = spot_prefix + text_start + ' ' + "{0}"
            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }
            
            yield example

    def load_ES_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        # ES = Entity Span
        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "ES", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('ES')
            instruction = self._get_prompt(instruction, labels_str)
            entities = []

            for entity in instance['entities']:
                entities.append(entity["name"])

            if len(entities) > 0:
                label = " " + ", ".join([entity_name for entity_name in entities])
            else:
                label = " None"

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }

            if random.random() < AUX_PROB:
                yield example

    def load_ET_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        # ET = Entity Type
        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Task": "ET", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('ET')
            entities = []
            kv_pairs = []

            for entity in instance['entities']:
                if entity['type'] == 'NA' or entity['type'] == '':
                    continue
                kv_pair = [entity['name'], entity['type']]
                kv_pairs.append(kv_pair)
                entities.append(entity["name"])

            entities_str = ", ".join([entity_name for entity_name in entities])
            instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + " Entities: " + entities_str + " \n" + "Answer:"

            if len(kv_pairs) > 0:
                label = " " + "; ".join(["{}: {}".format(v, k) for (k, v) in kv_pairs])
            else:
                label = " None"

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }

            if random.random() < AUX_PROB:
                yield example

    def load_EP_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        # EP = Entity Pair
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task": "EP", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('EP')
            instruction = self._get_prompt(instruction, labels_str)
            relation_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    continue
                relation_pair = [relation['head']['name'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)

            if len(relation_pairs) > 0:
                label = " " + "; ".join(["{}, {}".format(h, t) for (h, t) in relation_pairs])
            else:
                label = ' None'

            if len(ground_truth_pairs) > 0:
                ground_truth = " " + "; ".join(["{}, {}".format(h, t) for (h, t) in ground_truth_pairs])
            else:
                ground_truth = ' None'

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": ground_truth,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }

            if random.random() < AUX_PROB:
                yield example

    def load_EPR_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        # EPR = Entity Pair Relationship
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task": "EPR", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('EPR')
            relation_pairs = []
            entity_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    ground_truth_pairs.append([relation['head']['name'], 'NA', relation['tail']['name']])
                    continue
                relation_pair = [relation['head']['name'], relation['type'], relation['tail']['name']]
                entity_pair = [relation['head']['name'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)
                entity_pairs.append(entity_pair)

            ep_name = ' ' + "; ".join(["{}, {}".format(h, t) for (h, t) in entity_pairs])
            instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + " Entity Pairs: " + ep_name + ' \n' + "Answer:"

            if len(relation_pairs) > 0:
                label = ' ' + "; ".join(["{}: {}, {}".format(r, h, t) for (h, r, t) in relation_pairs])
            else:
                label = ' None'

            if len(ground_truth_pairs) > 0:
                ground_truth = ' ' + "; ".join(["{}: {}, {}".format(r, h, t) for (h, r, t) in ground_truth_pairs])
            else:
                logger.error("******Error item: {}******".format(instance))
                raise Exception('Dataset Error:{}, No ground truth!'.format(dataset_name))

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": ground_truth,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }

            if random.random() < AUX_PROB:
                yield example

    def load_RE_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task": "RE", "Dataset": dataset_name, "Samples": [], "subset": subset}
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        uie_labels, record_schema = self._construct_uie_labels(instances)
        
        record_schema.type_list = labels
        
        uie_sampler = DynamicSSIGenerator(
            tokenizer=self.tokenizer,
            schema=record_schema,
            positive_rate=1,
            negative=-1,
            ordered_prompt=True
        )

        for (idx, instance), label in zip(enumerate(instances), uie_labels):
            positive = [ ent['type'] for ent in instance['relations'] if ent['type'] ]
            asoc_prefix_ids, asoc_prefix, negative_asoc = uie_sampler.sample_asoc(positive)
            example = sample_template.copy()

            instruction = asoc_prefix + text_start + ' ' + "{0}"

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }

            yield example

    def load_EE_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task": "EE", "Dataset": dataset_name, "Samples": [], "subset": subset}

        # TODO, reconstruct Event Instruction to two stage
        # TODO, check
        labels_str = f'Event type: {labels[0]}, Arguments type: {labels[1]}.'
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction('EE')
            instruction = self._get_prompt(instruction)
            event_pairs = []

            for k, event in enumerate(instance['events']):
                instance['events'][k]['trigger'] = event['trigger'].replace("'", SINGLE_QUOTES_SUBSTITUTE)
                instance['events'][k]['type'] = event['type'].replace("'", SINGLE_QUOTES_SUBSTITUTE)

                if event['type'] == 'NA' or event['type'] == '':
                    continue
                event_type = event['type']
                event_trigger = event['trigger']
                event_arguments = [" {}: {}".format(argument['name'], argument['role']) for
                                   argument in event['arguments']]

                event_arguments = "None" if not event_arguments else ",".join(event_arguments)
                event_pair = [event_type, event_trigger, event_arguments]
                event_pairs.append(event_pair)

            if len(event_pairs) > 0:
                label = ",".join([" ( {}: {}, {}) ".format(type, trigger, arguments)
                                   for (type, trigger, arguments) in event_pairs])
            else:
                label = ' None'
            
            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }

            yield example

    def load_EET_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Task": "EET", "Dataset": dataset_name, "Samples": [], "subset": subset}

        # TODO, reconstruct Event Instruction to two stage
        # TODO, check
        print('EETRETET')
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)
        instances = self._split_by_events(instances)

        # Empty arguments to prevent UIE SSI capture undesired feature
        for instance in instances:
            for ev in instance["events"]:
                ev['arguments'] = []
        
        uie_labels, record_schema = self._construct_uie_labels(instances)
        if len(record_schema.type_list) < len(labels):
            record_schema.type_list = labels
        
        uie_sampler = DynamicSSIGenerator(
            tokenizer=self.tokenizer,
            schema=record_schema,
            positive_rate=1,
            negative=-1,
            ordered_prompt=True
        )

        for (idx, instance), label in zip(enumerate(instances), uie_labels):
            example = sample_template.copy()
            positive = [ ev['type'] for ev in instance['events'] ]
            spot_prefix_ids, spot_prefix, positive_spot, negative_spot = uie_sampler.sample_spot(positive)
            example = sample_template.copy()
            
            instruction = spot_prefix + text_start + ' ' + "{0}"
            
            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }

            yield example
    
    def _split_by_events(self, instances):
        splitted = []

        for instance in instances:
            for event in instance['events']:
                new_instance = {
                    **instance,
                    "events": [event]
                }
                splitted.append(new_instance)
        
        print('Split instances by event: {}'.format(len(splitted)))
        return splitted

    def load_EEA_dataset(self, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):

        instances, labels = self._load_dataset(dataset_path, labels_path)
        role_list = list(set(v for value in labels.values() for v in value))
        sample_template = {"Task": "EEA", "Dataset": dataset_name, "Samples": [], "subset": subset}

        # TODO, reconstruct Event Instruction to two stage
        # TODO, check
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)
        instances = self._split_by_events(instances)
        uie_labels, record_schema = self._construct_uie_labels(instances)
        if len(record_schema.type_list) < len(labels):
            record_schema.role_list = role_list

        uie_sampler = DynamicSSIGenerator(
            tokenizer=self.tokenizer,
            schema=record_schema,
            positive_rate=1,
            negative=-1,
            ordered_prompt=True
        )
        
        for (idx, instance), label in zip(enumerate(instances), uie_labels):
            if len(instance['events']) > 1:
                raise "Error: EEA dataset should only have one event."
            example = sample_template.copy()

            positive_ev = [ ev['type'] for ev in instance['events'] ]
            spot_prefix_ids, spot_prefix, positive_spot, negative_spot = uie_sampler.sample_spot(positive_ev, negative=0)

            positive_roles = [ arg['role'] for ev in instance['events'] for arg in ev['arguments'] ]
            asoc_prefix_ids, asoc_prefix, negative_asoc = uie_sampler.sample_asoc(positive=positive_roles, candidates=labels[positive_ev[0]])

            instruction = spot_prefix + asoc_prefix + text_start + ' ' + "{0}"

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction,
                "answer_prefix": self.config.prompt["response_split"]
            }
            yield example

    
    def _generate_examples(self, path=None, task_config=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            cache_dir=self.config.model_cache_dir,
            use_fast=True,
        )
        self.tokenizer = tokenizer
        for task in task_config:
            if task == "NER":
                load_func = self.load_NER_dataset
            elif task == 'RE':
                load_func = self.load_RE_dataset
            elif task == 'EE':
                load_func = self.load_EE_dataset
            elif task == 'ES':
                load_func = self.load_ES_dataset
            elif task == 'ET':
                load_func = self.load_ET_dataset
            elif task == 'EP':
                load_func = self.load_EP_dataset
            elif task == 'EPR':
                load_func = self.load_EPR_dataset
            elif task == 'EET':
                load_func = self.load_EET_dataset
            elif task == 'EEA':
                load_func = self.load_EEA_dataset
            else:
                raise ValueError("Unsupport {} task, plz check {} task config!".format(task, subset))

            # load dataset
            for dataset in task_config[task]:
                ds_name = dataset["dataset name"]
                sampling_strategy = dataset.get("sampling strategy", "random")
                ds_path = os.path.join(path, task, ds_name, subset + '.json')
                labels_path = os.path.join(path, task, ds_name, 'labels.json')
                assert os.path.exists(ds_path)
                assert os.path.exists(labels_path)
                
                idx = -1
                instances = []
                for sample in load_func(ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                        subset):
                    idx += 1
                    instances.append(sample)
                    sample['input_ids'] = tokenizer.encode(sample['Instance']['sentence'], add_special_tokens=True)
                    yield f"{task}##{ds_path}##{idx}", sample
