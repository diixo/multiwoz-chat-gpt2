
import json
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from tqdm import tqdm, trange
import sys
from tokenizer import GPT2Tokenizer


class DialogLoader(Dataset):
    IGNORE_INDEX = -100

    def __init__(self, tokenizer: GPT2Tokenizer, data_type: str, max_length=512):
        assert data_type in ['train', 'dev', 'test']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_type = data_type
        self._get_special_token_ids()
        self._process_dst(data_type)
        self._create_examples()


    def _process_dst(self, split):

        dst_files = {
            'train': self.config.train_data_path,
            'dev': self.config.validation_data_path,
            'test': self.config.test_data_path
        }
        dst_f = dst_files[split]

        dst_cont, dial_n, example_n = iterate_dst_file(dst_f)
        print('{} -> # of dialogues: {}, examples: {}'.format(torch.split, dial_n, example_n))
        self.dst_data = dst_cont



    def _create_examples(self):
        self.examples = []
        for example_num, example_id in enumerate(tqdm(sorted(self.dst_data.keys()))):
            if self.data_size != -1 and example_num == self.data_size:
				break

			context = self.dst_data[example_id]['context'] # str of word token
			turn_utt = self.dst_data[example_id]['turn_utt']
			bs_dict = self.dst_data[example_id]['belief_state'] # dict
			self.normalize_value(bs_dict)
			bs_str = self.dict2sorted_str(bs_dict)
			# if self.args.remove_dontcare:
			# 	assert 'dontcare' not in bs_str
		
			context_ids = self.tokenizer(context)['input_ids']
			target_ids = self.tokenizer(bs_str)['input_ids'] # TODO: Q3
			target_len = len(target_ids)
			if not self.generation:
				# dialogue_context <BOS> belief_state <EOS>
				input_ids = context_ids + [self.bos_id] + target_ids + [self.eos_id]
				ignore_len = len(input_ids) - target_len - 1 # eos_id
				label_ids = [-100] * ignore_len + target_ids + [self.eos_id]
				assert len(input_ids) == len(label_ids)
				if len(input_ids) >= 1024: # handle over-length example
					input_ids = input_ids[-1023:]
					label_ids = label_ids[-1023:]
			else:
				input_ids = context_ids + [self.bos_id] # give bos for generate() api
				label_ids = None
				if len(input_ids) >= 1024:
					input_ids = input_ids[-1023:]

			assert len(input_ids) < 1024
			self.examples.append({
				'input_ids': input_ids, # list of ids
				'label_ids': label_ids, # list of ids
				'context': context,
				'turn_utt': turn_utt,
				'bs_dict': bs_dict,
				'bs_str': bs_str, 
				'example_id': example_id,
			})

		if self.data_type != 'demo':
			print('Data Statistics: {} -> {} examples'.format(self.data_type, len(self.examples)))
			print('Data Statistics: {} -> {} examples'.format(self.data_type, len(self.examples)), file=sys.stderr)


	def _pad(self, sentences, pad_id):

		max_len = max((map(len, sentences)))
		attention_mask = []
		sentences_pad = []
		for sent in sentences:
			pad_len = max_len - len(sent)
			sentences_pad.append( sent + [pad_id]*pad_len )
			attention_mask.append( [1]*len(sent) + [0]*pad_len)
		return sentences_pad, attention_mask


	def __len__(self): # required
		return len(self.examples)


	def __getitem__(self, index): # required
		return self.examples[index]


	def collate_fn(self, batch): # optional but useful

		input_ids = [example['input_ids'] for example in batch]
		input_ids, attention_mask = self._pad(input_ids, self.pad_id)
		input_ids, attention_mask = torch.tensor(input_ids).long().to(self.config.device), torch.tensor(attention_mask).long().to(self.config.device)

		if not self.generation:
			label_ids = [example['label_ids'] for example in batch]
			label_ids, _ = self._pad(label_ids, -100)
			label_ids = torch.tensor(label_ids).long().to(self.config.device)
		else:
			label_ids = None
		token_type_ids = None

		# store info for scoring
		context = [ex['context'] for ex in batch]
		bs_dict = [ex['bs_dict'] for ex in batch]
		bs_str = [ex['bs_str'] for ex in batch]
		example_id = [ex['example_id'] for ex in batch]
		turn_utt = [ex['turn_utt'] for ex in batch]

		return {
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'token_type_ids': token_type_ids,
			'label_ids': label_ids,
			'context': context,
			'bs_dict': bs_dict,
			'bs_str': bs_str,
			'example_id': example_id,
			'turn_utt': turn_utt
		}

