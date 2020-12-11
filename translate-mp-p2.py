import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.dataset import Dataset
from utils.misc import save_config, validate_config
from utils.misc import get_memory_alloc, plot_alignment, check_device
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor
from utils.config import PAD, EOS
from modules.loss import NLLLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq

import logging
logging.basicConfig(level=logging.INFO)


def load_arguments(parser):

	""" Seq2seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, default='None', help='test src dir')
	parser.add_argument('--path_vocab_src', type=str, default='None', help='vocab src dir, not needed')
	parser.add_argument('--path_vocab_tgt', type=str, default='None', help='vocab tgt dir, not needed')
	parser.add_argument('--load', type=str, required=True, help='1st pass model load dir')
	parser.add_argument('--load_p2', type=str, required=True, help='2nd pass model load dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--seqrev', type=str, default='False', help='whether or not to reverse sequence')
	parser.add_argument('--use_type', type=str, default='word', help='word | char')

	return parser


def translate(test_set, model, model_p2, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			use_gpu: on gpu/cpu
	"""

	# reset batch_size:
	model.max_seq_len, model_p2.max_seq_len = max_seq_len, max_seq_len
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		model_p2.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):
				batch_items = evaliter.next()
				print(idx+1, len(evaliter))

				# load data
				src_ids = batch_items['srcid'][0]
				src_lengths = batch_items['srclen']
				tgt_ids = batch_items['tgtid'][0]
				tgt_lengths = batch_items['tgtlen']
				src_len = max(src_lengths)
				tgt_len = max(tgt_lengths)
				src_ids = src_ids[:,:src_len].to(device=device)
				tgt_ids = tgt_ids[:,:tgt_len].to(device=device)

				decoder_outputs_p1, decoder_hidden_p1, ret_dict_p1 = model(src=src_ids,
					src_lens=src_lengths, is_training=False,
					beam_width=beam_width, use_gpu=use_gpu)

				output_p1 = torch.stack(ret_dict_p1['sequence'], dim=1).squeeze(-1).to(device=device)
				output_p1_lens = [min((x==EOS).nonzero()).to(device='cpu')+1 if EOS in x else torch.tensor([len(x)]) for x in output_p1]
				output_p1 = output_p1[:,:int(max(output_p1_lens))]

				decoder_outputs, decoder_hidden, other = model_p2(src=src_ids,
					src_lens=src_lengths, is_training=False,
					beam_width=beam_width, use_gpu=use_gpu,
					output_p1=output_p1, output_p1_lens=output_p1_lens)

				# write to file
				seqlist = other['sequence']
				seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
				for i in range(len(seqwords)):
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '<spc>':
							words.append(' ')
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						if seqrev:
							words = words[::-1]
						if test_set.use_type == 'word':
							outline = ' '.join(words)
						elif test_set.use_type == 'char':
							outline = ''.join(words)
					f.write('{}\n'.format(outline))
					# import pdb; pdb.set_trace()

				sys.stdout.flush()


def translate_batch(test_set, model, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# reset batch_size:
	model.max_seq_len = max_seq_len
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	print('batch_size: {}'.format(test_set.batch_size))

	model.eval()
	with torch.no_grad():

		# select batch
		n_total = len(evaliter)
		iter_idx = 0
		per_iter = 500 # 1892809 lines; 100/batch; 38 iterations
		st = iter_idx * per_iter
		ed = min((iter_idx + 1) * per_iter, n_total)
		f = open(os.path.join(test_path_out, '{:04d}.txt'.format(iter_idx)), 'w', encoding="utf8")

		for idx in range(len(evaliter)):
			batch_items = evaliter.next()
			if idx < st:
				continue
			elif idx >= ed:
				break
			print(idx, ed)

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids[:,:tgt_len].to(device=device)

			decoder_outputs, decoder_hidden, other = model(src=src_ids, src_lens=src_lengths,
				is_training=False, beam_width=beam_width, use_gpu=use_gpu)

			# memory usage
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))

			# write to file
			seqlist = other['sequence']
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			for i in range(len(seqwords)):
				if src_lengths[i] == 0:
					continue
				words = []
				for word in seqwords[i]:
					if word == '<pad>':
						continue
					elif word == '<spc>':
						words.append(' ')
					elif word == '</s>':
						break
					else:
						words.append(word)
				if len(words) == 0:
					outline = ''
				else:
					if seqrev:
						words = words[::-1]
					if test_set.use_type == 'word':
						outline = ' '.join(words)
					elif test_set.use_type == 'char':
						outline = ''.join(words)
				f.write('{}\n'.format(outline))

			sys.stdout.flush()


def att_plot(test_set, model, model_p2, plot_path, use_gpu, max_seq_len, beam_width, device):

	"""
		generate attention alignment plots
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu
			max_seq_len
		Returns:

	"""

	beam_width = 1
	model.max_seq_len, model_p2.max_seq_len = max_seq_len, max_seq_len
	print('max seq len {}'.format(model.max_seq_len))


	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)

	# start eval
	model.eval()
	model_p2.eval()
	count = 0
	with torch.no_grad():
		for idx in range(len(evaliter)):
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids[:,:tgt_len].to(device=device)

			# decoder_outputs_p1, decoder_hidden_p1, other_p1 = model(src_ids,
			# 	src_lens=src_lengths, tgt=tgt_ids,
			# 	is_training=False, beam_width=beam_width)

			decoder_outputs_p1, decoder_hidden_p1, other_p1 = model(src_ids,
				src_lens=src_lengths, is_training=False, 
				beam_width=beam_width, use_gpu=use_gpu)

			output_p1 = torch.stack(other_p1['sequence'], dim=1).squeeze(-1).to(device=device)
			output_p1_lens = [min((x==EOS).nonzero()).to(device='cpu')+1 if EOS in x else torch.tensor([len(x)]) for x in output_p1]
			output_p1 = output_p1[:,:int(max(output_p1_lens))]

			decoder_outputs, decoder_hidden, other = model_p2(src=src_ids,
				src_lens=src_lengths, is_training=False,
				beam_width=beam_width, use_gpu=use_gpu,
				output_p1=output_p1, output_p1_lens=output_p1_lens)

			# Evaluation
			# default batch_size = 1
			# attention: 31 * [1 x 1 x 32]
			# 	(tgt_len(query_len) * [ batch_size x 1 x src_len(key_len)]
			attention_p1 = other_p1['attention_score']
			attention = other['attention_score']
			attention_o1 = other['attention_score_o1']
			seqlist_p1 = other_p1['sequence'] # traverse over time not batch
			seqlist = other['sequence'] # traverse over time not batch
			bsize = test_set.batch_size
			max_seq = test_set.max_seq_len
			vocab_size = len(test_set.tgt_word2id)

			# Print sentence by sentence
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)
			seqwords_p1 = _convert_to_words(seqlist_p1, test_set.tgt_id2word)
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)

			n_q_p1 = len(attention_p1)
			n_k_p1 = attention_p1[0].size(2)
			b_size_p1 =  attention_p1[0].size(0)
			att_score_p1 = torch.empty(n_q_p1, n_k_p1, dtype=torch.float)
			n_q = len(attention)
			n_k = attention[0].size(2)
			b_size =  attention[0].size(0)
			att_score = torch.empty(n_q, n_k, dtype=torch.float)
			# att_score = np.empty([n_q, n_k])
			n_q_o1 = len(attention_o1)
			n_k_o1 = attention_o1[0].size(2)
			b_size_o1 =  attention_o1[0].size(0)
			att_score_o1 = torch.empty(n_q_o1, n_k_o1, dtype=torch.float)

			for i in range(len(seqwords)): # loop over sentences
				outline_src = ' '.join(srcwords[i])
				outline_ref = ' '.join(refwords[i])
				outline_gen_p1 = ' '.join(seqwords_p1[i])
				outline_gen = ' '.join(seqwords[i])
				print('SRC: {}'.format(outline_src))
				print('REF: {}'.format(outline_ref))
				print('GEN_P1: {}'.format(outline_gen_p1))
				print('GEN_P2: {}'.format(outline_gen))
				for j in range(len(attention_p1)):
					# record att scores
					att_score_p1[j] = attention_p1[j][i]
				for j in range(len(attention)):
					# i: idx of batch
					# j: idx of query
					gen = seqwords[i][j]
					# if j==len(refwords[i]):
					# 	print(src_ids.size(), tgt_ids.size(), output_p1.size(), len(decoder_outputs), len(seqwords[i]))
					# 	import pdb; pdb.set_trace()
					# ref = refwords[i][j]
					att = attention[j][i]
					# record att scores
					att_score[j] = att
				for j in range(len(attention_o1)):
					# record att scores
					att_score_o1[j] = attention_o1[j][i]

				# plotting
				loc_eos_k = srcwords[i].index('</s>') + 1
				loc_eos_q_p1 = seqwords_p1[i].index('</s>') + 1 if '</s>' in seqwords_p1[i] else len(seqwords_p1[i])
				loc_eos_q = seqwords[i].index('</s>') + 1 if '</s>' in seqwords[i] else len(seqwords[i])
				loc_eos_ref = refwords[i].index('</s>') + 1
				print('eos_k: {}, eos_q_p1: {}, eos_q: {}'.format(loc_eos_k, loc_eos_q_p1, loc_eos_q))
				att_score_trim_p1 = att_score_p1[:loc_eos_q_p1, :loc_eos_k]
				att_score_trim = att_score[:loc_eos_q, :loc_eos_k]
				att_score_trim_o1 = att_score_o1[:loc_eos_q, :loc_eos_q_p1]
				# each row (each query) sum up to 1
				# print(att_score_trim)
				print('\n')
				# import pdb; pdb.set_trace()

				choice = input('Plot or not ? - y/n\n')
				if choice:
					if choice.lower()[0] == 'y':
						print('plotting ...')
						src = srcwords[i][:loc_eos_k]
						hyp_p1 = seqwords_p1[i][:loc_eos_q_p1]
						hyp = seqwords[i][:loc_eos_q]
						ref = refwords[i][:loc_eos_ref]
						# x-axis: src; y-axis: hyp
						# plot_alignment(att_score_trim.numpy(),
						# 	plot_dir, src=src, hyp=hyp, ref=ref)
						plot_dir = os.path.join(plot_path, 'attn_p1_{}.png'.format(count))
						plot_alignment(att_score_trim_p1.numpy(), plot_dir, src=src, hyp=hyp_p1, ref=None) # no ref
						plot_dir = os.path.join(plot_path, 'attn_p2_{}.png'.format(count))
						plot_alignment(att_score_trim.numpy(), plot_dir, src=src, hyp=hyp, ref=None) # no ref
						plot_dir = os.path.join(plot_path, 'attn_p2_{}_correction.png'.format(count))
						plot_alignment(att_score_trim_o1.numpy(), plot_dir, src=hyp_p1, hyp=hyp, ref=None) # no ref
						count += 1
						input('Press enter to continue ...')


def translate_tf(test_set, model, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			use_gpu: on gpu/cpu
	"""

	# reset batch_size:
	model.max_seq_len = max_seq_len
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):
				batch_items = evaliter.next()
				print(idx+1, len(evaliter))

				# load data
				src_ids = batch_items['srcid'][0]
				src_lengths = batch_items['srclen']
				tgt_ids = batch_items['tgtid'][0]
				tgt_lengths = batch_items['tgtlen']
				src_len = max(src_lengths)
				tgt_len = max(tgt_lengths)
				src_ids = src_ids[:,:src_len].to(device=device)
				tgt_ids = tgt_ids[:,:tgt_len].to(device=device)

				# import pdb; pdb.set_trace()
				decoder_outputs, decoder_hidden, other = model(
					src=src_ids, src_lens=src_lengths, tgt=tgt_ids,
					is_training=True, teacher_forcing_ratio=1.0,
					beam_width=beam_width, use_gpu=use_gpu)

				# write to file
				seqlist = other['sequence']
				seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
				for i in range(len(seqwords)):
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '<spc>':
							words.append(' ')
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						if seqrev:
							words = words[::-1]
						if test_set.use_type == 'word':
							outline = ' '.join(words)
						elif test_set.use_type == 'char':
							outline = ''.join(words)
					f.write('{}\n'.format(outline))

				sys.stdout.flush()


def main():

	# load config
	parser = argparse.ArgumentParser(description='Seq2seq Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = config['test_path_tgt'] # dummy
	if type(test_path_tgt) == type(None):
		test_path_tgt = test_path_src
	test_path_out = config['test_path_out']
	load_dir = config['load']
	load_dir_p2 = config['load_p2']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	use_type = config['use_type']

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)
	config_save_dir = os.path.join(test_path_out, 'eval.cfg')
	save_config(config, config_save_dir)

	# set test mode: 1 = translate; 3 = plot
	MODE = config['eval_mode']
	if MODE == 3:
		max_seq_len = 32
		batch_size = 1
		beam_width = 1
		# use_gpu = False

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model = resume_checkpoint.model.to(device)
	latest_checkpoint_path = load_dir_p2
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model_p2 = resume_checkpoint.model.to(device)
	vocab_src = resume_checkpoint.input_vocab
	vocab_tgt = resume_checkpoint.output_vocab
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# load test_set
	test_set = Dataset(test_path_src, test_path_tgt,
						vocab_src_list=vocab_src, vocab_tgt_list=vocab_tgt,
						seqrev=seqrev,
						max_seq_len=max_seq_len,
						batch_size=batch_size,
						use_gpu=use_gpu,
						use_type=use_type)
	print('Test dir: {}'.format(test_path_src))
	print('Testset loaded')
	sys.stdout.flush()

	# run eval
	if MODE == 1: # FR
		translate(test_set, model, model_p2, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev)

	if MODE == 2:
		translate_batch(test_set, model, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev)

	elif MODE == 3:
		# plotting
		att_plot(test_set, model, model_p2, test_path_out, use_gpu,
			max_seq_len, beam_width, device)

	elif MODE == 4: # TF
		translate_tf(test_set, model, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev)


if __name__ == '__main__':
	main()
