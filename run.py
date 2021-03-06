from helper import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from data_loader import *
from model import *
from cagnn import train_cagnn, load_encoder_data
from cagnn_encoder import Latent_Learning

class Main(object):

	def __init__(self, params, Corpus_, entity_embed, relation_embed):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		self.p = params
		self.entity_embed = entity_embed
		self.relation_embed = relation_embed
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data(Corpus_.entity2id, Corpus_.relation2id)
		self.model        = self.add_model()
		self.optimizer    = self.add_optimizer(self.model.parameters())

	def load_data(self, ent2id, rel2id):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset  (FB15k-237, WN18RR, YAGO3-10)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""

		# ent_set, rel_set = OrderedSet(), OrderedSet()
		# for split in ['train', 'test', 'valid']:
		# 	for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
		# 		sub, rel, obj = map(str.lower, line.strip().split('\t'))
		# 		ent_set.add(sub)
		# 		rel_set.add(rel)
		# 		ent_set.add(obj)
		#
		# self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		# self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		# self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for rel, idx in self.rel2id.items()})

		self.ent2id = ent2id
		self.rel2id = rel2id
		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim


		self.data	= ddict(list)
		sr2o		= ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.data, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

		self.triples = ddict(list)

		if self.p.train_strategy == 'one_to_n':
			for (sub, rel), obj in self.sr2o.items():
				self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
		else:
			for sub, rel, obj in self.data['train']:
				rel_inv		= rel + self.p.num_rel
				sub_samp	= len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
				sub_samp	= np.sqrt(1/sub_samp)

				self.triples['train'].append({'triple':(sub, rel, obj),     'label': self.sr2o[(sub, rel)],     'sub_samp': sub_samp})
				self.triples['train'].append({'triple':(obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)], 'sub_samp': sub_samp})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train'		:   get_data_loader(TrainDataset, 'train', 	self.p.batch_size),
			'valid_head'	:   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail'	:   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head'	:   get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail'	:   get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}


	def add_model(self):
		"""
		Creates the computational graph

		Parameters
		----------
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		print("Defining model")
		model_encoder = Latent_Learning(self.entity_embed, self.relation_embed, self.p.out_dim, self.p.out_dim, \
										self.p.gat_drop, self.p.gat_alpha, self.p.gat_layers)
		model_encoder.to(self.device)
		print("Only Conv model trained")
		model = AcrE(self.p)
		model.to(self.device)

		model_encoder.load_state_dict(torch.load(
        '{}/trained.pth'.format(self.p.outfolder)))

		model.ent_embed = model_encoder.final_entity_embeddings
		model.rel_embed = model_encoder.final_relation_embeddings

		params = [value.numel() for value in model.parameters()]
		print(params)
		print(f'model_para : {np.sum(params)}')


		return model

	def add_optimizer(self, parameters):
		"""
		Creates an optimizer for training the parameters

		Parameters
		----------
		parameters:         The parameters of the model
		
		Returns
		-------
		Returns an optimizer for learning the parameters of the model
		
		"""
		if self.p.opt == 'adam': return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
		else:			 return torch.optim.SGD(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'valid' or 'test' split

		
		Returns
		-------
		triples:	The triples used for this split
		labels:		The label for each triple
		"""
		if split == 'train':
			if self.p.train_strategy == 'one_to_x':
				triple, label, neg_ent, sub_samp = [ _.to(self.device) for _ in batch]
				return triple[:, 0], triple[:, 1], triple[:, 2], label, neg_ent, sub_samp
			else:
				triple, label = [ _.to(self.device) for _ in batch]
				return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

		Parameters
		----------
		save_path: path where the model is saved
		
		Returns
		-------
		"""
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		"""
		Function to load a saved model

		Parameters
		----------
		load_path: path to the saved model
		
		Returns
		-------
		"""
		state				= torch.load(load_path)
		state_dict			= state['state_dict']
		self.best_val_mrr 		= state['best_val']['mrr']
		self.best_val 			= state['best_val']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch=0):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""		
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		self.logger.info(
			'[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mr'],
																				 results['right_mr'], results['mr']))
		self.logger.info(
			'[Epoch {} {}]: hits@10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_hits@10'],
																				 results['right_hits@10'], results['hits@10']))
		self.logger.info(
			'[Epoch {} {}]: hits@3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_hits@3'],
																		 results['right_hits@3'], results['hits@3']))
		self.logger.info(
			'[Epoch {} {}]: hits@1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split,
																					results['left_hits@1'],
																					results['right_hits@1'],
																					results['hits@1']))
		return results

	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel, None, 'one_to_n')
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

		return results


	def run_epoch(self, epoch):
		"""
		Function to run one epoch of training

		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()

			sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')

			pred	= self.model.forward(sub, rel, neg_ent, self.p.train_strategy)
			loss	= self.model.loss(pred, label, sub_samp)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}, \t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss

	def fit(self):
		"""
		Function to run training and evaluation of model

		Parameters
		----------
		
		Returns
		-------
		"""
		self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0.
		val_mrr = 0
		save_path = os.path.join('./torch_saved', self.p.name)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', verbose=1, patience=15)
		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		for epoch in range(self.p.max_epochs):
			train_loss	= self.run_epoch(epoch)
			val_results	= self.evaluate('valid', epoch)
			scheduler.step(val_results['mrr'])
			if val_results['mrr'] > self.best_val_mrr:
				self.best_val		= val_results
				self.best_val_mrr	= val_results['mrr']
				self.best_epoch		= epoch
				self.save_model(save_path)
			self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5}, \n\n\n'.format(epoch, train_loss, self.best_val_mrr))

		
		# Restoring model corresponding to the best validation performance and evaluation on test data
		self.logger.info('Loading best model, evaluating on test data')
		self.load_model(save_path)		
		self.evaluate('test')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Dataset and Experiment name
	parser.add_argument('--data',           dest="data",         default='WN18RR/',            		help='Dataset to use for the experiment')
	parser.add_argument("--name",            			default='testrun_'+str(uuid.uuid4())[:8],	help='Name of the experiment')

	# Training parameters
	parser.add_argument("--gpu",		type=str,               default='0',					help='GPU to use, set -1 for CPU')
	parser.add_argument("--train_strategy", type=str,               default='one_to_x',				help='Training strategy to use')
	parser.add_argument("--opt", 		type=str,               default='adam',					help='Optimizer to use for training')
	parser.add_argument('--neg_num',        dest="neg_num",         default=1000,    	type=int,       	help='Number of negative samples to use for loss calculation')
	parser.add_argument('--batch',          dest="batch_size",      default=128,    	type=int,       	help='Batch size')
	parser.add_argument("--l2",		type=float,             default=0.0,					help='L2 regularization')
	parser.add_argument("--lr",		type=float,             default=0.0001,					help='Learning Rate')
	parser.add_argument("--epoch",		dest='max_epochs', 	default=500,		type=int,  		help='Maximum number of epochs')
	parser.add_argument("--num_workers",	type=int,               default=10,                      		help='Maximum number of workers used in DataLoader')
	parser.add_argument('--seed',           dest="seed",            default=42,   		type=int,       	help='Seed to reproduce results')
	parser.add_argument('--restore',   	dest="restore",       	action='store_true',            		help='Restore from the previously saved model')

	# Model parameters
	parser.add_argument("--lbl_smooth",     dest='lbl_smooth',	default=0.1,		type=float,		help='Label smoothing for true labels')
	parser.add_argument("--embed_dim",	type=int,              	default=None,                   		help='Embedding dimension for entity and relation, ignored if k_h and k_w are set')
	parser.add_argument('--bias',      	dest="bias",          	action='store_true',            		help='Whether to use bias in the model')
	parser.add_argument('--k_w',	  	dest="k_w", 		default=10,   		type=int, 		help='Width of the reshaped matrix')
	parser.add_argument('--k_h',	  	dest="k_h", 		default=20,   		type=int, 		help='Height of the reshaped matrix')
	parser.add_argument('--hid_drop',  	dest="hid_drop",      	default=0.5,    	type=float,     	help='Dropout for Hidden layer')
	parser.add_argument('--feat_drop', 	dest="feat_drop",     	default=0.5,    	type=float,     	help='Dropout for Feature')
	parser.add_argument('--inp_drop',  	dest="inp_drop",      	default=0.2,    	type=float,     	help='Dropout for Input layer')
	parser.add_argument('--channel',	dest="channel",	default=32,	type=int,	help='Number of out channel')
	parser.add_argument("--way",	type=str,	default='t',	help='Serial or Parallel')
	parser.add_argument("--first_atrous",	dest="first_atrous",	default=1,	type=int,	help="First layer expansion coefficient")
	parser.add_argument("--second_atrous", dest="second_atrous", default=2, type=int,	help="Second layer expansion coefficient")
	parser.add_argument("--third_atrous", dest="third_atrous", default=5, type=int,	help="Third layer expansion coefficient")

	parser.add_argument('--init_dim', dest="init_dim", default=50, type=int, help='Size of embeddings (if pretrained not used)')
	parser.add_argument('--out_dim', dest="out_dim", default=100, type=int, help='Entity output embedding dimensions')
	parser.add_argument('--gat_layers', dest="gat_layers", default=2, type=int, help='Multihead att CAGNN')
	parser.add_argument('--gat_drop', dest="gat_drop", default=0.3, type=float, help='Dropout probability for SpGAT layer')
	parser.add_argument('--gat_alpha', dest="gat_alpha", default=0.1,  type=float, help='Dropout probability for SpGAT layer')
	parser.add_argument('--gat_wc', dest="gat_wc", default=5e-6, type=float, help='L2 reglarization for cagnn')
	parser.add_argument('--margin', dest="margin", default=5, type=int, help='Size of embeddings (if pretrained not used)')
	parser.add_argument('--epoch_c', dest="epoch_c", default=1, type=int, help='Number of epochs in cagnn')
	parser.add_argument('--outfolder', dest="outfolder", default="./checkpoints/wn/", help="Folder name to save the models.")
	parser.add_argument('--batch_cagnn', dest="batch_cagnn", default=86835, type=int,
						help='Size of embeddings (if pretrained not used)')
	parser.add_argument('--neg_s_cagnn', dest="valid_invalid_ratio_cagnn", default=2, type=int,
						help='Size of embeddings (if pretrained not used)')


	# Logging parameters
	parser.add_argument('--logdir',    	dest="log_dir",       	default='./log/',               		help='Log directory')
	parser.add_argument('--config',    	dest="config_dir",    	default='./config/',            		help='Config directory')
	

	args = parser.parse_args()
	
	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	Corpus_, entity_embeddings, relation_embeddings = load_encoder_data(args)
	train_cagnn(args, Corpus_, entity_embeddings, relation_embeddings)

	model = Main(args, Corpus_, entity_embeddings, relation_embeddings)
	model.fit()
