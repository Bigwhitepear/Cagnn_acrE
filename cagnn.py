from helper import *
from create_batch import Corpus
from layers import Cross_Att
from cagnn_encoder import Latent_Learning
from torch.autograd import Variable

CUDA = torch.cuda.is_available()

def save_model(model, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained.pth"))
    print("Done saving Model")

def load_encoder_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, \
        unique_entities_train = build_data(args.data, is_unweigted=False, directed=True)

    relation2id.update({rel + '_reverse': idx + len(relation2id) for rel, idx in relation2id.items()})

    entity_embeddings = np.random.randn(
        len(entity2id), args.init_dim)
    #逆关系也需要进行更新
    relation_embeddings = np.random.randn(
        len(relation2id), args.init_dim)
    print("Initialised relations and entities randomly")
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_cagnn, args.valid_invalid_ratio_cagnn, unique_entities_train)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


def batch_gat_loss(args, gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_cagnn) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_cagnn), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    ##使用点乘的方式计算hr + rr - tr
    # x = torch.mul(source_embeds, relation_embeds) - torch.mul(tail_embeds, relation_embeds)
    # x = torch.mul(source_embeds, relation_embeds) + torch.mul(relation_embeds, relation_embeds) \
    #     - torch.mul(tail_embeds, relation_embeds)
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    ##使用点乘的方式计算
    # x = torch.mul(source_embeds, relation_embeds) - torch.mul(tail_embeds, relation_embeds)
    # x = torch.mul(source_embeds, relation_embeds) + torch.mul(relation_embeds, relation_embeds) \
    #     - torch.mul(tail_embeds, relation_embeds)
    neg_norm = torch.norm(x, p=1, dim=1)

    # y = -torch.ones(int(args.valid_invalid_ratio_cagnn) * len_pos_triples).cuda()
    y = -torch.ones(int(args.valid_invalid_ratio_cagnn) * len_pos_triples)

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss
def train_cagnn(args, Corpus_, entity_embeddings, relation_embeddings):
    print("Initial entity dimensions {} , relation dimensions {}".format(
        entity_embeddings.size(), relation_embeddings.size()))
    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.gat_layers))
    model_gat = Latent_Learning(entity_embeddings, relation_embeddings, args.out_dim, args.out_dim,
                                args.gat_drop, args.gat_alpha, args.gat_layers)
    if CUDA:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.gat_wc)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = torch.nn.MarginRankingLoss(margin=args.margin)

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epoch_c))
    min_loss = 99.9
    for epoch in range(args.epoch_c):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_cagnn == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_cagnn
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_cagnn) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, update_rel=True)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                args, gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
        # if min_loss > (sum(epoch_loss) / len(epoch_loss)):
        #     min_loss = (sum(epoch_loss) / len(epoch_loss))
        #     save_model(model_gat, args.output_folder)
    save_model(model_gat, args.outfolder)