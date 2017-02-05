require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'

local timer = torch.Timer()
local CharLMMinibatchLoader = require 'data.CharLMMinibatchLoader'
--local LSTM = require 'LSTM'
local HMRNNL1 = require 'HMRNNL1'
local HMRNNL2 = require 'HMRNNL2'
local Z = require 'Z'
require 'Embedding'
local model_utils=require 'model_utils'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a Hierarchical Multiscale RNN language model')
cmd:text()
cmd:text('Options')
cmd:option('-vocabfile','vocab.t7','filename of the string->int table')
cmd:option('-datafile','data.t7','filename of the serialized torch ByteTensor to load')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-seq_length',1,'number of timesteps to unroll to')
cmd:option('-rnn_size',256,'size of LSTM internal state')
cmd:option('-max_epochs',2,'number of full passes through the training data')
cmd:option('-savefile','model','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-prev_z',0,'default state of the detection is not a space')
cmd:option('-t',1,'get z near 0 or 1')
cmd:text()

-- parse parametres d'entrée
local opt = cmd:parse(arg)

-- preparation :
torch.manualSeed(opt.seed)
opt.savefile = cmd:string(opt.savefile, opt,
    {save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true})
    .. '.t7'

local loader = CharLMMinibatchLoader.create(
        opt.datafile, opt.vocabfile, opt.batch_size, opt.seq_length)
local vocab_size = loader.vocab_size  -- the number of distinct characters
-- vocab_size = 80
-- definir un prototype de modele pour un pas de temps, puis le cloner

local protos = {}
protos.embed = Embedding(vocab_size, opt.rnn_size)
-- entrées du HMRNN : {x, prev_c, prev_h}, sorties: {next_c, next_h}
protos.hmrnnl1 = HMRNNL1.hmrnnl1(opt)
protos.hmrnnl2 = HMRNNL2.hmrnnl2(opt)
protos.z = Z.z(opt)
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
-- dimension = 80
protos.criterion = nn.ClassNLLCriterion() --train a classication problem with n classes. The input given through a forward() is expected to contain log-probabilities of each class

-- mettre les modules ci-dessus dans un tenseur de paramètres plat
local params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.hmrnnl1, protos.hmrnnl2, protos.z, protos.softmax)
params:uniform(-0.08, 0.08)

-- faire des clones, après applatissement, pour réallouer de la mémoire
local clones = {}
for name,proto in pairs(protos) do
    --print('cloning '..name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- Etat initial du HMRNN (initialement 0, mais l'état final est renvoyé à l'état initial quand on  applique la backpropagation dans le temps)
local initstate_h1 = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate_h2 = initstate_h1:clone()
local one = {[1]=torch.ones(opt.batch_size, opt.rnn_size)}
local un = {[1]=torch.ones(1)}
local T = {[1]=torch.ones(opt.batch_size, opt.rnn_size)}
-- Le message du backward de l'état final du HMRNN (dloss/dfinalstate) est 0, puisqu'il n'influence pas la prédiction
local dfinalstate_h1 = initstate_h1:clone()
local dfinalstate_h2 = initstate_h2:clone()

-- faire fwd/bwd and retourner loss, grad_params
function feval(params_)
    local unpack = unpack or table.unpack
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch()

    ------------------- passe forward -------------------
    local embeddings = {}            -- input embeddings
    local embedSpace = {}            -- input space for embeddings
    local hmrnn_h1 = {[0]=initstate_h1} -- valeurs de sortie du HMRNN (etat du premier niveau)
    local hmrnn_h2 = {[0]=initstate_h2} -- valeurs de sortie du HMRNN (etat du deuxieme niveau)
    local hmrnn_z = {[0]=initstate_h2} -- valeurs de sortie du HMRNN (prediction d'espace)

    local predictions = {}           -- softmax outputs
    local loss = 0
    local lz = 0
    local c= nn.MSECriterion()

    for t=1,opt.seq_length do
        embeddings[t] = clones.embed[t]:forward(x[{{}, t}])

        --print (hmrnn_z[t-1])
        hmrnn_h1[t] = clones.hmrnnl1[t]:forward{embeddings[t], hmrnn_h1[t-1], hmrnn_h2[t-1], hmrnn_z[t-1], one[1]}
        hmrnn_z[t] = clones.z[t]:forward({hmrnn_h1[t], T[1]})
        -- non augmentons T afin d'assuer que le softmax donne des valeurs les plus proches de 0 et de 1
        T[1] = T[1] * 1.001
        local space = {[1]=torch.ones(opt.batch_size, opt.rnn_size)}
        local notSpace = {[0]=initstate_h1}
        if x[{{}, t}] == 3 then      --pour apprendre l'espace
            lz = lz+c:forward(hmrnn_z[t],space[1])
        else
            lz = lz+c:forward(hmrnn_z[t],notSpace[0])
        end
        hmrnn_h2[t] = clones.hmrnnl2[t]:forward{hmrnn_h1[t], hmrnn_h2[t-1], hmrnn_z[t], one[1]}
        predictions[t] = clones.softmax[t]:forward(hmrnn_h1[t])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end

    ------------------ passe backward -------------------
    -- l'inverse complet des opération effectuées dans la passe forward
    local dembeddings = {}                              -- d loss / d input embeddings
    local dhmrnn_h1 = {[0]=dfinalstate_h1}              -- valeur de sortie du premier niveau du HMRNN
    for t=opt.seq_length,1,-1 do
        --print (t)
        -- backprop à travers loss, et softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        -- Deux cas pour dloss/dh_t:
        --   1. h_T est utilisé uniquement une fois, envoyé au softmax (mais pas pour le prochain pas du HMRNN)
        --   2. h_t est utilisé deux fois, pour le softmax et pour le prochain pas. Pour obeir à la règle de chaine multivariée, on les ajoute.
        if t == opt.seq_length then
            assert(dhmrnn_h1[t] == nil)
            dhmrnn_h1[t] = clones.softmax[t]:backward(hmrnn_h1[t], doutput_t)
        else
            --print (clones.softmax[t]:backward(hmrnn_h1[t], doutput_t))
            dhmrnn_h1[t]=clones.softmax[t]:backward(hmrnn_h1[t], doutput_t)
        end
        local space = {[1]=torch.ones(opt.batch_size, opt.rnn_size)}
        local notSpace = {[0]=initstate_h1}
        if x[{{}, t}] == 3 then      --pour apprendre l'espace
            gradZ = c:backward(hmrnn_z[t],space[1])
        else
            gradZ = c:backward(hmrnn_z[t],notSpace[0])
        end
        clones.z[t]:backward({hmrnn_h1[t],T[1]},gradZ)
        -- backprop through LSTM timestep
        dembeddings[t],d2,d3 = unpack(clones.hmrnnl1[t]:backward({embeddings[t], hmrnn_h1[t-1], hmrnn_h2[t-1], hmrnn_z[t-1], one[1]},dhmrnn_h1[t]))
        -- backprop through embeddings
        clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end

    ------------------------ autre ----------------------
    -- transfere de l'état final à l'état initial (backpopagation through time)
    initstate_h1:copy(hmrnn_h1[#hmrnn_h1])
    initstate_h2:copy(hmrnn_h2[#hmrnn_h2])

    -- clip le gradient élément par élément
    grad_params:clamp(-5, 5)

    --retourne lz, grad_params  --loss de z
    return loss, grad_params
end

-- optimisation
local losses = {}
local optim_state = {learningRate = 1e-1}
local iterations = opt.max_epochs * loader.nbatches
print (iterations)
for i = 1, iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end
    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f", i, loss[1]))
    end
    if #losses <=2000 then
        gnuplot.title('Loss of HM-RNN for first 2000 characters')
        gnuplot.plot(torch.Tensor(losses))
    end
end
print('Done in time (seconds): ', timer:time().real)
