require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'

local unpack = unpack or table.unpack

cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-vocabfile','vocab.t7','filename of the string->int table')
cmd:option('-model','model.t7','contains just the protos table, and nothing else')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',true,'false to use max at each timestep, true to sample at each timestep')
cmd:option('-text',"hello my name is ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample. set to a space " " to disable')
cmd:option('-length',50,'number of characters to sample')
cmd:text()

-- parser les paramètres d'entrée du système
opt = cmd:parse(arg)

-- preparation and chargement
torch.manualSeed(opt.seed)

local vocab = torch.load(opt.vocabfile)
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- chargement du modèle
protos = torch.load(opt.model)
opt.rnn_size = protos.embed.weight:size(2)

--protos.embed = Embedding(vocab_size, opt.rnn_size)
---- paramètres du HMRNN : {x, prev_c, prev_h}, sorties: {next_c, next_h}


-- Etat initial du HMRNN, NB nous utilisons des minibatches de TAILLE UN ici
local prev_h1 = torch.zeros(1, opt.rnn_size)
local prev_h2 = prev_h1:clone()
local prev_z = prev_h1:clone()
local one = torch.ones(1,opt.rnn_size)
local T = torch.ones(1,opt.rnn_size)

local seed_text = opt.text
local prev_char

-- effectuer quelques pas
for c in seed_text:gmatch'.' do

    prev_char = torch.Tensor{vocab[c]}

    local embedding = protos.embed:forward(prev_char)
    local next_h1 = protos.hmrnnl1:forward{embedding, prev_h1, prev_h2, prev_z, one}
    local next_z = protos.z:forward{next_h1, T}
    T = T * 1.001
    local next_h2 = protos.hmrnnl2:forward{next_h1, prev_h2, next_z, one}
    prev_h1:copy(next_h1)
    prev_h2:copy(next_h2)
    prev_z:copy(next_z)
end
--print(prev_char)
-- maintement nous commencons à echantillonner et faire le argmax
for i=1, opt.length do
    -- embedding et HMRNN
    local embedding = protos.embed:forward(prev_char)
    local next_h1 = protos.hmrnnl1:forward{embedding, prev_h1, prev_h2, prev_z, one}
    local next_z = protos.z:forward{next_h1, T}
    T = T * 1.001
    local next_h2 = protos.hmrnnl2:forward{next_h1, prev_h2, next_z, one}
    prev_h1:copy(next_h1)
    prev_h2:copy(next_h2)
    prev_z:copy(next_z)
    -- softmax du pas precedent
    local log_probs = protos.softmax:forward(next_h1)

    if not opt.sample then
        -- use argmax
        local _, prev_char_ = log_probs:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        local probs = torch.exp(log_probs):squeeze()
        prev_char = torch.multinomial(probs, 1):resize(1)
    end

    --print('OUTPUT:', ivocab[prev_char[1]])
    io.write(ivocab[prev_char[1]])
end
io.write('\n') io.flush()
