local HMRNNL1 = {}

-- Créé un timestep d'un HMRNNL1
function HMRNNL1.hmrnnl1(opt)
    local x = nn.Identity()()
    local prev_h1 = nn.Identity()()
    local prev_h2 = nn.Identity()()
    local prev_z = nn.Identity()()
    local one = nn.Identity()()

    local lin_prev_h1 = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h1)
    local lin_prev_h2 = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h2)
    local lin_x = nn.Linear(opt.rnn_size, opt.rnn_size)(x)

    local sub = nn.CSubTable()({one,prev_z})

    local mul1 = nn.CMulTable()({sub, lin_prev_h1})
    local mul2 = nn.CMulTable()({prev_z, lin_prev_h2})

    local add1 = nn.CAddTable()({mul1, mul2})
    local add2 = nn.CAddTable()({add1, lin_x})
    local next_h1 = nn.Tanh()(add2)
    return nn.gModule({x, prev_h1, prev_h2, prev_z, one}, {next_h1})
end

return HMRNNL1
