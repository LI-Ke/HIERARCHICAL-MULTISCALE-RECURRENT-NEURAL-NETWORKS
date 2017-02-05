local HMRNNL2 = {}

-- Créé un timestep d'un HMRNNL2
function HMRNNL2.hmrnnl2(opt)
    local prev_h1 = nn.Identity()()
    local prev_h2 = nn.Identity()()
    local prev_z = nn.Identity()()
    local one = nn.Identity()()

    local lin_prev_h1 = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h1)
    local lin_prev_h2 = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h2)
    local add1 = nn.CAddTable()({lin_prev_h1, lin_prev_h2})
    local activ = nn.Tanh()(add1)

    local sub = nn.CSubTable()({one,prev_z})

    local mul1 = nn.CMulTable()({prev_z, activ})
    local mul2 = nn.CMulTable()({sub, prev_h2})

    local next_h2 = nn.CAddTable()({mul1, mul2})

    return nn.gModule({prev_h1, prev_h2, prev_z, one}, {next_h2})
end

return HMRNNL2
