local Z = {}

function Z.z(opt)
    local next_h1 = nn.Identity()()
    local T = nn.Identity()()
    local lin_next_h1 = nn.Linear(opt.rnn_size, opt.rnn_size)(next_h1)
    local mul = nn.CMulTable()({T, lin_next_h1})
    local z = nn.Sigmoid()(mul)
    return nn.gModule({next_h1, T}, {z})
end

return Z
