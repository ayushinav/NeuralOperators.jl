@inline function __project(b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3},
        additional::Nothing) where {T1, T2}
    # b : p x nb
    # t : p x N x nb
    o = zeros(eltype(b), size(t)[2:end]...) # N x nb
    for L in indices((o, b, t), (2, 2, 3))
        batched_mat_vec_mul!(view(o, :, L), view(b, :, L), view(t, :, :, L))
    end

    return o
end

@inline function __project(b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3},
        additional::Nothing) where {T1, T2}
    # b : p x u x nb
    # t : p x N x nb
    o = zeros(eltype(b), size(b, 2), size(t)[2:end]...) # u x N x nb

    for L in indices((o, b, t), 3)
        batched_mat_mul!(view(o, :, :, L), adjoint(view(b, :, :, L)), view(t, :, :, L))
    end

    return o
end

@inline function __project(b::AbstractArray{T1, N}, t::AbstractArray{T2, 3},
        additional::Nothing) where {T1, T2, N}
    # b : p x u_size x nb
    # t : p x N x nb

    u_size = size(b)[2:(end - 1)]
    o = zeros(eltype(b), u_size..., size(t)[2:end]...) # u_size x N x nb
    u_slice = fill(:, length(u_size) + 1)

    for L in indices(t, 3)
        batched_mat_mul!(reshape(view(o, u_slice..., L), :, size(t, 2)),
            reshape(view(b, u_slice..., L), size(t, 1), :), view(t, :, :, L))
    end

    return o # u_size x N x nb
end

function batched_mat_vec_mul!(o::AbstractArray{T1, 1}, b::AbstractArray{T2, 1},
        t::AbstractArray{T3, 2}) where {T1, T2, T3}
    @turbo for I in indices((o, t), (1, 2))
        C = zero(T1)
        for J in indices((b, t), (1, 1))
            C = C + b[J] * t[J, I]
        end
        o[I] = C
    end
end

function batched_mat_mul!(o::AbstractArray{T1, 2}, b::AbstractArray{T2, 2},
        t::AbstractArray{T3, 2}) where {T1, T2, T3}
    @turbo for J in indices((o, b), (1, 1))
        for I in indices((o, t), (2, 2))
            C = zero(T1)
            for K in indices((t, b), (1, 2))
                C = C + b[J, K] * t[K, I]
            end
            o[J, I] = C
        end
    end
end

@inline function __project(
        b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3}, additional::T) where {T1, T2, T}
    # b : p x nb
    # t : p x N x nb
    b_ = reshape(b, size(b, 1), 1, size(b, 2)) # p x 1 x nb
    return additional(b_ .* t) # p x N x nb => out_dims x N x nb
end

@inline function __project(
        b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3}, additional::T) where {T1, T2, T}
    # b : p x u x nb
    # t : p x N x nb

    if size(b, 2) == 1 || size(t, 2) == 1
        return additional(b .* t) # p x N x nb => out_dims x N x nb
    else
        b_ = reshape(b, size(b)[1:2]..., 1, size(b, 3)) # p x u x 1 x nb
        t_ = reshape(t, size(t, 1), 1, size(t)[2:end]...) # p x 1 x N x nb

        return additional(b_ .* t_) # p x u x N x nb => out_size x N x nb
    end
end

@inline function __project(b::AbstractArray{T1, N}, t::AbstractArray{T2, 3},
        additional::T) where {T1, T2, N, T}
    # b : p x u_size x nb
    # t : p x N x nb
    u_size = size(b)[2:(end - 1)]

    b_ = reshape(b, size(b, 1), u_size..., 1, size(b)[end])
    # p x u_size x 1 x nb

    t_ = reshape(t, size(t, 1), ones(eltype(u_size), length(u_size))..., size(t)[2:end]...)
    # p x (1,1,1...) x N x nb

    return additional(b_ .* t_) # p x u_size x N x nb => out_size x N x nb
end
