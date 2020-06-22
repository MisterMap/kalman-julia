module KalmanFilterTools
using LinearAlgebra
using BenchmarkTools
using StaticArrays: SMatrix, SVector

import Base.filter

struct KalmanFilter{T<:AbstractFloat,N,K}
    motion_model::SMatrix{N,N,T}
    motion_error_model::SMatrix{N,N,T}
    measurement_model::SMatrix{K,N,T}
    measurement_error_model::SMatrix{K,K,T}
end

struct KalmanFilterState{T<:AbstractFloat,N}
    state::SVector{N,T}
    covariance::SMatrix{N,N,T}
end

function Base.filter(
    self::KalmanFilter{T,N,K},
    measurement::SVector{K,T},
    previous_state::KalmanFilterState{T,N},
) where {T<:AbstractFloat,N,K}
    # Prediction
    state = self.motion_model * previous_state.state
    covariance =
        self.motion_model * previous_state.covariance * transpose(self.motion_model) +
        self.motion_error_model

    # Filtration
    error_covariance =
        self.measurement_model * covariance * transpose(self.measurement_model) +
        self.measurement_error_model
    kalman_gain = covariance * transpose(self.measurement_model) * inv(error_covariance)
    state = state + kalman_gain * (measurement - self.measurement_model * state)
    covariance = (I - kalman_gain * self.measurement_model) * covariance
    KalmanFilterState{T,N}(state, covariance)
end

function Base.filter(
    self::KalmanFilter{T,N,K},
    measurements::Array{SVector{K,T}},
    initial_state::KalmanFilterState{T,N},
) where {T<:AbstractFloat,N,K}
    current_state = initial_state
    states = KalmanFilterState{T,N}[]
    for measurement in measurements
        current_state = filter(self, measurement, current_state)
        push!(states, current_state)
    end
    states
end
export KalmanFilter, KalmanFilterState, filter
end
