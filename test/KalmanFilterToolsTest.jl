
using KalmanFilterTools
using Test
using StaticArrays: SMatrix, SVector
using BenchmarkTools
state = KalmanFilterState(SVector(1.0, 1.0), SMatrix{2,2}(rand(2, 2)))
filter_object = KalmanFilter{Float64,2,2}(
    SMatrix{2,2}(rand(2, 2)),
    SMatrix{2,2}(rand(2, 2)),
    SMatrix{2,2}(rand(2, 2)),
    SMatrix{2,2}(rand(2, 2)),
)

@test filter(filter_object, SVector{2}(rand(2)), state) !== nothing

measurements = SVector{2, Float64}[SVector{2}(rand(2)) for x = 1:100]
@test filter(filter_object, measurements, state) |> length == 100

@benchmark filter(filter_object, x, state) setup = (x = [SVector{2}(rand(2)) for x = 1:100])
