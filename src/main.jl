using Flux
using FileIO
using Images


## IMPORT DATA
basePath = "./Data/T0/"
digit = rand(30:39)
currentPath = string(basePath, digit, "/")
digits = readdir(currentPath)

chosenPicture = rand(digits)
picturePath = string(currentPath, chosenPicture)
digit -= 30
image = Gray.(FileIO.load(picturePath))

resizedImage = ImageTransformations.imresize(image, (64, 64), method=ImageTransformations.Linear())

pixels = convert.(Float64, resizedImage)

w1 = rand(Float32,1024,64*64)
b1 = rand(Float32,1024)
w2 = rand(Float32,256,1024)
b2 = rand(Float32,256)
w3 = rand(Float32,64,256)
b3 = rand(Float32,64)
w4 = rand(Float32,1,64)
b4 = rand(Float32,1)


discriminator = Flux.Chain(
    Flux.Dense(w1,b1,Flux.tanh),
    Flux.Dense(w2,b2,Flux.tanh),
    Flux.Dense(w3,b3,Flux.tanh),
    Flux.Dense(w4,b4,Flux.tanh),
) |> gpu

w1 = (rand(Float32,256,128).-0.5).*2
b1 = (rand(Float32,256).-0.5).*2
w2 = (rand(Float32,512,256).-0.5).*2
b2 = (rand(Float32,512).-0.5).*2
w3 = (rand(Float32,1024,512).-0.5).*2
b3 = (rand(Float32,1024).-0.5).*2
w4 = (rand(Float32,64*64,1024).-0.5).*2
b4 = (rand(Float32,64*64).-0.5).*2

generator = Flux.Chain(
    Flux.Dense(w1,b1,Flux.tanh),
    Flux.Dense(w2,b2,Flux.tanh),
    Flux.Dense(w3,b3,Flux.tanh),
    Flux.Dense(w4,b4,Flux.tanh),
) |> gpu

