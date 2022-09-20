using Flux
using FileIO
using Images


## IMPORT DATA

function importPictureData()

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
    return reshape(pixels, 64 * 64)

end

batchSize = 128

w1 = randn(Float32, 1024, 64 * 64)|> gpu
b1 = randn(Float32, 1024)|> gpu
w2 = randn(Float32, 256, 1024)|> gpu
b2 = randn(Float32, 256)|> gpu
w3 = randn(Float32, 64, 256)|> gpu
b3 = randn(Float32, 64)|> gpu
w4 = randn(Float32, 1, 64)|> gpu
b4 = randn(Float32, 1)|> gpu


discriminator = Flux.Chain(
    Flux.Dense(w1, b1, Flux.tanh),
    Flux.Dense(w2, b2, Flux.tanh),
    Flux.Dense(w3, b3, Flux.tanh),
    Flux.Dense(w4, b4, Flux.tanh),
) |> gpu


w1 = randn(Float32, 256, 128) |> gpu
b1 = randn(Float32, 256)|> gpu
w2 = randn(Float32, 512, 256)|> gpu
b2 = randn(Float32, 512)|> gpu
w3 = randn(Float32, 1024, 512)|> gpu
b3 = randn(Float32, 1024)|> gpu
w4 = randn(Float32, 64 * 64, 1024)|> gpu
b4 = randn(Float32, 64 * 64)|> gpu

generator = Flux.Chain(
    Flux.Dense(w1, b1, Flux.tanh),
    Flux.Dense(w2, b2, Flux.tanh),
    Flux.Dense(w3, b3, Flux.tanh),
    Flux.Dense(w4, b4, Flux.tanh),
) |> gpu


opt = Descent()
loss(x, y) = sum((x .- y) .^ 2) |> gpu


function train_dscr!(discriminator,real_data,fake_data)
    allData = vcat(real_data,fake_data) |> gpu

    ps = Flux.params(discriminator)

    for i = 1:256
        currentData = [(allData[1+(i-1)*64*64:i*64*64],i<129 ? 1 : 0 )]|> gpu
        Flux.train!(loss,ps,currentData,opt) |> gpu
    end
end


function train_gen!(discriminator,generator)
    noise = randn(Float32,128,batchSize)|> gpu
    ps = Flux.params(generator)
    for i = 1:128
        currentData = [(discriminator(generator(noise[:,i])),1)] |> gpu
        Flux.train!(loss,ps,currentData,opt)
      end
end

function train!(epoch,discriminator,generator)
    for n = 1:epoch
    
        real_data = reshape(hcat(map(_ -> importPictureData(),1:128)...),128*64*64) |> gpu
        #println(real_data)
        noise = randn(Float32,128,batchSize)|> gpu
        fake_data = reshape(generator(noise),128*64*64)

        train_dscr!(discriminator,real_data,fake_data)

        train_gen!(discriminator,generator)
        println(n)
    end
end