using Pkg
Pkg.activate(".")
using Flux
using FileIO
using Images
using CUDA
using Zygote
using BSON

## IMPORT DATA

function importPictureData()

    basePath = "./Data/"
    subFolderPath = string("T",rand(1:6),"/")
    #digit = rand(30:39)
    digit = rand(30:34)
    currentPath = string(basePath,subFolderPath, digit, "/")
    digits = readdir(currentPath)
    chosenPicture = rand(digits)
    picturePath = string(currentPath, chosenPicture)
    digit -= 30
    image = Gray.(FileIO.load(picturePath))
    resizedImage = ImageTransformations.imresize(image, (32, 32), method=ImageTransformations.Linear())
    pixels = convert.(Float16, resizedImage)
    pixels = (pixels.-0.5).*2
    return reshape(pixels, 32 * 32)

end

batchSize = 128
latent_space = 64

discriminator = Chain(
    Flux.Dense(1024 => 512, x -> Flux.leakyrelu(x, 0.2f0), bias=true, init=Flux.glorot_normal),
    Flux.Dropout(0.3),
    Flux.Dense(512 => 256, x -> Flux.leakyrelu(x, 0.2f0), bias=true, init=Flux.glorot_normal),
    Flux.Dropout(0.3),
    Flux.Dense(256 => 64, x -> Flux.leakyrelu(x, 0.2f0), bias=true, init=Flux.glorot_normal),
    Flux.Dropout(0.3),
    Flux.Dense(64 => 1, Flux.sigmoid, bias=true, init=Flux.glorot_normal),
) |> gpu

generator = Chain(
    Flux.Dense(latent_space => 128, x -> Flux.leakyrelu(x, 0.2f0), bias=true, init=Flux.glorot_normal),
    Flux.Dense(128 => 256, x -> Flux.leakyrelu(x, 0.2f0), bias=true, init=Flux.glorot_normal),
    Flux.Dense(256 => 512, x -> Flux.leakyrelu(x, 0.2f0), bias=true, init=Flux.glorot_normal),
    Flux.Dense(512 => 1024, Flux.tanh, bias=true, init=Flux.glorot_normal),
) |> gpu


opt = Flux.Adam(2e-4)

function train_dscr!(discriminator,real_data,fake_data)
    allData = (hcat(real_data,fake_data)) |> gpu
    allTarget = permutedims([ones(Float16,128); zeros(Float16,128)]) |> gpu

    ps = Flux.params(discriminator)


    loss, pullback = Zygote.pullback(ps) do
        preds = discriminator(allData)
        loss = Flux.Losses.binarycrossentropy(preds, allTarget)
    end

    grads = pullback(1f0)
    Flux.update!(opt,Flux.params(discriminator),grads)
    return loss
end


function train_gen!(discriminator,generator)
    noise = randn(Float16,latent_space,batchSize)|> gpu
    ps = Flux.params(generator)
    testmode!(discriminator)
    loss, pullback = Zygote.pullback(ps) do
        preds = discriminator(generator(noise))
        loss = Flux.Losses.binarycrossentropy(preds, 1.0)
    end
    grads = pullback(1.0)
    Flux.update!(opt,Flux.params(generator),grads)
    Flux.trainmode!(discriminator, :auto)
    return loss
end

function train!(epoch,discriminator,generator)
    lossVector = zeros(epoch,2)
    for n = 1:epoch
    
        real_data = hcat(map(_ -> importPictureData(),1:128)...) |> gpu
        noise = randn(Float16,latent_space,batchSize) |> gpu
        fake_data = generator(noise) 

        loss_dscr = train_dscr!(discriminator,real_data,fake_data)

        loss_gen = train_gen!(discriminator,generator)
        println(string("Epoch:",n))
        println(string("Loss for generator epoch",n,":",loss_gen))
        lossVector[n,1] = loss_gen
        println(string("Loss for discriminator epoch",n,":",loss_dscr))
        lossVector[n,2] = loss_dscr        
    end
    return lossVector
end

function toPicture(pixels,i) 
    pixels = reshape(pixels,(32,32))
    Gray.(pixels)
    FileIO.save(string("test",i,".png"),pixels)
end
function generatePicture(i) 
    noise = randn(Float16,latent_space) |> gpu
    pixels = Array(generator(noise))
    pixels = reshape(pixels,(32,32))
    pixels = (pixels)./2 .+ 0.5
    Gray.(pixels)
    FileIO.save(string("test",i,".png"),pixels)
end
function saveNetwork(discriminator,generator)
    BSON.@save("discriminator.BSON",discriminator)
    BSON.@save("generator.BSON",generator)
end

#println(train!(1000,discriminator,generator))