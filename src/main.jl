using Flux
using FileIO
using Images
using Zygote
using BSON

## IMPORT DATA

function importPictureData()

    basePath = "./Data/"
    subFolderPath = string("T",rand(1:6),"/")
    digit = rand(30:39)
    currentPath = string(basePath,subFolderPath, digit, "/")
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

discriminator = Chain(
    Flux.Dense(64*64 => 1024, Flux.tanh, bias=true, init=Flux.glorot_normal),
    Flux.Dense(1024 => 256, Flux.tanh, bias=true, init=Flux.glorot_normal),
    Flux.Dense(256 => 64, Flux.tanh, bias=true, init=Flux.glorot_normal),
    Flux.Dense(64 => 1, Flux.sigmoid, bias=true, init=Flux.glorot_normal),
) |> gpu

generator = Chain(
    Flux.Dense(128 => 256, Flux.tanh, bias=true, init=Flux.glorot_normal),
    Flux.Dense(256 => 512, Flux.tanh, bias=true, init=Flux.glorot_normal),
    Flux.Dense(512 => 1024, Flux.tanh, bias=true, init=Flux.glorot_normal),
    Flux.Dense(1024 => 4096, Flux.sigmoid, bias=true, init=Flux.glorot_normal),
) |> gpu


opt = Flux.Adam(2e-4)

function train_dscr!(discriminator,real_data,fake_data)
    allData = (hcat(real_data,fake_data)) |> gpu
    allTarget = permutedims([ones(Float32,128); zeros(Float32,128)]) |> gpu

    ps = Flux.params(discriminator)


    loss, pullback = Zygote.pullback(ps) do
        preds = discriminator(allData)
        loss = Flux.Losses.mse(preds, allTarget)
    end

    grads = pullback(1f0)
    Flux.update!(opt,Flux.params(discriminator),grads)
    return loss
end


function train_gen!(discriminator,generator)
    noise = rand(Float32,128,batchSize)|> gpu
    ps = Flux.params(generator)
    
    loss, pullback = Zygote.pullback(ps) do
        preds = discriminator(generator(noise))
        loss = Flux.Losses.mse(preds, 1.0)
    end
    grads = pullback(1.0)
    Flux.update!(opt,Flux.params(generator),grads)
    return loss
end

function train!(epoch,discriminator,generator)
    lossVector = zeros(epoch,2)
    for n = 1:epoch
    
        real_data = hcat(map(_ -> importPictureData(),1:128)...) |> gpu
        noise = rand(Float32,128,batchSize) |> gpu
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
    pixels = reshape(pixels,(64,64))
    Gray.(pixels)
    FileIO.save(string("test",i,".png"),pixels)
end
function saveNetwork(discriminator,generator)
    BSON.@save("discriminator.BSON",discriminator)
    BSON.@save("generator.BSON",generator)
end

println(train!(1,discriminator,generator))