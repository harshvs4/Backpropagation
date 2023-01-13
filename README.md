# MNIST Dataset

# Part 1

## Backpropagation on a Neural Network

![image](https://user-images.githubusercontent.com/63489899/212275539-e32cce6e-8b77-4336-a2b7-5419cd739905.png)

![image](https://user-images.githubusercontent.com/63489899/212283083-3b38be21-9faa-4219-b1eb-66dacc053906.png)

## Major Steps

     1. h1 = w1*i1 + w2*i2
        h2 = w3*i1 + w4*i2
        out_h1 = σ(h1) = 1/(1+exp(-h1))
        out_h2 = σ(h2)
        o1 = w5*out_h1 + w6*out_h2
        o2 = w7*out_h1 + w8*out_h2
        out_o1 = σ(o1)
        out_o1 = σ(o2)
        E_Total = E1+E2
        E1 = 1/2 * (t1 - out_o1)^2
        E2 = 1/2 * (t2 - out_o2)^2 
     2. ∂E_total/∂w5 = ∂(E1+E2)/∂w5
        ∂E_total/∂w5 = ∂E1/∂w5
        ∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂out_o1 * ∂out_o1/∂o1 * ∂o1/∂w5
        ∂E1/∂out_01= ∂(1/2 * (t1 - out_o1)^2)/∂out_o1 = (out_o1 - t1)
        ∂out_o1/∂01 = out_o1 * (1 - out_o1)
        ∂o1/∂w5 = out_h1
     3. ∂E_total/∂w5 = (out_o1 - t1) * out_o1*(1 - out_o1) * out_h1
        ∂E_total/∂w6 = (out_o1 - t1) * out_o1*(1 - out_o1) * out_h2
        ∂E_total/∂w7 = (out_o2 - t2) * out_o2*(1 - out_o2) * out_h1
        ∂E_total/∂w6 = (out_o1 - t1) * out_o1*(1 - out_o1) * out_h2
     4. ∂E1/∂out_h1 = (out_o1 * t1) * out_o1 * (1 - out_01) * w5
        ∂E2/∂out_h1 = (out_o2 * t1) * out_o2 * (1 - out_02) * w7
        ∂E_total/∂out_h1 = (out_o1 * t1) * out_o1 * (1 - out_01) * w5 + (out_o2 * t1) * out_o2 * (1 - out_02) * w7
        ∂E_total/∂out_h1 = (out_o1 * t1) * out_o1 * (1 - out_01) * w5 + (out_o2 * t1) * out_o2 * (1 - out_02) * w7
     5. ∂E_total/∂w1 = ∂E1/out_h1 * ∂out_h1/∂h1 * ∂h1/∂w1
        ∂E_total/∂w2 = ∂E1/out_h1 * ∂out_h1/∂h1 * ∂h1/∂w2
        ∂E_total/∂w3 = ∂E1/out_h2 * ∂out_h2/∂h2 * ∂h2/∂w3
     6. ∂E_total/∂w1 = ((out_o1 * t1) * out_o1 * (1 - out_01) * w5 + (out_o2 * t1) * out_o2 * (1 - out_02) * w7) * out_h1 * (1 - out_h1) * i1
        ∂E_total/∂w2 = ((out_o1 * t1) * out_o1 * (1 - out_01) * w5 + (out_o2 * t1) * out_o2 * (1 - out_02) * w7) * out_h1 * (1 - out_h1) * i2
        ∂E_total/∂w3 = ((out_o1 * t1) * out_o1 * (1 - out_01) * w6 + (out_o2 * t1) * out_o2 * (1 - out_02) * w8) * out_h2 * (1 - out_h2) * i1
        ∂E_total/∂w4 = ((out_o1 * t1) * out_o1 * (1 - out_01) * w6 + (out_o2 * t1) * out_o2 * (1 - out_02) * w8) * out_h2 * (1 - out_h2) * i2

## Total Loss graph

1. At ͷ = 1

![image](https://user-images.githubusercontent.com/63489899/212284018-c65c96a8-0579-49d0-b2c0-464fb582baed.png)

2. At ͷ = 2

![image](https://user-images.githubusercontent.com/63489899/212284229-40638af5-fa37-46e0-89ba-b364c12ba62f.png)

3. At ͷ = 0.1

![image](https://user-images.githubusercontent.com/63489899/212284373-a7c46148-f19f-4ac9-8e44-cc8c88b369af.png)

4. At ͷ = 0.5

![image](https://user-images.githubusercontent.com/63489899/212284497-30b48c0c-72ee-4fbd-ad87-45b07156730c.png)


# Reference
https://towardsdatascience.com/how-to-reduce-training-parameters-in-cnns-while-keeping-accuracy-99-a213034a9777
