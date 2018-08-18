import numpy as np
import sys

class initializer:
    def xavier(self, shape, prev_out_dim):
        return np.random.randn(*shape) / np.sqrt(prev_out_dim)
    def he(self, shape, prev_out_dim):
        return np.random.randn(*shape) * np.sqrt(2/prev_out_dim)
    def normal(self, shape):
        return np.random.randn(*shape)*0.1

class conv2d_tool:
    def pad(self, x, pad_h_2, pad_w_2):
        pad_x = np.pad(x, [(0, 0), (0, 0), 
                (int(pad_h_2/2), (pad_h_2)-int(pad_h_2/2)), 
                (int(pad_w_2/2), (pad_w_2)-int(pad_w_2/2))], 'constant')
       
        return pad_x


    def unpad(self, x, pad_h_2, pad_w_2):
        unpad_x =  x[:, :, 
                        int(pad_h_2/2) : x.shape[2]-((pad_h_2)-int(pad_h_2/2)),
                        int(pad_w_2/2) : x.shape[3]-((pad_w_2)-int(pad_w_2/2))
                    ]
     
        return unpad_x


    def im2col(self, pad_x, kernel_size, strides, out_h, out_w):
        N, C = pad_x.shape[0], pad_x.shape[1]
              
        col = np.zeros([N, out_h*out_w, np.prod(kernel_size)*C]) #batch, 가능한 모든경우의 수, 필터사이즈*채널
        
        col_count = 0
        for h in range(out_h): # 가능한 out_h만큼 도는데 out_h 한번당 모든 out_w만큼 체크
            for w in range(out_w): 
                col[:, col_count, :] = pad_x[:, :, 
                                                strides[0]*h:kernel_size[0]+strides[0]*h,
                                                strides[1]*w:kernel_size[1]+strides[1]*w
                                            ].reshape(N, -1)
                col_count += 1

        return col.reshape(-1, np.prod(kernel_size)*C) #배치도 다 row로 묶음. 결과적으로 2차원


    def col2im(self, col, pad_x_shape, kernel_size, strides, out_h, out_w):
        N, C, H, W = pad_x_shape
       
        im = np.zeros(pad_x_shape) #batch, 가능한 모든경우의 수, 필터사이즈*채널
        col = col.reshape(N, out_h*out_w, np.prod(kernel_size)*C) # batch별로 분할
        
        col_count = 0        
        for h in range(out_h):
            for w in range(out_w):
                im[:, :,
                        strides[0]*h:kernel_size[0]+strides[0]*h,
                        strides[1]*w:kernel_size[1]+strides[1]*w
                    ] = col[:, col_count, :].reshape(N, C, *kernel_size)
                col_count += 1
     
        return im


class conv2d:
    def __init__(self, kernel_size, strides, out_dim, w_init, b_init=0):#, padding='SAME'):
        self.kernel_size = kernel_size
        self.strides = strides
        self.out_dim = out_dim
        self.w_init = w_init
        self.w = None
        self.bias = np.full(out_dim, b_init).astype(np.float32) # bias
       
        self.out_h = None
        self.out_w = None
        self.pad_h_2 = None
        self.pad_w_2 = None
        self.output_shape = None
        self.pad_x_shape = None

        self.col_x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        #data.shape: Batch, height, width, channel
        N, H, W, C = x.shape
        x = x.transpose(0, 3, 1, 2) # batch, channel, height, width

        #ex) pad_h_2 = 1인 경우 좌변은 0, 우변은 1.     2인경우 좌변은 1 우변은 1.   
        out_h = int(np.ceil(H / self.strides[0]))
        self.out_h = out_h
        
        out_w = int(np.ceil(W / self.strides[1]))
        self.out_w = out_w

        pad_h_2 = int(out_h * self.strides[0] - H + self.kernel_size[0] - self.strides[0])  #padding to H #2*pad_h
        pad_h_2 = max(0, pad_h_2)
        self.pad_h_2 = pad_h_2

        pad_w_2 = int(out_w * self.strides[1] - W + self.kernel_size[1] - self.strides[1])  #padding to w #2*pad_w   
        pad_w_2 = max(0, pad_w_2)
        self.pad_w_2 = pad_w_2
        
        tool = conv2d_tool()
        
        #pad => im2col
        pad_x = tool.pad(x, pad_h_2, pad_w_2)
        col_x = tool.im2col(pad_x, self.kernel_size, self.strides, out_h, out_w)
        self.pad_x_shape = pad_x.shape
        self.col_x = col_x

        #weight reshape
        col_w = self.w.reshape(self.out_dim, -1).T # 이렇게 해야 한 열에 1 out channel에 대한 kernel_size*c 의 의미가 들어있음.

        #convolution
        out = np.matmul(col_x, col_w) + self.bias
        out = out.reshape(N, out_h, out_w, self.out_dim)
        self.output_shape = out.shape

        return out

    def backward(self, grad):
        if grad.ndim == 2: #2차원인 경우 == FCNN 계층에서 넘어온 grad 임.
            grad = grad.reshape(self.output_shape) # [N, H, W, out_dim] 
  
        tool = conv2d_tool()
        
        #grad shape은 np.matmul(col_x, col_w)+bias 한걸 reshape했던거니까  col_grad의 행 size는 col_x.shape[0]이 되고, 열은 out_dim 임.
        col_grad = grad.reshape(self.col_x.shape[0], self.out_dim) 

        self.db = np.sum(col_grad, axis=0)
        self.dw = np.matmul((self.col_x).T, col_grad)

        col_w = self.w.reshape(self.out_dim, -1).T # [-1, out_dim]
        dx = np.matmul(col_grad, col_w.T)
        dx = tool.col2im(dx, self.pad_x_shape, self.kernel_size, self.strides, self.out_h, self.out_w)      
        dx = tool.unpad(dx, self.pad_h_2, self.pad_w_2)
        dx = dx.transpose(0,2,3,1) # [N, H, W, out_dim]

        return dx


class maxpool2d:
    def __init__(self, kernel_size, strides):#, padding='SAME'):
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.maxindex = None
        self.pad_h_2 = None
        self.pad_w_2 = None
        self.out_h = None
        self.out_w = None
        self.output_shape = None
        self.pad_x_shape = None
        self.out_dim = None


    def forward(self, x):
        #data.shape: Batch, height, width, channel
        N, H, W, C = x.shape
        x = x.transpose(0, 3, 1, 2) # batch, channel, height, width

        #ex) pad_h_2 = 1인 경우 좌변은 0, 우변은 1.     2인경우 좌변은 1 우변은 1.   
        out_h = int(np.ceil(H / self.strides[0]))
        self.out_h = out_h
        
        out_w = int(np.ceil(W / self.strides[1]))
        self.out_w = out_w

        pad_h_2 = int(out_h * self.strides[0] - H + self.kernel_size[0] - self.strides[0])  #padding to H #2*pad_h
        pad_h_2 = max(0, pad_h_2)
        self.pad_h_2 = pad_h_2

        pad_w_2 = int(out_w * self.strides[1] - W + self.kernel_size[1] - self.strides[1])  #padding to w #2*pad_w   
        pad_w_2 = max(0, pad_w_2)
        self.pad_w_2 = pad_w_2
        

        tool = conv2d_tool()
        
        #pad => im2col
        pad_x = tool.pad(x, pad_h_2, pad_w_2)
        col_x = tool.im2col(pad_x, self.kernel_size, self.strides, out_h, out_w)
        self.pad_x_shape = pad_x.shape

        #im2col된 데이터(shape: [N*out_h*out_w, np.multiply(*kernel_size)*C])를 maxpooling이기 때문에 filter단위로 다시 reshape해줌
        col_x = col_x.reshape(-1, np.prod(self.kernel_size))       
    
        #max index
        col_maxindex = np.argmax(col_x, axis=1)       
        self.maxindex = col_maxindex

        #max
        col_max = np.max(col_x, axis=1, keepdims=True)
        col_max = col_max.reshape(N, out_h, out_w, C)
        self.output_shape = col_max.shape

        return col_max        
        
    def backward(self, grad):
        if grad.ndim == 2: #2차원인 경우 == FCNN 계층에서 넘어온 grad 임.
            # self.maxindex는 모든필터에관해서 pooling된 첫칸에 들어갈 값의 maxindex 구하고,
            # 또 모든필터에관해서 다음칸 처리하므로 아래와같이 W, C와 에관한 shape로 다뤄야함.
            grad = grad.reshape(self.output_shape) # [N, H, W, C] 
            
        mask = np.zeros((self.maxindex.shape[0], np.prod(self.kernel_size))).astype(np.float32)
        mask[np.arange(self.maxindex.shape[0]), self.maxindex] = grad.flatten()

        tool = conv2d_tool()
        mask = tool.col2im(mask, self.pad_x_shape, self.kernel_size, self.strides, self.out_h, self.out_w)
        mask = tool.unpad(mask, self.pad_h_2, self.pad_w_2) # [N, out_dim, out_h, out_w]
        mask = mask.transpose(0,2,3,1) # [N, H, W, out_dim , ]

        return mask


class flatten():
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return x

    def backward(self, grad):
        return grad


class dropout:
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, x, is_train=True):
        if is_train == True:
            uniform = np.random.uniform(0, 1, size=x.shape) # [0, 1)
            mask = uniform > self.keep_prob # keep_prob보다 작으면 false, 크면 true
            #0.6으로 치면 0.6보다 작은 값들(==60%)는 false, 0.6보다 큰 값들(==40%)는 true
            self.mask = mask #즉 mask는 지울 값들을 true로 mask 함.
            x[mask] = 0 #true인 위치의 값을 0으로 dropout.
            return x
        else:
            return x

    def backward(self, grad):
        grad[self.mask] = 0 #dropout 시켰던 부분은 미분값이 0이므로 grad도 0으로 할당.
        return grad


class affine:
    def __init__(self, out_dim, w_init, b_init=0):
        self.x = None # input
        self.w = None
        self.out_dim = out_dim
        self.w_init = w_init
        self.b = np.full(out_dim, b_init).astype(np.float32) # bias
        self.dw = None # w gradient
        self.db = None # bias gradient

    def forward(self, x):
        self.x = x # input
        out = np.matmul(x, self.w) + self.b
        return out

    def backward(self, grad=1):
        #x.T = [w_shape[0], batch], grad = [batch, w_shape[1]]  즉 np.matmul(x.T, grad) 하면 batch전체에 관해 dw가 계산됨.
        self.dw = np.matmul(self.x.T, grad) #shape 때문에 이렇게 됨. 계산그래프 그려보면 이해됨.
        self.db = np.mean(grad, axis=0) #batch별 평균.
        return np.matmul(grad, self.w.T) # x에 관해서 계속 backpropagation 되기 때문에 x에관한 미분을 리턴해서 이전 layer에 전파.


class relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        mask = (x<=0) # 0이하인 값 mask,  if x = [10, -1, 3] => mask = [false, true, false]
        self.mask = mask # backward할 때, mask된 부분(0이하인 값)은 미분 0
        x[mask] = 0 # 0이하인 값에 0 할당.
        return x

    def backward(self, grad):
        grad[self.mask] = 0 #forward 값이 0이하였던 부분은 미분값 0으로 할당.
        return grad


class sigmoid:
    def __init__(self):
        self.sigvalue = None 

    def forward(self, x):
        sigvalue = 1/(1+np.exp(-x))
        self.sigvalue = sigvalue
        return sigvalue

    def backward(self, grad=1):
        return grad*self.sigvalue*(1-self.sigvalue)


class softmax_cross_entropy_with_logits:
    def __init__(self):
        self.target = None
        self.pred = None #softmax 결과
        self.loss = None 

    def forward(self, x, target):
        target = np.array(target)
        self.target = target

        #softmax
        max_value = np.max(x, axis=1, keepdims=True)
        exp = np.exp(x - max_value) #max값을 빼도, 빼지 않은 것과 결과는 동일하며, 빼지 않으면 값 overflow 발생 가능. 
        pred = exp / np.sum(exp, axis=1, keepdims=True)
        self.pred = pred

        #cross_entropy
        epsilon = 1e-07
        loss = -target*np.log(pred + epsilon) # pred가 0이면 np.log = -inf
        loss = np.mean(np.sum(loss, axis=1), axis=0) #data별로 sum 하고, batch별로 mean
        self.loss = loss
        return loss

    def backward(self, grad=1):
        #return np.mean(self.pred-self.target, axis=0) #배치별로 gradient 평균냄.
        return (self.pred-self.target)/self.target.shape[0] #배치사이즈로 나눠줌. 여기서 안나누고 affine.backward에서 나눠도되긴함
        #근데 affine.backward에서 나누면 affine 레이어마다 나눠야해서 계산량이 더 많음.


class model:
    def __init__(self, x_shape, y_shape=None):
        self.graph = []
        self.loss_graph = None
        self.x_shape = x_shape
        self.y_shape = y_shape

    def weight_initialize(self):
        forward_value = np.zeros([1, *self.x_shape[1:]]).astype(np.float32) #1 batch data

        #first node init
        node = self.graph[0]
        if 'conv2d' in str(node): #image cnn
            prev_out_dim = self.x_shape[-1]
            node.w = node.w_init([node.out_dim, prev_out_dim, *(node.kernel_size)], prev_out_dim * np.prod(node.kernel_size))

        elif 'affine' in str(node): #affine fcnn
            prev_out_dim = self.x_shape[-1]
            node.w = node.w_init([prev_out_dim, node.out_dim], prev_out_dim)

        forward_value = node.forward(forward_value)
        node_name = str(node).split()[0].split('.')[1]
        print(node_name, forward_value.shape)

        for node in self.graph[1:]:
            if 'conv2d' in str(node): #image cnn
                prev_out_dim = forward_value.shape[-1]
                node.w = node.w_init([node.out_dim, prev_out_dim, *(node.kernel_size)], prev_out_dim * np.prod(node.kernel_size))

            elif 'affine' in str(node):
                prev_out_dim = forward_value.shape[-1]
                node.w = node.w_init([prev_out_dim, node.out_dim], prev_out_dim)
                
            forward_value = node.forward(forward_value)
            node_name = str(node).split()[0].split('.')[1]
            print(node_name, forward_value.shape)


    def connect(self, node):
        self.graph.append(node)

    def connect_loss(self, node):
        self.loss_graph = node

    def forward(self, x, is_train=True):
        x = np.array(x)
        
        logits = x
        for node in self.graph:
            if 'dropout' in str(node):
                logits = node.forward(logits, is_train)
            else:
                logits = node.forward(logits)
        
        #self.output_layer = logits
        return logits

    def backward(self, logits, y, lr=0.002): #train
        y = np.array(y)
        #if y.shape[1:] != self.y_shape[1:]:
            #print("model에 설정한 y_shape와 입력된 데이터의 shape가 다릅니다.", self.y_shape[1:], '!=', y.shape[1:])
            #sys.exit(1)

        #calc loss
        loss = self.calc_loss(logits, y)
        #loss = self.loss_graph.forward(logits, y)
        
        #backpropagation
        grad = self.loss_graph.backward()
        for index in range(len(self.graph)-1, -1, -1):
            grad = self.graph[index].backward(grad) #grad 계산하면 dw,db 갱신됨.
            #계산그래프여서 backpropagation에 필요한건 grad에 계산되어있음. 그러므로 grad 구하자마자 w, b 업데이트해도됨. 
            #계산그래프 안쓰면 모든 w', b' 계산이 끝난 후 update해야함.
            if 'affine' in str(self.graph[index]):
                self.graph[index].w -= lr * self.graph[index].dw
                self.graph[index].b -= lr * self.graph[index].db
        return loss

    def calc_loss(self, logits, y):
        loss = self.loss_graph.forward(logits, y)
        return loss

    def correct(self, logits, y, axis=1):
        compare = (np.argmax(logits, axis) == np.argmax(y, axis))
        return np.sum(compare)



