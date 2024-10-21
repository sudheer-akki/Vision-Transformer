import torch.nn as nn
from torch.nn import MultiheadAttention
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_patches(patches):
    # Squeeze the batch dimension
    tensor = patches.squeeze(0)  # Shape now becomes (3, 10, 10, 40, 40)

    # Create a 10x10 grid of subplots
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    # Loop through each of the 10x10 grid and plot the images
    for i in range(10):
        for j in range(10):
            # Extract the 40x40 image for RGB channels
            image = tensor[:, i, j, :, :].permute(1, 2, 0)  # Shape becomes (40, 40, 3)

            # Plot the image in the corresponding subplot
            axes[i, j].imshow(image.detach().cpu().numpy())
            axes[i, j].axis('off')  # Turn off the axis for cleaner visualization

    # Adjust layout so the images don't overlap
    plt.tight_layout()
    plt.show()


def Make_patches(image, patch_size, visualize_patch = False):
    #image shape (1,3,400,400)
    P, C = patch_size, image.shape[1] 
    num_patches = image.shape[2] * image.shape[3] // patch_size**2 # N = HW/P**2 = 100

    last_dimension_shape = image.shape[1]*patch_size**2  # 4800

    #dividing patches along Height and Width
    patches = image.unfold(2,P,P).unfold(3,P,P) #(1,3,10,10,40,40)
    # The shape of patches is (1, C, num_patches_x, num_patches_y, patch_size, patch_size) 

    if visualize_patch:
        visualize_patches(patches)
        
    """reshaped = patches.reshape(1,1,num_patches,last_dimension_shape) # reshape computationally expensive use view"""

    """Note: View must expect Tensor to be Contiguous (one single memory block instead of Scattered underlying Memory)"""

    reshaped = patches.contiguous().view(patches.size(0), -1, C * P * P).float()
    # target shape (1,100,4800)

    # target shape N× (P2·C) = (number of patches, Patch_size**2xColour channel) = (100,40x40X3) = (100,4800) 
    return reshaped



class MakePatchEmbedding(nn.Module):
    def __init__(self, in_channel, out_channel,patch_size):
        super(MakePatchEmbedding,self).__init__()
        self.patches = nn.Conv2d(in_channels=in_channel,
                                 out_channels=out_channel,
                                 kernel_size=patch_size,
                                 stride = patch_size,
                                 padding=0)
        """Linear Flattening: Grid of pixel values, is flattened into a 1D vector (a sequence of numbers)
        The flatten operation works on the last two dimensions of the tensor, converting the height and width 
        of the patches into a single vector (flattening each 2D patch into 1D)."""
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.patch_size = patch_size
    
    def forward(self,images):
        assert images.shape[-1] % self.patch_size == 0, f"Image is not squared"
        patches = self.patches(images)
        flattened = self.flatten(patches)
        """(batch_size, out_channel, num_patches): (batch_size, num_patches, out_channel)"""
        return flattened.permute(0,2,1)
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embedding_dim,num_heads:int=12,attn_dropout:float=0):
        super(MultiHeadSelfAttention, self).__init__()


        # MultiHeadAttention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )

    def forward(self,x):
        attn_output, _ = self.multihead_attn(query=x,
                                             key=x,
                                             value=x, 
                                             need_weights =False) # we need just only layer outputs
        return attn_output


class MultiLayerPerceptron(nn.Module):
    def __init__(self,embedding_dim, mlp_ratio = 4):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=embedding_dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(in_features= embedding_dim*mlp_ratio,
                      out_features=embedding_dim)
        )

    def forward(self,x):
        x = self.mlp(x)
        return x

class TransformerEncoder(nn.Module):

    def __init__(self,embedding_dim, mlp_ratio:int= 4, num_heads:int=12,attn_dropout: float= 0.0, mlp_dropout:float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(mlp_dropout)
        # Layer Norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa_block = MultiHeadSelfAttention(embedding_dim=embedding_dim,num_heads=num_heads,attn_dropout=attn_dropout)
        self.mlp_block = MultiLayerPerceptron(embedding_dim=embedding_dim,mlp_ratio=mlp_ratio)


    def forward(self,x):
        # apply dropout
        out_1 = self.dropout(x)
        # apply layer normalization
        out_2 = self.layer_norm(out_1)
        # compute multi-head self-attention
        msa_out = self.msa_block(out_1)
        # apply dropout 
        out_2 = self.dropout(msa_out)
        # apply residual connection
        res_out = x + out_2
        # apply layer normalization
        out_4 = self.layer_norm(res_out)
        #residual block for MLP block (adding input to the output)
        out_5 = self.mlp_block(out_4)
        # apply dropout 
        out_6 = self.dropout(out_5)
        # apply residual connection
        out = res_out + out_6
        return out

class MLPHead(nn.Module):
    def __init__(self, embedding_dim,mlp_ratio, num_classes=36,fine_tune=False):
        super(MLPHead, self).__init__()
        self.num_classes = num_classes
        
        if not fine_tune:
            # hidden layer with tanh activation function 
            self.mlp_head = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim*mlp_ratio),  # hidden layer
                                    nn.Tanh(),
                                    nn.Linear(embedding_dim*mlp_ratio, num_classes)    # output layer
                            )
        else:
            # single linear layer
            self.mlp_head = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.mlp_head(x)
        return x
    
def get_sinusoid_encoding(num_tokens=100+1, token_len=768):
    """ Make Sinusoid Encoding Table

        Args:
            num_tokens (int): number of tokens
            token_len (int): length of a token
            
        Returns:
            (torch.FloatTensor) sinusoidal position encoding table
    """

    def get_position_angle_vec(i):
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ViT_Model(nn.Module):
    def __init__(self,num_classes,in_channels=3,patch_size=40,embedding_dim:int = 768, num_transformer_layers: int = 12,embedding_dropout: float = 0.1):
        super(ViT_Model, self).__init__()
        c,h,w = 3,400,400
        """ Intialization """
        self.patch_size = patch_size
        self.no_of_patches = h*w//patch_size**2  #100
        self.embedding_dim = embedding_dim #fixed 768
        self.num_classes = num_classes

        self.sinusoidal_embedding = nn.Parameter(data=get_sinusoid_encoding(num_tokens=self.no_of_patches+1, token_len=self.embedding_dim), requires_grad=False)
        #self.sinusoidal_embedding = get_sinusoid_encoding(num_tokens=self.no_of_patches+1,token_len=self.embedding_dim)

        """ Patch Embeddings: Making Linear Patches + Flattening + Mapping to D (Latent vector size) dimension 
        Def: Patch embeddings are generated by applying a simple linear transformation to the flattened pixel values of the patch"""

        """Step1: = Making Linear Patches + Flattening """
        """self.linear_patch_embedding = MakePatchEmbedding(in_channel=in_channels,
                                   out_channel=self.no_of_patches,
                                   patch_size=patch_size)"""
     
        """Step2: Linear projection layer: 
        input size = patch_size^2 * in_channels, 
        output size = flattened patches is multiplied with embedding tensor, E of shape (P²*C × d). 
        The final embedded patches is now having shape of (1×d). d is the model dimension.
        Mapping of 40x40x1 -> 768 """
        # trainable linear projection for mapping dimnesion of patches (weight matrix E)
        #self.linear_projection_matrix = nn.Linear(patch_size * patch_size * in_channels, self.embedding_dim) #(1,100,768)
        self.linear_projection_matrix = nn.Parameter(
                    torch.randn( patch_size * patch_size * in_channels, embedding_dim)) #(1600,768)

        """Step3: Intializing Class token """
        self.class_token = nn.Parameter(torch.zeros(1,1, self.embedding_dim),requires_grad=True) # (1,1,768)
        #print(f"Class token embedding shape: {self.class_token.shape}")



        """ Step5: Intializing Position Embedding for Patches """
        self.positional_embedding = nn.Parameter(
            torch.rand(1, self.no_of_patches+1, self.embedding_dim), requires_grad = True
        )


        """Step7: Intialize Transformer Encoder:
        Embedded Patch 
        =                
        LayerNorm
        >
        MultiHeadSelfAttention
        >
        output + Embedded Patch
        >
        LayerNorm
        >
        MLP Block
        >
        output + Previous Embedded Patch
        """
        # stack transformer encoder layers 
        transformer_encoder_list = [
            TransformerEncoder(self.embedding_dim) 
                    for _ in range(num_transformer_layers)] 
        self.transformer_encoder_layers = nn.Sequential(*transformer_encoder_list)

        """Step9: Final MLP Classifier """
        self.final_classifier = MLPHead(embedding_dim=self.embedding_dim, mlp_ratio=4, num_classes=self.num_classes)


    def forward(self, image):

        assert image.shape[2] % self.patch_size == 0 , f"Patch shape is not divisible with Image shape"


        """Step1: Making Linear Patches + Flattening """
        #linear_patch_embedded = self.linear_patch_embedding(image) # (batch,num_of_patches,embeddings(3x40x40)) = (1,100,4800)
        flattened_patch_embeddings= Make_patches(image=image, patch_size=self.patch_size) #(1,100,4800)

        """Step2: Linear projection of Flattened patches """
        # linearly embed patches
        linear_projected = torch.matmul(flattened_patch_embeddings , self.linear_projection_matrix) # (1,100,4800) x (4800,768) = (1,100,768)
        
        """Step4: Adding intialized Class token to Patch embeddings"""
        patch_embedding_with_class_token = torch.cat((self.class_token,linear_projected), dim=1) # (1,1,768) + (1,100,768) = (1,101,768)
        print(f"Patch embedding shape with class: {patch_embedding_with_class_token.shape}")

      
        """Step6: linear_patches_with_class_token (step4) + Adding Sinusoidal encodings or Intialized Patch positional Embeddings (step5)"""
        #5. Add class token to positional embeddings
        positional_embedding_with_class = patch_embedding_with_class_token + self.sinusoidal_embedding #self.positional_embedding # (1,101,768) + (1,101,768)
        print(f"class token to positional embedding shape:{positional_embedding_with_class.shape}")

        """ Step8: Passing Patch + Positional Embedding -> Transformer Encoder """
        transformer_output = self.transformer_encoder_layers(positional_embedding_with_class)

     
        # extract [class] token from encoder output
        output_class_token  = transformer_output[:,0] 

        """ Step10: pass token through mlp head for classification"""
        output = self.final_classifier(output_class_token )

        return output
    

def set_seeds(seed:int =42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__=="__main__":

    #Make_patches(image=torch.rand(1,1,400,400), patch_size=40)
    set_seeds()
    model = ViT_Model(num_classes=36)

    x = torch.rand((1,3,400,400))

    model(x)




