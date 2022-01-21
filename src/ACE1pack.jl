module ACE1pack

# load and reexport JuLIP, ACE1, IPFitting 
using Reexport 
@reexport using JuLIP
@reexport using ACE1
@reexport using IPFitting
export JuLIP, ACE1, IPFitting 

# TODO: produce the convenience layers 

end
