
import EquivariantTensors as ET 
import Polynomials4ML as P4ML

function splinify(et_pair::ETPairModel, et_ps, et_st; 
                  Nspl = 30)
   
   # polynomial basis taking y = y(r) as input 
   trans_y = et_pair.rembed.layer.rbasis.trans
   polys_y = et_pair.rembed.layer.rbasis.basis
   # weights for learnable radials 
   WW = et_ps.rembed.rbasis.post.W
   # use P4ML to generate individual cubic splines 
   splines = [ 
         P4ML.splinify( y -> WW[:, :, i] * polys_y(y), -1.0, 1.0, Nspl ) 
         for i in 1:size(WW, 3)  ]
   # extract the spline parameters into an array of parameter sets          
   states = [ P4ML._init_luxstate(spl) for spl in splines ]
   # selects the correct spline based on the (Zi, Zj) pair 
   selector2 = et_pair.rembed.layer.rbasis.post.selector
   # envelope multiplying the spline 
   envelope = et_pair.rembed.layer.envelope

   spl_rbasis = ET.TransSelSplines(trans_y, envelope, selector2, splines[1], states)

   return ETPairModel( ET.EdgeEmbed(spl_rbasis), et_pair.readout )
end