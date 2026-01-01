
import EquivariantTensors as ET 
import Polynomials4ML as P4ML

# These implementations of `splinify` expect a very specific structure of the 
# pair potential basis. In principle it is possible to relax this 
# considerably but it needs a little bit of thinking and planning/design 
# work before just diving in. To be discussed when needed.

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
   # selects the correct spline based on the (Zi, Zj) pair 
   selector2 = et_pair.rembed.layer.rbasis.post.selector
   # envelope multiplying the spline 
   envelope = et_pair.rembed.layer.envelope

   spl_rbasis = ET.trans_splines(trans_y, splines, selector2, envelope)

   return ETPairModel( ET.EdgeEmbed(spl_rbasis), et_pair.readout )
end


function splinify(et_model::ETACE, et_ps, et_st; Nspl = 50)

   rembed = et_model.rembed.layer   # radial embedding, edgeembed stripped
   trans = rembed.trans             # x -> y dp_transform 
   rpolys_env = rembed.basis        # polynomials * envelope
   polys_y = rpolys_env.l.layers.layer_1    # polynomial basis 
   yenv_func = rpolys_env.l.layers.layer_2.func   # envelope function

   # envelope multiplying the spline, apply the transformation a second 
   # time until we figure out how to reuse it conveniently 
   trans_yenv = ET.dp_transform( 
            (x, st) -> yenv_func(trans.f(x, st)), 
            trans.refstate )
   # selects the correct spline based on the (Zi, Zj) pair 
   selector2 = rembed.post.selector
   # generate the splines using P4ML 
   WW = et_ps.rembed.post.W
   splines = [ 
         P4ML.splinify( y -> WW[:, :, i] * polys_y(y), -1.0, 1.0, Nspl ) 
         for i in 1:size(WW, 3)  ]

   rembed_spl = ET.trans_splines(trans, splines, selector2, trans_yenv)
   ace_spl = ETACE( ET.EdgeEmbed(rembed_spl), 
                    et_model.yembed, 
                    et_model.basis, 
                    et_model.readout )
   return ace_spl
end