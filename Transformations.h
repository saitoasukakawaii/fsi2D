using namespace dealii;
// First, we define tensors for solution variables
// v (velocity), u (displacement), p (pressure).
// Moreover, we define 
// corresponding tensors for derivatives (e.g., gradients, 
// deformation gradients) and
// linearized tensors that are needed to solve the 
// non-linear problem with Newton's method.  
namespace ALE_Transformations
{    
  template <int dim> 
    inline
    Tensor<2,dim> 
    get_pI (unsigned int q,
	    std::vector<Vector<double> > old_solution_values)
    {
      Tensor<2,dim> tmp;
      tmp[0][0] =  old_solution_values[q](dim+dim);
      tmp[1][1] =  old_solution_values[q](dim+dim);
      
      return tmp;      
    }

  template <int dim> 
    inline
    Tensor<2,dim> 
    get_pI_LinP (const double phi_i_p)
    {
      Tensor<2,dim> tmp;
      tmp.clear();
      tmp[0][0] = phi_i_p;    
      tmp[1][1] = phi_i_p;
      
      return tmp;
   }

 template <int dim> 
   inline
   Tensor<1,dim> 
   get_grad_p (unsigned int q,
	       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
   {     
     Tensor<1,dim> grad_p;     
     grad_p[0] =  old_solution_grads[q][dim+dim][0];
     grad_p[1] =  old_solution_grads[q][dim+dim][1];
      
     return grad_p;
   }

 template <int dim> 
  inline
  Tensor<1,dim> 
  get_grad_p_LinP (const Tensor<1,dim> phi_i_grad_p)	 
    {
      Tensor<1,dim> grad_p;      
      grad_p[0] =  phi_i_grad_p[0];
      grad_p[1] =  phi_i_grad_p[1];
	   
      return grad_p;
   }

 template <int dim> 
   inline
   Tensor<2,dim> 
   get_grad_u (unsigned int q,
	       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
   {   
      Tensor<2,dim> structure_continuation;     
      structure_continuation[0][0] = old_solution_grads[q][dim][0];
      structure_continuation[0][1] = old_solution_grads[q][dim][1];
      structure_continuation[1][0] = old_solution_grads[q][dim+1][0];
      structure_continuation[1][1] = old_solution_grads[q][dim+1][1];

      return structure_continuation;
   }

  template <int dim> 
  inline
  Tensor<2,dim> 
  get_grad_v (unsigned int q,
	      std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
    {      
      Tensor<2,dim> grad_v;      
      grad_v[0][0] =  old_solution_grads[q][0][0];
      grad_v[0][1] =  old_solution_grads[q][0][1];
      grad_v[1][0] =  old_solution_grads[q][1][0];
      grad_v[1][1] =  old_solution_grads[q][1][1];
      
      return grad_v;
   }

  template <int dim> 
    inline
    Tensor<2,dim> 
    get_grad_v_T (const Tensor<2,dim> tensor_grad_v)
    {   
      Tensor<2,dim> grad_v_T;
      grad_v_T = transpose (tensor_grad_v);
            
      return grad_v_T;      
    }
  
  template <int dim> 
    inline
    Tensor<2,dim> 
    get_grad_v_LinV (const Tensor<2,dim> phi_i_grads_v)	 
    {     
        Tensor<2,dim> tmp;		 
	tmp[0][0] = phi_i_grads_v[0][0];
	tmp[0][1] = phi_i_grads_v[0][1];
	tmp[1][0] = phi_i_grads_v[1][0];
	tmp[1][1] = phi_i_grads_v[1][1];
      
	return tmp;
    }

  template <int dim> 
    inline
    Tensor<2,dim> 
    get_Identity ()
    {   
      Tensor<2,dim> identity;
      identity[0][0] = 1.0;
      identity[0][1] = 0.0;
      identity[1][0] = 0.0;
      identity[1][1] = 1.0;
            
      return identity;      
   }

 template <int dim> 
 inline
 Tensor<2,dim> 
 get_F (unsigned int q,
	std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
    {     
      Tensor<2,dim> F;
      F[0][0] = 1.0 +  old_solution_grads[q][dim][0];
      F[0][1] = old_solution_grads[q][dim][1];
      F[1][0] = old_solution_grads[q][dim+1][0];
      F[1][1] = 1.0 + old_solution_grads[q][dim+1][1];
      return F;
   }

 template <int dim> 
 inline
 Tensor<2,dim> 
 get_F_T (const Tensor<2,dim> F)
    {
      return  transpose (F);
    }

 template <int dim> 
 inline
 Tensor<2,dim> 
 get_F_Inverse (const Tensor<2,dim> F)
    {     
      return invert (F);    
    }

 template <int dim> 
 inline
 Tensor<2,dim> 
 get_F_Inverse_T (const Tensor<2,dim> F_Inverse)
   { 
     return transpose (F_Inverse);
   }

 template <int dim> 
   inline
   double
   get_J (const Tensor<2,dim> tensor_F)
   {     
     return determinant (tensor_F);
   }


 template <int dim> 
 inline
 Tensor<1,dim> 
 get_v (unsigned int q,
	std::vector<Vector<double> > old_solution_values)
    {
      Tensor<1,dim> v;	    
      v[0] = old_solution_values[q](0);
      v[1] = old_solution_values[q](1);
      
      return v;    
   }

 template <int dim> 
   inline
   Tensor<1,dim> 
   get_v_LinV (const Tensor<1,dim> phi_i_v)
   {
     Tensor<1,dim> tmp;
     tmp[0] = phi_i_v[0];
     tmp[1] = phi_i_v[1];
     
     return tmp;    
   }

 template <int dim> 
 inline
 Tensor<1,dim> 
 get_u (unsigned int q,
	std::vector<Vector<double> > old_solution_values)
   {
     Tensor<1,dim> u;     
     u[0] = old_solution_values[q](dim);
     u[1] = old_solution_values[q](dim+1);
     
     return u;          
   }

 template <int dim> 
   inline
   Tensor<1,dim> 
   get_u_LinU (const Tensor<1,dim> phi_i_u)
   {
     Tensor<1,dim> tmp;     
     tmp[0] = phi_i_u[0];
     tmp[1] = phi_i_u[1];
     
     return tmp;    
   }
 

 template <int dim> 
 inline
 double
 get_J_LinU (unsigned int q, 
	     const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
	     const Tensor<2,dim> phi_i_grads_u)	    
{
  return (phi_i_grads_u[0][0] * (1 + old_solution_grads[q][dim+1][1]) +
		   (1 + old_solution_grads[q][dim][0]) * phi_i_grads_u[1][1] -
		   phi_i_grads_u[0][1] * old_solution_grads[q][dim+1][0] - 
		   old_solution_grads[q][dim][1] * phi_i_grads_u[1][0]);  
}

  template <int dim> 
  inline
  double
  get_J_Inverse_LinU (const double J,
		      const double J_LinU)
    {
      return (-1.0/std::pow(J,2) * J_LinU);
    }

template <int dim> 
 inline
 Tensor<2,dim>
  get_F_LinU (const Tensor<2,dim> phi_i_grads_u)  
  {
    Tensor<2,dim> tmp;
    tmp[0][0] = phi_i_grads_u[0][0];
    tmp[0][1] = phi_i_grads_u[0][1];
    tmp[1][0] = phi_i_grads_u[1][0];
    tmp[1][1] = phi_i_grads_u[1][1];
    
    return tmp;
  }

template <int dim> 
 inline
 Tensor<2,dim>
  get_F_Inverse_LinU (const Tensor<2,dim> phi_i_grads_u,
		       const double J,
		       const double J_LinU,
		       unsigned int q,
		       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads
		       )  
  {
    Tensor<2,dim> F_tilde;
    F_tilde[0][0] = 1.0 + old_solution_grads[q][dim+1][1];
    F_tilde[0][1] = -old_solution_grads[q][dim][1];
    F_tilde[1][0] = -old_solution_grads[q][dim+1][0];
    F_tilde[1][1] = 1.0 + old_solution_grads[q][dim][0];
    
    Tensor<2,dim> F_tilde_LinU;
    F_tilde_LinU[0][0] = phi_i_grads_u[1][1];
    F_tilde_LinU[0][1] = -phi_i_grads_u[0][1];
    F_tilde_LinU[1][0] = -phi_i_grads_u[1][0];
    F_tilde_LinU[1][1] = phi_i_grads_u[0][0];

    return (-1.0/(J*J) * J_LinU * F_tilde +
	    1.0/J * F_tilde_LinU);
 
  }

 template <int dim> 
   inline
   Tensor<2,dim>
   get_J_F_Inverse_T_LinU (const Tensor<2,dim> phi_i_grads_u)  
   {
     Tensor<2,dim> tmp;
     tmp[0][0] = phi_i_grads_u[1][1];
     tmp[0][1] = -phi_i_grads_u[1][0];
     tmp[1][0] = -phi_i_grads_u[0][1];
     tmp[1][1] = phi_i_grads_u[0][0];
     
     return  tmp;
   }


 template <int dim> 
 inline
 double
 get_tr_C_LinU (unsigned int q, 
		 const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
		 const Tensor<2,dim> phi_i_grads_u)	    
{
  return ((1 + old_solution_grads[q][dim][0]) *
	  phi_i_grads_u[0][0] + 
	  old_solution_grads[q][dim][1] *
	  phi_i_grads_u[0][1] +
	  (1 + old_solution_grads[q][dim+1][1]) *
	  phi_i_grads_u[1][1] + 
	  old_solution_grads[q][dim+1][0] *
	  phi_i_grads_u[1][0]);
}

 
}

// Second, we define the ALE transformations rules. These
// are used to transform the fluid equations from the Eulerian
// coordinate system to an arbitrary fixed reference 
// configuration.
namespace NSE_in_ALE
{
  template <int dim> 
 inline
 Tensor<2,dim>
 get_stress_fluid_ALE (const double density,
		       const double viscosity,	
		       const Tensor<2,dim>  pI,
		       const Tensor<2,dim>  grad_v,
		       const Tensor<2,dim>  grad_v_T,
		       const Tensor<2,dim>  F_Inverse,
		       const Tensor<2,dim>  F_Inverse_T)
  {    
    return (-pI + density * viscosity *
	   (grad_v * F_Inverse + F_Inverse_T * grad_v_T ));
  }

  template <int dim> 
  inline
  Tensor<2,dim>
  get_stress_fluid_except_pressure_ALE (const double density,
					const double viscosity,	
					const Tensor<2,dim>  grad_v,
					const Tensor<2,dim>  grad_v_T,
					const Tensor<2,dim>  F_Inverse,
					const Tensor<2,dim>  F_Inverse_T)
  {
    return (density * viscosity * (grad_v * F_Inverse + F_Inverse_T * grad_v_T));
  }

  template <int dim> 
  inline
  Tensor<2,dim> 
  get_stress_fluid_ALE_1st_term_LinAll (const Tensor<2,dim>  pI,
					const Tensor<2,dim>  F_Inverse_T,
					const Tensor<2,dim>  J_F_Inverse_T_LinU,					    
					const Tensor<2,dim>  pI_LinP,
					const double J)
  {          
    return (-J * pI_LinP * F_Inverse_T - pI * J_F_Inverse_T_LinU);	     
  }
  
  template <int dim> 
  inline
  Tensor<2,dim> 
  get_stress_fluid_ALE_2nd_term_LinAll_short (const Tensor<2,dim> J_F_Inverse_T_LinU,					    
					      const Tensor<2,dim> stress_fluid_ALE,
					      const Tensor<2,dim> grad_v,
					      const Tensor<2,dim> grad_v_LinV,					    
					      const Tensor<2,dim> F_Inverse,
					      const Tensor<2,dim> F_Inverse_LinU,					    
					      const double J,
					      const double viscosity,
					      const double density 
					      )  
{
    Tensor<2,dim> sigma_LinV;
    Tensor<2,dim> sigma_LinU;

    sigma_LinV = grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
    sigma_LinU = grad_v *  F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);
 
    return (density * viscosity * 
	    (sigma_LinV + sigma_LinU) * J * transpose(F_Inverse) +
	    stress_fluid_ALE * J_F_Inverse_T_LinU);    
  }

template <int dim> 
inline
Tensor<2,dim> 
get_stress_fluid_ALE_3rd_term_LinAll_short (const Tensor<2,dim> F_Inverse,			   
					    const Tensor<2,dim> F_Inverse_LinU,					     
					    const Tensor<2,dim> grad_v,
					    const Tensor<2,dim> grad_v_LinV,					    
					    const double viscosity,
					    const double density,
					    const double J,
					    const Tensor<2,dim> J_F_Inverse_T_LinU)		    		  			     
{
  return density * viscosity * 
    (J_F_Inverse_T_LinU * transpose(grad_v) * transpose(F_Inverse) +
     J * transpose(F_Inverse) * transpose(grad_v_LinV) * transpose(F_Inverse) +
     J * transpose(F_Inverse) * transpose(grad_v) * transpose(F_Inverse_LinU));  
}



template <int dim> 
inline
double
get_Incompressibility_ALE (unsigned int q,
			   std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
{
  return (old_solution_grads[q][0][0] +
	  old_solution_grads[q][dim+1][1] * old_solution_grads[q][0][0] -
	  old_solution_grads[q][dim][1] * old_solution_grads[q][1][0] -
	  old_solution_grads[q][dim+1][0] * old_solution_grads[q][0][1] +
	  old_solution_grads[q][1][1] +
	  old_solution_grads[q][dim][0] * old_solution_grads[q][1][1]); 

}

template <int dim> 
inline
double
get_Incompressibility_ALE_LinAll (const Tensor<2,dim> phi_i_grads_v,
				  const Tensor<2,dim> phi_i_grads_u,
				  unsigned int q, 				
				  const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	     	    
{
  return (phi_i_grads_v[0][0] + phi_i_grads_v[1][1] + 
	  phi_i_grads_u[1][1] * old_solution_grads[q][0][0] + old_solution_grads[q][dim+1][1] * phi_i_grads_v[0][0] -
	  phi_i_grads_u[0][1] * old_solution_grads[q][1][0] - old_solution_grads[q][dim+0][1] * phi_i_grads_v[1][0] -
	  phi_i_grads_u[1][0] * old_solution_grads[q][0][1] - old_solution_grads[q][dim+1][0] * phi_i_grads_v[0][1] +
	  phi_i_grads_u[0][0] * old_solution_grads[q][1][1] + old_solution_grads[q][dim+0][0] * phi_i_grads_v[1][1]);
}


  template <int dim> 
  inline
  Tensor<1,dim> 
  get_Convection_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
			       const Tensor<1,dim> phi_i_v,
			       const double J,
			       const double J_LinU,
			       const Tensor<2,dim> F_Inverse,
			       const Tensor<2,dim> F_Inverse_LinU,			    			
			       const Tensor<1,dim> v,
			       const Tensor<2,dim> grad_v,				
			       const double density	   			     
			       )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)v = rho J grad(v)F^{-1}v
    
    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * v +
		       J * grad_v * F_Inverse_LinU * v);
    
    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * (phi_i_grads_v * F_Inverse * v + 
			    grad_v * F_Inverse * phi_i_v));
    
    return density * (convection_LinU + convection_LinV);
  }
  

  template <int dim> 
  inline
  Tensor<1,dim> 
  get_Convection_u_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
				 const Tensor<1,dim> phi_i_u,
				 const double J,
				 const double J_LinU,			    
				 const Tensor<2,dim>  F_Inverse,
				 const Tensor<2,dim>  F_Inverse_LinU,
				 const Tensor<1,dim>  u,
				 const Tensor<2,dim>  grad_v,				
				 const double density	   			     
				 )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u
    
    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * u +
		       J * grad_v * F_Inverse_LinU * u +
		       J * grad_v * F_Inverse * phi_i_u);
    
    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * u); 
        
    return density * (convection_LinU + convection_LinV);
}


  
  template <int dim> 
  inline
  Tensor<1,dim> 
  get_Convection_u_old_LinAll_short (const Tensor<2,dim> phi_i_grads_v,				 
				     const double J,
				     const double J_LinU,				 
				     const Tensor<2,dim>  F_Inverse,
				     const Tensor<2,dim>  F_Inverse_LinU,				 			
				     const Tensor<1,dim>  old_timestep_solution_displacement,	
				     const Tensor<2,dim>  grad_v,				
				     const double density					     		       	     
				     )
  {
    // Linearization of fluid convection term
    // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u
    
    Tensor<1,dim> convection_LinU;
    convection_LinU = (J_LinU * grad_v * F_Inverse * old_timestep_solution_displacement +
		       J * grad_v * F_Inverse_LinU * old_timestep_solution_displacement);
    
    Tensor<1,dim> convection_LinV;
    convection_LinV = (J * phi_i_grads_v * F_Inverse * old_timestep_solution_displacement); 
    
    
    return density * (convection_LinU  + convection_LinV);
  }

template <int dim> 
inline
Tensor<1,dim> 
get_accelaration_term_LinAll (const Tensor<1,dim> phi_i_v,
			      const Tensor<1,dim> v,
			      const Tensor<1,dim> old_timestep_v,
			      const double J_LinU,
			      const double J,
			      const double old_timestep_J,
			      const double density)
{   
  return density/2.0 * (J_LinU * (v - old_timestep_v) + (J + old_timestep_J) * phi_i_v);
  
}


}


// In the third namespace, we summarize the 
// constitutive relations for the solid equations.
namespace Structure_Terms_in_ALE
{
  // Green-Lagrange strain tensor
  template <int dim> 
  inline
  Tensor<2,dim> 
  get_E (const Tensor<2,dim> F_T,
	 const Tensor<2,dim> F,
	 const Tensor<2,dim> Identity)
  {    
    return 0.5 * (F_T * F - Identity);
  }

  template <int dim> 
  inline
  double
  get_tr_E (const Tensor<2,dim> E)
  {     
    return trace (E);
  }

  template <int dim> 
  inline
  double
  get_tr_E_LinU (unsigned int q, 
		 const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
		 const Tensor<2,dim> phi_i_grads_u)	    
  {
    return ((1 + old_solution_grads[q][dim][0]) *
	    phi_i_grads_u[0][0] + 
	    old_solution_grads[q][dim][1] *
	    phi_i_grads_u[0][1] +
	    (1 + old_solution_grads[q][dim+1][1]) *
	    phi_i_grads_u[1][1] + 
	    old_solution_grads[q][dim+1][0] *
	    phi_i_grads_u[1][0]); 
  }
  
}
