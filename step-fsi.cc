/**
 Thomas Wick 
 Technische Universitaet Muenchen / RICAM Linz
 Date: March 18, 2016
 E-mail: wick@ma.tum.de


 This code is a modification of 
 the ANS article open-source version:

 http://media.archnumsoft.org/10305/

 while using a nonlinear harmonic MMPDE
 in contrast to a (linear) biharmonic model.

 This code is based on the deal.II.8.3.0 version


 deal.II step: fluid-structure interaction
 Keywords: fluid-structure interaction, nonlinear harmonic MMPDE, 
           finite elements, benchmark computation, 
	   monolithic framework 



*/

/**
  This code is licensed under the "GNU GPL version 2 or later". See
  license.txt or https://www.gnu.org/licenses/gpl-2.0.html

  Copyright 2011-2016: Thomas Wick 
*/


// Include files
//--------------

// The first step, as always, is to include
// the functionality of these 
// deal.II library files and some C++ header
// files.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>  
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>


// C++
#include <fstream>
#include <sstream>
#include "Transformations.h"
#include "Parameters.h"
// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:				
using namespace dealii;


const long double PI = 3.141592653589793238462643;
const double R = 1.2e-2;

// First, we define tensors for solution variables
// v (velocity), u (displacement), p (pressure).
// Moreover, we define 
// corresponding tensors for derivatives (e.g., gradients, 
// deformation gradients) and
// linearized tensors that are needed to solve the 
// // non-linear problem with Newton's method.   
// namespace ALE_Transformations
// {    
//   template <int dim> 
//     inline
//     Tensor<2,dim> 
//     get_pI (unsigned int q,
// 	    std::vector<Vector<double> > old_solution_values)
//     {
//       Tensor<2,dim> tmp;
//       tmp[0][0] =  old_solution_values[q](dim+dim);
//       tmp[1][1] =  old_solution_values[q](dim+dim);
      
//       return tmp;      
//     }

//   template <int dim> 
//     inline
//     Tensor<2,dim> 
//     get_pI_LinP (const double phi_i_p)
//     {
//       Tensor<2,dim> tmp;
//       tmp.clear();
//       tmp[0][0] = phi_i_p;    
//       tmp[1][1] = phi_i_p;
      
//       return tmp;
//    }

//  template <int dim> 
//    inline
//    Tensor<1,dim> 
//    get_grad_p (unsigned int q,
// 	       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
//    {     
//      Tensor<1,dim> grad_p;     
//      grad_p[0] =  old_solution_grads[q][dim+dim][0];
//      grad_p[1] =  old_solution_grads[q][dim+dim][1];
      
//      return grad_p;
//    }

//  template <int dim> 
//   inline
//   Tensor<1,dim> 
//   get_grad_p_LinP (const Tensor<1,dim> phi_i_grad_p)	 
//     {
//       Tensor<1,dim> grad_p;      
//       grad_p[0] =  phi_i_grad_p[0];
//       grad_p[1] =  phi_i_grad_p[1];
	   
//       return grad_p;
//    }

//  template <int dim> 
//    inline
//    Tensor<2,dim> 
//    get_grad_u (unsigned int q,
// 	       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
//    {   
//       Tensor<2,dim> structure_continuation;     
//       structure_continuation[0][0] = old_solution_grads[q][dim][0];
//       structure_continuation[0][1] = old_solution_grads[q][dim][1];
//       structure_continuation[1][0] = old_solution_grads[q][dim+1][0];
//       structure_continuation[1][1] = old_solution_grads[q][dim+1][1];

//       return structure_continuation;
//    }

//   template <int dim> 
//   inline
//   Tensor<2,dim> 
//   get_grad_v (unsigned int q,
// 	      std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
//     {      
//       Tensor<2,dim> grad_v;      
//       grad_v[0][0] =  old_solution_grads[q][0][0];
//       grad_v[0][1] =  old_solution_grads[q][0][1];
//       grad_v[1][0] =  old_solution_grads[q][1][0];
//       grad_v[1][1] =  old_solution_grads[q][1][1];
      
//       return grad_v;
//    }

//   template <int dim> 
//     inline
//     Tensor<2,dim> 
//     get_grad_v_T (const Tensor<2,dim> tensor_grad_v)
//     {   
//       Tensor<2,dim> grad_v_T;
//       grad_v_T = transpose (tensor_grad_v);
            
//       return grad_v_T;      
//     }
  
//   template <int dim> 
//     inline
//     Tensor<2,dim> 
//     get_grad_v_LinV (const Tensor<2,dim> phi_i_grads_v)	 
//     {     
//         Tensor<2,dim> tmp;		 
// 	tmp[0][0] = phi_i_grads_v[0][0];
// 	tmp[0][1] = phi_i_grads_v[0][1];
// 	tmp[1][0] = phi_i_grads_v[1][0];
// 	tmp[1][1] = phi_i_grads_v[1][1];
      
// 	return tmp;
//     }

//   template <int dim> 
//     inline
//     Tensor<2,dim> 
//     get_Identity ()
//     {   
//       Tensor<2,dim> identity;
//       identity[0][0] = 1.0;
//       identity[0][1] = 0.0;
//       identity[1][0] = 0.0;
//       identity[1][1] = 1.0;
            
//       return identity;      
//    }

//  template <int dim> 
//  inline
//  Tensor<2,dim> 
//  get_F (unsigned int q,
// 	std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)
//     {     
//       Tensor<2,dim> F;
//       F[0][0] = 1.0 +  old_solution_grads[q][dim][0];
//       F[0][1] = old_solution_grads[q][dim][1];
//       F[1][0] = old_solution_grads[q][dim+1][0];
//       F[1][1] = 1.0 + old_solution_grads[q][dim+1][1];
//       return F;
//    }

//  template <int dim> 
//  inline
//  Tensor<2,dim> 
//  get_F_T (const Tensor<2,dim> F)
//     {
//       return  transpose (F);
//     }

//  template <int dim> 
//  inline
//  Tensor<2,dim> 
//  get_F_Inverse (const Tensor<2,dim> F)
//     {     
//       return invert (F);    
//     }

//  template <int dim> 
//  inline
//  Tensor<2,dim> 
//  get_F_Inverse_T (const Tensor<2,dim> F_Inverse)
//    { 
//      return transpose (F_Inverse);
//    }

//  template <int dim> 
//    inline
//    double
//    get_J (const Tensor<2,dim> tensor_F)
//    {     
//      return determinant (tensor_F);
//    }


//  template <int dim> 
//  inline
//  Tensor<1,dim> 
//  get_v (unsigned int q,
// 	std::vector<Vector<double> > old_solution_values)
//     {
//       Tensor<1,dim> v;	    
//       v[0] = old_solution_values[q](0);
//       v[1] = old_solution_values[q](1);
      
//       return v;    
//    }

//  template <int dim> 
//    inline
//    Tensor<1,dim> 
//    get_v_LinV (const Tensor<1,dim> phi_i_v)
//    {
//      Tensor<1,dim> tmp;
//      tmp[0] = phi_i_v[0];
//      tmp[1] = phi_i_v[1];
     
//      return tmp;    
//    }

//  template <int dim> 
//  inline
//  Tensor<1,dim> 
//  get_u (unsigned int q,
// 	std::vector<Vector<double> > old_solution_values)
//    {
//      Tensor<1,dim> u;     
//      u[0] = old_solution_values[q](dim);
//      u[1] = old_solution_values[q](dim+1);
     
//      return u;          
//    }

//  template <int dim> 
//    inline
//    Tensor<1,dim> 
//    get_u_LinU (const Tensor<1,dim> phi_i_u)
//    {
//      Tensor<1,dim> tmp;     
//      tmp[0] = phi_i_u[0];
//      tmp[1] = phi_i_u[1];
     
//      return tmp;    
//    }
 

//  template <int dim> 
//  inline
//  double
//  get_J_LinU (unsigned int q, 
// 	     const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
// 	     const Tensor<2,dim> phi_i_grads_u)	    
// {
//   return (phi_i_grads_u[0][0] * (1 + old_solution_grads[q][dim+1][1]) +
// 		   (1 + old_solution_grads[q][dim][0]) * phi_i_grads_u[1][1] -
// 		   phi_i_grads_u[0][1] * old_solution_grads[q][dim+1][0] - 
// 		   old_solution_grads[q][dim][1] * phi_i_grads_u[1][0]);  
// }

//   template <int dim> 
//   inline
//   double
//   get_J_Inverse_LinU (const double J,
// 		      const double J_LinU)
//     {
//       return (-1.0/std::pow(J,2) * J_LinU);
//     }

// template <int dim> 
//  inline
//  Tensor<2,dim>
//   get_F_LinU (const Tensor<2,dim> phi_i_grads_u)  
//   {
//     Tensor<2,dim> tmp;
//     tmp[0][0] = phi_i_grads_u[0][0];
//     tmp[0][1] = phi_i_grads_u[0][1];
//     tmp[1][0] = phi_i_grads_u[1][0];
//     tmp[1][1] = phi_i_grads_u[1][1];
    
//     return tmp;
//   }

// template <int dim> 
//  inline
//  Tensor<2,dim>
//   get_F_Inverse_LinU (const Tensor<2,dim> phi_i_grads_u,
// 		       const double J,
// 		       const double J_LinU,
// 		       unsigned int q,
// 		       std::vector<std::vector<Tensor<1,dim> > > old_solution_grads
// 		       )  
//   {
//     Tensor<2,dim> F_tilde;
//     F_tilde[0][0] = 1.0 + old_solution_grads[q][dim+1][1];
//     F_tilde[0][1] = -old_solution_grads[q][dim][1];
//     F_tilde[1][0] = -old_solution_grads[q][dim+1][0];
//     F_tilde[1][1] = 1.0 + old_solution_grads[q][dim][0];
    
//     Tensor<2,dim> F_tilde_LinU;
//     F_tilde_LinU[0][0] = phi_i_grads_u[1][1];
//     F_tilde_LinU[0][1] = -phi_i_grads_u[0][1];
//     F_tilde_LinU[1][0] = -phi_i_grads_u[1][0];
//     F_tilde_LinU[1][1] = phi_i_grads_u[0][0];

//     return (-1.0/(J*J) * J_LinU * F_tilde +
// 	    1.0/J * F_tilde_LinU);
 
//   }

//  template <int dim> 
//    inline
//    Tensor<2,dim>
//    get_J_F_Inverse_T_LinU (const Tensor<2,dim> phi_i_grads_u)  
//    {
//      Tensor<2,dim> tmp;
//      tmp[0][0] = phi_i_grads_u[1][1];
//      tmp[0][1] = -phi_i_grads_u[1][0];
//      tmp[1][0] = -phi_i_grads_u[0][1];
//      tmp[1][1] = phi_i_grads_u[0][0];
     
//      return  tmp;
//    }


//  template <int dim> 
//  inline
//  double
//  get_tr_C_LinU (unsigned int q, 
// 		 const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
// 		 const Tensor<2,dim> phi_i_grads_u)	    
// {
//   return ((1 + old_solution_grads[q][dim][0]) *
// 	  phi_i_grads_u[0][0] + 
// 	  old_solution_grads[q][dim][1] *
// 	  phi_i_grads_u[0][1] +
// 	  (1 + old_solution_grads[q][dim+1][1]) *
// 	  phi_i_grads_u[1][1] + 
// 	  old_solution_grads[q][dim+1][0] *
// 	  phi_i_grads_u[1][0]);
// }

 
// }

// // Second, we define the ALE transformations rules. These
// // are used to transform the fluid equations from the Eulerian
// // coordinate system to an arbitrary fixed reference 
// // configuration.
// namespace NSE_in_ALE
// {
//   template <int dim> 
//  inline
//  Tensor<2,dim>
//  get_stress_fluid_ALE (const double density,
// 		       const double viscosity,	
// 		       const Tensor<2,dim>  pI,
// 		       const Tensor<2,dim>  grad_v,
// 		       const Tensor<2,dim>  grad_v_T,
// 		       const Tensor<2,dim>  F_Inverse,
// 		       const Tensor<2,dim>  F_Inverse_T)
//   {    
//     return (-pI + density * viscosity *
// 	   (grad_v * F_Inverse + F_Inverse_T * grad_v_T ));
//   }

//   template <int dim> 
//   inline
//   Tensor<2,dim>
//   get_stress_fluid_except_pressure_ALE (const double density,
// 					const double viscosity,	
// 					const Tensor<2,dim>  grad_v,
// 					const Tensor<2,dim>  grad_v_T,
// 					const Tensor<2,dim>  F_Inverse,
// 					const Tensor<2,dim>  F_Inverse_T)
//   {
//     return (density * viscosity * (grad_v * F_Inverse + F_Inverse_T * grad_v_T));
//   }

//   template <int dim> 
//   inline
//   Tensor<2,dim> 
//   get_stress_fluid_ALE_1st_term_LinAll (const Tensor<2,dim>  pI,
// 					const Tensor<2,dim>  F_Inverse_T,
// 					const Tensor<2,dim>  J_F_Inverse_T_LinU,					    
// 					const Tensor<2,dim>  pI_LinP,
// 					const double J)
//   {          
//     return (-J * pI_LinP * F_Inverse_T - pI * J_F_Inverse_T_LinU);	     
//   }
  
//   template <int dim> 
//   inline
//   Tensor<2,dim> 
//   get_stress_fluid_ALE_2nd_term_LinAll_short (const Tensor<2,dim> J_F_Inverse_T_LinU,					    
// 					      const Tensor<2,dim> stress_fluid_ALE,
// 					      const Tensor<2,dim> grad_v,
// 					      const Tensor<2,dim> grad_v_LinV,					    
// 					      const Tensor<2,dim> F_Inverse,
// 					      const Tensor<2,dim> F_Inverse_LinU,					    
// 					      const double J,
// 					      const double viscosity,
// 					      const double density 
// 					      )  
// {
//     Tensor<2,dim> sigma_LinV;
//     Tensor<2,dim> sigma_LinU;

//     sigma_LinV = grad_v_LinV * F_Inverse + transpose(F_Inverse) * transpose(grad_v_LinV);
//     sigma_LinU = grad_v *  F_Inverse_LinU + transpose(F_Inverse_LinU) * transpose(grad_v);
 
//     return (density * viscosity * 
// 	    (sigma_LinV + sigma_LinU) * J * transpose(F_Inverse) +
// 	    stress_fluid_ALE * J_F_Inverse_T_LinU);    
//   }

// template <int dim> 
// inline
// Tensor<2,dim> 
// get_stress_fluid_ALE_3rd_term_LinAll_short (const Tensor<2,dim> F_Inverse,			   
// 					    const Tensor<2,dim> F_Inverse_LinU,					     
// 					    const Tensor<2,dim> grad_v,
// 					    const Tensor<2,dim> grad_v_LinV,					    
// 					    const double viscosity,
// 					    const double density,
// 					    const double J,
// 					    const Tensor<2,dim> J_F_Inverse_T_LinU)		    		  			     
// {
//   return density * viscosity * 
//     (J_F_Inverse_T_LinU * transpose(grad_v) * transpose(F_Inverse) +
//      J * transpose(F_Inverse) * transpose(grad_v_LinV) * transpose(F_Inverse) +
//      J * transpose(F_Inverse) * transpose(grad_v) * transpose(F_Inverse_LinU));  
// }



// template <int dim> 
// inline
// double
// get_Incompressibility_ALE (unsigned int q,
// 			   std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	 
// {
//   return (old_solution_grads[q][0][0] +
// 	  old_solution_grads[q][dim+1][1] * old_solution_grads[q][0][0] -
// 	  old_solution_grads[q][dim][1] * old_solution_grads[q][1][0] -
// 	  old_solution_grads[q][dim+1][0] * old_solution_grads[q][0][1] +
// 	  old_solution_grads[q][1][1] +
// 	  old_solution_grads[q][dim][0] * old_solution_grads[q][1][1]); 

// }

// template <int dim> 
// inline
// double
// get_Incompressibility_ALE_LinAll (const Tensor<2,dim> phi_i_grads_v,
// 				  const Tensor<2,dim> phi_i_grads_u,
// 				  unsigned int q, 				
// 				  const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads)	     	    
// {Postproce
//   return (phi_i_grads_v[0][0] + phi_i_grads_v[1][1] + 
// 	  phi_i_grads_u[1][1] * old_solution_grads[q][0][0] + old_solution_grads[q][dim+1][1] * phi_i_grads_v[0][0] -
// 	  phi_i_grads_u[0][1] * old_solution_grads[q][1][0] - old_solution_grads[q][dim+0][1] * phi_i_grads_v[1][0] -
// 	  phi_i_grads_u[1][0] * old_solution_grads[q][0][1] - old_solution_grads[q][dim+1][0] * phi_i_grads_v[0][1] +
// 	  phi_i_grads_u[0][0] * old_solution_grads[q][1][1] + old_solution_grads[q][dim+0][0] * phi_i_grads_v[1][1]);
// }


//   template <int dim> 
//   inline
//   Tensor<1,dim> 
//   get_Convection_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
// 			       const Tensor<1,dim> phi_i_v,
// 			       const double J,
// 			       const double J_LinU,
// 			       const Tensor<2,dim> F_Inverse,
// 			       const Tensor<2,dim> F_Inverse_LinU,			    			
// 			       const Tensor<1,dim> v,
// 			       const Tensor<2,dim> grad_v,				
// 			       const double density	   			     
// 			       )
//   {
//     // Linearization of fluid convection term
//     // rho J(F^{-1}v\cdot\grad)v = rho J grad(v)F^{-1}v
    
//     Tensor<1,dim> convection_LinU;
//     convection_LinU = (J_LinU * grad_v * F_Inverse * v +
// 		       J * grad_v * F_Inverse_LinU * v);
    
//     Tensor<1,dim> convection_LinV;
//     convection_LinV = (J * (phi_i_grads_v * F_Inverse * v + 
// 			    grad_v * F_Inverse * phi_i_v));
    
//     return density * (convection_LinU + convection_LinV);
//   }
  

//   template <int dim> 
//   inline
//   Tensor<1,dim> 
//   get_Convection_u_LinAll_short (const Tensor<2,dim> phi_i_grads_v,
// 				 const Tensor<1,dim> phi_i_u,Postproce
// 				 const double J,
// 				 const double J_LinU,			    
// 				 const Tensor<2,dim>  F_Inverse,
// 				 const Tensor<2,dim>  F_Inverse_LinU,
// 				 const Tensor<1,dim>  u,
// 				 const Tensor<2,dim>  grad_v,				
// 				 const double density	   			     
// 				 )
//   {
//     // Linearization of fluid convection term
//     // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u
    
//     Tensor<1,dim> convection_LinU;
//     convection_LinU = (J_LinU * grad_v * F_Inverse * u +
// 		       J * grad_v * F_Inverse_LinU * u +
// 		       J * grad_v * F_Inverse * phi_i_u);
    
//     Tensor<1,dim> convection_LinV;
//     convection_LinV = (J * phi_i_grads_v * F_Inverse * u); 
        
//     return density * (convection_LinU + convection_LinV);
// }


  
//   template <int dim> 
//   inline
//   Tensor<1,dim> 
//   get_Convection_u_old_LinAll_short (const Tensor<2,dim> phi_i_grads_v,				 
// 				     const double J,
// 				     const double J_LinU,				 
// 				     const Tensor<2,dim>  F_Inverse,
// 				     const Tensor<2,dim>  F_Inverse_LinU,				 			
// 				     const Tensor<1,dim>  old_timestep_solution_displacement,	
// 				     const Tensor<2,dim>  grad_v,				
// 				     const double density					     		       	     
// 				     )
//   {
//     // Linearization of fluid convection term
//     // rho J(F^{-1}v\cdot\grad)u = rho J grad(v)F^{-1}u
    
//     Tensor<1,dim> convection_LinU;
//     convection_LinU = (J_LinU * grad_v * F_Inverse * old_timestep_solution_displacement +
// 		       J * grad_v * F_Inverse_LinU * old_timestep_solution_displacement);
    
//     Tensor<1,dim> convection_LinV;
//     convection_LinV = (J * phi_i_grads_v * F_Inverse * old_timestep_solution_displacement); 
    
    
//     return density * (convection_LinU  + convection_LinV);
//   }

// template <int dim> 
// inline
// Tensor<1,dim> 
// get_accelaration_term_LinAll (const Tensor<1,dim> phi_i_v,
// 			      const Tensor<1,dim> v,
// 			      const Tensor<1,dim> old_timestep_v,
// 			      const double J_LinU,
// 			      const double J,
// 			      const double old_timestep_J,
// 			      const double density)
// {   
//   return density/2.0 * (J_LinU * (v - old_timestep_v) + (J + old_timestep_J) * phi_i_v);
  
// }


// }


// // In the third namespace, we summarize the 
// // constitutive relations for the solid equations.
// namespace Structure_Terms_in_ALE
// {
//   // Green-Lagrange strain tensor
//   template <int dim> 
//   inline
//   Tensor<2,dim> 
//   get_E (const Tensor<2,dim> F_T,
// 	 const Tensor<2,dim> F,
// 	 const Tensor<2,dim> Identity)
//   {    
//     return 0.5 * (F_T * F - Identity);
//   }

//   template <int dim> 
//   inline
//   double
//   get_tr_E (const Tensor<2,dim> E)
//   {     
//     return trace (E);
//   }

//   template <int dim> 
//   inline
//   double
//   get_tr_E_LinU (unsigned int q, 
// 		 const std::vector<std::vector<Tensor<1,dim> > > old_solution_grads,
// 		 const Tensor<2,dim> phi_i_grads_u)	    
//   {
//     return ((1 + old_solution_grads[q][dim][0]) *
// 	    phi_i_grads_u[0][0] + 
// 	    old_solution_grads[q][dim][1] *
// 	    phi_i_grads_u[0][1] +
// 	    (1 + old_solution_grads[q][dim+1][1]) *
// 	    phi_i_grads_u[1][1] + 
// 	    old_solution_grads[q][dim+1][0] *
// 	    phi_i_grads_u[1][0]); 
//   }
  
// }



// In this class, we define a function
// that deals with the boundary values.
// For our configuration, 
// we impose of parabolic inflow profile for the
// velocity at the left hand side of the channel. We choose
// a time dependent inflow profile with smooth 
// increase, to avoid irregularities in the initial data.			
template <int dim>
class BoundaryParabel : public Function<dim> 
{
  public:
  BoundaryParabel (const double time, const double inflow_velocity)    
    : Function<dim>(dim+dim+1) 
    {
      _time = time;    
	  _inflow_velocity= inflow_velocity;  
    }
    
  virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

  virtual void vector_value (const Point<dim> &p, 
			     Vector<double>   &value) const;

private:
  double _time;
  double _inflow_velocity;

};

// The boundary values are given to component 
// with number 0 (namely the x-velocity)
template <int dim>
double
BoundaryParabel<dim>::value (const Point<dim>  &p,
			     const unsigned int component) const
{
  Assert (component < this->n_components,
	  ExcIndexRange (component, 0, this->n_components));

//   const long double PI = 3.141592653589793238462643;
  // The maximum inflow depends on the configuration
  // for the different test cases:
  // FSI 1: 0.2; 
  // FSI 2: 1.0; 
  // FSI 3: 2.0;
  //
  // For the two unsteady test cases FSI 2 and FSI 3, it
  // is recommanded to start with a smooth increase of 
  // the inflow. Hence, we use the cosine function 
  // to control the inflow at the beginning until
  // the total time 2.0 has been reached. 
  // double inflow_velocity = 0.2;

//   if (component == 0)   
//     {
//       if (_time < 2.0)
// 	{
// 	  return   ( (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * _inflow_velocity * 
// 		     (1.0 - std::cos(PI/2.0 * _time))/2.0 * 
// 		     (4.0/0.1681) * 		     		    
// 		     (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
// 	}
//       else 
// 	{
// 	  return ( (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * _inflow_velocity * 			
// 		   (4.0/0.1681) * 		     		    
// 		   (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );
	  
// 	}

//     }
 
 if (component == 0)   
	{
	  return ( (p(0) == 0) && (p(1) <= R) ? _inflow_velocity * 			
		   (11/9) * (1 - std::pow(p(1)/R,9)) : 0 );
	  
	}

  return 0;
}



template <int dim>
void
BoundaryParabel<dim>::vector_value (const Point<dim> &p,
				    Vector<double>   &values) const 
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = BoundaryParabel<dim>::value (p, c);
}


// In the next class, we define the main problem at hand.
// Here, we implement
// the top-level logic of solving a
// time dependent FSI problem in a 
// variational-monolithic ALE framework.
//
// The initial framework of our program is based on the 
// deal.II step-22 tutorial program and the 
// FSI ANS-step (T. Wick; Archive for Numerical Software, 2013) 
// based on biharmonic mesh motion. Step-22
// explains best how to deal with vector-valued problems in
// deal.II. We extend that program by several additional elements:
//
// i)   additional non-linearity in the fluid (convection term)
//      -> requires non-linear solution algorithm
// ii)  non-linear structure problem that is fully coupled to the fluid
//      -> second source of non-linearities due to the transformation
// iii) implementation of a Newton-like method to solve the non-linear problem  
//
// To construct the ALE mapping for the fluid mesh motion, we 
// solve an additional partial differential equation that 
// is given by a nonlinear harmonic equation.
//
// All equations are written in a common global system that 
// is referred to as a variational-monolithic solution algorithm.
// 
// The discretization of the continuous problem is organized
// with Rothe's method (first time, then space): 
// - time discretization is based on finite differences 
// - spatial discretization is based on a Galerkin finite element scheme
// - the non-linear problem is solved by a Newton-like method 
//
// The  program is organized as follows. First, we set up
// runtime parameters and the system as done in other deal.II tutorial steps. 
// Then, we assemble
// the system matrix (Jacobian of Newton's method) 
// and system right hand side (residual of Newton's method) for the non-linear
// system. Two functions for the boundary values are provided because
// we are only supposed to apply boundary values in the first Newton step. In the
// subsequent Newton steps all Dirichlet values have to be equal zero.
// Afterwards, the routines for solving the linear 
// system and the Newton iteration are self-explaining. The following
// function is standard in deal.II tutorial steps:
// writing the solutions to graphical output. 
// The last three functions provide the framework to compute 
// functional values of interest. For the given fluid-structure
// interaction problem, we compute the displacement in the x- and y-directions 
// of the structure at a certain point. We are also interested in the observation
// of the drag- and lift evaluations, which are achieved by line-integration over faces
// or alternatively via domain integration.  
template <int dim>
class FSI_ALE_Problem 
{
public:
  
  FSI_ALE_Problem (const unsigned int &degree, const unsigned int &temperature_degree);
  ~FSI_ALE_Problem (); 
  void run ();
  
private:

  struct AssemblyScratchData
  {
    AssemblyScratchData (const FESystem<dim> &fe, const unsigned int &degree);
    AssemblyScratchData (const AssemblyScratchData &scratch_data);
    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;
  };
  struct AssemblyMatrixCopyData
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;
  };
  struct AssemblyRhsCopyData
  {
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
  };
	// for mass transfer
	struct AssemblyTemperatureScratchData
  {
    AssemblyTemperatureScratchData (const FiniteElement<dim> &Temperature_fe, 
																		const unsigned int &Temperature_degree,
															 			const FESystem<dim> &fe, 
																		const unsigned int &degree);
    AssemblyTemperatureScratchData (const AssemblyTemperatureScratchData &scratch_data);
    FEValues<dim>     fe_values;
		FEFaceValues<dim> fe_face_values;
		FEValues<dim> 		temperature_fe_values;
		FEFaceValues<dim> temperature_fe_face_values;
  };
  struct AssemblyTemperatureMatrixCopyData
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;
  };
  struct AssemblyTemperatureRhsCopyData
  {
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
  };

  // Setup of material parameters, time-stepping scheme
  // spatial grid, etc.
  void set_runtime_parameters ();

  // Create system matrix, rhs and distribute degrees of freedom.
  void setup_system ();

  // Assemble left and right hand side for Newton's method
  void assemble_system_matrix ();   
  void assemble_system_rhs ();
	void assemble_temperature_system_matrix ();   
  void assemble_temperature_system_rhs ();

  void local_assemble_system_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     AssemblyScratchData                                  &scratch_data,
                                     AssemblyMatrixCopyData                               &copy_data);
  void local_assemble_system_rhs (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  AssemblyScratchData                                  &scratch_data,
                                  AssemblyRhsCopyData                                  &copy_data);
	void local_assemble_temperature_system_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     						 AssemblyTemperatureScratchData                       &scratch_data,
                                     						 AssemblyTemperatureMatrixCopyData                    &copy_data);
  void local_assemble_temperature_system_rhs (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  						AssemblyTemperatureScratchData                       &scratch_data,
                                  						AssemblyTemperatureRhsCopyData                       &copy_data);

  void copy_local_to_global_matrix (const AssemblyMatrixCopyData &copy_data);
  void copy_local_to_global_rhs (const AssemblyRhsCopyData &copy_data);
	void copy_local_to_global_temperature_matrix (const AssemblyTemperatureMatrixCopyData &copy_data);
  void copy_local_to_global_temperature_rhs (const AssemblyTemperatureRhsCopyData &copy_data);

  // Boundary conditions (bc)
  void set_initial_bc (const double time);
  void set_newton_bc ();
  double inlet_flow (const double t);
  // Linear solver
  void solve ();
	void solve_temperature ();

  // Nonlinear solver
  void newton_iteration(const double time);			  

  // Graphical visualization of output
  void output_results (const unsigned int &time_step,
											 const BlockVector<double> &solution,
											 const Vector<double> &temperature_solution) const;


//   // Evaluation of functional values  
//   double compute_point_value (Point<dim> p,
// 			      const unsigned int component) const;
  
//   void compute_drag_lift_fsi_fluid_tensor ();
//   void compute_drag_lift_fsi_fluid_tensor_domain ();
//   void compute_drag_lift_fsi_fluid_tensor_domain_structure();

  void compute_functional_values ();
  void compute_minimal_J();

  // class to output F and E tensor
  class Postprocessor : public DataPostprocessor<dim>
  {
    public:
    //   Postprocessor ();
      virtual
      void
      compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                         const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                         const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                         const std::vector<Point<dim> >                  &normals,
                                         const std::vector<Point<dim> >                  &evaluation_points,
                                         std::vector<Vector<double> >                    &computed_quantities) const;
      virtual std::vector<std::string> get_names () const;
      virtual
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation () const;
      virtual UpdateFlags get_needed_update_flags () const;
  };

  // Local mesh refinement
//   void refine_mesh();
 
  const unsigned int   degree;
  
  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  ConstraintMatrix     constraints;

  // mass transfer
  const unsigned int        temperature_degree;
  FE_Q<dim>                 temperature_fe;
  DoFHandler<dim>           temperature_dof_handler;
  ConstraintMatrix 					temperature_constraints;
  SparseMatrix<double>      temperature_system_matrix;
  Vector<double>            temperature_solution;
  Vector<double>            old_temperature_solution;
  Vector<double>    				temperature_rhs;
	SparsityPattern 					tsp;

  BlockSparsityPattern      sparsity_pattern; 
  BlockSparseMatrix<double> system_matrix; 
  
  BlockVector<double> solution, newton_update, old_timestep_solution;
  BlockVector<double> system_rhs;
  
  TimerOutput         timer;
  
  // Global variables for timestepping scheme   
  unsigned int timestep_number;
  unsigned int max_no_timesteps;  
  double timestep, theta, time; 
  std::string time_stepping_scheme;

  // Fluid parameters 
  double density_fluid, viscosity; 
  
  // Structure parameters
  double density_structure; 
  double lame_mu[3], lame_lambda[3];
  double poisson_ratio_nu;  

  // Other parameters to control the fluid mesh motion 
  double cell_diameter;  
  double alpha_u;
 
  double force_structure_x, force_structure_y;
  
  double global_drag_lift_value;
//   SparseDirectMUMPS A_direct;
// SparseDirectUMFPACK A_direct;
  unsigned int solid_id[3] = { 15,16,17 }; 
  unsigned int fluid_id = 14, 		   
			   fixed_id = 20, 
			   inlet_id = 18, 
			   outlet_id= 19,
			   symmetry_id = 21;
	// NO transfer parameter.
	double k_ery, k_w;
  double R_basal, R_max, a_c, RT;
	double D_l, D_w;
};


// The constructor of this class is comparable 
// to other tutorials steps, e.g., step-22, and step-31. 
// We are going to use the following finite element discretization: 
// Q_2^c for the fluid, Q_2^c for the solid, P_1^dc for the pressure. 
template <int dim>
FSI_ALE_Problem<dim>::FSI_ALE_Problem (const unsigned int &degree, const unsigned int &temperature_degree)
                :
                degree (degree),
								triangulation (Triangulation<dim>::maximum_smoothing),
                fe (FE_Q<dim>(degree), dim,  // velocities                  
		    				FE_Q<dim>(degree), dim,  // displacements		    
		    				FE_DGP<dim>(degree-1), 1),   // pressure
                dof_handler (triangulation),
								temperature_degree(temperature_degree),
    						temperature_fe(temperature_degree),
    						temperature_dof_handler(triangulation),
								timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)		
{}


// This is the standard destructor.
template <int dim>
FSI_ALE_Problem<dim>::~FSI_ALE_Problem () 
{}


template <int dim>
void
FSI_ALE_Problem<dim>::Postprocessor::
compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
																		const std::vector<std::vector<Tensor<1,dim> > > &duh,
																		const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
																		const std::vector<Point<dim> >                  &/*normals*/,
																		const std::vector<Point<dim> >                  &/*evaluation_points*/,
																		std::vector<Vector<double> >                    &computed_quantities) const
{
	const unsigned int n_quadrature_points = uh.size();
	Assert (duh.size() == n_quadrature_points,
					ExcInternalError())

	Assert (computed_quantities.size() == n_quadrature_points,
					ExcInternalError());
// assert dim*2+1component
	Assert (uh[0].size() == (dim+dim+1),
					ExcInternalError());
	
	Assert (computed_quantities[0].size() == 10, ExcInternalError())
	Tensor<2,dim> identity;
	identity.clear();
	identity[0][0] = 1.0;
	identity[0][1] = 0.0;
	identity[1][0] = 0.0;
	identity[1][1] = 1.0;

	for (unsigned int q=0; q<n_quadrature_points; ++q)
		{
		Tensor<2, dim> F;
		F.clear();
		F[0][0] = duh[q][dim  ][0] + 1.0;
		F[0][1] = duh[q][dim  ][1];
		F[1][0] = duh[q][dim+1][0];
		F[1][1] = duh[q][dim+1][1] + 1.0;
		Tensor<2, dim> E;
		E.clear();
		E = transpose (F) * F - identity;
		double output_trace_E = trace(E);
		double output_J = determinant(F);
		computed_quantities[q](0) = F[0][0];
		computed_quantities[q](1) = F[0][1];
		computed_quantities[q](2) = F[1][0];
		computed_quantities[q](3) = F[1][1];
		computed_quantities[q](4) = E[0][0];
		computed_quantities[q](5) = E[0][1];
		computed_quantities[q](6) = E[1][0];
		computed_quantities[q](7) = E[1][1];
		computed_quantities[q](8) = output_trace_E;
		computed_quantities[q](9) = output_J;
		}
}

template <int dim>
std::vector<std::string>
FSI_ALE_Problem<dim>::Postprocessor::
get_names () const
{
	std::vector<std::string> names;
	names.push_back ("Fxx");
	names.push_back ("Fxy");
	names.push_back ("Fyx");
	names.push_back ("Fyy");
	names.push_back ("Exx");
	names.push_back ("Exy");
	names.push_back ("Eyx");
	names.push_back ("Eyy");
	names.push_back ("trace_E");
	names.push_back ("J");
	return names;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
FSI_ALE_Problem<dim>::Postprocessor::
get_data_component_interpretation () const
{
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	interpretation (10,
									DataComponentInterpretation::component_is_scalar);

	return interpretation;
}

template <int dim>
UpdateFlags
FSI_ALE_Problem<dim>::Postprocessor::
get_needed_update_flags () const
{
	return update_values | update_gradients;
}

// In this method, we set up runtime parameters that 
// could also come from a paramter file. We propose
// three different configurations FSI 1, FSI 2, and FSI 3.
// The reader is invited to change these values to obtain
// other results. 
template <int dim>
void FSI_ALE_Problem<dim>::set_runtime_parameters ()
{
   // Fluid parameters
  density_fluid = 1.05e+3;

  // FSI 1 & 3: 1.0e+3; FSI 2: 1.0e+4
  density_structure =  2.2e+3; 
  viscosity = 3.45e-3;  


	k_ery 	= -23;    	// s^{-1}  negative for consume
	k_w 		= -0.01;		// s^{-1}
	R_basal = 2.13; 		// nM/s
	R_max 	= 457.5;  	// nM/s
	a_c 		= 3.5;			// Pa or kg/m/s^2
	D_w 		= 8.48e-10; // m^2/s	
	D_l 		= 3.3e-9;		// m^2/s
	RT   		= 2e-6;			// m or 2 \mu m
  // Structure parameters
  // FSI 1 & 2: 0.5e+6; FSI 3: 2.0e+6
  double E[3];
  E[0] = 2e+6;
  E[1] = 6e+6;
  E[2] = 4e+6; 

  poisson_ratio_nu = 0.45; 

  double tmp1 = poisson_ratio_nu/((1+poisson_ratio_nu)*(1-2*poisson_ratio_nu));
  double tmp2 = 2*(1+poisson_ratio_nu);

  for(int i=0;i<3;++i){
	  lame_mu[i] = E[i]/tmp2;
	  lame_lambda[i]= E[i]/tmp1;
  }
  
  // Force on beam
  force_structure_x = 0; 
  force_structure_y = 0; 


  // Diffusion parameters to control the fluid mesh motion
  // The higher these parameters the stiffer the fluid mesh.
  alpha_u = 1e-8; 

  // Timestepping schemes
  //BE, CN, CN_shifted
  time_stepping_scheme = "CN_shifted";

  // Timestep size:
  // FSI 1: 1.0 (quasi-stationary)
  // FSI 2: <= 1.0e-2 (non-stationary)
  // FSI 3: <= 1.0e-3 (non-stationary)
  timestep = 5e-3;

  // Maximum number of timesteps:
  // FSI 1: 25 , T= 25   (timestep == 1.0)
  // FSI 2: 1500, T= 15  (timestep == 1.0e-2)
  // FSI 3: 10000, T= 10 (timestep == 1.0e-3)
  max_no_timesteps = 800;
  
  // A variable to count the number of time steps
  timestep_number = 0;

  // Counts total time  
  time = 0;
 
  // Here, we choose a time-stepping scheme that
  // is based on finite differences:
  // BE         = backward Euler scheme 
  // CN         = Crank-Nicolson scheme
  // CN_shifted = time-shifted Crank-Nicolson scheme 
  // For further properties of these schemes,
  // we refer to standard literature.
//   if (time_stepping_scheme == "BE")
//     theta = 1.0;
//   else if (time_stepping_scheme == "CN")
//     theta = 0.5;
//   else if (time_stepping_scheme == "CN_shifted")
//     theta = 0.5 + timestep;
//   else 
//     {
// 		std::cout << "No such timestepping scheme!" << std::endl;
// 		std::cout << "Use Crank-Nicolson as default." << std::endl;
// 		theta = 0.5;
// 		}
  theta = 0.5 + timestep;
  // In the following, we read a *.inp grid from a file.
  // The geometry information is based on the 
  // fluid-structure interaction benchmark problems 
  // (Lit. J. Hron, S. Turek, 2006)
  std::string grid_name;
//   grid_name  = parameters.mesh_file; 
  grid_name = "Khanafer_half.msh";
  std::cout << "\n================================\n"
 	        << "Mesh file: " << grid_name << std::endl;
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  Assert (dim==2, ExcInternalError());
//   if( grid_name.find('.inp')!=std::string::npos )
// 	std::cout << ".inp mesh file, using 'read_ucd' to read mesh..." << std::endl;
//   grid_in.read_ucd (input_file); 
  grid_in.read_msh(input_file);

//   Point<dim> p(0.2, 0.2);
//   double radius = 0.05;
//   static const HyperBallBoundary<dim> boundary(p,radius);
//   triangulation.set_boundary (80, boundary);
//   triangulation.set_boundary (81, boundary);
    
//   triangulation.refine_global (parameters.no_of_refinements); 
 
  std::cout << "\n==============================" 
	    << "====================================="  << std::endl;
  std::cout << "Parameters\n" 
	    << "==========\n"
	    << "Density fluid:        "   <<  density_fluid << "\n"
	    << "Density structure:    "   <<  density_structure << "\n"  
	    << "Viscosity fluid:      "   <<  viscosity << "\n"
	    << "alpha_u:              "   <<  alpha_u << "\n"
	    << "Lame coeff. mu: "   << lame_mu[0] << " " << lame_mu[1] << " " << lame_mu[2] << "\n\n"
		<< "time scheme:          "   <<  time_stepping_scheme << "\n"
		<< "max timesteps:        "   <<  max_no_timesteps << "\n"
		<< "length of time step:  "   <<  timestep << "\n"
		<< "Degree of polynomial: "   <<  degree << " order.\n"
	    << std::endl;
}



// This function is similar to many deal.II tuturial steps.
template <int dim>
void FSI_ALE_Problem<dim>::setup_system ()
{
  timer.enter_section("Setup system.");

  system_matrix.clear ();
  
  dof_handler.distribute_dofs (fe);  
//   DoFRenumbering::Cuthill_McKee (dof_handler);
  DoFRenumbering::boost::king_ordering (dof_handler);

  {
      temperature_dof_handler.distribute_dofs(temperature_fe);
	  DoFRenumbering::boost::king_ordering (temperature_dof_handler);
      temperature_constraints.clear();
      DoFTools::make_hanging_node_constraints(temperature_dof_handler,
                                              temperature_constraints);
      temperature_constraints.close();
    }

  // We are dealing with 7 components for this 
  // two-dimensional fluid-structure interacion problem
  // Precisely, we use:
  // velocity in x and y:                0
  // structure displacement in x and y:  1
  // scalar pressure field:              2
  std::vector<unsigned int> block_component (5,0);
  block_component[dim] = 1;
  block_component[dim+1] = 1;
  block_component[dim+dim] = 2;
 
  DoFRenumbering::component_wise (dof_handler, block_component);

  {				 
    constraints.clear ();
    set_newton_bc ();
    DoFTools::make_hanging_node_constraints (dof_handler,
					     constraints);
  }
  constraints.close ();
  
  std::vector<unsigned int> dofs_per_block (3);
  DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);  
  const unsigned int n_v = dofs_per_block[0],
    n_u = dofs_per_block[1],
    n_p =  dofs_per_block[2];
  const unsigned int n_T = temperature_dof_handler.n_dofs();
  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl  	  
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << " (" << n_v << "+" << n_u << "+" << n_p <<  ")\n"
						<< "DoF of T is: " << n_T
            << std::endl;


 
      
 {
    BlockCompressedSimpleSparsityPattern csp (3,3);

    csp.block(0,0).reinit (n_v, n_v);
    csp.block(0,1).reinit (n_v, n_u);
    csp.block(0,2).reinit (n_v, n_p);
  
    csp.block(1,0).reinit (n_u, n_v);
    csp.block(1,1).reinit (n_u, n_u);
    csp.block(1,2).reinit (n_u, n_p);
  
    csp.block(2,0).reinit (n_p, n_v);
    csp.block(2,1).reinit (n_p, n_u);
    csp.block(2,2).reinit (n_p, n_p);
 
    csp.collect_sizes();    
  

    DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);

    sparsity_pattern.copy_from (csp);
  }
 
 system_matrix.reinit (sparsity_pattern);

  {
		temperature_system_matrix.clear();
		CompressedSimpleSparsityPattern tcsp (n_T, n_T);
    DoFTools::make_sparsity_pattern (temperature_dof_handler, tcsp,
                                     temperature_constraints, false);
		temperature_constraints.condense (tcsp);
		
		tsp.copy_from (tcsp);
    temperature_system_matrix.reinit (tsp);
		// temperature_system_matrix.clear();
		// CompressedSimpleSparsityPattern tcsp (n_T, n_T);
    // DoFTools::make_sparsity_pattern (temperature_dof_handler, tcsp,
    //                                  temperature_constraints, false);
		// temperature_system_matrix.reinit (tcsp);
  }



  // Actual solution at time step n
  solution.reinit (3);
  solution.block(0).reinit (n_v);
  solution.block(1).reinit (n_u);
  solution.block(2).reinit (n_p);
 
  solution.collect_sizes ();
 
  // Old timestep solution at time step n-1
  old_timestep_solution.reinit (3);
  old_timestep_solution.block(0).reinit (n_v);
  old_timestep_solution.block(1).reinit (n_u);
  old_timestep_solution.block(2).reinit (n_p);
 
  old_timestep_solution.collect_sizes ();


  // Updates for Newton's method
  newton_update.reinit (3);
  newton_update.block(0).reinit (n_v);
  newton_update.block(1).reinit (n_u);
  newton_update.block(2).reinit (n_p);
 
  newton_update.collect_sizes ();
 
  // Residual for  Newton's method
  system_rhs.reinit (3);
  system_rhs.block(0).reinit (n_v);
  system_rhs.block(1).reinit (n_u);
  system_rhs.block(2).reinit (n_p);

  system_rhs.collect_sizes ();
  

  temperature_solution.reinit (n_T);
  old_temperature_solution.reinit (n_T);
  temperature_rhs.reinit (n_T);

  timer.exit_section(); 
}



template <int dim>
FSI_ALE_Problem<dim>::AssemblyScratchData::
AssemblyScratchData (const FESystem<dim> &fe, const unsigned int &degree)
  :
  fe_values (fe,
             QGauss<dim>(degree+2),
             update_values   | update_gradients |
             update_quadrature_points | update_JxW_values),
  fe_face_values (fe,
                  QGauss<dim-1>(degree+2),
                  update_values         | update_quadrature_points  |
									update_normal_vectors | update_gradients |
									update_JxW_values)
{}

template <int dim>
FSI_ALE_Problem<dim>::AssemblyScratchData::
AssemblyScratchData (const AssemblyScratchData &scratch_data)
  :
  fe_values (scratch_data.fe_values.get_fe(),
             scratch_data.fe_values.get_quadrature(),
             update_values   | update_gradients |
             update_quadrature_points | update_JxW_values),
  fe_face_values (scratch_data.fe_face_values.get_fe(),
                  scratch_data.fe_face_values.get_quadrature(),
                  update_values         | update_quadrature_points  |
									update_normal_vectors | update_gradients |
									update_JxW_values)
{}

template <int dim>
void
FSI_ALE_Problem<dim>::copy_local_to_global_matrix (const AssemblyMatrixCopyData &copy_data)
{
	constraints.distribute_local_to_global (copy_data.cell_matrix,
                                            copy_data.local_dof_indices,
                                            system_matrix);

}

template <int dim>
void
FSI_ALE_Problem<dim>::copy_local_to_global_rhs (const AssemblyRhsCopyData &copy_data)
{
	constraints.distribute_local_to_global (copy_data.cell_rhs,
                                            copy_data.local_dof_indices,
                                            system_rhs);

}

template <int dim>
void FSI_ALE_Problem<dim>::assemble_system_matrix ()
{
  timer.enter_section("Assemble Matrix.");
  system_matrix=0;
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &FSI_ALE_Problem::local_assemble_system_matrix,
                  &FSI_ALE_Problem::copy_local_to_global_matrix,
                  AssemblyScratchData(fe,degree),
                  AssemblyMatrixCopyData());
  timer.exit_section();
}


template <int dim>
void FSI_ALE_Problem<dim>::assemble_system_rhs ()
{
  timer.enter_section("Assemble Rhs.");
  system_rhs=0;
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &FSI_ALE_Problem::local_assemble_system_rhs,
                  &FSI_ALE_Problem::copy_local_to_global_rhs,
                  AssemblyScratchData(fe,degree),
                  AssemblyRhsCopyData());
  timer.exit_section();
}

template <int dim>
FSI_ALE_Problem<dim>::AssemblyTemperatureScratchData::
AssemblyTemperatureScratchData (const FiniteElement<dim> &Temperature_fe, 
																const unsigned int &Temperature_degree,
																const FESystem<dim> &fe, 
																const unsigned int &degree)
  :
  fe_values (fe,
             QGauss<dim>(degree+2),
             update_values   | update_gradients |
             update_quadrature_points | update_JxW_values),
  fe_face_values (fe,
                  QGauss<dim-1>(degree+2),
                  update_values         | update_quadrature_points  |
				  				update_normal_vectors | update_gradients |
				  				update_JxW_values),
	temperature_fe_values (Temperature_fe,
											QGauss<dim>(Temperature_degree+2),
											update_values   | update_gradients |
											update_quadrature_points | update_JxW_values),
	temperature_fe_face_values (Temperature_fe,
															QGauss<dim-1>(Temperature_degree+2),
															update_values         | update_quadrature_points  |
															update_normal_vectors | update_gradients |
															update_JxW_values)
{}

template <int dim>
FSI_ALE_Problem<dim>::AssemblyTemperatureScratchData::
AssemblyTemperatureScratchData (const AssemblyTemperatureScratchData &scratch_data)
  :
  fe_values (scratch_data.fe_values.get_fe(),
             scratch_data.fe_values.get_quadrature(),
             update_values   | update_gradients |
             update_quadrature_points | update_JxW_values),
  fe_face_values (scratch_data.fe_face_values.get_fe(),
                  scratch_data.fe_face_values.get_quadrature(),
                  update_values         | update_quadrature_points  |
									update_normal_vectors | update_gradients |
									update_JxW_values),
	temperature_fe_values (scratch_data.temperature_fe_values.get_fe(),
													scratch_data.temperature_fe_values.get_quadrature(),
													update_values   | update_gradients |
													update_quadrature_points | update_JxW_values),
  temperature_fe_face_values (scratch_data.temperature_fe_face_values.get_fe(),
															scratch_data.temperature_fe_face_values.get_quadrature(),
															update_values         | update_quadrature_points  |
															update_normal_vectors | update_gradients |
															update_JxW_values)
{}

template <int dim>
void
FSI_ALE_Problem<dim>::copy_local_to_global_temperature_matrix (const AssemblyTemperatureMatrixCopyData &copy_data)
{
	temperature_constraints.distribute_local_to_global (copy_data.cell_matrix,
																											copy_data.local_dof_indices,
																											temperature_system_matrix);

}

template <int dim>
void
FSI_ALE_Problem<dim>::copy_local_to_global_temperature_rhs (const AssemblyTemperatureRhsCopyData &copy_data)
{
	temperature_constraints.distribute_local_to_global (copy_data.cell_rhs,
																											copy_data.local_dof_indices,
																											temperature_rhs);
}

template <int dim>
void FSI_ALE_Problem<dim>::assemble_temperature_system_matrix ()
{
  timer.enter_section("Assemble Temperature Matrix.");
	std::cout << "start assemble T matrix1" << std::endl;
  temperature_system_matrix=0;
	std::cout << "start assemble T matrix2" << std::endl;
  WorkStream::run(temperature_dof_handler.begin_active(),
                  temperature_dof_handler.end(),
                  *this,
                  &FSI_ALE_Problem::local_assemble_temperature_system_matrix,
                  &FSI_ALE_Problem::copy_local_to_global_temperature_matrix,
                  AssemblyTemperatureScratchData(temperature_fe,temperature_degree,fe,degree),
                  AssemblyTemperatureMatrixCopyData());
	std::cout << "Assemble T matrix over" << std::endl;
  timer.exit_section();
}



template <int dim>
void FSI_ALE_Problem<dim>::assemble_temperature_system_rhs ()
{
  timer.enter_section("Assemble Temperature Rhs.");
  temperature_rhs=0;
  WorkStream::run(temperature_dof_handler.begin_active(),
                  temperature_dof_handler.end(),
                  *this,
                  &FSI_ALE_Problem::local_assemble_temperature_system_rhs,
                  &FSI_ALE_Problem::copy_local_to_global_temperature_rhs,
                  AssemblyTemperatureScratchData(temperature_fe,temperature_degree,fe,degree),
                  AssemblyTemperatureRhsCopyData());
  timer.exit_section();
}

// In this function, we assemble the Jacobian matrix
// for the Newton iteration. The fluid and the structure 
// equations are computed on different sub-domains
// in the mesh and ask for the corresponding 
// material ids. The fluid equations are defined on 
// mesh cells with the material id == 0 and the structure
// equations on cells with the material id == 1. 
//
// To compensate the well-known problem in fluid
// dynamics on the outflow boundary, we also
// add some correction term on the outflow boundary.
// This relation is known as `do-nothing' condition.
// In the inner loops of the local_cell_matrix, the 
// time dependent equations are discretized with
// a finite difference scheme. 
// Quasi-stationary processes (FSI 1) can be computed 
// by the BE scheme. The other two schemes are useful 
// for non-stationary computations (FSI 2 and FSI 3).
//
// Assembling of the inner most loop is treated with help of 
// the fe.system_to_component_index(j).first function from
// the library. 
// Using this function makes the assembling process much faster
// than running over all local degrees of freedom. 
template <int dim>
void
FSI_ALE_Problem<dim>::
local_assemble_system_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                              AssemblyScratchData                                  &scratch_data,
                              AssemblyMatrixCopyData                               &copy_data)
{

  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int   n_q_points      = scratch_data.fe_values.get_quadrature().size();
  const unsigned int   n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

  copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);

  copy_data.local_dof_indices.resize(dofs_per_cell); 
  cell->get_dof_indices (copy_data.local_dof_indices);

  // Now, we are going to use the 
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector velocities (0); // 0
  const FEValuesExtractors::Vector displacements (dim); // 2
  const FEValuesExtractors::Scalar pressure (dim+dim); // 4

  // We declare Vectors and Tensors for 
  // the solutions at the previous Newton iteration:
  std::vector<Vector<double> > old_solution_values (n_q_points, 
				 		    Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								std::vector<Tensor<1,dim> > (dim+dim+1));

  std::vector<Vector<double> >  old_solution_face_values (n_face_q_points, 
							  Vector<double>(dim+dim+1));
       
  std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points, 
								     std::vector<Tensor<1,dim> > (dim+dim+1));
    

  // We declare Vectors and Tensors for 
  // the solution at the previous time step:
  std::vector<Vector<double> > old_timestep_solution_values (n_q_points, 
				 		    Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points, 
  					  std::vector<Tensor<1,dim> > (dim+dim+1));

  std::vector<Vector<double> >   old_timestep_solution_face_values (n_face_q_points, 
								    Vector<double>(dim+dim+1));
  
  std::vector<std::vector<Tensor<1,dim> > >  old_timestep_solution_face_grads (n_face_q_points, 
									       std::vector<Tensor<1,dim> > (dim+dim+1));
   
  // Declaring test functions:
  std::vector<Tensor<1,dim> > phi_i_v (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_v(dofs_per_cell);
  std::vector<double>         phi_i_p(dofs_per_cell);   
  std::vector<Tensor<1,dim> > phi_i_u (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_u(dofs_per_cell);

  // This is the identity matrix in two dimensions:
  const Tensor<2,dim> Identity = ALE_Transformations
    ::get_Identity<dim> ();
 				     				   
      scratch_data.fe_values.reinit (cell);
      copy_data.cell_matrix = 0;
      
      // We need the cell diameter to control the fluid mesh motion
      cell_diameter = cell->diameter();
      
      // Old Newton iteration values
      scratch_data.fe_values.get_function_values (solution, old_solution_values);
      scratch_data.fe_values.get_function_gradients (solution, old_solution_grads);
      
      // Old_timestep_solution values
      scratch_data.fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      scratch_data.fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
      // Next, we run over all cells for the fluid equations
      if (cell->material_id() == fluid_id)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_i_v[k]       = scratch_data.fe_values[velocities].value (k, q);
		  phi_i_grads_v[k] = scratch_data.fe_values[velocities].gradient (k, q);
		  phi_i_p[k]       = scratch_data.fe_values[pressure].value (k, q);			      			 
		  phi_i_u[k]       = scratch_data.fe_values[displacements].value (k, q);
		  phi_i_grads_u[k] = scratch_data.fe_values[displacements].gradient (k, q);
		}
	      
	      // We build values, vectors, and tensors
	      // from information of the previous Newton step. These are introduced 
	      // for two reasons:
	      // First, these are used to perform the ALE mapping of the 
	      // fluid equations. Second, these terms are used to 
	      // make the notation as simple and self-explaining as possible:
	      const Tensor<2,dim> pI = ALE_Transformations		
		::get_pI<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> v = ALE_Transformations
		::get_v<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> u = ALE_Transformations
		::get_u<dim> (q,old_solution_values);
	      	    	      
	      const Tensor<2,dim> grad_v = ALE_Transformations
		::get_grad_v<dim> (q, old_solution_grads);	
	      
	      const Tensor<2,dim> grad_v_T = ALE_Transformations
		::get_grad_v_T<dim> (grad_v);

	      const Tensor<2,dim> grad_u = ALE_Transformations
		::get_grad_u<dim> (q, old_solution_grads);
	      
	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);	    
	      
	      const Tensor<2,dim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (F);
	      
	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (F_Inverse);
	      
	      const double J = ALE_Transformations
		::get_J<dim> (F);

	      
	      // Stress tensor for the fluid in ALE notation	      
	      const Tensor<2,dim> sigma_ALE = NSE_in_ALE
		::get_stress_fluid_ALE<dim> (density_fluid, viscosity, pI,
					     grad_v, grad_v_T, F_Inverse, F_Inverse_T );
	      
	      // Further, we also need some information from the previous time steps
	      const Tensor<1,dim> old_timestep_v = ALE_Transformations
		::get_v<dim> (q, old_timestep_solution_values);

	      const Tensor<1,dim> old_timestep_u = ALE_Transformations
		::get_u<dim> (q, old_timestep_solution_values);
	      
	      const Tensor<2,dim> old_timestep_F = ALE_Transformations
		::get_F<dim> (q, old_timestep_solution_grads);
	      
	      const double old_timestep_J = ALE_Transformations
		::get_J<dim> (old_timestep_F);
	      
	      // Outer loop for dofs
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	
		  const Tensor<2,dim> pI_LinP = ALE_Transformations
		    ::get_pI_LinP<dim> (phi_i_p[i]);
		  
		  const Tensor<2,dim> grad_v_LinV = ALE_Transformations
		    ::get_grad_v_LinV<dim> (phi_i_grads_v[i]);
		  
		  const double J_LinU =  ALE_Transformations
		    ::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]);
		  
		  const Tensor<2,dim> J_F_Inverse_T_LinU = ALE_Transformations
		    ::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);
		  
		  const Tensor<2,dim> F_Inverse_LinU = ALE_Transformations
		    ::get_F_Inverse_LinU (phi_i_grads_u[i], J, J_LinU, q, old_solution_grads);
		    
		  const Tensor<2,dim>  stress_fluid_ALE_1st_term_LinAll = NSE_in_ALE			
		    ::get_stress_fluid_ALE_1st_term_LinAll<dim> 
		    (pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);
						      
		  const Tensor<2,dim> stress_fluid_ALE_2nd_term_LinAll = NSE_in_ALE
		    ::get_stress_fluid_ALE_2nd_term_LinAll_short 
		    (J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,								      
		     F_Inverse, F_Inverse_LinU,	J, viscosity, density_fluid);  

		  const Tensor<1,dim> convection_fluid_LinAll_short = NSE_in_ALE		    
		    ::get_Convection_LinAll_short<dim> 
		    (phi_i_grads_v[i], phi_i_v[i], J,J_LinU,						
		     F_Inverse, F_Inverse_LinU, v, grad_v, density_fluid);
	   
		  const double incompressibility_ALE_LinAll = NSE_in_ALE
		    ::get_Incompressibility_ALE_LinAll<dim> 
		    (phi_i_grads_v[i], phi_i_grads_u[i], q, old_solution_grads); 
	     	    	      	    	     
		  const Tensor<1,dim> accelaration_term_LinAll = NSE_in_ALE
		    ::get_accelaration_term_LinAll 
		    (phi_i_v[i], v, old_timestep_v, J_LinU,
		     J, old_timestep_J, density_fluid);
	      
		  const Tensor<1,dim> convection_fluid_u_LinAll_short =  NSE_in_ALE
		    ::get_Convection_u_LinAll_short<dim> 
		    (phi_i_grads_v[i], phi_i_u[i], J,J_LinU, F_Inverse,
		     F_Inverse_LinU, u, grad_v, density_fluid);

		  const Tensor<1,dim> convection_fluid_u_old_LinAll_short = NSE_in_ALE
		    ::get_Convection_u_old_LinAll_short<dim> 
		    (phi_i_grads_v[i], J, J_LinU, F_Inverse,
		     F_Inverse_LinU, old_timestep_u, grad_v, density_fluid);

		  // Inner loop for dofs
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {	
		      // Fluid , NSE in ALE
		      const unsigned int comp_j = fe.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1)
			{		
			  copy_data.cell_matrix(j,i) += (accelaration_term_LinAll * phi_i_v[j] +   
						timestep * theta *					  
						convection_fluid_LinAll_short * phi_i_v[j] - 					      
						convection_fluid_u_LinAll_short * phi_i_v[j] +
						convection_fluid_u_old_LinAll_short * phi_i_v[j] +
						timestep * scalar_product(stress_fluid_ALE_1st_term_LinAll, phi_i_grads_v[j]) +
						timestep * theta *
						scalar_product(stress_fluid_ALE_2nd_term_LinAll, phi_i_grads_v[j]) 					 
						) * scratch_data.fe_values.JxW(q);
			}					    
		      else if (comp_j == 2 || comp_j == 3)
			{
			  // Nonlinear harmonic MMPDE
			  copy_data.cell_matrix(j,i) += (-alpha_u/(J*J) * J_LinU * scalar_product(grad_u, phi_i_grads_u[j]) 
						+ alpha_u/J * scalar_product(phi_i_grads_u[i], phi_i_grads_u[j])
						) * scratch_data.fe_values.JxW(q);

			}
		      else if (comp_j == 4)
			{
			  copy_data.cell_matrix(j,i) += (incompressibility_ALE_LinAll *  phi_i_p[j] 
						) * scratch_data.fe_values.JxW(q);		
			}
		      // end j dofs  
		    }   
		  // end i dofs	  
		}   
	      // end n_q_points  
	    }    
	  	  
	  // We compute in the following
	  // one term on the outflow boundary. 
	  // This relation is well-know in the literature 
	  // as "do-nothing" condition. Therefore, we only
	  // ask for the corresponding color at the outflow 
	  // boundary that is 1 in our case.
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->face(face)->at_boundary() &&		  
		  (cell->face(face)->boundary_indicator() == outlet_id) 
		  )
		{
		  
		  scratch_data.fe_face_values.reinit (cell, face);
		  
		  scratch_data.fe_face_values.get_function_values (solution, old_solution_face_values);
		  scratch_data.fe_face_values.get_function_gradients (solution, old_solution_face_grads);	
		  
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {
		      for (unsigned int k=0; k<dofs_per_cell; ++k)
			{
			  phi_i_v[k]       = scratch_data.fe_face_values[velocities].value (k, q);
			  phi_i_grads_v[k] = scratch_data.fe_face_values[velocities].gradient (k, q);		
			  phi_i_grads_u[k] = scratch_data.fe_face_values[displacements].gradient (k, q);
			}
		      
		      const Tensor<2,dim>  grad_v = ALE_Transformations
			::get_grad_v<dim> (q, old_solution_face_grads);
		      
		      const Tensor<2,dim> F = ALE_Transformations
			::get_F<dim> (q, old_solution_face_grads);
		      
		      const Tensor<2,dim> F_Inverse = ALE_Transformations
			::get_F_Inverse<dim> (F);
		      
		      const double J = ALE_Transformations
			::get_J<dim> (F);
		      
		      
		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  const Tensor<2,dim> grad_v_LinV = ALE_Transformations
			    ::get_grad_v_LinV<dim> (phi_i_grads_v[i]);
			  
			  const double J_LinU = ALE_Transformations
			    ::get_J_LinU<dim> (q, old_solution_face_grads, phi_i_grads_u[i]);
					       			  
			  const Tensor<2,dim> J_F_Inverse_T_LinU = ALE_Transformations
			    ::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);
			  
			  const Tensor<2,dim> F_Inverse_LinU = ALE_Transformations
			    ::get_F_Inverse_LinU 
			    (phi_i_grads_u[i], J, J_LinU, q, old_solution_face_grads);
			  
			  const Tensor<2,dim> stress_fluid_ALE_3rd_term_LinAll =  NSE_in_ALE
			    ::get_stress_fluid_ALE_3rd_term_LinAll_short<dim> 
			    (F_Inverse, F_Inverse_LinU, grad_v, grad_v_LinV,
			     viscosity, density_fluid, J, J_F_Inverse_T_LinU);
			  	
			  // Here, we multiply the symmetric part of fluid's stress tensor
			  // with the normal direction.
			  const Tensor<1,dim> neumann_value
			    = (stress_fluid_ALE_3rd_term_LinAll * scratch_data.fe_face_values.normal_vector(q));
			  
			  for (unsigned int j=0; j<dofs_per_cell; ++j)
			    {		     
			      const unsigned int comp_j = fe.system_to_component_index(j).first; 
			      if (comp_j == 0 || comp_j == 1)
				{
				  copy_data.cell_matrix(j,i) -= 1.0 * (timestep * theta *
							neumann_value * phi_i_v[j] 
							) * scratch_data.fe_face_values.JxW(q);
				}
			      // end j    
			    } 
			  // end i
			}   
		      // end q_face_points
		    } 
		  // end if-routine face integrals
		}  	      
	      // end face integrals do-nothing
	    }   

	  
	  // This is the same as discussed in step-22:
	//   cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_matrix, local_dof_indices,
	// 					  system_matrix);
	  
	  // Finally, we arrive at the end for assembling the matrix
	  // for the fluid equations and step to the computation of the 
	  // structure terms:
	} 
      else if (cell->material_id() == solid_id[0] || cell->material_id() == solid_id[1] || cell->material_id() == solid_id[2])
	{	  
		double lame_coefficient_mu, lame_coefficient_lambda;
		 if (cell->material_id() == solid_id[0]){
		 	lame_coefficient_mu = lame_mu[0];
			lame_coefficient_lambda = lame_lambda[0];
		 }
		 if (cell->material_id() == solid_id[1]){
		 	lame_coefficient_mu = lame_mu[1];
			lame_coefficient_lambda = lame_lambda[1];
		 }
		if (cell->material_id() == solid_id[2]){
		 	lame_coefficient_mu = lame_mu[2];
			lame_coefficient_lambda = lame_lambda[2];
		}
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	      
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_i_v[k]       = scratch_data.fe_values[velocities].value (k, q);
		  phi_i_grads_v[k] = scratch_data.fe_values[velocities].gradient (k, q);
		  phi_i_p[k]       = scratch_data.fe_values[pressure].value (k, q);			      			 
		  phi_i_u[k]       = scratch_data.fe_values[displacements].value (k, q);
		  phi_i_grads_u[k] = scratch_data.fe_values[displacements].gradient (k, q);
		}
	      
	      // It is here the same as already shown for the fluid equations.
	      // First, we prepare things coming from the previous Newton
	      // iteration...
	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);
	      
	      const Tensor<2,dim> F_T = ALE_Transformations
		::get_F_T<dim> (F);
	      

	      const Tensor<2,dim> E = Structure_Terms_in_ALE 
		::get_E<dim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (E);
std::vector<unsigned int> local_dof_indices (dofs_per_cell);
	      	      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	    	     		
		  const Tensor<2,dim> F_LinU = ALE_Transformations		  
		    ::get_F_LinU<dim> (phi_i_grads_u[i]);
		  
		     		       
		  // STVK: Green-Lagrange strain tensor derivatives
		  const Tensor<2,dim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);
		  
		  const double tr_E_LinU = Structure_Terms_in_ALE
		    ::get_tr_E_LinU<dim> (q,old_solution_grads, phi_i_grads_u[i]);
		  
		       
		  // STVK
		  // Piola-kirchhoff stress structure STVK linearized in all directions 		  
		  Tensor<2,dim> piola_kirchhoff_stress_structure_STVK_LinALL;
		  piola_kirchhoff_stress_structure_STVK_LinALL = lame_coefficient_lambda * 
		    (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) 
		    + 2 * lame_coefficient_mu * (F_LinU * E + F * E_LinU);
		       
			   
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      // STVK 
		      const unsigned int comp_j = fe.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1)
			{
			  copy_data.cell_matrix(j,i) += (density_structure * phi_i_v[i] * phi_i_v[j] +   						   
						timestep * theta * scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL, 
										  phi_i_grads_v[j]) 
						) * scratch_data.fe_values.JxW(q);      	
			}		     
		      else if (comp_j == 2 || comp_j == 3)
			{
			  copy_data.cell_matrix(j,i) += (density_structure * 
						(phi_i_u[i] * phi_i_u[j] - timestep * theta * phi_i_v[i] * phi_i_u[j])						
						) *  scratch_data.fe_values.JxW(q);			  
			}
		      else if (comp_j == 4)
			{
			  copy_data.cell_matrix(j,i) += (phi_i_p[i] * phi_i_p[j]) * scratch_data.fe_values.JxW(q);      
			}
		      // end j dofs
		    }  
		  // end i dofs		     
		}   
	      // end n_q_points 
	    }    

	// cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_matrix, local_dof_indices,
	// 					  system_matrix);
	  // end if (second PDE: STVK material)  
	} 
      // end cell
}



// In this function we assemble the semi-linear 
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the 
// system matrix.
template <int dim>
void
FSI_ALE_Problem<dim>::
local_assemble_system_rhs (const typename DoFHandler<dim>::active_cell_iterator &cell,
                              AssemblyScratchData                               &scratch_data,
                              AssemblyRhsCopyData                               &copy_data)
{
  const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int   n_q_points      = scratch_data.fe_values.get_quadrature().size();
  const unsigned int   n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

  copy_data.cell_rhs.reinit (dofs_per_cell);

  copy_data.local_dof_indices.resize(dofs_per_cell);
  cell->get_dof_indices (copy_data.local_dof_indices);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim); 
  const FEValuesExtractors::Scalar pressure (dim+dim); 
 
  std::vector<Vector<double> > 
    old_solution_values (n_q_points, Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));


  std::vector<Vector<double> > 
    old_solution_face_values (n_face_q_points, Vector<double>(dim+dim+1));
  
  std::vector<std::vector<Tensor<1,dim> > > 
    old_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));
  
  std::vector<Vector<double> > 
    old_timestep_solution_values (n_q_points, Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));

  std::vector<Vector<double> > 
    old_timestep_solution_face_values (n_face_q_points, Vector<double>(dim+dim+1));
     
  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_face_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));
 

      scratch_data.fe_values.reinit (cell);	 
      copy_data.cell_rhs = 0;   	
      
      cell_diameter = cell->diameter();
      
      // old Newton iteration
      scratch_data.fe_values.get_function_values (solution, old_solution_values);
      scratch_data.fe_values.get_function_gradients (solution, old_solution_grads);
            
      // old timestep iteration
      scratch_data.fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      scratch_data.fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
      // Again, material_id == 0 corresponds to 
      // the domain for fluid equations
      if (cell->material_id() == fluid_id)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	      
	      const Tensor<2,dim> pI = ALE_Transformations
		::get_pI<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> v = ALE_Transformations
		::get_v<dim> (q, old_solution_values);
	      
	      const Tensor<2,dim> grad_v = ALE_Transformations 
		::get_grad_v<dim> (q, old_solution_grads);
	      
	      const Tensor<2,dim> grad_u = ALE_Transformations 
		::get_grad_u<dim> (q, old_solution_grads);
	      
	      const Tensor<2,dim> grad_v_T = ALE_Transformations
		::get_grad_v_T<dim> (grad_v);
	      
	      const Tensor<1,dim> u = ALE_Transformations
		::get_u<dim> (q, old_solution_values); 
	      
	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);	       	     
	      
	      const Tensor<2,dim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (F);
	      
	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (F_Inverse);
	      
	      const double J = ALE_Transformations
		::get_J<dim> (F);

	      
	      // This is the fluid stress tensor in ALE formulation
	      const Tensor<2,dim> sigma_ALE = NSE_in_ALE
		::get_stress_fluid_except_pressure_ALE<dim> 
		(density_fluid, viscosity, grad_v, grad_v_T, F_Inverse, F_Inverse_T );
	      	      	    	      
	      // We proceed by catching the previous time step values
	      const Tensor<1,dim> old_timestep_v = ALE_Transformations
		::get_v<dim> (q, old_timestep_solution_values);
	      
	      const Tensor<2,dim> old_timestep_grad_v = ALE_Transformations
		::get_grad_v<dim> (q, old_timestep_solution_grads);

	      const Tensor<2,dim> old_timestep_grad_v_T = ALE_Transformations
		::get_grad_v_T<dim> (old_timestep_grad_v);

	      const Tensor<1,dim> old_timestep_u = ALE_Transformations
		     ::get_u<dim> (q, old_timestep_solution_values);		 
	       
	      const Tensor<2,dim> old_timestep_F = ALE_Transformations
		::get_F<dim> (q, old_timestep_solution_grads);
	       
	      const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (old_timestep_F);
	       
	      const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
	      
	      const double old_timestep_J = ALE_Transformations
		::get_J<dim> (old_timestep_F);
	      		   
	      // This is the fluid stress tensor in the ALE formulation
	      // at the previous time step
	      const Tensor<2,dim> old_timestep_sigma_ALE = NSE_in_ALE
		::get_stress_fluid_except_pressure_ALE<dim> 
		(density_fluid, viscosity, old_timestep_grad_v, old_timestep_grad_v_T, 
		 old_timestep_F_Inverse, old_timestep_F_Inverse_T );
		  	
	      Tensor<2,dim> stress_fluid;
	      stress_fluid.clear();
	      stress_fluid = (J * sigma_ALE * F_Inverse_T);
	      
	      Tensor<2,dim> fluid_pressure;
	      fluid_pressure.clear();
	      fluid_pressure = (-pI * J * F_Inverse_T);
	      	      	      
	      Tensor<2,dim> old_timestep_stress_fluid;
	      old_timestep_stress_fluid.clear();
	      old_timestep_stress_fluid = 
		(old_timestep_J * old_timestep_sigma_ALE * old_timestep_F_Inverse_T);
	  
	      // Divergence of the fluid in the ALE formulation
	      const double incompressiblity_fluid = NSE_in_ALE
		::get_Incompressibility_ALE<dim> (q, old_solution_grads);
	    
	      // Convection term of the fluid in the ALE formulation.
	      // We emphasize that the fluid convection term for
	      // non-stationary flow problems in ALE
	      // representation is difficult to derive.  	      
	      // For adequate discretization, the convection term will be 
	      // split into three smaller terms:
	      Tensor<1,dim> convection_fluid;
	      convection_fluid.clear();
	      convection_fluid = density_fluid * J * (grad_v * F_Inverse * v);
	    	     
	      // The second convection term for the fluid in the ALE formulation	      
	      Tensor<1,dim> convection_fluid_with_u;
	      convection_fluid_with_u.clear();
	      convection_fluid_with_u = 
		density_fluid * J * (grad_v * F_Inverse * u);
	      
	      // The third convection term for the fluid in the ALE formulation	      
	      Tensor<1,dim> convection_fluid_with_old_timestep_u;
	      convection_fluid_with_old_timestep_u.clear();
	      convection_fluid_with_old_timestep_u = 
		density_fluid * J * (grad_v * F_Inverse * old_timestep_u);
	      
	      // The convection term of the previous time step
	      Tensor<1,dim> old_timestep_convection_fluid;
	      old_timestep_convection_fluid.clear();
	      old_timestep_convection_fluid = 
		(density_fluid * old_timestep_J * 
		 (old_timestep_grad_v * old_timestep_F_Inverse * old_timestep_v));
	    
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  // Fluid, NSE in ALE
		  const unsigned int comp_i = fe.system_to_component_index(i).first; 
		  if (comp_i == 0 || comp_i == 1)
		    {   		  
		      const Tensor<1,dim> phi_i_v = scratch_data.fe_values[velocities].value (i, q);
		      const Tensor<2,dim> phi_i_grads_v = scratch_data.fe_values[velocities].gradient (i, q);
		      
		      copy_data.cell_rhs(i) -= (density_fluid * (J + old_timestep_J)/2.0 * 
				       (v - old_timestep_v) * phi_i_v +				
				       timestep * theta * convection_fluid * phi_i_v +	
				       timestep * (1.0-theta) *
				       old_timestep_convection_fluid * phi_i_v -
				       (convection_fluid_with_u -
					convection_fluid_with_old_timestep_u) * phi_i_v +
				       timestep * scalar_product(fluid_pressure, phi_i_grads_v) +
				       timestep * theta * scalar_product(stress_fluid, phi_i_grads_v) +
				       timestep * (1.0-theta) *
				       scalar_product(old_timestep_stress_fluid, phi_i_grads_v) 			
				       ) *  scratch_data.fe_values.JxW(q);
		      
		    }		
		  else if (comp_i == 2 || comp_i == 3)
		    {	
		      const Tensor<2,dim> phi_i_grads_u = scratch_data.fe_values[displacements].gradient (i, q);
		      
		      // Nonlinear harmonic MMPDE
		      copy_data.cell_rhs(i) -= (alpha_u/J * scalar_product(grad_u, phi_i_grads_u)
				       ) * scratch_data.fe_values.JxW(q);


		    }  
		  else if (comp_i == 4)
		    {
		      const double phi_i_p = scratch_data.fe_values[pressure].value (i, q);
		      copy_data.cell_rhs(i) -= (incompressiblity_fluid * phi_i_p) *  scratch_data.fe_values.JxW(q);
		    }
		  // end i dofs  
		}  	     	   
	      // close n_q_points  
	    } 
	  	  	  	  
	  // As already discussed in the assembling method for the matrix,
	  // we have to integrate some terms on the outflow boundary:
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->face(face)->at_boundary() && 		  
		  (cell->face(face)->boundary_indicator() == outlet_id) 
		  )
		{
		  
		  scratch_data.fe_face_values.reinit (cell, face);
		  
		  scratch_data.fe_face_values.get_function_values (solution, old_solution_face_values);
		  scratch_data.fe_face_values.get_function_gradients (solution, old_solution_face_grads);
		  
		  scratch_data.fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
		  scratch_data.fe_face_values.get_function_gradients (old_timestep_solution, old_timestep_solution_face_grads);			
		  
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {	
		      // These are terms coming from the
		      // previous Newton iterations ...
		      const Tensor<2,dim> grad_v = ALE_Transformations
			::get_grad_v<dim> (q, old_solution_face_grads);
		      
		      const Tensor<2,dim> grad_v_T = ALE_Transformations
			::get_grad_v_T<dim> (grad_v);
		      
		      const Tensor<2,dim> F = ALE_Transformations
			::get_F<dim> (q, old_solution_face_grads);
		      
		      const Tensor<2,dim> F_Inverse = ALE_Transformations
			::get_F_Inverse<dim> (F);
		      
		      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
			::get_F_Inverse_T<dim> (F_Inverse);
		      
		      const double J = ALE_Transformations
			::get_J<dim> (F);
		      
		      // ... and here from the previous time step iteration
		      const Tensor<2,dim> old_timestep_grad_v = ALE_Transformations
			::get_grad_v<dim> (q, old_timestep_solution_face_grads);
		      
		      const Tensor<2,dim> old_timestep_grad_v_T = ALE_Transformations
			::get_grad_v_T<dim> (old_timestep_grad_v);
		      
		      const Tensor<2,dim> old_timestep_F = ALE_Transformations
			::get_F<dim> (q, old_timestep_solution_face_grads);
		      
		      const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations
			::get_F_Inverse<dim> (old_timestep_F);
		      
		      const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations
			::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
		      
		      const double old_timestep_J = ALE_Transformations
			::get_J<dim> (old_timestep_F);
		  		      
		      Tensor<2,dim> sigma_ALE_tilde;
		      sigma_ALE_tilde.clear();
		      sigma_ALE_tilde = 
			(density_fluid * viscosity * F_Inverse_T * grad_v_T);
		      
		      Tensor<2,dim> old_timestep_sigma_ALE_tilde;
		      old_timestep_sigma_ALE_tilde.clear();
		      old_timestep_sigma_ALE_tilde = 
			(density_fluid * viscosity * old_timestep_F_Inverse_T * old_timestep_grad_v_T);
		      
		      // Neumann boundary integral
		      Tensor<2,dim> stress_fluid_transposed_part;
		      stress_fluid_transposed_part.clear();
		      stress_fluid_transposed_part = (J * sigma_ALE_tilde * F_Inverse_T);
		      
		      Tensor<2,dim> old_timestep_stress_fluid_transposed_part;
		      old_timestep_stress_fluid_transposed_part.clear();		      
		      old_timestep_stress_fluid_transposed_part = 
			(old_timestep_J * old_timestep_sigma_ALE_tilde * old_timestep_F_Inverse_T);

		      const Tensor<1,dim> neumann_value
			= (stress_fluid_transposed_part * scratch_data.fe_face_values.normal_vector(q));
		      
		      const Tensor<1,dim> old_timestep_neumann_value
			= (old_timestep_stress_fluid_transposed_part * scratch_data.fe_face_values.normal_vector(q));
		      		     
		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  const unsigned int comp_i = fe.system_to_component_index(i).first; 
			  if (comp_i == 0 || comp_i == 1)
			    {  
			      copy_data.cell_rhs(i) +=  1.0 * (timestep * theta * 
						 neumann_value * scratch_data.fe_face_values[velocities].value (i, q) +
						 timestep * (1.0-theta) *
						 old_timestep_neumann_value * 
						 scratch_data.fe_face_values[velocities].value (i, q)
						 ) * scratch_data.fe_face_values.JxW(q);					   
			    }
			  // end i
			}  
		      // end face_n_q_points    
		    }                                     
		} 
	    }  // end face integrals do-nothing condition

	  
	//   cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_rhs, local_dof_indices,
	// 					  system_rhs);
	 
	  // Finally, we arrive at the end for assembling 
	  // the variational formulation for the fluid part and step to
	  // the assembling process of the structure terms:
	}   
      else if (cell->material_id() == solid_id[0] || cell->material_id() == solid_id[1] || cell->material_id() == solid_id[2])
	{	  
		 double lame_coefficient_mu, lame_coefficient_lambda;
		 if (cell->material_id() == solid_id[0]){
		 	lame_coefficient_mu = lame_mu[0];
			lame_coefficient_lambda = lame_lambda[0];
		 }
		 if (cell->material_id() == solid_id[1]){
		 	lame_coefficient_mu = lame_mu[1];
			lame_coefficient_lambda = lame_lambda[1];
		 }
		if (cell->material_id() == solid_id[2]){
		 	lame_coefficient_mu = lame_mu[2];
			lame_coefficient_lambda = lame_lambda[2];
		}  
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {		 		 	      
	      const Tensor<1,dim> v = ALE_Transformations
		::get_v<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> u = ALE_Transformations
		::get_u<dim> (q, old_solution_values);
	      
	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);
	      
	      const Tensor<2,dim> F_T = ALE_Transformations
		::get_F_T<dim> (F);
	      
	      const Tensor<2,dim> Identity = ALE_Transformations
		::get_Identity<dim> ();
	      
	      const Tensor<2,dim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (F);
	      
	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (F_Inverse);
	      
	      const double J = ALE_Transformations
		::get_J<dim> (F);
	      
	      const Tensor<2,dim> E = Structure_Terms_in_ALE
		::get_E<dim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (E);
	      
	      // Previous time step values
	      const Tensor<1,dim> old_timestep_v = ALE_Transformations
		::get_v<dim> (q, old_timestep_solution_values);
	      
	      const Tensor<1,dim> old_timestep_u = ALE_Transformations
		::get_u<dim> (q, old_timestep_solution_values);
	      
	      const Tensor<2,dim> old_timestep_F = ALE_Transformations
		::get_F<dim> (q, old_timestep_solution_grads);
	      
	      const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (old_timestep_F);
	      
	      const Tensor<2,dim> old_timestep_F_T = ALE_Transformations
		::get_F_T<dim> (old_timestep_F);
	      
	      const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
	      
	      const double old_timestep_J = ALE_Transformations
		::get_J<dim> (old_timestep_F);
	      
	      const Tensor<2,dim> old_timestep_E = Structure_Terms_in_ALE
		::get_E<dim> (old_timestep_F_T, old_timestep_F, Identity);
	      
	      const double old_timestep_tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (old_timestep_E);
	      
	      
	      // STVK structure model
	      Tensor<2,dim> sigma_structure_ALE;
	      sigma_structure_ALE.clear();
	      sigma_structure_ALE = (1.0/J *
				     F * (lame_coefficient_lambda * tr_E * Identity +
					  2 * lame_coefficient_mu * E) * F_T);
	      
	      
	      Tensor<2,dim> stress_term;
	      stress_term.clear();
	      stress_term = (J * sigma_structure_ALE * F_Inverse_T);
	      
	      Tensor<2,dim> old_timestep_sigma_structure_ALE;
	      old_timestep_sigma_structure_ALE.clear();
	      old_timestep_sigma_structure_ALE = (1.0/old_timestep_J *
						  old_timestep_F * (lame_coefficient_lambda *
								    old_timestep_tr_E * Identity +
								    2 * lame_coefficient_mu *
								    old_timestep_E) * 
						  old_timestep_F_T);
	      
	      Tensor<2,dim> old_timestep_stress_term;
	      old_timestep_stress_term.clear();
	      old_timestep_stress_term = (old_timestep_J * old_timestep_sigma_structure_ALE * old_timestep_F_Inverse_T);
	      	
	      // Attention: normally no time
	      Tensor<1,dim> structure_force;
	      structure_force.clear();
	      structure_force[0] = density_structure * force_structure_x;
	      structure_force[1] = density_structure * force_structure_y;
	      
	      Tensor<1,dim> old_timestep_structure_force;
	      old_timestep_structure_force.clear();
	      old_timestep_structure_force[0] = density_structure * force_structure_x;
	      old_timestep_structure_force[1] = density_structure * force_structure_y;
	    
      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  // STVK structure model
		  const unsigned int comp_i = fe.system_to_component_index(i).first; 
		  if (comp_i == 0 || comp_i == 1)
		    { 
		      const Tensor<1,dim> phi_i_v = scratch_data.fe_values[velocities].value (i, q);
		      const Tensor<2,dim> phi_i_grads_v = scratch_data.fe_values[velocities].gradient (i, q);
		      
		      copy_data.cell_rhs(i) -= (density_structure * (v - old_timestep_v) * phi_i_v +
				       timestep * theta * scalar_product(stress_term,phi_i_grads_v) +  
				       timestep * (1.0-theta) * scalar_product(old_timestep_stress_term, phi_i_grads_v) 
				       - timestep * theta * structure_force * phi_i_v   
				       - timestep * (1.0 - theta) * old_timestep_structure_force * phi_i_v 
				       ) * scratch_data.fe_values.JxW(q);    
		      
		    }		
		  else if (comp_i == 2 || comp_i == 3)
		    {
		      const Tensor<1,dim> phi_i_u = scratch_data.fe_values[displacements].value (i, q);
		      copy_data.cell_rhs(i) -=  (density_structure * 
					((u - old_timestep_u) * phi_i_u -
					 timestep * (theta * v + (1.0-theta) * 
						     old_timestep_v) * phi_i_u)
					) * scratch_data.fe_values.JxW(q);    
		      
		    }
		  else if (comp_i == 4)
		    {
		      const double phi_i_p = scratch_data.fe_values[pressure].value (i, q);
		      copy_data.cell_rhs(i) -= (old_solution_values[q](dim+dim) * phi_i_p) * scratch_data.fe_values.JxW(q);  
		      
		    }
		  // end i	  
		} 	
	      // end n_q_points 		   
	    } 
	  
	//   cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_rhs, local_dof_indices,
	// 					  system_rhs);
	  
	// end if (for STVK material)  
	}   
      
}
template <int dim>
void
FSI_ALE_Problem<dim>::
local_assemble_temperature_system_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
																					AssemblyTemperatureScratchData                       &scratch_data,
																					AssemblyTemperatureMatrixCopyData                    &copy_data)
{
	// fe for temperature
	// the temperature dof is different than system dof
  const unsigned int   dofs_per_cell   = temperature_fe.dofs_per_cell;
	// the q points should be same between system and temperature
  const unsigned int   n_q_points      = scratch_data.temperature_fe_values.get_quadrature().size();
  // const unsigned int   n_face_q_points = scratch_data.temperature_fe_face_values.get_quadrature().size();

	const unsigned int   n_q_points_all      = scratch_data.fe_values.get_quadrature().size();
  // const unsigned int   n_face_q_points_all = scratch_data.fe_face_values.get_quadrature().size();

  copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);

  copy_data.local_dof_indices.resize(dofs_per_cell); 
  cell->get_dof_indices (copy_data.local_dof_indices);

  // Now, we are going to use the 
  // FEValuesExtractors to determine
  // the four principle variables
  const FEValuesExtractors::Vector velocities (0); // 0
  const FEValuesExtractors::Vector displacements (dim); // 2
  const FEValuesExtractors::Scalar pressure (dim+dim); // 4

  // We declare Vectors and Tensors for 
  // the solutions at the current time for v,u,p:
  std::vector<Vector<double> > old_solution_values (n_q_points_all, 
				 		    Vector<double>(dim+dim+1));

  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points_all, 
								std::vector<Tensor<1,dim> > (dim+dim+1));

  // std::vector<Vector<double> >  old_solution_face_values (n_face_q_points_all, 
	// 						  Vector<double>(dim+dim+1));
       
  // std::vector<std::vector<Tensor<1,dim> > > old_solution_face_grads (n_face_q_points_all, 
	// 							     std::vector<Tensor<1,dim> > (dim+dim+1));
  

  // We declare Vectors and Tensors for 
  // the solution at the previous time step:
  std::vector<Vector<double> > old_timestep_solution_values (n_q_points_all, 
				 		    Vector<double>(dim+dim+1));

  // std::vector<std::vector<Tensor<1,dim> > > old_timestep_solution_grads (n_q_points_all, 
  // 					  std::vector<Tensor<1,dim> > (dim+dim+1));

  // std::vector<Vector<double> >   old_timestep_solution_face_values (n_face_q_points_all, 
	// 							    Vector<double>(dim+dim+1));
  
  // std::vector<std::vector<Tensor<1,dim> > >  old_timestep_solution_face_grads (n_face_q_points_all, 
	// 								       std::vector<Tensor<1,dim> > (dim+dim+1));
   
  // Declaring test functions for mass transfer:
  std::vector<double > phi_i_T (dofs_per_cell); 
	std::vector<double > phi_i_T_SUPG (dofs_per_cell); 
  std::vector<Tensor<1,dim> > phi_i_grads_T(dofs_per_cell);
	     				   
      scratch_data.temperature_fe_values.reinit (cell);
			// goto the same cell, dof_handler is system not temperature dof_handler
			typename DoFHandler<dim>::active_cell_iterator
    	fsi_cell (&triangulation,
								cell->level(),
								cell->index(),
								&dof_handler);
      copy_data.cell_matrix = 0;
      scratch_data.fe_values.reinit (fsi_cell);
      // We need the cell diameter to control the fluid mesh motion
      cell_diameter = cell->diameter();
      
      // N+1 values
      scratch_data.fe_values.get_function_values (solution, old_solution_values);
      scratch_data.fe_values.get_function_gradients (solution, old_solution_grads);
      
      // N values
      scratch_data.fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      // scratch_data.fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
      // Next, we run over all cells for the fluid equations
      if (cell->material_id() == fluid_id)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
				{
					phi_i_grads_T[k] = scratch_data.temperature_fe_values.shape_grad  (k, q);
					phi_i_T[k]       = scratch_data.temperature_fe_values.shape_value (k, q);
				}
	      
	      // We build values, vectors, and tensors
	      // from information of the previous Newton step. These are introduced 
	      // for two reasons:
	      // First, these are used to perform the ALE mapping of the 
	      // fluid equations. Second, these terms are used to 
	      // make the notation as simple and self-explaining as possible:
	      
	      const Tensor<1,dim> v = ALE_Transformations
		::get_v<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> u = ALE_Transformations
		::get_u<dim> (q,old_solution_values);
	      
	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);	    
	      
	      const Tensor<2,dim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (F);
	      
	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (F_Inverse);
	      
	      const double J = ALE_Transformations
		::get_J<dim> (F);
	      
				const Tensor<1,dim> old_timestep_u = ALE_Transformations
		::get_u<dim> (q, old_timestep_solution_values);
	      // ALE velocity
				const Tensor<1,dim> w = v-(u-old_timestep_u)/timestep;
				const double det_w = w.norm();
				if (det_w <1e-8){
					for( unsigned int j=0; j<dofs_per_cell; ++j )
					{
						phi_i_T_SUPG[j] = 0.0;
					}
				}
				else {
					cell_diameter *= std::sqrt(J);
					const double Pe = det_w*cell_diameter/2/D_l;
					const double PePe = Pe*Pe;
					if(Pe<0.3){
						const double a_Pe = Pe*(1/3+PePe*(
							-1/45+PePe*(
								2/945+PePe*(
									-1/4725+PePe*2/93555
								)
							)
						)
					);
					}
					else{
						const double a_Pe = 1/std::tanh(Pe)-1/Pe;
					}
	      	
					for( unsigned int j=0; j<dofs_per_cell; ++j )
					{
						phi_i_T_SUPG[j] = 0.5 * a_Pe * cell_diameter / det_w
														* ( w * ( F_Inverse_T * phi_i_grads_T[j] ) );
					}
				}
				// Outer loop for dofs
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	
			
			// v-(u-old_timestep_u)/timestep
		  const double convection_temperature = J * phi_i_grads_T[i] * F_Inverse * v;
			const double convection_temperature_u = J * phi_i_grads_T[i] * F_Inverse * u;
			const double convection_temperature_u_old = J * phi_i_grads_T[i] * F_Inverse * old_timestep_u;
			const Tensor<1,dim> diffusion_temperature = J * D_l * phi_i_grads_T[i] * F_Inverse * F_Inverse_T;
			const double convection_SUPG = J * phi_i_grads_T[i] * F_Inverse * w;
		  // Inner loop for dofs
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {	
				copy_data.cell_matrix(j,i) += (J * phi_i_T[i] * phi_i_T[j] + 
																			 timestep * theta * 
																			 convection_temperature * phi_i_T[j] -
																			 convection_temperature_u * phi_i_T[j] + 
																			 convection_temperature_u_old * phi_i_T[j] + 
																			 timestep * theta * diffusion_temperature * phi_i_grads_T[j] -
																			 timestep * theta * k_ery * J * phi_i_T[i] * phi_i_T[j] +
																			 timestep * convection_SUPG * phi_i_T_SUPG[j] -
																			 timestep * k_ery * J * phi_i_T[i] * phi_i_T_SUPG[j]
																			 ) * scratch_data.temperature_fe_values.JxW(q);
		      // end j dofs  
		    }   
		  // end i dofs	  
		}   
	      // end n_q_points  
	    }    
	  	  
	  // We compute in the following
	  // one term on the outflow boundary. 
	  // This relation is well-know in the literature 
	  // as "do-nothing" condition. Therefore, we only
	  // ask for the corresponding color at the outflow 
	  // boundary that is 1 in our case.
	  // for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	  //   {
		// 	if (cell->neighbor_index(face) != -1)	       
	  //      if (cell->material_id() != cell->neighbor(face)->material_id() &&
		// 		 		cell->neighbor(face)->material_id() == solid_id[0])
		// {
		  
		//   scratch_data.temperature_fe_face_values.reinit (cell, face);
		//   scratch_data.fe_face_values.reinit (fsi_cell, face);
		//   scratch_data.fe_face_values.get_function_values (solution, old_solution_face_values);
		//   scratch_data.fe_face_values.get_function_gradients (solution, old_solution_face_grads);	
		  
		//   for (unsigned int q=0; q<n_face_q_points; ++q)
		//     {
		//       for (unsigned int k=0; k<dofs_per_cell; ++k)
		// 	{
		// 		phi_i_grads_T[k] = scratch_data.temperature_fe_values.shape_grad  (k, q);
		// 		phi_i_T[k]       = scratch_data.temperature_fe_values.shape_value (k, q);
		// 	}
		//       const Tensor<1,dim> v = ALE_Transformations
		// ::get_v<dim> (q, old_solution_face_values);
	      
	  //     const Tensor<1,dim> u = ALE_Transformations
		// ::get_u<dim> (q,old_solution_face_values);
	      
	  //     const Tensor<2,dim> F = ALE_Transformations
		// ::get_F<dim> (q, old_solution_face_grads);	    
	      
	  //     const Tensor<2,dim> F_Inverse = ALE_Transformations
		// ::get_F_Inverse<dim> (F);
	      
	  //     const Tensor<2,dim> F_Inverse_T = ALE_Transformations
		// ::get_F_Inverse_T<dim> (F_Inverse);
	      
	  //     const double J = ALE_Transformations
		// ::get_J<dim> (F);

		//       const Tensor<1,dim> old_timestep_u = ALE_Transformations
		// ::get_u<dim> (q, old_timestep_solution_face_values);

		// 			const Tensor<2,dim> Rotation = ALE_Transformations
		// ::get_Rotation_PI_half();

		//       Tensor<2,dim> sigma_ALE_tilde;
		//       sigma_ALE_tilde.clear();
		//       sigma_ALE_tilde = 
		// 	(density_fluid * viscosity * F_Inverse_T * grad_v_T);

		// 			// Tensor<2,dim> stress_fluid_transposed_part;
		//       // stress_fluid_transposed_part.clear();
		//       // stress_fluid_transposed_part = (J * sigma_ALE_tilde * F_Inverse_T);
		// 			const Tensor<1,dim> Normal_Vector = scratch_data.fe_face_values.normal_vector(q);
    //       const Tensor<1,dim> Tangential = Rotation*Normal_Vector;
		// 			// direct is wrong?
		// 			const Tensor<1,dim> neumann_value
		// 	= (sigma_ALE_tilde * Tangential);
		//       const double WSS = neumann_value.norm();
		// 			const double R_T = RT*( R_basal+R_max*( WSS/(WSS+a_c) ) );

		//       for (unsigned int i=0; i<dofs_per_cell; ++i)
		// 	{
		// 			const Tensor<1,dim>
		// 	  	diffusion_temperature_face1 = (D_w-D_l) * J * phi_i_grad_T[i] * F_Inverse * F_Inverse_T + J * phi_i_T[i] * v * F_Inverse_T;
		// 			const Tensor<1,dim>
		// 	  	diffusion_temperature_face2 = J * phi_i_T[i] * ( - u + old_timestep_u) * F_Inverse_T;
		// 	  // Here, we multiply the symmetric part of fluid's stress tensor
		// 	  // with the normal direction.
		// 	  const Tensor<1,dim> neumann_value_T
		// 	    = ( timestep * theta * diffusion_temperature_face1 + diffusion_temperature_face2 ) 
		// 			* Normal_Vector;
			  
		// 	  for (unsigned int j=0; j<dofs_per_cell; ++j)
		// 	    {		     

		// 		  copy_data.cell_matrix(j,i) += ( ( neumann_value_T - timestep * theta * J * R_T ) * phi_i_T[j] ) 
		// 																	* scratch_data.temperature_fe_face_values.JxW(q);
		// 	      // end j    
		// 	    } 
		// 	  // end i
		// 	}   
		//       // end q_face_points
		//     } 
		//   // end if-routine face integrals
		// }  	      
	  //     // end face integrals do-nothing
	  //   }   

	  
	  // This is the same as discussed in step-22:
	//   cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_matrix, local_dof_indices,
	// 					  system_matrix);
	  
	  // Finally, we arrive at the end for assembling the matrix
	  // for the fluid equations and step to the computation of the 
	  // structure terms:
	} 
      else if (cell->material_id() == solid_id[0] || cell->material_id() == solid_id[1] || cell->material_id() == solid_id[2])
	{	  
		for (unsigned int q=0; q<n_q_points; ++q)
	    {
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
				{
					phi_i_grads_T[k] = scratch_data.temperature_fe_values.shape_grad  (k, q);
					phi_i_T[k]       = scratch_data.temperature_fe_values.shape_value (k, q);
				}
	      
	      // We build values, vectors, and tensors
	      // from information of the previous Newton step. These are introduced 
	      // for two reasons:
	      // First, these are used to perform the ALE mapping of the 
	      // fluid equations. Second, these terms are used to 
	      // make the notation as simple and self-explaining as possible:
	          	      
	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);	    
	      
	      const Tensor<2,dim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (F);
	      
	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (F_Inverse);
	      
	      const double J = ALE_Transformations
		::get_J<dim> (F);
	      
	      // Outer loop for dofs
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	
	
			const Tensor<1,dim> diffusion_temperature = J * D_w * phi_i_grads_T[i] * F_Inverse * F_Inverse_T;

		  // Inner loop for dofs
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {	
				copy_data.cell_matrix(j,i) += (J * phi_i_T[i] * phi_i_T[j] + 
																			 timestep * theta * diffusion_temperature * phi_i_grads_T[j] -
																			 timestep * theta * k_w * J * phi_i_T[i] * phi_i_T[j] 
																			 ) * scratch_data.temperature_fe_values.JxW(q);

		      // end j dofs  
		    }   
		  // end i dofs	  
		}   
	      // end n_q_points  
	    }  

	// cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_matrix, local_dof_indices,
	// 					  system_matrix);
	  // end if (second PDE: STVK material)  
	} 
      // end cell
}



// In this function we assemble the semi-linear 
// of the right hand side of Newton's method (its residual).
// The framework is in principal the same as for the 
// system matrix.
template <int dim>
void
FSI_ALE_Problem<dim>::
local_assemble_temperature_system_rhs (const typename DoFHandler<dim>::active_cell_iterator &cell,
																			 AssemblyTemperatureScratchData                       &scratch_data,
																			 AssemblyTemperatureRhsCopyData                    		&copy_data)
{
	// fe for temperature
	// the temperature dof is different than system dof
  const unsigned int   dofs_per_cell   = temperature_fe.dofs_per_cell;
	// the q points should be same between system and temperature
  const unsigned int   n_q_points      = scratch_data.temperature_fe_values.get_quadrature().size();
  const unsigned int   n_face_q_points = scratch_data.temperature_fe_face_values.get_quadrature().size();

	const unsigned int   n_q_points_all      = scratch_data.fe_values.get_quadrature().size();
  const unsigned int   n_face_q_points_all = scratch_data.fe_face_values.get_quadrature().size();
	// reinit rhs for T
  copy_data.cell_rhs.reinit (dofs_per_cell);

  copy_data.local_dof_indices.resize(dofs_per_cell); 
  cell->get_dof_indices (copy_data.local_dof_indices);


  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Vector displacements (dim); 
  const FEValuesExtractors::Scalar pressure (dim+dim); 

  // time N+1, v u p face
  std::vector<Vector<double> >
    old_solution_face_values (n_face_q_points_all, Vector<double>(dim+dim+1));
  // time N+1, grad v u p face
  std::vector<std::vector<Tensor<1,dim> > >
    old_solution_face_grads (n_face_q_points_all, std::vector<Tensor<1,dim> > (dim+dim+1));
  // time N, v u p
  std::vector<Vector<double> >
    old_timestep_solution_values (n_q_points_all, Vector<double>(dim+dim+1));
	// time N, v u p grad
  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_grads (n_q_points_all, std::vector<Tensor<1,dim> > (dim+dim+1));
	// time N, v u p face
  std::vector<Vector<double> > 
    old_timestep_solution_face_values (n_face_q_points_all, Vector<double>(dim+dim+1));
  // time N, grad v u p face
  std::vector<std::vector<Tensor<1,dim> > > 
    old_timestep_solution_face_grads (n_face_q_points_all, std::vector<Tensor<1,dim> > (dim+dim+1));
 
	// time N, temperature
 	std::vector<double > 
    old_timestep_T_values (n_q_points);
	// time N, temperature grad
	std::vector<Tensor<1,dim>  > 
    old_timestep_T_grads (n_q_points);
	// time N, temperature face
 	// std::vector<double > 
  //   old_timestep_T_face_values (n_face_q_points);
	// // time N, temperature grad face
	// std::vector<Tensor<1,dim>  > 
  //   old_timestep_T_face_grads (n_face_q_points, Tensor<1,dim> );

      scratch_data.temperature_fe_values.reinit (cell);
			// goto the same cell, dof_handler is system not temperature dof_handler
			typename DoFHandler<dim>::active_cell_iterator
    	fsi_cell (&triangulation,
								cell->level(),
								cell->index(),
								&dof_handler); 
      copy_data.cell_rhs = 0;
      scratch_data.fe_values.reinit (fsi_cell);

      cell_diameter = cell->diameter();
      
      // // old Newton iteration
      // scratch_data.fe_values.get_function_values (solution, old_solution_values);
      // scratch_data.fe_values.get_function_gradients (solution, old_solution_grads);
            
      // old timestep iteration
      scratch_data.fe_values.get_function_values (old_timestep_solution, old_timestep_solution_values);
      scratch_data.fe_values.get_function_gradients (old_timestep_solution, old_timestep_solution_grads);
      
			scratch_data.temperature_fe_values.get_function_values (old_temperature_solution, old_timestep_T_values);
      scratch_data.temperature_fe_values.get_function_gradients (old_temperature_solution, old_timestep_T_grads);
      // Again, material_id == 0 corresponds to 
      // the domain for fluid equations
      if (cell->material_id() == fluid_id)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	     
				const double C =  old_timestep_T_values[q];
	      const Tensor<1,dim> grad_C = old_timestep_T_grads[q];
	      
	      	      	    	      
	      // We proceed by catching the previous time step values
	      const Tensor<1,dim> old_timestep_v = ALE_Transformations
		::get_v<dim> (q, old_timestep_solution_values);
	    		     
	      const Tensor<2,dim> old_timestep_F = ALE_Transformations
		::get_F<dim> (q, old_timestep_solution_grads);
	       
	      const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (old_timestep_F);
	       
	      const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
	      
	      const double old_timestep_J = ALE_Transformations
		::get_J<dim> (old_timestep_F);
	      		   
	      // This is the fluid stress tensor in the ALE formulation
	      // at the previous time step
			const double convection_temperature_old = old_timestep_J * grad_C * old_timestep_F_Inverse * old_timestep_v;

			const Tensor<1,dim> diffusion_temperature_old = old_timestep_J * D_l * grad_C * old_timestep_F_Inverse * old_timestep_F_Inverse_T;
	    
	    
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{  		  
		      const double phi_i_T = scratch_data.temperature_fe_values.shape_value (i, q);
		      const Tensor<1,dim> phi_i_grads_T = scratch_data.temperature_fe_values.shape_grad  (i, q);
		      
		      copy_data.cell_rhs(i) -=(- old_timestep_J * C * phi_i_T 
																   + timestep * (1.0-theta) * convection_temperature_old * phi_i_T
																	 + timestep * (1.0-theta) * diffusion_temperature_old * phi_i_grads_T
																	 - timestep * (1.0-theta) * old_timestep_J * k_ery * C * phi_i_T
																	) * scratch_data.temperature_fe_values.JxW(q);
		  // end i dofs  
		}  	     	   
	      // close n_q_points  
	    } 
	  	  	  	  
	  // As already discussed in the assembling method for the matrix,
	  // we have to integrate some terms on the outflow boundary:
	  for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->neighbor_index(face) != -1)	       
	       if (cell->material_id() != cell->neighbor(face)->material_id() &&
				 		cell->neighbor(face)->material_id() == solid_id[0])
		{
		  
		  scratch_data.temperature_fe_face_values.reinit (cell, face);
		  scratch_data.fe_face_values.reinit (fsi_cell, face);
			// N for T face
			// scratch_data.temperature_fe_face_values.get_function_values (old_temperature_solution, old_timestep_T_face_values);
		  // scratch_data.temperature_fe_face_values.get_function_gradients (old_temperature_solution, old_timestep_T_face_grads);
			// N for v u p face
		  scratch_data.fe_face_values.get_function_values (solution, old_solution_face_values);
		  scratch_data.fe_face_values.get_function_gradients (solution, old_solution_face_grads);	

		  // N for v u p face
		  scratch_data.fe_face_values.get_function_values (old_timestep_solution, old_timestep_solution_face_values);
		  scratch_data.fe_face_values.get_function_gradients (old_timestep_solution, old_timestep_solution_face_grads);			
		  
		  for (unsigned int q=0; q<n_face_q_points; ++q)
		    {	
		      // These are terms coming from the
		      // previous Newton iterations ...
					const Tensor<2,dim> grad_v = ALE_Transformations
			::get_grad_v<dim> (q, old_solution_face_grads);
		      
		      const Tensor<2,dim> grad_v_T = ALE_Transformations
			::get_grad_v_T<dim> (grad_v);
		      
		      const Tensor<2,dim> F = ALE_Transformations
			::get_F<dim> (q, old_solution_face_grads);
		      
		      const Tensor<2,dim> F_Inverse = ALE_Transformations
			::get_F_Inverse<dim> (F);
		      
		      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
			::get_F_Inverse_T<dim> (F_Inverse);
		      
		      const double J = ALE_Transformations
			::get_J<dim> (F);
		      
		      // ... and here from the previous time step iteration
		      const Tensor<2,dim> old_timestep_grad_v = ALE_Transformations
			::get_grad_v<dim> (q, old_timestep_solution_face_grads);
		      
		      const Tensor<2,dim> old_timestep_grad_v_T = ALE_Transformations
			::get_grad_v_T<dim> (old_timestep_grad_v);
		      
		      const Tensor<2,dim> old_timestep_F = ALE_Transformations
			::get_F<dim> (q, old_timestep_solution_face_grads);
		      
		      const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations
			::get_F_Inverse<dim> (old_timestep_F);
		      
		      const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations
			::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
		      
		      const double old_timestep_J = ALE_Transformations
			::get_J<dim> (old_timestep_F);

		  		const Tensor<2,dim> Rotation = ALE_Transformations
			::get_Rotation_PI_half<dim>();
	
		      Tensor<2,dim> sigma_ALE_tilde;
		      sigma_ALE_tilde.clear();
		      sigma_ALE_tilde = 
			(density_fluid * viscosity * F_Inverse_T * grad_v_T);

		      Tensor<2,dim> old_timestep_sigma_ALE_tilde;
		      old_timestep_sigma_ALE_tilde.clear();
		      old_timestep_sigma_ALE_tilde = 
			(density_fluid * viscosity * old_timestep_F_Inverse_T * old_timestep_grad_v_T);
		      
		      // Neumann boundary integral
		     
		      
		  //     Tensor<2,dim> old_timestep_stress_fluid_transposed_part;
		  //     old_timestep_stress_fluid_transposed_part.clear();		      
		  //     old_timestep_stress_fluid_transposed_part = 
			// (old_timestep_J * old_timestep_sigma_ALE_tilde * old_timestep_F_Inverse_T);

				// // direct is wrong?
				const Tensor<1,dim> Normal_Vector = scratch_data.fe_face_values.normal_vector(q);
				
		    const Tensor<1,dim> Tangential = Rotation * Normal_Vector;
				const Tensor<1,dim> neumann_value = (sigma_ALE_tilde * Tangential);
				const Tensor<1,dim> neumann_value_old
			= (old_timestep_sigma_ALE_tilde * Tangential);
		  //     const Tensor<1,dim> old_timestep_neumann_value
			// = (old_timestep_stress_fluid_transposed_part * scratch_data.fe_face_values.normal_vector(q));
					const double WSS     = neumann_value.norm(); 
		      const double WSS_old = neumann_value_old.norm();
					const double R_T     = RT*( R_basal+R_max*( WSS/(WSS+a_c) ) );
					const double R_T_old = RT*( R_basal+R_max*( WSS_old/(WSS_old+a_c) ) );	  

		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
					const double phi_i_T = scratch_data.temperature_fe_values.shape_value (i, q);
		      
			     copy_data.cell_rhs(i) += (- timestep * theta * J * R_T * phi_i_T 
						 												 - timestep * (1.0-theta) * old_timestep_J * R_T_old * phi_i_T
																		) * scratch_data.temperature_fe_face_values.JxW(q);					   
			  // end i
			}  
		      // end face_n_q_points    
		    }                                     
		} 
	    }  // end face integrals do-nothing condition

	  
	//   cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_rhs, local_dof_indices,
	// 					  system_rhs);
	 
	  // Finally, we arrive at the end for assembling 
	  // the variational formulation for the fluid part and step to
	  // the assembling process of the structure terms:
	}   
      else if (cell->material_id() == solid_id[0] || cell->material_id() == solid_id[1] || cell->material_id() == solid_id[2])
	{	  
		 for (unsigned int q=0; q<n_q_points; ++q)
	    {	     
				const double C =  old_timestep_T_values[q];
	      const Tensor<1,dim> grad_C = old_timestep_T_grads[q];
	      
	      	      	    	      
	      // We proceed by catching the previous time step values
	    		     
	      const Tensor<2,dim> old_timestep_F = ALE_Transformations
		::get_F<dim> (q, old_timestep_solution_grads);
	       
	      const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (old_timestep_F);
	       
	      const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (old_timestep_F_Inverse);
	      
	      const double old_timestep_J = ALE_Transformations
		::get_J<dim> (old_timestep_F);
	      		   

			const Tensor<1,dim> diffusion_temperature_old = old_timestep_J * D_w * grad_C * old_timestep_F_Inverse * old_timestep_F_Inverse_T;
	    
	    
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{  		  
		      const double phi_i_T = scratch_data.temperature_fe_values.shape_value (i, q);
		      const Tensor<1,dim> phi_i_grads_T = scratch_data.temperature_fe_values.shape_grad  (i, q);
		      
		      copy_data.cell_rhs(i) -=(- old_timestep_J * C * phi_i_T 
																	 + timestep * (1.0-theta) * diffusion_temperature_old * phi_i_grads_T
																	 - timestep * (1.0-theta) * old_timestep_J * k_w * C * phi_i_T
																	) * scratch_data.temperature_fe_values.JxW(q);
		  // end i dofs  
		}  	     	   
	      // close n_q_points  
	    } 
	  
	//   cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_rhs, local_dof_indices,
	// 					  system_rhs);
	  
	// end if (for STVK material)  
	}   
      
}

// inlet fow profile
template <int dim>
double 
FSI_ALE_Problem<dim>::inlet_flow (const double t)
{
//   const long double PI = 3.141592653589793238462643;
  const double T0 = 1.0;
  const double qmax = 250;
  const double a = 2.0/3.0;
  double qnew = 0;
  double temp, fi;
  if ( t <= T0 ) {
	if ( t <= a ){
	  fi=3*PI*t-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}else{
	  fi = 3*PI*a-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}
  }else{
	temp=t;
	while(temp>T0){ temp=temp-T0; }
	if(temp<=a){
	  fi = 3*PI*temp-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}else{
	  fi = 3*PI*a-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}	
  }
  return 1e-6*qnew/(PI*std::pow(R,2));// u=Q/A, A=PI*r^2, Q:mL/s,
}
// Here, we impose boundary conditions
// for the whole system. The fluid inflow 
// is prescribed by a parabolic profile. The usual
// structure displacement shall be fixed  
// at all outer boundaries. 
// The pressure variable is not subjected to any
// Dirichlet boundary conditions and is left free 
// in this method. Please note, that 
// the interface between fluid and structure has no
// physical boundary due to our formulation. Interface
// conditions are automatically fulfilled: that is 
// one major advantage of the `variational-monolithic' formulation.
template <int dim>
void
FSI_ALE_Problem<dim>::set_initial_bc (const double time)
{ 
	double inflow_velocity = inlet_flow (time);
    std::map<unsigned int,double> boundary_values;  
    std::vector<bool> component_mask (dim+dim+1, true);
    // (Scalar) pressure
	// vx vy ux uy p
	//  1  1  1  0 0
	component_mask[dim+1] = false; 
    component_mask[dim+dim] = false;  
	// inlet boundary 
    VectorTools::interpolate_boundary_values (dof_handler,
						inlet_id,
						BoundaryParabel<dim>(time, inflow_velocity),
						boundary_values,
						component_mask);    
	// vx vy ux uy p
	//  0  1  0  1 0
	// symmetry boundary
	component_mask[0]   = false; // vx
    component_mask[dim] = false; // ux
	component_mask[dim+1] = true; // uy
    VectorTools::interpolate_boundary_values (dof_handler,
						symmetry_id,
						ZeroFunction<dim>(dim+dim+1),  
						boundary_values,
						component_mask);
	// vx vy ux uy p
	//  1  0  1  0 0
	// fixed boundary
	// vy and uv do not fix
	component_mask[0]   = true;  // vx
	component_mask[1]   = false;  // vy
    component_mask[dim] = true;  // ux 
	component_mask[dim+1]   = false;  // uy
    VectorTools::interpolate_boundary_values (dof_handler,
						fixed_id,
						ZeroFunction<dim>(dim+dim+1),  
						boundary_values,
						component_mask);
    // vx vy ux uy p
	//  0  0  1  0 0
    // outlet boundary
    component_mask[0] = false;		// vx
    component_mask[1] = false; 		// vy
	component_mask[dim] = true;  	// ux 
	// component_mask[dim+1]   = true;  // uy  
    VectorTools::interpolate_boundary_values (dof_handler,
						outlet_id,
						ZeroFunction<dim>(dim+dim+1),  
						boundary_values,
						component_mask);
    
    for (typename std::map<unsigned int, double>::const_iterator
	   i = boundary_values.begin();
	   i != boundary_values.end();
	   ++i)
      solution(i->first) = i->second;
    
}

// This function applies boundary conditions 
// to the Newton iteration steps. For all variables that
// have Dirichlet conditions on some (or all) parts
// of the outer boundary, we apply zero-Dirichlet
// conditions, now. 
template <int dim>
void
FSI_ALE_Problem<dim>::set_newton_bc ()
{
    std::vector<bool> component_mask (dim+dim+1, true);
	component_mask[dim+1]   = false;  // uy
    component_mask[dim+dim] = false;  // p
	// vx vy ux uy p
	//  1  1  1  0 0
    // inlet boundary
    VectorTools::interpolate_boundary_values (dof_handler,
						  inlet_id,
						  ZeroFunction<dim>(dim+dim+1),                                             
						  constraints,
						  component_mask);
	// vx vy ux uy p
	//  0  1  0  1 0
	// symmetry boundary
	component_mask[0]   = false; // vx 
    component_mask[dim] = false; // ux
	component_mask[dim+1] = true;  // uy
    VectorTools::interpolate_boundary_values (dof_handler,
						  symmetry_id,
					      ZeroFunction<dim>(dim+dim+1),  
						  constraints,
						  component_mask);
	// vx vy ux uy p
	//  1  0  1  0 0
    // fixed boundary
	// vy and uv do not fix
	component_mask[0]   = true;  // vx
	component_mask[1]   = false;  // vy
    component_mask[dim] = true;  // ux 
	component_mask[dim+1]   = false;  // uy
    VectorTools::interpolate_boundary_values (dof_handler,
						fixed_id,
						ZeroFunction<dim>(dim+dim+1),  
						constraints,
						component_mask);
    // vx vy ux uy p
	//  0  0  1  0 0
    // outlet boundary
    component_mask[0] = false;		// vx
    component_mask[1] = false; 		// vy
	component_mask[dim] = true;  	// ux 
	// component_mask[dim+1]   = true;  // uy  
    VectorTools::interpolate_boundary_values (dof_handler,
						outlet_id,
						ZeroFunction<dim>(dim+dim+1),  
						constraints,
						component_mask);
}  

// In this function, we solve the linear systems
// inside the nonlinear Newton iteration. For simplicity we
// use a direct solver from UMFPACK.
template <int dim>
void 
FSI_ALE_Problem<dim>::solve () 
{
  timer.enter_section("Solve linear system.");
  Vector<double> sol, rhs;    
  sol = newton_update;    
  rhs = system_rhs;
  
  // SparseDirectUMFPACK A_direct;
  // A_direct.factorize(system_matrix);     
//   A_direct.vmult(sol,rhs); 
  SparseDirectMUMPS A_direct;
  A_direct.initialize (system_matrix);
  A_direct.vmult(sol,rhs); 
  newton_update = sol;
  
  constraints.distribute (newton_update);
  timer.exit_section();
}

template <int dim>
void 
FSI_ALE_Problem<dim>::solve_temperature () 
{
  timer.enter_section("Solve linear system of temperature.");
  Vector<double> sol, rhs;    
  sol = temperature_solution;    
  rhs = temperature_rhs;
  
  // SparseDirectUMFPACK temperature_A_direct;
  // temperature_A_direct.factorize(temperature_system_matrix);     
  // temperature_A_direct.vmult(sol,rhs); 
  SparseDirectMUMPS temperature_A_direct;
  temperature_A_direct.initialize (temperature_system_matrix);
  temperature_A_direct.vmult(sol,rhs); 
  temperature_solution = sol;
  
  temperature_constraints.distribute (temperature_solution);
  timer.exit_section();
}

// This is the Newton iteration with simple linesearch backtracking 
// to solve the 
// non-linear system of equations. First, we declare some
// standard parameters of the solution method. Addionally,
// we also implement an easy line search algorithm. 
template <int dim>
void FSI_ALE_Problem<dim>::newton_iteration (const double time) 
					       
{ 
  Timer timer_newton;
  const double lower_bound_newton_residuum = 1.0e-8; 
  const unsigned int max_no_newton_steps  = 60;

  // Decision whether the system matrix should be build
  // at each Newton step
  const double nonlinear_rho = 0.1; 
 
  // Line search parameters
  unsigned int line_search_step;
  const unsigned int  max_no_line_search_steps = 10;
  const double line_search_damping = 0.6;
  double new_newton_residuum;
  
  // Application of the initial boundary conditions to the 
  // variational equations:
  std::cout << "Setting initial_bc...\n" << std::endl;
  set_initial_bc (time);
  std::cout << "Assembling rhs...\n" << std::endl;
  assemble_system_rhs();

  double newton_residuum = system_rhs.linfty_norm(); 
  double old_newton_residuum= newton_residuum;
  unsigned int newton_step = 1;
  // prevent call A_direct.sovle() before initialized.
//    assemble_system_matrix ();
//    A_direct.initialize (system_matrix);

  if (newton_residuum < lower_bound_newton_residuum)
    {
      std::cout << '\t' 
		<< std::scientific 
		<< newton_residuum 
		<< std::endl;     
    }

  std::cout << "Start newton procedure...\n" << std::endl;

  while (newton_residuum > lower_bound_newton_residuum &&
	 newton_step < max_no_newton_steps)
    {
      timer_newton.start();
      old_newton_residuum = newton_residuum;
      
      assemble_system_rhs();
      newton_residuum = system_rhs.linfty_norm();

      if (newton_residuum < lower_bound_newton_residuum)
	{
	  std::cout << '\t' 
		    << std::scientific 
		    << newton_residuum << std::endl;
	  break;
	}
  
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	{
	  assemble_system_matrix ();
	  // using mumps
	//   A_direct.initialize (system_matrix);
	  // using UMFPACK
  	//   A_direct.factorize(system_matrix);
	}

      // Solve Ax = b
      solve ();	  
        
      line_search_step = 0;	  
      for ( ; 
	    line_search_step < max_no_line_search_steps; 
	    ++line_search_step)
	{	     					 
	  solution += newton_update;
	  
	  assemble_system_rhs ();			
	  new_newton_residuum = system_rhs.linfty_norm();
	  
	  if (new_newton_residuum < newton_residuum)
	      break;
	  else 	  
	    solution -= newton_update;
	  
	  newton_update *= line_search_damping;
	}	   
     
      timer_newton.stop();
      
      std::cout << std::setprecision(5) <<newton_step << '\t' 
		<< std::scientific << newton_residuum << '\t'
		<< std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	std::cout << "r" << '\t' ;
      else 
	std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t' 
		<< std::scientific << timer_newton ()
		<< std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;      
    }
	std::cout << "End newton procedure...\n" << std::endl;
}

// This function is known from almost all other 
// tutorial steps in deal.II.
template <int dim>
void
FSI_ALE_Problem<dim>::output_results (const unsigned int &time_step,
			      													const BlockVector<double> &output_vector, 
																			const Vector<double> &temperature_solution)  const
{
  Postprocessor postprocessor;

  std::vector<std::string> solution_names; 
  solution_names.push_back ("x_velo");
  solution_names.push_back ("y_velo"); 
  solution_names.push_back ("x_dis");
  solution_names.push_back ("y_dis");
  solution_names.push_back ("p_fluid");
   
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim+dim+1, DataComponentInterpretation::component_is_scalar);


  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler); 

  data_out.add_data_vector (output_vector, solution_names,
			    DataOut<dim>::type_dof_data,
			    data_component_interpretation);

  data_out.add_data_vector (output_vector, postprocessor);

 	data_out.add_data_vector (temperature_dof_handler, temperature_solution,
                              "mass_T");

  data_out.build_patches (std::min(degree, temperature_degree));

  std::string filename_basis;
  filename_basis  = "solution_fsi_2d_"; 
   
  std::ostringstream filename;

  std::cout << "------------------" << std::endl;
  std::cout << "Write solution" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << std::endl;
  filename << filename_basis
	   << Utilities::int_to_string (time_step, 5)
	   << ".vtu";
  
  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);

}

// With help of this function, we extract 
// point values for a certain component from our
// discrete solution. We use it to gain the 
// // displacements of the structure in the x- and y-directions.
// template <int dim>
// double FSI_ALE_Problem<dim>::compute_point_value (Point<dim> p, 
// 					       const unsigned int component) const  
// {
 
//   Vector<double> tmp_vector(dim+dim+1);
//   VectorTools::point_value (dof_handler, 
// 			    solution, 
// 			    p, 
// 			    tmp_vector);
  
//   return tmp_vector(component);
// }

// Now, we arrive at the function that is responsible 
// to compute the line integrals for the drag and the lift. Note, that 
// by a proper transformation via the Gauss theorem, the both 
// quantities could also be achieved by domain integral computation. 
// Nevertheless, we choose the line integration because deal.II provides
// all routines for face value evaluation. 
// template <int dim>
// void FSI_ALE_Problem<dim>::compute_drag_lift_fsi_fluid_tensor()
// {
    
//   const QGauss<dim-1> face_quadrature_formula (3);
//   FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula, 
// 				    update_values | update_gradients | update_normal_vectors | 
// 				    update_JxW_values);
  
//   const unsigned int dofs_per_cell = fe.dofs_per_cell;
//   const unsigned int n_face_q_points = face_quadrature_formula.size();

//   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
//   std::vector<Vector<double> >  face_solution_values (n_face_q_points, 
// 						      Vector<double> (dim+dim+1));

//   std::vector<std::vector<Tensor<1,dim> > > 
//     face_solution_grads (n_face_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));
  
//   Tensor<1,dim> drag_lift_value;
  
//   typename DoFHandler<dim>::active_cell_iterator
//     cell = dof_handler.begin_active(),
//     endc = dof_handler.end();

//    for (; cell!=endc; ++cell)
//      {

//        // First, we are going to compute the forces that
//        // act on the cylinder. We notice that only the fluid 
//        // equations are defined here.
//        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
// 	 if (cell->face(face)->at_boundary() && 
// 	     cell->face(face)->boundary_indicator()==80)
// 	   {
// 	     fe_face_values.reinit (cell, face);
// 	     fe_face_values.get_function_values (solution, face_solution_values);
// 	     fe_face_values.get_function_gradients (solution, face_solution_grads);
	 	      
// 	     for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
// 	       {	       
// 		 const Tensor<2,dim> pI = ALE_Transformations
// 		   ::get_pI<dim> (q_point, face_solution_values);
		 
// 		 const Tensor<2,dim> grad_v = ALE_Transformations 
// 		   ::get_grad_v<dim> (q_point, face_solution_grads);
		 
// 		 const Tensor<2,dim> grad_v_T = ALE_Transformations
// 		   ::get_grad_v_T<dim> (grad_v);
		 
// 		 const Tensor<2,dim> F = ALE_Transformations
// 		   ::get_F<dim> (q_point, face_solution_grads);	       	     
		 
// 		 const Tensor<2,dim> F_Inverse = ALE_Transformations
// 		   ::get_F_Inverse<dim> (F);
		 
// 		 const Tensor<2,dim> F_Inverse_T = ALE_Transformations
// 		   ::get_F_Inverse_T<dim> (F_Inverse);
		 
// 		 const double J = ALE_Transformations
// 		   ::get_J<dim> (F);
		 
// 		 const Tensor<2,dim> sigma_ALE = NSE_in_ALE
// 		   ::get_stress_fluid_except_pressure_ALE<dim> 
// 		   (density_fluid, viscosity, 
// 		    grad_v, grad_v_T, F_Inverse, F_Inverse_T );
		 
// 		 Tensor<2,dim> stress_fluid;
// 		 stress_fluid.clear();
// 		 stress_fluid = (J * sigma_ALE * F_Inverse_T);
		 
// 		 Tensor<2,dim> fluid_pressure;
// 		 fluid_pressure.clear();
// 		 fluid_pressure = (-pI * J * F_Inverse_T);
		 
// 		 drag_lift_value -= (stress_fluid + fluid_pressure) * 
// 		   fe_face_values.normal_vector(q_point)* fe_face_values.JxW(q_point); 
		 
// 	       }
// 	   } // end boundary 80 for fluid
       
//        // Now, we compute the forces that act on the beam. Here,
//        // we have two possibilities as already discussed in the paper.
//        // We use again the fluid tensor to compute 
//        // drag and lift:
//        if (cell->material_id() == fluid_id)
// 	 {	   
// 	   for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
// 	     if (cell->neighbor_index(face) != -1)	       
// 	       if (cell->material_id() !=  cell->neighbor(face)->material_id() &&
// 		   cell->face(face)->boundary_indicator()!=80)
// 		 {
		   
// 		   fe_face_values.reinit (cell, face);
// 		   fe_face_values.get_function_values (solution, face_solution_values);
// 		   fe_face_values.get_function_gradients (solution, face_solution_grads);
		   		  
// 		   for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
// 		     {
// 		       const Tensor<2,dim> pI = ALE_Transformations
// 			 ::get_pI<dim> (q_point, face_solution_values);
		       
// 		       const Tensor<2,dim> grad_v = ALE_Transformations 
// 			 ::get_grad_v<dim> (q_point, face_solution_grads);
		       
// 		       const Tensor<2,dim> grad_v_T = ALE_Transformations
// 			 ::get_grad_v_T<dim> (grad_v);
		       
// 		       const Tensor<2,dim> F = ALE_Transformations
// 			 ::get_F<dim> (q_point, face_solution_grads);	       	     
		       
// 		       const Tensor<2,dim> F_Inverse = ALE_Transformations
// 			 ::get_F_Inverse<dim> (F);
		       
// 		       const Tensor<2,dim> F_Inverse_T = ALE_Transformations
// 			 ::get_F_Inverse_T<dim> (F_Inverse);
		       
// 		       const double J = ALE_Transformations
// 			 ::get_J<dim> (F);
		       
// 		       const Tensor<2,dim> sigma_ALE = NSE_in_ALE
// 			 ::get_stress_fluid_except_pressure_ALE<dim> 
// 			 (density_fluid, viscosity, grad_v, grad_v_T, F_Inverse, F_Inverse_T );
		       
// 		       Tensor<2,dim> stress_fluid;
// 		       stress_fluid.clear();
// 		       stress_fluid = (J * sigma_ALE * F_Inverse_T);
		       
// 		       Tensor<2,dim> fluid_pressure;
// 		       fluid_pressure.clear();
// 		       fluid_pressure = (-pI * J * F_Inverse_T);
		       
// 		       drag_lift_value -= 1.0 * (stress_fluid + fluid_pressure) * 
// 			 fe_face_values.normal_vector(q_point)* fe_face_values.JxW(q_point); 		       		       
// 		     }
// 		 }	   
// 	 }               
//      } 
   
//    std::cout << "Face drag:   " << time << "   " << drag_lift_value[0] << std::endl;
//    std::cout << "Face lift:   " << time << "   " << drag_lift_value[1] << std::endl;
// }

// template <int dim>
// void FSI_ALE_Problem<dim>::compute_drag_lift_fsi_fluid_tensor_domain()
// {

//   unsigned int drag_lift_select = 0;
//   double drag_lift_constant = 1.0;

//   double  value = 0.0;
//   system_rhs = 0;
//   const QGauss<dim> quadrature_formula (3);
//    FEValues<dim>     fe_values (fe, quadrature_formula,
// 				update_values |
// 				update_gradients |
// 				update_JxW_values |
// 				update_q_points);

//    const unsigned int dofs_per_cell = fe.dofs_per_cell;
//    const unsigned int n_q_points    = quadrature_formula.size();

//    const FEValuesExtractors::Vector velocities (0);
//   const FEValuesExtractors::Vector displacements (dim); 
//   const FEValuesExtractors::Scalar pressure (dim+dim); 



//    Vector<double> local_rhs (dofs_per_cell);
//    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

//    std::vector<Vector<double> > old_solution_values (n_q_points, Vector<double> (dim+dim+1));

//    std::vector<std::vector<Tensor<1,dim> > > 
//      old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));

//      typename DoFHandler<dim>::active_cell_iterator
//      cell = dof_handler.begin_active(),
//      endc = dof_handler.end();

//    for (; cell!=endc; ++cell)
//    {
//      local_rhs = 0;

//      fe_values.reinit (cell);
//      fe_values.get_function_values (solution, old_solution_values);
//      fe_values.get_function_gradients (solution, old_solution_grads);

//      for (unsigned int q=0; q<n_q_points; ++q)
//        {
// 	 const Tensor<2,dim> pI = ALE_Transformations
// 		::get_pI<dim> (q, old_solution_values);
	      
// 	      const Tensor<1,dim> v = ALE_Transformations
// 		::get_v<dim> (q, old_solution_values);
	      
// 	      const Tensor<2,dim> grad_v = ALE_Transformations 
// 		::get_grad_v<dim> (q, old_solution_grads);
	      
// 	      const Tensor<2,dim> grad_v_T = ALE_Transformations
// 		::get_grad_v_T<dim> (grad_v);
	      
// 	      const Tensor<2,dim> F = ALE_Transformations
// 		::get_F<dim> (q, old_solution_grads);	       	     
	      
// 	      const Tensor<2,dim> F_Inverse = ALE_Transformations
// 		::get_F_Inverse<dim> (F);
	      
// 	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
// 		::get_F_Inverse_T<dim> (F_Inverse);
	      
// 	      const double J = ALE_Transformations
// 		::get_J<dim> (F);
	     

// 	      // This is the fluid stress tensor in ALE formulation
// 	      const Tensor<2,dim> sigma_ALE = NSE_in_ALE
// 		::get_stress_fluid_except_pressure_ALE<dim> 
// 		(density_fluid, viscosity, grad_v, grad_v_T, F_Inverse, F_Inverse_T );
	      

// 	   Tensor<2,dim> stress_fluid;
// 	      stress_fluid.clear();
// 	      stress_fluid = (J * sigma_ALE * F_Inverse_T);
	      

// 	   Tensor<2,dim> fluid_pressure;
// 	      fluid_pressure.clear();
// 	      fluid_pressure = (-pI * J * F_Inverse_T);
	      

// 	 Tensor<1,dim> convection_fluid;
// 	 convection_fluid.clear();
// 	 convection_fluid = density_fluid * J * (grad_v * F_Inverse * v);

// 	 // Divergence of the fluid in the ALE formulation
// 	 const double incompressiblity_fluid = NSE_in_ALE
// 	   ::get_Incompressibility_ALE<dim> (q, old_solution_grads);
	    

	 
// 	 for (unsigned int i=0; i<dofs_per_cell; ++i)
// 	   {
// 	     const unsigned int comp_i = fe.system_to_component_index(i).first;
	     
// 	     if (comp_i == 0 || comp_i == 1)
// 	       {   		  
// 		 const Tensor<1,dim> phi_i_v = fe_values[velocities].value (i, q);
// 		 const Tensor<2,dim> phi_i_grads_v = fe_values[velocities].gradient (i, q);
		 
// 		 local_rhs(i) -= (convection_fluid * phi_i_v +
// 				  scalar_product(fluid_pressure, phi_i_grads_v) +
// 				  scalar_product(stress_fluid, phi_i_grads_v) 						
// 				  ) *  fe_values.JxW(q);
		 
// 	       }		
// 	     else if (comp_i == 2 || comp_i == 3)
// 	       {	
		 
// 	       }  
// 	     else if (comp_i == 4)
// 	       {
// 		 const double phi_i_p = fe_values[pressure].value (i, q);
// 		 local_rhs(i) -= (incompressiblity_fluid * phi_i_p) *  fe_values.JxW(q);
// 	       }
// 	   }
//        }  // end q_points
     
//      cell->get_dof_indices (local_dof_indices);
     
//      for (unsigned int i=0; i<dofs_per_cell; ++i)
//        system_rhs(local_dof_indices[i]) += local_rhs(i);

//    }  // end cell



//    std::vector<bool> component_mask (dim+dim+1, true);
//    component_mask[dim] = false;
//     component_mask[dim+1] = true;
//     component_mask[dim+dim] = true;  //pressure

//    std::map<unsigned int,double> boundary_values;
//    VectorTools::interpolate_boundary_values (dof_handler,
// 					     80,
// 					     ComponentSelectFunction<dim>(drag_lift_select, drag_lift_constant,dim+dim+1),
// 					     boundary_values,
// 					     component_mask);


//    VectorTools::interpolate_boundary_values (dof_handler,
//        0,
//        ZeroFunction<dim>(dim+dim+1),
//        boundary_values,
//        component_mask);

//    VectorTools::interpolate_boundary_values (dof_handler,
//        1,
//        ZeroFunction<dim>(dim+dim+1),
//        boundary_values,
//        component_mask);

//    VectorTools::interpolate_boundary_values (dof_handler,
//        2,
//        ZeroFunction<dim>(dim+dim+1),
//        boundary_values,
//        component_mask);

//    VectorTools::interpolate_boundary_values (dof_handler,
//        81,
//        ZeroFunction<dim>(dim+dim+1),
//        boundary_values,
//        component_mask);

//    value = 0.;

//    for(std::map<unsigned int,double>::const_iterator 
// p=boundary_values.begin(); p!=boundary_values.end(); p++)
//    {
//      value += p->second * system_rhs(p->first);
//    }


//    global_drag_lift_value += value;

// }



// template <int dim>
// void FSI_ALE_Problem<dim>::compute_drag_lift_fsi_fluid_tensor_domain_structure()
// {

//   unsigned int drag_lift_select = 0;
//   double drag_lift_constant = 1.0;

//   double  value = 0.0;
//   system_rhs = 0;
//   const QGauss<dim> quadrature_formula (3);
//    FEValues<dim>     fe_values (fe, quadrature_formula,
// 				update_values |
// 				update_gradients |
// 				update_JxW_values |
// 				update_q_points);

//    const unsigned int dofs_per_cell = fe.dofs_per_cell;
//    const unsigned int n_q_points    = quadrature_formula.size();

//    const FEValuesExtractors::Vector velocities (0);
//   const FEValuesExtractors::Vector displacements (dim); 
//   const FEValuesExtractors::Scalar pressure (dim+dim); 



//    Vector<double> local_rhs (dofs_per_cell);
//    std::vector<unsigned int> local_dof_indices (dofs_per_cell);

//    std::vector<Vector<double> > old_solution_values (n_q_points, Vector<double> (dim+dim+1));

//    std::vector<std::vector<Tensor<1,dim> > > 
//      old_solution_grads (n_q_points, std::vector<Tensor<1,dim> > (dim+dim+1));

//      typename DoFHandler<dim>::active_cell_iterator
//      cell = dof_handler.begin_active(),
//      endc = dof_handler.end();

//    for (; cell!=endc; ++cell)
//    {
//      local_rhs = 0;

//      fe_values.reinit (cell);
//      fe_values.get_function_values (solution, old_solution_values);
//      fe_values.get_function_gradients (solution, old_solution_grads);

//      if (cell->material_id() == solid_id)
//      for (unsigned int q=0; q<n_q_points; ++q)
//        {
// 	      const Tensor<2,dim> F = ALE_Transformations
// 		::get_F<dim> (q, old_solution_grads);
	      
// 	      const Tensor<2,dim> F_T = ALE_Transformations
// 		::get_F_T<dim> (F);
	      
// 	      const Tensor<2,dim> Identity = ALE_Transformations
// 		::get_Identity<dim> ();
	      
// 	      const Tensor<2,dim> F_Inverse = ALE_Transformations
// 		::get_F_Inverse<dim> (F);
	      
// 	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
// 		::get_F_Inverse_T<dim> (F_Inverse);
	      
// 	      const double J = ALE_Transformations
// 		::get_J<dim> (F);
	      
// 	      const Tensor<2,dim> E = Structure_Terms_in_ALE
// 		::get_E<dim> (F_T, F, Identity);
	      
// 	      const double tr_E = Structure_Terms_in_ALE
// 		::get_tr_E<dim> (E);


// 	  // STVK structure model
// 	      Tensor<2,dim> sigma_structure_ALE;
// 	      sigma_structure_ALE.clear();
// 	      sigma_structure_ALE = (1.0/J *
// 				     F * (lame_coefficient_lambda *
// 					  tr_E * Identity +
// 					  2 * lame_coefficient_mu *
// 					  E) * 
// 				     F_T);
	      

// 	 Tensor<2,dim> stress_term;
// 	 stress_term.clear();
// 	 stress_term = (J * sigma_structure_ALE * F_Inverse_T);
	      

	 
// 	 for (unsigned int i=0; i<dofs_per_cell; ++i)
// 	   {
// 	     const unsigned int comp_i = fe.system_to_component_index(i).first;
	     
// 	     if (comp_i == 0 || comp_i == 1)
// 	       {   		  
// 		 const Tensor<2,dim> phi_i_grads_v = fe_values[velocities].gradient (i, q);
		 
// 		 local_rhs(i) -= (scalar_product(stress_term,phi_i_grads_v) 
// 				  ) * fe_values.JxW(q);
		 
// 	       }		
// 	     else if (comp_i == 2 || comp_i == 3)
// 	       {	
		 
// 	       }  
// 	     else if (comp_i == 4)
// 	       {
	
// 	       }
// 	   }
//        }  // end q_points
     
//      cell->get_dof_indices (local_dof_indices);
     
//      for (unsigned int i=0; i<dofs_per_cell; ++i)
//        system_rhs(local_dof_indices[i]) += local_rhs(i);

//    }  // end cell



//    std::vector<bool> component_mask (dim+dim+1, true);
//    component_mask[dim] = false;
//     component_mask[dim+1] = true;
//     component_mask[dim+dim] = true;  //pressure

//    std::map<unsigned int,double> boundary_values;

//    VectorTools::interpolate_boundary_values (dof_handler,
// 					     81,
// 					     ComponentSelectFunction<dim>(drag_lift_select, drag_lift_constant,dim+dim+1),
// 					     boundary_values,
// 					     component_mask);

//    VectorTools::interpolate_boundary_values (dof_handler,
// 					     80,
// 					     ZeroFunction<dim>(dim+dim+1),
// 					     boundary_values,
// 					     component_mask);


//    VectorTools::interpolate_boundary_values (dof_handler,
//        0,
//        ZeroFunction<dim>(dim+dim+1),
//        boundary_values,
//        component_mask);

//    VectorTools::interpolate_boundary_values (dof_handler,
//        1,
//        ZeroFunction<dim>(dim+dim+1),
//        boundary_values,
//        component_mask);

//    VectorTools::interpolate_boundary_values (dof_handler,
//        2,
//        ZeroFunction<dim>(dim+dim+1),
//        boundary_values,
//        component_mask);


//    value = 0.;

//    for(std::map<unsigned int,double>::const_iterator 
// p=boundary_values.begin(); p!=boundary_values.end(); p++)
//    {
//      value += p->second * system_rhs(p->first);
//    }


//    global_drag_lift_value += value;


// }







template <int dim>
void FSI_ALE_Problem<dim>::compute_minimal_J()
{
  QGauss<dim>   quadrature_formula(degree+2);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |
                           update_quadrature_points  |
                           update_JxW_values |
                           update_gradients);
  const unsigned int   n_q_points      = quadrature_formula.size();
  
  
  std::vector<std::vector<Tensor<1,dim> > > old_solution_grads (n_q_points, 
								std::vector<Tensor<1,dim> > (dim+dim+1));
  
  double min_J= 1.0e+5;
  double J=1.0e+5;


  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  
  for (; cell!=endc; ++cell)
    { 
      fe_values.reinit (cell);
            
      fe_values.get_function_gradients (solution, old_solution_grads);
      
      if (cell->material_id() == fluid_id)
	{
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {
	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);
	      
	      J = ALE_Transformations::get_J<dim> (F);
	      if (J < min_J)
		min_J = J;
	      
	    }
	}
      
    }
  
  std::cout << "Min J: " << time << "   "  << min_J << std::endl;
}









// Here, we compute the four quantities of interest:
// the x and y-displacements of the structure, the drag, and the lift.
template<int dim>
void FSI_ALE_Problem<dim>::compute_functional_values()
{
//   double x1,y1;
//   x1 = compute_point_value(Point<dim>(0.6,0.2), dim);
//   y1 = compute_point_value(Point<dim>(0.6,0.2), dim+1);
  
//   std::cout << "------------------" << std::endl;
//   std::cout << "DisX: " << time << "   " << x1 << std::endl;
//   std::cout << "DisY: " << time << "   " << y1 << std::endl;
//   std::cout << "------------------" << std::endl;
  
//   // Compute drag and lift via line integral
//   compute_drag_lift_fsi_fluid_tensor();

//   // Compute drag and lift via domain integral
//   global_drag_lift_value = 0.0;
//   compute_drag_lift_fsi_fluid_tensor_domain();
//   compute_drag_lift_fsi_fluid_tensor_domain_structure();
//   std::cout << "Domain drag: " << time << "   "  << global_drag_lift_value << std::endl;

  std::cout << "------------------" << std::endl;
  compute_minimal_J();
  
  std::cout << std::endl;
}


// template<int dim>
// void FSI_ALE_Problem<dim>::refine_mesh()
// {
//   typename DoFHandler<dim>::active_cell_iterator
//     cell = dof_handler.begin_active(),
//     endc = dof_handler.end();
  
//   for (; cell!=endc; ++cell)
//     {
//       // Only Refine the solid
//       if (cell->material_id() == solid_id)
// 	cell->set_refine_flag();
//     }


//   BlockVector<double> tmp_solution;
//   tmp_solution = solution;
  
//   SolutionTransfer<dim, BlockVector<double> > solution_transfer (dof_handler);
  
//   triangulation.prepare_coarsening_and_refinement();
//   solution_transfer.prepare_for_coarsening_and_refinement(tmp_solution);
  
//   triangulation.execute_coarsening_and_refinement ();
//   setup_system ();
  
//   solution_transfer.interpolate(tmp_solution, solution); 

// }





// As usual, we have to call the run method. It handles
// the output stream to the terminal.
// Second, we define some output skip that is necessary 
// (and really useful) to avoid to much printing 
// of solutions. For large time dependent problems it is 
// sufficient to print only each tenth solution. 
// Third, we perform the time stepping scheme of 
// the solution process.
template <int dim>
void FSI_ALE_Problem<dim>::run () 
{  

  // We set runtime parameters to drive the problem.
  // These parameters could also be read from a parameter file that
  // can be handled by the ParameterHandler object (see step-19)
  set_runtime_parameters ();


  setup_system();

  
  const unsigned int output_skip = 1;


//   unsigned int refine_mesh_1st = 1;
//   unsigned int refine_mesh_2nd = 2;
//   unsigned int refine_mesh_3rd = 3;

//   unsigned int refinement_cycle = 0;
//   const bool mesh_refinement = false;


  do
    { 
      std::cout << "Timestep " << timestep_number 
		<< " (" << time_stepping_scheme 
		<< ")" <<    ": " << time
		<< " (" << timestep << ")"
		<< "\n==============================" 
		<< "=====================================" 
		<< std::endl; 
      
      std::cout << std::endl;
      
      // Compute next time step
      old_timestep_solution = solution;
			old_temperature_solution = temperature_solution;
      newton_iteration (time); 
			VectorTools::interpolate_boundary_values	(	temperature_dof_handler,
																									inlet_id,
																									ConstantFunction (1),
																									temperature_constraints
																								);
			assemble_temperature_system_matrix ();  
			// ofstream outTM("testTM.dat");
			// temperature_system_matrix(outTM);
			// outTM.close();
			assemble_temperature_system_rhs ();
			solve_temperature ();
      time += timestep;
	
      // Compute functional values: dx, dy, drag, lift
      std::cout << std::endl;
      compute_functional_values();
      
      // Write solutions 
      if ((timestep_number % output_skip == 0))
	    output_results (timestep_number,solution,temperature_solution);


    //   if (mesh_refinement && (timestep_number  == refine_mesh_1st ||
	// 		      timestep_number  == refine_mesh_2nd ||
	// 		      timestep_number  == refine_mesh_3rd))			      			      			     
	//   {
	//     std::cout << "Refinement cycle " 
	// 	          << refinement_cycle 
	// 	          << "\n================== "
	// 	          << std::endl;
	  
	//     refine_mesh ();
	//     ++refinement_cycle;		
	//   }
      
      ++timestep_number;

    }
    while (timestep_number <= max_no_timesteps);
   
}

// The main function looks almost the same
// as in all other deal.II tuturial steps.

int main (int argc, char **argv) 
{
Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv); 

  try
    {
      deallog.depth_console (0);
	  	const unsigned int degree = 2;
			const unsigned int temperature_degree = 2;
      FSI_ALE_Problem<2> flow_problem(degree, temperature_degree);
      flow_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      
      return 1;
    }
  catch (...) 
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}




