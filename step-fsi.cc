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
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>

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
#include <set>
#include "Transformations.h"
#include "Parameters.h"
#include "SolidParameters.h"
// #include "solver_mumps.h"
// At the end of this top-matter, we import
// all deal.II names into the global
// namespace:				
using namespace dealii;


const long double pi = 3.141592653589793238462643;
const double R = 1.2e-2;




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

 // component 0 vx
 // component 1 vy
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


template <int dim>
class FSI_ALE_Problem 
{
public:
  
  FSI_ALE_Problem (const unsigned int &degree);
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

  // Setup of material parameters, time-stepping scheme
  // spatial grid, etc.
  void set_runtime_parameters ();

  // Create system matrix, rhs and distribute degrees of freedom.
  void setup_system ();

  // Assemble left and right hand side for Newton's method
  void assemble_system_matrix ();   
  void assemble_system_rhs ();

  void local_assemble_system_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                     AssemblyScratchData                                  &scratch,
                                     AssemblyMatrixCopyData                               &copy_data);
  void local_assemble_system_rhs (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  AssemblyScratchData                                  &scratch,
                                  AssemblyRhsCopyData                                  &copy_data);
  void copy_local_to_global_matrix (const AssemblyMatrixCopyData &copy_data);
  void copy_local_to_global_rhs (const AssemblyRhsCopyData &copy_data);

  // Boundary conditions (bc)
  void set_initial_bc (const double time);
  void set_newton_bc ();
  double inlet_flow (const double t);
  // Linear solver
  void solve ();

  // Nonlinear solver
  void newton_iteration(const double time);			  

  // Graphical visualization of output
  void output_results (const unsigned int refinement_cycle,
		       const BlockVector<double> solution) const;



  void compute_functional_values ();
  void compute_minimal_J();

//   // class to output F and E tensor
//   class Postprocessor : public DataPostprocessor<dim>
//   {
//     public:
//     //   Postprocessor ();
//       virtual
//       void
//       compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
//                                          const std::vector<std::vector<Tensor<1,dim> > > &duh,
//                                          const std::vector<std::vector<Tensor<2,dim> > > &dduh,
//                                          const std::vector<Point<dim> >                  &normals,
//                                          const std::vector<Point<dim> >                  &evaluation_points,
//                                          std::vector<Vector<double> >                    &computed_quantities) const;
//       virtual std::vector<std::string> get_names () const;
//       virtual
//       std::vector<DataComponentInterpretation::DataComponentInterpretation>
//       get_data_component_interpretation () const;
//       virtual UpdateFlags get_needed_update_flags () const;
//   };


 
  const unsigned int   degree;
  
  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  AffineConstraints<double>    constraints; 
  
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
  double lame_mu[4], lame_lambda[4];
  double solid_bulk[3];
  double poisson_ratio_nu;  
	double density_poro;
	double kappa;
  // Other parameters to control the fluid mesh motion 
  double cell_diameter; 

  // modified lapalace
  double alpha_u;
  // linear elastic
  double alpha_lambda;
  double alpha_mu;
 
  double force_structure_x, force_structure_y;
  
  double global_drag_lift_value;
//   SparseDirectMUMPS A_direct;
  SparseDirectUMFPACK A_direct;

  unsigned int solid_id[4] = { 2,3,4,5 }; 
  unsigned int fluid_id = 1, 		   
			   fixed_id = 6, 
			   inlet_id = 7, 
			   outlet_id= 9,
			   symmetry_id = 10;
  
};


// The constructor of this class is comparable 
// to other tutorials steps, e.g., step-22, and step-31. 
// We are going to use the following finite element discretization: 
// Q_2^c for the fluid, Q_2^c for the solid, P_1^dc for the pressure. 
template <int dim>
FSI_ALE_Problem<dim>::FSI_ALE_Problem (const unsigned int &degree)
                :
                degree (degree),
				triangulation (Triangulation<dim>::maximum_smoothing),
                fe (FE_Q<dim>(degree), dim,  // velocities                  
		    		FE_Q<dim>(degree), dim,  // displacements		    
		    		FE_DGP<dim>(degree-1), 1),   // pressure
                dof_handler (triangulation),
				timer (std::cout, TimerOutput::summary, TimerOutput::cpu_times)		
{}


// This is the standard destructor.
template <int dim>
FSI_ALE_Problem<dim>::~FSI_ALE_Problem () 
{}


// template <int dim>
//   void
//   FSI_ALE_Problem<dim>::Postprocessor::
//   compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
//                                      const std::vector<std::vector<Tensor<1,dim> > > &duh,
//                                      const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
//                                      const std::vector<Point<dim> >                  &/*normals*/,
//                                      const std::vector<Point<dim> >                  &/*evaluation_points*/,
//                                      std::vector<Vector<double> >                    &computed_quantities) const
//   {
//     const unsigned int n_quadrature_points = uh.size();
//     Assert (duh.size() == n_quadrature_points,
//             ExcInternalError())

//     Assert (computed_quantities.size() == n_quadrature_points,
//             ExcInternalError());
// 	// assert dim*2+1component
//     Assert (uh[0].size() == (dim+dim+1),
//             ExcInternalError());
    
//     Assert (computed_quantities[0].size() == 10, ExcInternalError())
//     Tensor<2,dim> identity;
// 	identity.clear();
//     identity[0][0] = 1.0;
//     identity[0][1] = 0.0;
//     identity[1][0] = 0.0;
//     identity[1][1] = 1.0;

//     for (unsigned int q=0; q<n_quadrature_points; ++q)
//       {
// 		  Tensor<2, dim> F;
// 		  F.clear();
// 		  F[0][0] = duh[q][dim  ][0] + 1.0;
// 		  F[0][1] = duh[q][dim  ][1];
// 		  F[1][0] = duh[q][dim+1][0];
// 		  F[1][1] = duh[q][dim+1][1] + 1.0;
// 		  Tensor<2, dim> E;
// 		  E.clear();
// 		  E = 0.5 * (transpose (F) * F - identity);
// 		  double output_trace_E = trace(E);
// 		  double output_J = determinant(F);
// 		  computed_quantities[q](0) = F[0][0];
// 		  computed_quantities[q](1) = F[0][1];
// 		  computed_quantities[q](2) = F[1][0];
// 		  computed_quantities[q](3) = F[1][1];
// 		  computed_quantities[q](4) = E[0][0];
// 		  computed_quantities[q](5) = E[0][1];
// 		  computed_quantities[q](6) = E[1][0];
// 		  computed_quantities[q](7) = E[1][1];
// 		  computed_quantities[q](8) = output_trace_E;
// 		  computed_quantities[q](9) = output_J;
//       }
//   }

//   template <int dim>
//   std::vector<std::string>
//   FSI_ALE_Problem<dim>::Postprocessor::
//   get_names () const
//   {
//     std::vector<std::string> names;
//     names.push_back ("Fxx");
// 	names.push_back ("Fxy");
// 	names.push_back ("Fyx");
// 	names.push_back ("Fyy");
// 	names.push_back ("Exx");
// 	names.push_back ("Exy");
// 	names.push_back ("Eyx");
// 	names.push_back ("Eyy");
// 	names.push_back ("trace_E");
//     names.push_back ("J");
//     return names;
//   }

//   template <int dim>
//   std::vector<DataComponentInterpretation::DataComponentInterpretation>
//   FSI_ALE_Problem<dim>::Postprocessor::
//   get_data_component_interpretation () const
//   {
//     std::vector<DataComponentInterpretation::DataComponentInterpretation>
//     interpretation (10,
//                     DataComponentInterpretation::component_is_scalar);
 
//     return interpretation;
//   }

//   template <int dim>
//   UpdateFlags
//   FSI_ALE_Problem<dim>::Postprocessor::
//   get_needed_update_flags () const
//   {
//     return update_values | update_gradients;
//   }


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

  // Structure parameters
  // FSI 1 & 2: 0.5e+6; FSI 3: 2.0e+6
  double E[3];
  E[0] = 2.e+6;
  E[1] = 6.e+6;
  E[2] = 4.e+6; 
  E[3] = 5.e+6;

  poisson_ratio_nu = 0.45; 

  // double tmp1 = poisson_ratio_nu/((1+poisson_ratio_nu)*(1-2*poisson_ratio_nu));
  // double tmp2 = 2*(1+poisson_ratio_nu);

  for(int i=0;i<3;++i){
	  lame_mu[i] = E[i]/(2*(1+poisson_ratio_nu));
	  lame_lambda[i]= E[i]*poisson_ratio_nu/((1+poisson_ratio_nu)*(1-2*poisson_ratio_nu));
	  solid_bulk[i] = E[i]/3/(1-2*poisson_ratio_nu);
  }


	double nu_poro = 0.49;
	// tmp1 = nu_poro/((1+nu_poro)*(1-2*nu_poro));
	// tmp2 = 2*(1+nu_poro);

    lame_mu[3] = E[3]/(2*(1+nu_poro));
	lame_lambda[3]= E[3]*nu_poro/((1+nu_poro)*(1-2*nu_poro));
	density_poro = 1.3e+3; 				// kg/m^3

	kappa = 1e-13;  							// m^3 s kg^{-1}


	double mesh_nu = 0.3;
	double mesh_E = 1;
  alpha_mu = mesh_E/(2*(1+mesh_nu));
  alpha_lambda = mesh_E*mesh_nu/((1+mesh_nu)*(1-2*mesh_nu));
  // Force on beam
  force_structure_x = 0; 
  force_structure_y = 0; 


  alpha_u = 1e-8; 

  // Timestepping schemes
  //BE, CN, CN_shifted
  time_stepping_scheme = "CN_shifted";


  timestep = 5e-3;


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
  grid_name = "Khanafer_half_hematoma.msh";
  std::cout << "\n================================\n"
 	        << "Mesh file: " << grid_name << std::endl;
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file(grid_name.c_str());      
  Assert (dim==2, ExcInternalError());

  grid_in.read_msh(input_file);

	std::set<int> allID;
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  
  for (; cell!=endc; ++cell){
	int id = cell->material_id();
	allID.insert(id);
	for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if (cell->face(face)->at_boundary())   // boundary_indicator boundary_id
		  {
			int id = cell->face(face)->boundary_id();
			allID.insert(id);
		  }
		}
  }
	for (auto i: allID){
		std::cout << i << " ";
	}
	std::cout << std::endl;

  // triangulation.refine_global (parameters.no_of_refinements); 
	// triangulation.refine_global (1); 
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
  TimerOutput::Scope t(timer, "setup");

  system_matrix.clear ();
  
  dof_handler.distribute_dofs (fe);  
//   DoFRenumbering::Cuthill_McKee (dof_handler);
  DoFRenumbering::boost::king_ordering (dof_handler);

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

  dofs_per_block = DoFTools::count_dofs_per_fe_block (dof_handler, block_component);
   
  const unsigned int n_v = dofs_per_block[0],
    n_u = dofs_per_block[1],
    n_p =  dofs_per_block[2];

  std::cout << "Cells:\t"
            << triangulation.n_active_cells()
            << std::endl  	  
            << "DoFs:\t"
            << dof_handler.n_dofs()
            << " (" << n_v << '+' << n_u << '+' << n_p <<  ')'
            << std::endl;


 
      
 {

	BlockDynamicSparsityPattern csp (3,3);

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
//   for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
//   {
//     for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
//       system_matrix.add (copy_data.local_dof_indices[j],
//                             copy_data.local_dof_indices[i],
//                             copy_data.cell_matrix(j,i));
//   }
}

template <int dim>
void
FSI_ALE_Problem<dim>::copy_local_to_global_rhs (const AssemblyRhsCopyData &copy_data)
{
	constraints.distribute_local_to_global (copy_data.cell_rhs,
                                            copy_data.local_dof_indices,
                                            system_rhs);
//   for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
//   {
//     system_rhs(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
//   }
}

template <int dim>
void FSI_ALE_Problem<dim>::assemble_system_matrix ()
{
  TimerOutput::Scope t(timer, "Assemble Matrix.");
  system_matrix=0;
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &FSI_ALE_Problem::local_assemble_system_matrix,
                  &FSI_ALE_Problem::copy_local_to_global_matrix,
                  AssemblyScratchData(fe,degree),
                  AssemblyMatrixCopyData());

}



template <int dim>
void FSI_ALE_Problem<dim>::assemble_system_rhs ()
{
  TimerOutput::Scope t(timer, "Assemble Rhs.");
  system_rhs=0;
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &FSI_ALE_Problem::local_assemble_system_rhs,
                  &FSI_ALE_Problem::copy_local_to_global_rhs,
                  AssemblyScratchData(fe,degree),
                  AssemblyRhsCopyData());
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
	std::vector<Tensor<1,dim> > phi_i_grads_p (dofs_per_cell);   
  std::vector<Tensor<1,dim> > phi_i_u (dofs_per_cell); 
  std::vector<Tensor<2,dim> > phi_i_grads_u(dofs_per_cell);

  // This is the identity matrix in two dimensions:
  const Tensor<2,dim> Identity = ALE_Transformations
    ::get_Identity<dim> ();
 				     				   
      scratch_data.fe_values.reinit (cell);
      copy_data.cell_matrix = 0;
      
      // We need the cell diameter to control the fluid mesh motion
      // cell_diameter = cell->diameter();
      
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

	      Tensor<2,dim> u_sym;
		  u_sym.clear();
	      u_sym = Structure_Terms_in_ALE::get_u_sym(q, old_solution_grads);

		  double trU = Structure_Terms_in_ALE::get_trU(q, old_solution_grads);

	      // Outer loop for dofs
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	
		  const Tensor<2,dim> pI_LinP = ALE_Transformations
		    ::get_pI_LinP<dim> (phi_i_p[i]);
		  
		  const Tensor<2,dim> grad_v_LinV = ALE_Transformations
		    ::get_grad_v_LinV<dim> (phi_i_grads_v[i]);
		  
		  const double J_LinU =  ALE_Transformations
		    ::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]);

		  const Tensor<2,dim> F_LinU = ALE_Transformations		  
		    ::get_F_LinU<dim> (phi_i_grads_u[i]);

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

          

		  Tensor<2,dim> u_sym_LinU;
		  u_sym_LinU.clear();
	      u_sym_LinU = Structure_Terms_in_ALE::get_u_sym_LinU(phi_i_grads_u[i]);
		  double trU_LinU = Structure_Terms_in_ALE::get_trU_LinU(phi_i_grads_u[i]);

		  Tensor<2,dim> sigma_LinU;
		  sigma_LinU.clear();
		  sigma_LinU = F_LinU*(alpha_lambda*trU*Identity+2*alpha_mu*u_sym)
		  			 + F*(alpha_lambda*trU_LinU*Identity+2*alpha_mu*u_sym_LinU);



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
			  // change to linear elastic
			  copy_data.cell_matrix(j,i) += (scalar_product(sigma_LinU, phi_i_grads_u[j]) 
						) * scratch_data.fe_values.JxW(q);
			//   copy_data.cell_matrix(j,i) += (-alpha_u/(J*J) * J_LinU * scalar_product(grad_u, phi_i_grads_u[j]) 
			// 			+ alpha_u/J * scalar_product(phi_i_grads_u[i], phi_i_grads_u[j])
			// 			) * scratch_data.fe_values.JxW(q);

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
		  (cell->face(face)->boundary_id() == outlet_id)   // boundary_indicator boundary_id
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
	      
		  const double J = ALE_Transformations
		::get_J<dim> (F);

	      const Tensor<2,dim> old_timestep_F = ALE_Transformations
		::get_F<dim> (q, old_timestep_solution_grads);
	      
	      const double old_timestep_J = ALE_Transformations
		::get_J<dim> (old_timestep_F);

	      const Tensor<2,dim> E = Structure_Terms_in_ALE 
		::get_E<dim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (E);

	      const double solid_pressure_scalar = old_solution_values[q](dim+dim);
          const double old_timestep_solid_pressure_scalar = old_timestep_solution_values[q](dim+dim);

		  Tensor<2,dim> solid_pressure;
		  solid_pressure.clear();
		  solid_pressure = (-solid_pressure_scalar * Identity * J * F_Inverse_T);
		  Tensor<2,dim> old_timestep_solid_pressure;
		  old_timestep_solid_pressure.clear();
		  solid_pressure = (-old_timestep_solid_pressure * Identity * old_timestep_J * old_timestep_F_Inverse_T);

          const double betaTheta = get_betaTheta_ST91(solid_bulk, solid_pressure_scalar);
		  const double dbetaThetadp = get_dbetaThetadp_ST91(solid_bulk, solid_pressure_scalar);
		  const double old_timestep_betaTheta = get_betaTheta_ST91(solid_bulk, old_timestep_solid_pressure_scalar);
          const double density_structure_new = get_rho_ST91(solid_bulk, density_structure, solid_pressure_scalar);
		  const double drhodp = get_drhodp_ST91(solid_bulk, density_structure, solid_pressure_scalar);
		  const double old_timestep_density_structure_new = get_rho_ST91(solid_bulk, density_structure, old_timestep_solid_pressure_scalar);

	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	 

	      const Tensor<1,dim> v = ALE_Transformations
		::get_v<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> u = ALE_Transformations
		::get_u<dim> (q, old_solution_values);
			  // Previous time step values
	      const Tensor<1,dim> old_timestep_v = ALE_Transformations
		::get_v<dim> (q, old_timestep_solution_values);
	      
	      const Tensor<1,dim> old_timestep_u = ALE_Transformations
		::get_u<dim> (q, old_timestep_solution_values);

		  const Tensor<2,dim> F_LinU = ALE_Transformations		  
		    ::get_F_LinU<dim> (phi_i_grads_u[i]);

		  const double J_LinU = ALE_Transformations		  
		    ::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]); 

		  const double incompressibility_ALE_LinAll = NSE_in_ALE
		    ::get_Incompressibility_ALE_LinAll<dim> 
		    (phi_i_grads_v[i], phi_i_grads_u[i], q, old_solution_grads); 
			
		  // STVK: Green-Lagrange strain tensor derivatives
		  const Tensor<2,dim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);
		  
		  const double tr_E_LinU = Structure_Terms_in_ALE
		    ::get_tr_E_LinU<dim> (q,old_solution_grads, phi_i_grads_u[i]);
		  
		       
		  // STVK
		  // Piola-kirchhoff stress structure STVK linearized in all directions 		  
		//   Tensor<2,dim> piola_kirchhoff_stress_structure_STVK_LinALL;
		//   piola_kirchhoff_stress_structure_STVK_LinALL = lame_coefficient_lambda * 
		//     (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) 
		//     + 2 * lame_coefficient_mu * (F_LinU * E + F * E_LinU);
		Tensor<2,dim> 1st_PK_NH_LinU;
		1st_PK_NH_LinU = Structure_Terms_in_ALE::get_1st_PK_NH_LinU(lame_coefficient_mu,
                									J, J_LinU, trC, trC_LinU,
													F, F_LinU, F_Inverse_T, F_Inverse_T_LinU);



		double dJrho = J_LinU*density_structure_new+J*drhodp*phi_i_p[i];

		// linearized for solid
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      // STVK 
		      const unsigned int comp_j = fe.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1)
			{
			  copy_data.cell_matrix(j,i) += (0.5*dJrho*(v - old_timestep_v)* phi_i_v[j]+0.5*(J*density_structure_new+old_timestep_J*old_timestep_density_structure_new)*phi_i_v[i] * phi_i_v[j] +   						   
						timestep * theta * scalar_product(1st_PK_NH_LinU, 
										  phi_i_grads_v[j]) 
						) * scratch_data.fe_values.JxW(q);      	
			}		     
		      else if (comp_j == 2 || comp_j == 3)
			{
			  
			  copy_data.cell_matrix(j,i) += (0.5*dJrho*(u - old_timestep_u)* phi_i_u[j]+0.5*(J*density_structure_new+old_timestep_J*old_timestep_density_structure_new)*phi_i_u[i] * phi_i_u[j]
			  							    -timestep * theta * dJrho * v * phi_i_u[j]
										    -timestep * theta * J * density_structure_new * phi_i_v[i] * phi_i_u[j]
										    ) *  scratch_data.fe_values.JxW(q);	
			}
		      else if (comp_j == 4)
			{
			  copy_data.cell_matrix(j,i) += ( 0.5*J_LinU*betaTheta*(solid_pressure_scalar-old_timestep_solid_pressure_scalar) * phi_i_p[j]
											+ 0.5*J*dbetaThetadp*phi_i_p[i]*(solid_pressure_scalar-old_timestep_solid_pressure_scalar) * phi_i_p[j]
											+ 0.5*(J*betaTheta+old_timestep_J*old_timestep_betaTheta)*phi_i_p[i] * phi_i_p[j]
											+ timestep * theta * Incompressibility_ALE_LinAll * phi_i_p[j]
											) * scratch_data.fe_values.JxW(q);      
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
	 else if (cell->material_id() == solid_id[3] )
	{	  
		double lame_coefficient_mu, lame_coefficient_lambda;
		 
		 	lame_coefficient_mu = lame_mu[3];
			lame_coefficient_lambda = lame_lambda[3];
	
	  for (unsigned int q=0; q<n_q_points; ++q)
	    {	      
	      for (unsigned int k=0; k<dofs_per_cell; ++k)
		{
		  phi_i_v[k]       = scratch_data.fe_values[velocities].value (k, q);
		  phi_i_grads_v[k] = scratch_data.fe_values[velocities].gradient (k, q);
		  phi_i_p[k]       = scratch_data.fe_values[pressure].value (k, q);		
			phi_i_grads_p[k] = scratch_data.fe_values[pressure].gradient (k, q);		  
		  phi_i_u[k]       = scratch_data.fe_values[displacements].value (k, q);
		  phi_i_grads_u[k] = scratch_data.fe_values[displacements].gradient (k, q);
		}
	      
	      // It is here the same as already shown for the fluid equations.
	      // First, we prepare things coming from the previous Newton
	      // iteration...

		  const Tensor<2,dim> pI = ALE_Transformations		
		::get_pI<dim> (q, old_solution_values);

	      const Tensor<1,dim> v = ALE_Transformations
		::get_v<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> u = ALE_Transformations
		::get_u<dim> (q,old_solution_values);

		  const Tensor<1,dim> grad_P = ALE_Transformations
		::get_grad_p (q, old_solution_grads);

	      const Tensor<2,dim> F = ALE_Transformations
		::get_F<dim> (q, old_solution_grads);
	      
	      const Tensor<2,dim> F_T = ALE_Transformations
		::get_F_T<dim> (F);

	      const Tensor<2,dim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (F);
	      
	      const Tensor<2,dim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (F_Inverse);

				const double J = ALE_Transformations
		::get_J<dim> (F);

	      const Tensor<2,dim> E = Structure_Terms_in_ALE 
		::get_E<dim> (F_T, F, Identity);
		
	      const Tensor<2,dim> u_sym = Structure_Terms_in_ALE 
		::get_u_sym<dim>(q, old_solution_grads);

	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (E);

		  const double tr_U = Structure_Terms_in_ALE
		::get_tr_U<dim> (q, old_solution_grads);
	      	      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{	    
			const double J_LinU =  ALE_Transformations
		    ::get_J_LinU<dim> (q, old_solution_grads, phi_i_grads_u[i]);
				
			const Tensor<2,dim> F_Inverse_LinU = ALE_Transformations
		    ::get_F_Inverse_LinU (phi_i_grads_u[i], J, J_LinU, q, old_solution_grads);

			const Tensor<2,dim> J_F_Inverse_T_LinU = ALE_Transformations
		    ::get_J_F_Inverse_T_LinU<dim> (phi_i_grads_u[i]);

			const Tensor<2,dim> pI_LinP = ALE_Transformations
		    ::get_pI_LinP<dim> (phi_i_p[i]);

		  const Tensor<2,dim> F_LinU = ALE_Transformations		  
		    ::get_F_LinU<dim> (phi_i_grads_u[i]);

		  const Tensor<1,dim> grad_P_LinP = ALE_Transformations
			  ::get_grad_p_LinP (phi_i_grads_p[i]);	

          const Tensor<1,dim> accelaration_term_LinAll = NSE_in_ALE
		    ::get_accelaration_term_LinAll 
		    (phi_i_v[i], v, old_timestep_v, J_LinU,
		     J, old_timestep_J, density_poro);

	      const Tensor<1,dim> solidV_term_LinAll = NSE_in_ALE
		    ::get_solidV_term_LinAll 
		    (phi_i_u[i], u, old_timestep_u, J_LinU,
		     J, old_timestep_J, density_poro);
		    
		  // STVK: Green-Lagrange strain tensor derivatives
		//   const Tensor<2,dim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);
		  
		//   const double tr_E_LinU = Structure_Terms_in_ALE
		//     ::get_tr_E_LinU<dim> (q,old_solution_grads, phi_i_grads_u[i]);
		  const Tensor<2,dim> u_sym_LinU = Structure_Terms_in_ALE
		    ::get_u_sym_LinU(phi_i_grads_u[i]);

		  const double tr_U_LinU = Structure_Terms_in_ALE
		    ::get_trU_LinU<dim> (phi_i_grads_u[i]);

			const Tensor<2,dim>  stress_poro_ALE_1st_term_LinAll = NSE_in_ALE			
		    ::get_stress_fluid_ALE_1st_term_LinAll<dim> 
		    (pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);
		       
			const double incompressibility_ALE_LinAll = NSE_in_ALE
		    ::get_Incompressibility_ALE_LinAll<dim> 
		    (phi_i_grads_v[i], phi_i_grads_u[i], q, old_solution_grads); 

			const Tensor<1,dim> grad_p_LinAll = NSE_in_ALE
			::get_gradP_ALE_LinAll<dim>(J, grad_P, grad_P_LinP, F_Inverse, F_Inverse_T, 
		     F_Inverse_LinU, J_F_Inverse_T_LinU);
		  // linear elastic but still call STVK
		  // Piola-kirchhoff stress structure STVK linearized in all directions 	
		  // change to linear elastic	  
		  Tensor<2,dim> piola_kirchhoff_stress_structure_STVK_LinALL;
		  piola_kirchhoff_stress_structure_STVK_LinALL = lame_coefficient_lambda * 
		    (F_LinU * tr_U * Identity + F * tr_U_LinU * Identity) 
		    + 2 * lame_coefficient_mu * (F_LinU * u_sym + F * u_sym_LinU);
		       
			// linearized for Poroelasticity
		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		    {
		      // STVK 
		      const unsigned int comp_j = fe.system_to_component_index(j).first; 
		      if (comp_j == 0 || comp_j == 1)
			{
			  copy_data.cell_matrix(j,i) += ( accelaration_term_LinAll * phi_i_v[j] + 
				    	timestep * theta * scalar_product(stress_poro_ALE_1st_term_LinAll, phi_i_grads_v[j]) +			
						timestep * theta * scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL, 
										  phi_i_grads_v[j]) 
						) * scratch_data.fe_values.JxW(q);
			}		     
		      else if (comp_j == 2 || comp_j == 3)
			{
			  copy_data.cell_matrix(j,i) += (
						solidV_term_LinAll +
						density_poro * (
						  - timestep * theta * J_LinU * v
						  - timestep * theta * J * phi_i_v[i]
						)						
						) * phi_i_u[j] *  scratch_data.fe_values.JxW(q);			  
			}
		      else if (comp_j == 4)
			{
			  copy_data.cell_matrix(j,i) += ( incompressibility_ALE_LinAll * phi_i_p[j] 
											+ kappa * grad_p_LinAll * phi_i_grads_p[j]
											) * scratch_data.fe_values.JxW(q);      
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
	  // end if (second PDE: STVK-poro material)  
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
      
      // cell_diameter = cell->diameter();
      
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
		  const Tensor<2,dim> old_timestep_pI = ALE_Transformations
		::get_pI<dim> (q, old_timestep_solution_values);

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
	  
	      Tensor<2,dim> old_timestep_fluid_pressure;
	      old_timestep_fluid_pressure.clear();
	      old_timestep_fluid_pressure = (-old_timestep_pI * old_timestep_J * old_timestep_F_Inverse_T);

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

	      Tensor<2,dim> u_sym;
		  u_sym.clear();
	      u_sym = Structure_Terms_in_ALE::get_u_sym(q, old_solution_grads);

		  double trU = Structure_Terms_in_ALE::get_trU(q, old_solution_grads);

		  const Tensor<2,dim> Identity = ALE_Transformations
		::get_Identity<dim> ();

		  Tensor<2,dim> sigma_mesh;
		  sigma_mesh.clear();
		  sigma_mesh = F*(alpha_lambda*trU*Identity+2*alpha_mu*u_sym);

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
				       timestep * theta * scalar_product(fluid_pressure, phi_i_grads_v) +
					   timestep * (1.0-theta) * scalar_product(old_timestep_fluid_pressure, phi_i_grads_v) +
				       timestep * theta * scalar_product(stress_fluid, phi_i_grads_v) +
				       timestep * (1.0-theta) * scalar_product(old_timestep_stress_fluid, phi_i_grads_v) 			
				       ) *  scratch_data.fe_values.JxW(q);
		      
		    }		
		  else if (comp_i == 2 || comp_i == 3)
		    {	
		      const Tensor<2,dim> phi_i_grads_u = scratch_data.fe_values[displacements].gradient (i, q);
		      
		      // Nonlinear harmonic MMPDE
			  // change to linear elastic
		      copy_data.cell_rhs(i) -= ( scalar_product(sigma_mesh, phi_i_grads_u)
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
		  (cell->face(face)->boundary_id() == outlet_id) // boundary_indicator boundary_id
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
		 double lame_coefficient_mu, lame_coefficient_lambda, solid_coefficient_bulk;
		 if (cell->material_id() == solid_id[0]){
		 	lame_coefficient_mu = lame_mu[0];
			lame_coefficient_lambda = lame_lambda[0];
			solid_coefficient_bulk = solid_bulk[0];
		 }
		 if (cell->material_id() == solid_id[1]){
		 	lame_coefficient_mu = lame_mu[1];
			lame_coefficient_lambda = lame_lambda[1];
			solid_coefficient_bulk = solid_bulk[1];
		 }
		if (cell->material_id() == solid_id[2]){
		 	lame_coefficient_mu = lame_mu[2];
			lame_coefficient_lambda = lame_lambda[2];
			solid_coefficient_bulk = solid_bulk[2];
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

	      const Tensor<2,dim> C = Structure_Terms_in_ALE
		::get_C<dim> (F_T, F);

	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (E);

	      const double tr_C = Structure_Terms_in_ALE
		::get_tr_C<dim> (C);

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
	      const Tensor<2,dim> old_timestep_C = Structure_Terms_in_ALE
		::get_C<dim> (old_timestep_F_T, old_timestep_F);
	      const double old_timestep_tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (old_timestep_E);
		const double old_timestep_tr_C = Structure_Terms_in_ALE
		::get_tr_C<dim> (old_timestep_C);
	      
	      

	    //   // STVK structure model
	    //   Tensor<2,dim> sigma_structure_ALE; // cauthy stress
	    //   sigma_structure_ALE.clear();
	    //   sigma_structure_ALE = (1.0/J *
		// 		     F * (lame_coefficient_lambda * tr_E * Identity +
		// 			  2 * lame_coefficient_mu * E) * F_T);
	      
	      
	    //   Tensor<2,dim> stress_term; // 1st-PK
	    //   stress_term.clear();
	    //   stress_term = (J * sigma_structure_ALE * F_Inverse_T);
	      
	    //   Tensor<2,dim> old_timestep_sigma_structure_ALE;
	    //   old_timestep_sigma_structure_ALE.clear();
	    //   old_timestep_sigma_structure_ALE = (1.0/old_timestep_J *
		// 				  old_timestep_F * (lame_coefficient_lambda *
		// 						    old_timestep_tr_E * Identity +
		// 						    2 * lame_coefficient_mu *
		// 						    old_timestep_E) * 
		// 				  old_timestep_F_T);
	      
	    //   Tensor<2,dim> old_timestep_stress_term;
	    //   old_timestep_stress_term.clear();
	    //   old_timestep_stress_term = (old_timestep_J * old_timestep_sigma_structure_ALE * old_timestep_F_Inverse_T);
	      	
	    // neo-Hookean structure model
	      Tensor<2,dim> stress_term; // 1st-PK
	      stress_term.clear();
		  stress_term = get_1st_PK_NH(lame_coefficient_mu, J, tr_C, F, F_Inverse_T);

		  Tensor<2,dim> old_timestep_stress_term;
	      old_timestep_stress_term.clear();
		  old_timestep_stress_term = get_1st_PK_NH( lame_coefficient_mu, 
		  											old_timestep_J, 
		  											old_timestep_tr_C, 
		  											old_timestep_F, 
		  											old_timestep_F_Inverse_T);


	      // Attention: normally no time
	      Tensor<1,dim> structure_force;
	      structure_force.clear();
	      structure_force[0] = density_structure * force_structure_x;
	      structure_force[1] = density_structure * force_structure_y;
	      
	      Tensor<1,dim> old_timestep_structure_force;
	      old_timestep_structure_force.clear();
	      old_timestep_structure_force[0] = density_structure * force_structure_x;
	      old_timestep_structure_force[1] = density_structure * force_structure_y;
	    
          const double incompressiblity_solid = NSE_in_ALE
		::get_Incompressibility_ALE<dim> (q, old_solution_grads);

          const double old_timestep_incompressiblity_solid = NSE_in_ALE
		::get_Incompressibility_ALE<dim> (q, old_timestep_solution_grads);

		  const double solid_pressure_scalar = old_solution_values[q](dim+dim);
          const double old_timestep_solid_pressure_scalar = old_timestep_solution_values[q](dim+dim);

		  Tensor<2,dim> solid_pressure;
		  solid_pressure.clear();
		  solid_pressure = (-solid_pressure_scalar * Identity * J * F_Inverse_T);
		  Tensor<2,dim> old_timestep_solid_pressure;
		  old_timestep_solid_pressure.clear();
		  solid_pressure = (-old_timestep_solid_pressure * Identity * old_timestep_J * old_timestep_F_Inverse_T);

          const double betaTheta = get_betaTheta_ST91(solid_bulk, solid_pressure_scalar);
		  const double old_timestep_betaTheta = get_betaTheta_ST91(solid_bulk, old_timestep_solid_pressure_scalar);
          const double density_structure_new = get_rho_ST91(solid_bulk, density_structure, solid_pressure_scalar);
		  const double old_timestep_density_structure_new = get_rho_ST91(solid_bulk, density_structure, old_timestep_solid_pressure_scalar);

	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  // STVK structure model
		  // unified neo-Hookean
		  const unsigned int comp_i = fe.system_to_component_index(i).first; 
		  if (comp_i == 0 || comp_i == 1)
		    { 
		      const Tensor<1,dim> phi_i_v = scratch_data.fe_values[velocities].value (i, q);
		      const Tensor<2,dim> phi_i_grads_v = scratch_data.fe_values[velocities].gradient (i, q);
		      
		      copy_data.cell_rhs(i) -= ( 0.5*(J*density_structure_new+old_timestep_J*old_timestep_density_structure_new) * (v - old_timestep_v) * phi_i_v
				       + timestep * theta * scalar_product(stress_term,phi_i_grads_v)
				       + timestep * (1.0-theta) * scalar_product(old_timestep_stress_term, phi_i_grads_v)
					   + timestep * theta * scalar_product(solid_pressure, phi_i_grads_v)
					   + timestep * (1.0-theta) * scalar_product(old_timestep_solid_pressure, phi_i_grads_v)
				       - timestep * theta * structure_force * phi_i_v   
				       - timestep * (1.0 - theta) * old_timestep_structure_force * phi_i_v 
				       ) * scratch_data.fe_values.JxW(q);    
		      
		    }		
		  else if (comp_i == 2 || comp_i == 3)
		    {
		      const Tensor<1,dim> phi_i_u = scratch_data.fe_values[displacements].value (i, q);
		      copy_data.cell_rhs(i) -=  ( 0.5*(J*density_structure_new+old_timestep_J*old_timestep_density_structure_new) * (u - old_timestep_u)
					  - timestep * (theta *J * density_structure_new * v + 
					  (1.0-theta) * old_timestep_J * old_timestep_density_structure_new * old_timestep_v) 
					) * phi_i_u * scratch_data.fe_values.JxW(q);    
		      
		    }
		  else if (comp_i == 4)
		    {
		      const double phi_i_p = scratch_data.fe_values[pressure].value (i, q);
		      copy_data.cell_rhs(i) -= ( 0.5*(J*betaTheta+old_timestep_J*old_timestep_betaTheta)*(solid_pressure_scalar-old_timestep_solid_pressure_scalar) 
			  							+ theta*timestep*incompressiblity_solid
										+ (1-theta)*timestep*old_timestep_incompressiblity_solid
			  							) * phi_i_p * scratch_data.fe_values.JxW(q);  
		      
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
	 else if (cell->material_id() == solid_id[3])
	{	  
		 double lame_coefficient_mu, lame_coefficient_lambda;
		 
		 	lame_coefficient_mu = lame_mu[3];
			lame_coefficient_lambda = lame_lambda[3];

	  for (unsigned int q=0; q<n_q_points; ++q)
	    {		 		 	      
	      const Tensor<1,dim> v = ALE_Transformations
		::get_v<dim> (q, old_solution_values);
	      
	      const Tensor<1,dim> u = ALE_Transformations
		::get_u<dim> (q, old_solution_values);
	      
		  const Tensor<2,dim> pI = ALE_Transformations
		::get_pI<dim> (q, old_solution_values);

		  const Tensor<1,dim> grad_P = ALE_Transformations
		::get_grad_p<dim> (q, old_solution_grads);

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

	      const Tensor<2,dim> old_timestep_pI = ALE_Transformations
		::get_pI<dim> (q, old_timestep_solution_values);

	      const Tensor<2,dim> old_timestep_F = ALE_Transformations
		::get_F<dim> (q, old_timestep_solution_grads);
	      
	      const Tensor<2,dim> old_timestep_F_Inverse = ALE_Transformations
		::get_F_Inverse<dim> (old_timestep_F);
	      
	      const Tensor<2,dim> old_timestep_F_T = ALE_Transformations
		::get_F_T<dim> (old_timestep_F);
	      
	      const Tensor<2,dim> old_timestep_F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dim> (old_timestep_F_Inverse);

	      const Tensor<1,dim> old_timestep_grad_P = ALE_Transformations
		::get_grad_p<dim> (q, old_timestep_solution_grads);

	      const double old_timestep_J = ALE_Transformations
		::get_J<dim> (old_timestep_F);
	      
	      const Tensor<2,dim> old_timestep_E = Structure_Terms_in_ALE
		::get_E<dim> (old_timestep_F_T, old_timestep_F, Identity);
	      
	      const double old_timestep_tr_E = Structure_Terms_in_ALE
		::get_tr_E<dim> (old_timestep_E);
	      
	      
	      // STVK structure model
	      // change to linear elastic for poro
	      Tensor<2,dim> poro_pressure;
	      poro_pressure.clear();
	      poro_pressure = (-pI * J * F_Inverse_T);

		  Tensor<2,dim> old_timestep_poro_pressure;
	      old_timestep_poro_pressure.clear();
	      old_timestep_poro_pressure = (-old_timestep_pI * old_timestep_J * old_timestep_F_Inverse_T);


		//   Tensor<2,dim> sigma_structure_ALE;
	    //   sigma_structure_ALE.clear();
	    //   sigma_structure_ALE = (1.0/J *
		// 		     F * (lame_coefficient_lambda * tr_E * Identity +
		// 			  2 * lame_coefficient_mu * E) * F_T);
	      
	    //   Tensor<2,dim> stress_term;
	    //   stress_term.clear();
	    //   stress_term = (J * sigma_structure_ALE * F_Inverse_T);

	    //   Tensor<2,dim> old_timestep_sigma_structure_ALE;
	    //   old_timestep_sigma_structure_ALE.clear();
	    //   old_timestep_sigma_structure_ALE = (1.0/old_timestep_J *
		// 				  old_timestep_F * (lame_coefficient_lambda *
		// 						    old_timestep_tr_E * Identity +
		// 						    2 * lame_coefficient_mu *
		// 						    old_timestep_E) * 
		// 				  old_timestep_F_T);
	      
	    //   Tensor<2,dim> old_timestep_stress_term;
	    //   old_timestep_stress_term.clear();
	    //   old_timestep_stress_term = (old_timestep_J * old_timestep_sigma_structure_ALE * old_timestep_F_Inverse_T);

	      Tensor<2,dim> u_sym;
		  u_sym.clear();
		  u_sym = Structure_Terms_in_ALE::get_u_sym(q, old_solution_grads);
		  const double trU = Structure_Terms_in_ALE::get_trU(q, old_solution_grads);
		  Tensor<2,dim> stress_term;
	      stress_term.clear();
		  stress_term = F*(lame_coefficient_lambda*trU*Identity+2*lame_coefficient_mu*u_sym);

          Tensor<2,dim> old_timestep_u_sym;
		  old_timestep_u_sym.clear();
		  old_timestep_u_sym = Structure_Terms_in_ALE::get_u_sym(q, old_timestep_solution_grads);
		  const double old_timestep_trU = Structure_Terms_in_ALE::get_trU(q, old_timestep_solution_grads);
		  Tensor<2,dim> old_timestep_stress_term;
	      old_timestep_stress_term.clear();
		  old_timestep_stress_term = old_timestep_F*(lame_coefficient_lambda*old_timestep_trU*Identity+2*lame_coefficient_mu*old_timestep_u_sym);

		// Divergence of the poro in the ALE formulation
	      const double incompressiblity_poro = NSE_in_ALE
		::get_Incompressibility_ALE<dim> (q, old_solution_grads);

		// const double old_timestep_incompressiblity_poro = NSE_in_ALE
		// ::get_Incompressibility_ALE<dim> (q, old_timestep_solution_grads);
    
      
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  // STVK structure model
		  // change to linear elastic for poro
		  const unsigned int comp_i = fe.system_to_component_index(i).first; 
		  if (comp_i == 0 || comp_i == 1)
		    { 
		      const Tensor<1,dim> phi_i_v = scratch_data.fe_values[velocities].value (i, q);
		      const Tensor<2,dim> phi_i_grads_v = scratch_data.fe_values[velocities].gradient (i, q);
		      
		      copy_data.cell_rhs(i) -= (density_poro *0.5*(J+old_timestep_J) * (v - old_timestep_v) * phi_i_v +
					   timestep * theta * scalar_product(poro_pressure, phi_i_grads_v) +  
				       timestep * (1.0-theta) * scalar_product(old_timestep_poro_pressure, phi_i_grads_v) +
				       timestep * theta * scalar_product(stress_term, phi_i_grads_v) +  
				       timestep * (1.0-theta) * scalar_product(old_timestep_stress_term, phi_i_grads_v) 
				       ) * scratch_data.fe_values.JxW(q);    
		      
		    }		
		  else if (comp_i == 2 || comp_i == 3)
		    {
		      const Tensor<1,dim> phi_i_u = scratch_data.fe_values[displacements].value (i, q);
		      copy_data.cell_rhs(i) -=  (density_poro * 
										(0.5*(J+old_timestep_J)*(u - old_timestep_u) * phi_i_u -
					 					  timestep * (theta * J * v + 
										  			  (1.0-theta) * old_timestep_J * old_timestep_v) * phi_i_u)
					                    ) * scratch_data.fe_values.JxW(q);    
		      
		    }
		  else if (comp_i == 4)
		    {
		      const double phi_i_p = scratch_data.fe_values[pressure].value (i, q);
					const Tensor<1,dim> phi_i_grads_p = scratch_data.fe_values[pressure].gradient (i, q);
			copy_data.cell_rhs(i) -= ( incompressiblity_poro * phi_i_p + 
									kappa * J * grad_P * F_Inverse * F_Inverse_T * phi_i_grads_p
									) * scratch_data.fe_values.JxW(q);  
		    //   copy_data.cell_rhs(i) -= ( timestep * theta * incompressiblity_poro * phi_i_p + 
			// 							 timestep * theta * kappa * J * grad_P * F_Inverse * F_Inverse_T * phi_i_grads_p +
			// 							 timestep * (1-theta) * old_timestep_incompressiblity_poro * phi_i_p + 
			// 							 timestep * (1-theta) * kappa * old_timestep_J * old_timestep_grad_P * old_timestep_F_Inverse * old_timestep_F_Inverse_T * phi_i_grads_p
			// 							 ) * scratch_data.fe_values.JxW(q);  
		      
		    }
		  // end i	  
		} 	
	      // end n_q_points 		   
	    } 
	  
	//   cell->get_dof_indices (copy_data.local_dof_indices);
	//   constraints.distribute_local_to_global (local_rhs, local_dof_indices,
	// 					  system_rhs);
	  
	// end if (for STVK poro-material)  
	}
      
}


// inlet fow profile
template <int dim>
double 
FSI_ALE_Problem<dim>::inlet_flow (const double t)
{
//   const long double pi = 3.141592653589793238462643;
  const double T0 = 1.0;
  const double qmax = 250;
  const double a = 2.0/3.0;
  double qnew = 0;
  double temp, fi;
  if ( t <= T0 ) {
	if ( t <= a ){
	  fi=3*pi*t-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}else{
	  fi = 3*pi*a-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}
  }else{
	temp=t;
	while(temp>T0){ temp=temp-T0; }
	if(temp<=a){
	  fi = 3*pi*temp-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}else{
	  fi = 3*pi*a-1.4142;
	  qnew = qmax*(0.251+0.290*(std::cos(fi)+0.97*std::cos(2*fi)+0.47*std::cos(3*fi)+0.14*std::cos(4*fi)));
	}	
  }
  return 1e-6*qnew/(pi*std::pow(R,2));// u=Q/A, A=pi*r^2, Q:mL/s,
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
						dealii::Functions::ZeroFunction<dim>(dim+dim+1),  
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
						dealii::Functions::ZeroFunction<dim>(dim+dim+1),  
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
						dealii::Functions::ZeroFunction<dim>(dim+dim+1),  
						boundary_values,
						component_mask);
    
	// boundary value set to solution data
	// while \delta x is constraint by constraint class 
	// and set by set_newton_bc
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
						  dealii::Functions::ZeroFunction<dim>(dim+dim+1),                                             
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
					      dealii::Functions::ZeroFunction<dim>(dim+dim+1),  
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
						dealii::Functions::ZeroFunction<dim>(dim+dim+1),  
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
						dealii::Functions::ZeroFunction<dim>(dim+dim+1),  
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
  TimerOutput::Scope t(timer, "Solve linear system.");
  Vector<double> sol, rhs;    
  sol = newton_update;    
  rhs = system_rhs;
  
  A_direct.vmult(sol,rhs); 
  newton_update = sol;
  
  constraints.distribute (newton_update);
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
      assemble_system_matrix ();
	  // using mumps
	//   A_direct.initialize (system_matrix);
	  // using UMFPACK
  	  A_direct.factorize(system_matrix);
	  solve ();
	  solution += newton_update;
      assemble_system_rhs();
      newton_residuum = system_rhs.linfty_norm();

      if (newton_residuum < lower_bound_newton_residuum)
	  {
	    std::cout << '\t' 
			<< std::scientific 
			<< newton_residuum << std::endl;
	    break;
	  }
	
	  

    //   if (newton_residuum/old_newton_residuum > nonlinear_rho)
	// {
	//   assemble_system_matrix ();
	//   // using mumps
	// //   A_direct.initialize (system_matrix);
	//   // using UMFPACK
  	//   A_direct.factorize(system_matrix);
	// }

    //   // Solve Ax = b
    //   solve ();	  
        
    //   line_search_step = 0;	  
    //   for ( ; 
	//     line_search_step < max_no_line_search_steps; 
	//     ++line_search_step)
	// {	     					 
	//   solution += newton_update;
	  
	//   assemble_system_rhs ();			
	//   new_newton_residuum = system_rhs.linfty_norm();
	  
	//   if (new_newton_residuum < newton_residuum)
	//       break;
	//   else 	  
	//     solution -= newton_update;
	  
	//   newton_update *= line_search_damping;
	// }	   
     
      timer_newton.stop();
      
      std::cout << std::setprecision(5) <<newton_step << '\t' 
		<< std::scientific << newton_residuum << '\t'
		<< std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
      if (newton_residuum/old_newton_residuum > nonlinear_rho)
	std::cout << "r" << '\t' ;
      else 
	std::cout << " " << '\t' ;
      std::cout << line_search_step  << '\t' 
		<< std::scientific << timer_newton.cpu_time ()
		<< std::endl;


      // Updates
      timer_newton.reset();
      newton_step++;      
    }

	std::cout << "End newton procedure...\n" << std::endl;
}



// // origin newton_iteration
// template <int dim>
// void FSI_ALE_Problem<dim>::newton_iteration (const double time) 
					       
// { 
//   Timer timer_newton;
//   const double lower_bound_newton_residuum = 1.0e-8; 
//   const unsigned int max_no_newton_steps  = 60;

//   // Decision whether the system matrix should be build
//   // at each Newton step
//   const double nonlinear_rho = 0.1; 
 
//   // Line search parameters
//   unsigned int line_search_step;
//   const unsigned int  max_no_line_search_steps = 10;
//   const double line_search_damping = 0.6;
//   double new_newton_residuum;
  
//   // Application of the initial boundary conditions to the 
//   // variational equations:
//   std::cout << "Setting initial_bc...\n" << std::endl;
//   set_initial_bc (time);
//   std::cout << "Assembling rhs...\n" << std::endl;
//   assemble_system_rhs();

//   double newton_residuum = system_rhs.linfty_norm(); 
//   double old_newton_residuum= newton_residuum;
//   unsigned int newton_step = 1;
//   // prevent call A_direct.sovle() before initialized.
// //    assemble_system_matrix ();
// //    A_direct.initialize (system_matrix);

//   if (newton_residuum < lower_bound_newton_residuum)
//     {
//       std::cout << '\t' 
// 		<< std::scientific 
// 		<< newton_residuum 
// 		<< std::endl;     
//     }

//   std::cout << "Start newton procedure...\n" << std::endl;

//   while (newton_residuum > lower_bound_newton_residuum &&
// 	 newton_step < max_no_newton_steps)
//     {
//       timer_newton.start();
//       old_newton_residuum = newton_residuum;
      
//       assemble_system_rhs();
//       newton_residuum = system_rhs.linfty_norm();

//       if (newton_residuum < lower_bound_newton_residuum)
// 	{
// 	  std::cout << '\t' 
// 		    << std::scientific 
// 		    << newton_residuum << std::endl;
// 	  break;
// 	}
  
//       if (newton_residuum/old_newton_residuum > nonlinear_rho)
// 	{
// 	  assemble_system_matrix ();
// 	  // using mumps
// 	//   A_direct.initialize (system_matrix);
// 	  // using UMFPACK
//   	  A_direct.factorize(system_matrix);
// 	}

//       // Solve Ax = b
//       solve ();	  
        
//       line_search_step = 0;	  
//       for ( ; 
// 	    line_search_step < max_no_line_search_steps; 
// 	    ++line_search_step)
// 	{	     					 
// 	  solution += newton_update;
	  
// 	  assemble_system_rhs ();			
// 	  new_newton_residuum = system_rhs.linfty_norm();
	  
// 	  if (new_newton_residuum < newton_residuum)
// 	      break;
// 	  else 	  
// 	    solution -= newton_update;
	  
// 	  newton_update *= line_search_damping;
// 	}	   
     
//       timer_newton.stop();
      
//       std::cout << std::setprecision(5) <<newton_step << '\t' 
// 		<< std::scientific << newton_residuum << '\t'
// 		<< std::scientific << newton_residuum/old_newton_residuum  <<'\t' ;
//       if (newton_residuum/old_newton_residuum > nonlinear_rho)
// 	std::cout << "r" << '\t' ;
//       else 
// 	std::cout << " " << '\t' ;
//       std::cout << line_search_step  << '\t' 
// 		<< std::scientific << timer_newton.cpu_time ()
// 		<< std::endl;


//       // Updates
//       timer_newton.reset();
//       newton_step++;      
//     }
// 	std::cout << "End newton procedure...\n" << std::endl;
// }




// This function is known from almost all other 
// tutorial steps in deal.II.
template <int dim>
void
FSI_ALE_Problem<dim>::output_results (const unsigned int time_step,
			      const BlockVector<double> output_vector)  const
{
  // Postprocessor postprocessor;

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

  // data_out.add_data_vector (output_vector, postprocessor);
  // get cell stress
//   Vector<double> major_principle_stress(triangulation.n_active_cells());
//   Vector<double> minor_principle_stress(triangulation.n_active_cells());
//   Vector<double> von_mises_stress(triangulation.n_active_cells());
//   {
// 	for (auto &cell : triangulation.active_cell_iterators())
// 	if (cell->material_id() == solid_id[0] || cell->material_id() == solid_id[1] || cell->material_id() == solid_id[2])
// 	{	  
// 		 double lame_coefficient_mu, lame_coefficient_lambda;
// 		 if (cell->material_id() == solid_id[0]){
// 		 	lame_coefficient_mu = lame_mu[0];
// 			lame_coefficient_lambda = lame_lambda[0];
// 		 }
// 		 if (cell->material_id() == solid_id[1]){
// 		 	lame_coefficient_mu = lame_mu[1];
// 			lame_coefficient_lambda = lame_lambda[1];
// 		 }
// 		if (cell->material_id() == solid_id[2]){
// 		 	lame_coefficient_mu = lame_mu[2];
// 			lame_coefficient_lambda = lame_lambda[2];
// 		}  
// 			{
// 			SymmetricTensor<2, dim> accumulated_stress;
// 			for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
// 				accumulated_stress +=
// 				reinterpret_cast<PointHistory<dim> *>(cell->user_pointer())[q]
// 					.old_stress;
// 			norm_of_stress(cell->active_cell_index()) =
// 				(accumulated_stress / quadrature_formula.size()).norm();
// 			}
// 		else
// 			norm_of_stress(cell->active_cell_index()) = -1e+20;
// 	}
//   data_out.add_data_vector(major_principle_stress, "major_principle_stress");
//   data_out.add_data_vector(major_principle_stress, "major_principle_stress");
  data_out.build_patches ();

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
      newton_iteration (time);   
      time += timestep;
	
      // Compute functional values: dx, dy, drag, lift
      std::cout << std::endl;
      compute_functional_values();
      
      // Write solutions 
      if ((timestep_number % output_skip == 0))
	    output_results (timestep_number,solution);


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
      FSI_ALE_Problem<2> flow_problem(degree);
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




