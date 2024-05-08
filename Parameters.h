#include <deal.II/base/parameter_handler.h> 
using namespace dealii;
namespace Parameters
{
      struct GlobalValues
      {
        unsigned int degree;
	      unsigned int no_of_refinements;
        std::string mesh_file;
        static void declare_parameters(ParameterHandler &prm);
        void  parse_parameters(ParameterHandler &prm);
      };

      void GlobalValues::declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Global");
        {
          prm.declare_entry("degree", "2",
                        Patterns::Integer(0),
                        "degree" );   
	        prm.declare_entry("no_of_refinements", "1",
                        Patterns::Integer(0),
                        "no_of_refinements" ); 
          prm.declare_entry("mesh_file", "fsi.inp",
                        Patterns::Anything(),
                        "mesh file");     
        }
        prm.leave_subsection();
      }

      void GlobalValues::parse_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Global");
        {
          degree = prm.get_integer("degree");   
	        no_of_refinements = prm.get_integer("no_of_refinements");  
          mesh_file = prm.get("mesh_file");       
        }
        prm.leave_subsection();
      }
      
            
      
  struct BoundaryValue
    {
      double       inflow_velocity;     
      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void BoundaryValue::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("inflow_velocity_parameter");
      {
        prm.declare_entry("inflow_velocity", "0.3",
                        Patterns::Double(0),
                        "inlet velocity" );      
      }
      prm.leave_subsection();
    }

    void BoundaryValue::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("inflow_velocity_parameter");
      {
        inflow_velocity = prm.get_double("inflow_velocity");
        
      }
      prm.leave_subsection();
    }

    struct PhysicalConstants
    {
      double       density_fluid;  
      double       viscosity;
      double       density_structure;
      double       lame_coefficient_mu;
      double       poisson_ratio_nu;
      double       force_structure_x;
      double       force_structure_y;
      double       alpha_u;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void PhysicalConstants::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Physical constants");
      {
        prm.declare_entry("density_fluid", "1.0e+3",
                        Patterns::Double(0),
                        "density of fluid");
        prm.declare_entry("viscosity", "1.0e-3",
                        Patterns::Double(0),
                        "viscosity of fluid");						

        prm.declare_entry("density_structure", "1.0e+3",
                        Patterns::Double(0),
                        "density of structure");
        prm.declare_entry("lame_coefficient_mu", "0.5e+6",
                        Patterns::Double(0),
                        "mu");
        prm.declare_entry("poisson_ratio_nu", "0.4",
                        Patterns::Double(0),
                        "nu");

        prm.declare_entry("force_structure_x", "0.0",
                        Patterns::Double(0),
                        "fx");
        prm.declare_entry("force_structure_y", "0.0",
                        Patterns::Double(0),
                        "fy");

        prm.declare_entry("alpha_u", "1.0e-8",
                        Patterns::Double(0),
                        "alpha");                 
      }
      prm.leave_subsection();
    }

    void PhysicalConstants::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Physical constants");
      {
       density_fluid         = prm.get_double("density_fluid");
       viscosity             = prm.get_double("viscosity");
       density_structure     = prm.get_double("density_structure"); 
       lame_coefficient_mu   = prm.get_double("lame_coefficient_mu"); 
       poisson_ratio_nu      = prm.get_double("poisson_ratio_nu"); 
       force_structure_x     = prm.get_double("force_structure_x");
       force_structure_y     = prm.get_double("force_structure_y");
       alpha_u               = prm.get_double("alpha_u");
        
      }
      prm.leave_subsection();
    }

   struct Time
    {
      std::string  time_stepping_scheme;
      double       timestep; 
      int          max_no_timesteps; 

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Timestepping schemes & timestep & max_no_timesteps");
      {
        prm.declare_entry("time_stepping_scheme", "BE",
                        Patterns::Selection("BE|CN|CN_shifted"),
                        "stepping scheme");

        prm.declare_entry("timestep", "1.0",
                        Patterns::Double(0),
                        "each timestep");

	      prm.declare_entry("max_no_timesteps", "25",
                        Patterns::Integer(0),
                        "number of timesteps");		

      }
      prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Timestepping schemes & timestep & max_no_timesteps");
      {
       time_stepping_scheme =prm.get("time_stepping_scheme"); 
       timestep             =prm.get_double("timestep");
       max_no_timesteps     =prm.get_integer("max_no_timesteps"); 
      }
      prm.leave_subsection();
    }

    struct PhysicalGroup
    {
      unsigned int fluid_id;
      unsigned int solid_id;
    
      unsigned int inlet_id;
      unsigned int outlet_id;
      unsigned int fixed_id;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void PhysicalGroup::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("PhysicalGroup");
      {
        prm.declare_entry("fluid_id", "0",
                        Patterns::Integer(0),
                        "fluid id");

        prm.declare_entry("solid_id", "1",
                        Patterns::Integer(0),
                        "solid id");

	      prm.declare_entry("inlet_id", "0",
                        Patterns::Integer(0),
                        "inlet boundary id");		
        prm.declare_entry("outlet_id", "1",
                        Patterns::Integer(0),
                        "outlet boundary id");	
        prm.declare_entry("fixed_id", "80",
                        Patterns::Integer(0),
                        "fixed boundary id");	

      }
      prm.leave_subsection();
    }

    void PhysicalGroup::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("PhysicalGroup");
      {
       fluid_id  = prm.get_integer("fluid_id"); 
       solid_id  = prm.get_integer("solid_id");
       inlet_id  = prm.get_integer("inlet_id"); 
       outlet_id = prm.get_integer("outlet_id"); 
       fixed_id  = prm.get_integer("fixed_id"); 
      }
      prm.leave_subsection();
    }

    struct AllParameters 
    : public GlobalValues,
      public BoundaryValue,
      public PhysicalConstants, 
      public Time,
      public PhysicalGroup
    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      // prm.read_input(input_file); // 8.2.1 : read_input (<8.5.0); 9.3.0 : parse_input (>=9.0.0)
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      GlobalValues::declare_parameters(prm);
      BoundaryValue::declare_parameters(prm);
      PhysicalConstants::declare_parameters(prm); 
      Time::declare_parameters(prm);
      PhysicalGroup::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      GlobalValues::parse_parameters(prm);
      BoundaryValue::parse_parameters(prm);
      PhysicalConstants::parse_parameters(prm);
      Time::parse_parameters(prm);
      PhysicalGroup::parse_parameters(prm);
    }
}
