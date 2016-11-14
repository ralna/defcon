# Code to support geometric multigrid in defcon.
import backend
from petsc4py import PETSc

if backend.__name__ == "dolfin":
    from backend import as_backend_type, Function, MixedElement, VectorElement, FunctionSpace

# Set up multigrid support
def create_dm(V, problem=None):
    # firedrake does its own MG, we have nothing to do with it
    if backend.__name__ == "firedrake":
        return None

    comm = V.mesh().mpi_comm()
    coarse_meshes = problem.coarse_meshes(comm)
    coarse_fs = []
    for coarse_mesh in coarse_meshes:
        coarse_fs.append(FunctionSpace(coarse_mesh, V.ufl_element()))

    all_meshes = coarse_meshes + [V.mesh()]
    all_fs     = coarse_fs + [V]
    all_dms    = [create_fs_dm(W, problem) for W in all_fs]

    def fetcher(dm_, comm, j=None):
        return all_dms[j]

    # Now make DM i+1 out of DM i via refinement;
    # this builds PETSc's linked list of refinements and coarsenings
    for i in range(len(all_meshes)-1):
        dm = all_dms[i]
        dm.setRefine(fetcher, kargs=dict(j=i+1))

        rdm = dm.refine()
        all_dms[i+1] = rdm

    for i in range(len(all_meshes)-1, 0, -1):
        dm = all_dms[i]
        dm.setCoarsen(fetcher, kargs=dict(j=i-1))

    return all_dms[-1]

# This code is needed to set up shell DM's that hold the index
# sets and allow nice field-splitting to happen.
def create_fs_dm(V, problem=None):
    comm = V.mesh().mpi_comm()

    # this way the DM knows the function space it comes from
    dm = PETSc.DMShell().create(comm=comm)
    dm.setAttr('__fs__', V)
    dm.setAttr('__problem__', problem)

    # this gives the DM a template to create vectors inside snes
    dm.setGlobalVector(as_backend_type(Function(V).vector()).vec())

    # this tells the DM how to interpolate from mesh to mesh
    dm.setCreateInterpolation(create_interpolation)

    # if we have a mixed function space, then we need to tell PETSc
    # how to divvy up the different parts of the function space.
    # This is not needed for non-mixed elements.
    ufl_el = V.ufl_element()
    if isinstance(ufl_el, (MixedElement, VectorElement)):
        dm.setCreateSubDM(create_subdm)
        dm.setCreateFieldDecomposition(create_field_decomp)

    return dm

# This provides PETSc the information needed to decompose
# the field -- the set of names (currently blank, allowing petsc
# to simply enumerate them), the tuple of index sets, and the
# dms for the resulting subspaces.
def create_field_decomp(dm, *args, **kwargs):
    W = dm.getAttr('__fs__')
    problem = dm.getAttr('__problem__')
    Wsubs = [Wsub.collapse() for Wsub in W.split()]
    names = [None for Wsub in Wsubs]
    dms = [create_dm(Wsub, problem) for Wsub in Wsubs]
    return (names, funcspace_to_index_sets(W), dms)

# For a non-mixed function space, this converts the array of dofs
# into a PETSc IS.
# For a mixed (but not vector) function space, it returns a tuple
# of the PETSc IS'es for each field.
def funcspace_to_index_sets(fs):
    uflel = fs.ufl_element()
    comm = fs.mesh().mpi_comm()
    if isinstance(uflel, (MixedElement, VectorElement)):
        splitdofs = [V.dofmap().dofs() for V in fs.split()]
        ises = [PETSc.IS().createGeneral(sd, comm=comm)
                for sd in splitdofs]
        return tuple(ises)
    else:
        return (PETSc.IS().createGeneral(fs.dofmap().dofs(), comm=comm),)

# since field splitting occurs by having DM shells indicate
# which dofs belong to which field, we need to create DMs for
# the relevant subspaces in order to have recursive field splitting.
def create_subdm(dm, fields, *args, **kwargs):
    W = dm.getAttr('__fs__')
    problem = dm.getAttr('__problem__')
    comm = W.mesh().mpi_comm()
    if len(fields) == 1:
        f = int(fields[0])
        subel = W.sub(f).ufl_element()
        subspace = FunctionSpace(W.mesh(), subel)
        subdm = create_dm(subspace, problem)
        iset = PETSc.IS().createGeneral(W.sub(f).dofmap().dofs(), comm)
        return iset, subdm
    else:
        subel = MixedElement([W.sub(int(f)).ufl_element() for f in fields])
        subspace = FunctionSpace(W.mesh(), subel)
        subdm = create_dm(subspace, problem)

        alldofs = numpy.concatenate(
            [W.sub(int(f)).dofmap().dofs() for f in fields])
        iset = PETSc.IS().createGeneral(sorted(alldofs), comm=comm)

    return (iset, subdm)

def create_interpolation(dmc, dmf):
    """
    Create interpolation matrix interpolating from dmc -> dmf.

    Most of the heavy lifting is done in C++. The C++ code was written
    by Matteo Croci.
    """
    Vc = dmc.getAttr('__fs__') # coarse function space
    Vf = dmf.getAttr('__fs__') # fine function space

    pmat = create_transfer_matrix(Vc, Vf)
    return (pmat.mat(), None)

if backend.__name__ == "dolfin":
    from backend import compile_extension_module
    create_transfer_matrix_code = r'''
    #include <dolfin/geometry/BoundingBoxTree.h>
    #include <dolfin/fem/FiniteElement.h>
    #include <dolfin/fem/GenericDofMap.h>
    #include <dolfin/common/RangedIndexSet.h>
    #include <petscmat.h>

    namespace dolfin
    {
        // Coordinate comparison operator
        struct lt_coordinate
        {
          lt_coordinate(double tolerance) : TOL(tolerance) {}

          bool operator() (const std::vector<double>& x,
                           const std::vector<double>& y) const
          {
            const std::size_t n = std::max(x.size(), y.size());
            for (std::size_t i = 0; i < n; ++i)
            {
              double xx = 0.0;
              double yy = 0.0;
              if (i < x.size())
                xx = x[i];
              if (i < y.size())
                yy = y[i];

              if (xx < (yy - TOL))
                return true;
              else if (xx > (yy + TOL))
                return false;
            }
            return false;
          }

          // Tolerance
          const double TOL;
        };

        std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
        tabulate_coordinates_to_dofs(const FunctionSpace& V)
        {
          std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
            coords_to_dofs(lt_coordinate(1.0e-12));

          // Extract mesh, dofmap and element
          dolfin_assert(V.dofmap());
          dolfin_assert(V.element());
          dolfin_assert(V.mesh());
          const GenericDofMap& dofmap = *V.dofmap();
          const FiniteElement& element = *V.element();
          const Mesh& mesh = *V.mesh();

          // Geometric dimension
          const std::size_t gdim = mesh.geometry().dim();

          // Loop over cells and tabulate dofs
          boost::multi_array<double, 2> coordinates;
          std::vector<double> coordinate_dofs;
          std::vector<double> coors(gdim);

          // Speed up the computations by only visiting (most) dofs once
          const std::size_t local_size = dofmap.ownership_range().second
            - dofmap.ownership_range().first;
          RangedIndexSet already_visited(std::make_pair(0, local_size));

          for (CellIterator cell(mesh); !cell.end(); ++cell)
          {
            // Update UFC cell
            cell->get_coordinate_dofs(coordinate_dofs);

            // Get local-to-global map
            const ArrayView<const dolfin::la_index> dofs
              = dofmap.cell_dofs(cell->index());

            // Tabulate dof coordinates on cell
            element.tabulate_dof_coordinates(coordinates, coordinate_dofs, *cell);

            // Map dofs into coords_to_dofs
            for (std::size_t i = 0; i < dofs.size(); ++i)
            {
              const std::size_t dof = dofs[i];
              if (dof < local_size)
              {
                // Skip already checked dofs
                if (!already_visited.insert(dof))
                  continue;

                // Put coordinates in coors
                std::copy(coordinates[i].begin(), coordinates[i].end(), coors.begin());

                // Add dof to list at this coord
                const auto ins = coords_to_dofs.insert
                  (std::make_pair(coors, std::vector<std::size_t>{dof}));
                if (!ins.second)
                  ins.first->second.push_back(dof);
              }
            }
          }
          return coords_to_dofs;
        }

        std::shared_ptr<PETScMatrix> create_transfer_matrix(std::shared_ptr<const FunctionSpace> coarse_space, std::shared_ptr<const FunctionSpace> fine_space)
        {
        // Initialise PETSc Mat and error code
        PetscErrorCode ierr;
        Mat I;

        // Get coarse mesh and dimension of the domain
        const Mesh meshc = *coarse_space->mesh();
        std::size_t dim = meshc.geometry().dim();

        // MPI commpunicator, size and rank
        const MPI_Comm mpi_comm = meshc.mpi_comm();
        const unsigned int mpi_size = MPI::size(mpi_comm);
        const unsigned int mpi_rank = MPI::rank(mpi_comm); // mpi_rank is the rank of the current processor

        // Create and initialise the transfer matrix as MATMPIAIJ/MATSEQAIJ
        ierr = MatCreate(mpi_comm, &I); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        if (mpi_size > 1)
        {
            ierr = MatSetType(I, MATMPIAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else
        {
            ierr = MatSetType(I, MATSEQAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }

        // initialise bounding box tree and dofmaps
        std::shared_ptr<BoundingBoxTree> treec = meshc.bounding_box_tree();
        std::shared_ptr<const GenericDofMap> coarsemap = coarse_space->dofmap();
        std::shared_ptr<const GenericDofMap> finemap = fine_space->dofmap();

        // initialise local to global dof maps (these will be needed to allocate
        // the entries of the transfer matrix with the correct global indices)
        std::vector<std::size_t> coarse_local_to_global_dofs;
        coarsemap->tabulate_local_to_global_dofs(coarse_local_to_global_dofs);

        // Create map from coordinates to dofs sharing that coordinate
        std::map<std::vector<double>, std::vector<std::size_t>, lt_coordinate>
        coords_to_dofs = tabulate_coordinates_to_dofs(*fine_space);

        // Global dimensions of the dofs and of the transfer matrix (M-by-N, where
        // M is the fine space dimension, N is the coarse space dimension)
        std::size_t M = fine_space->dim();
        std::size_t N = coarse_space->dim();

        // Local dimension of the dofs and of the transfer matrix
        // we also keep track of the ownership range
        std::size_t mbegin = finemap->ownership_range().first;
        std::size_t mend = finemap->ownership_range().second;
        std::size_t m = mend - mbegin;

        std::size_t nbegin = coarsemap->ownership_range().first;
        std::size_t nend = coarsemap->ownership_range().second;
        std::size_t n = nend - nbegin;

        // we store the ownership range of the fine dofs so that
        // we can communicate it to the other workers.
        // This will be useful to check which dofs are owned by which processor
        std::vector<std::size_t> global_n_range(2,0);
        global_n_range[0] = nbegin;
        global_n_range[1] = nend;

        // Set transfer matrix sizes
        ierr = MatSetSizes(I, m, n, M, N); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Get finite element for the coarse space. This will be needed to evaluate
        // the basis functions for each cell.
        std::shared_ptr<const FiniteElement> el = coarse_space->element();

        // Check that it is the same kind of element on each space.
        {
            std::shared_ptr<const FiniteElement> elf = fine_space->element();
            // Check that function ranks match
            if (el->value_rank() != elf->value_rank())
            {
              dolfin_error("create_transfer_matrix",
                     "Creating interpolation matrix",
                         "Ranks of function spaces do not match: %d, %d.",
                     el->value_rank(), elf->value_rank());
            }

            // Check that function dims match
            for (std::size_t i = 0; i < el->value_rank(); ++i)
            {
              if (el->value_dimension(i) != elf->value_dimension(i))
              {
                dolfin_error("create_transfer_matrix",
                       "Creating interpolation matrix",
                       "Dimension %d of function space (%d) does not match dimension %d of function space (%d)",
                       i, el->value_dimension(i), i, elf->value_dimension(i));
              }
            }
        }
        // number of dofs per cell for the finite element.
        std::size_t eldim = el->space_dimension();
        // Number of dofs associated with each fine point
        int data_size = 1;
        for (unsigned data_dim = 0; data_dim < el->value_rank(); data_dim++)
            data_size *= el->value_dimension(data_dim);
        // Number of points in a fine cell
        int num_points = eldim / data_size;

        // Miscellaneous initialisations, these will all be needed later
        Point curr_point; // point variable
        Cell curr_cell; // cell variable
        std::vector<double> coordinate_dofs; // cell dofs coordinates vector
        ufc::cell ufc_cell; // ufc cell
        unsigned int id = 0; // cell id

        // The overall idea is: a fine point can be on a coarse cell in the current processor,
        // on a coarse cell in a different processor, or outside the coarse domain.
        // If the point is found on the processor, evaluate basis functions,
        // if found elsewhere, use the other processor to evaluate basis functions,
        // if not found at all, or if found in multiple processors,
        // use compute_closest_entity on all processors and find
        // which coarse cell is the closest entity to the fine point amongst all processors.

        std::vector<double> _x(dim); // vector with point coordinates
        // vector containing the ranks of the processors which might contain a fine point
        std::vector<unsigned int> found_ranks;

        // the next vectors we are defining here contain information relative to the
        // fine points for which a corresponding coarse cell owned by the current
        // processor was found.
        // found_ids[i] contains the coarse cell id relative to each fine point
        std::vector<std::size_t> found_ids;
        found_ids.reserve((std::size_t)M/mpi_size);
        // found_points[dim*i:dim*i + dim] contains the coordinates of the fine point i
        std::vector<double> found_points;
        found_points.reserve((std::size_t)dim*M/mpi_size);
        // global_row_indices[i] contains the global row indices of the fine point i
        // global_row_indices[data_size*i:data_size*i + data_size] are the rows associated with
        // this point
        std::vector<int> global_row_indices;
        global_row_indices.reserve((std::size_t) data_size*M/mpi_size);
        // found_points_senders[i] holds the rank of the process that owns the fine point
        // this is not strictly needed, but it is useful
        std::vector<std::size_t> found_points_senders;
        found_points_senders.reserve((std::size_t)M/mpi_size);

        // similar stuff for the not found points
        std::vector<unsigned int> not_found_points_senders;
        not_found_points_senders.reserve((std::size_t)M/mpi_size);
        std::vector<double> not_found_points;
        std::vector<int> not_found_global_row_indices;

        // same for the found elsewhere points
        std::vector<double> found_elsewhere;
        std::vector<int> found_elsewhere_global_row_indices;
        // which_processor contains the rank of the processor
        // that owns the coarse cell where the fine point was found
        std::vector<unsigned int> which_processor;

        // Loop over fine points owned by the current processor,
        // and find out who owns the coarse cell where the fine point lies.
        for (const auto &map_it : coords_to_dofs)
        {
             // Copy coordinates into buffer.
             std::copy(map_it.first.begin(), map_it.first.end(), _x.begin());

             // get the fine point into a Point variable
             if (dim == 3)
                 curr_point = Point(_x[0], _x[1], _x[2]);
             else if (dim == 2)
                 curr_point = Point(_x[0], _x[1]);
             else
                 curr_point = Point(_x[0]);

             // compute which processors share ownership of the coarse cell
             // that contains the fine point
             found_ranks = treec->compute_process_collisions(curr_point);

             // if the fine point is not in the domain or if more than one
             // processors share it, mark it as not found
             // (not found points will be searched by all the processors and
             // the processor that owns closest coarse cell to that point will be found,
             // so that even if multiple processes share the cell, we find the one that
             // actually owns it)
             if (found_ranks.empty() || found_ranks.size() > 1)
             {
                 // we store fine point coordinates, global row indices and the senders
                 // this information will be sent to all the processors
                 not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
                 not_found_global_row_indices.insert(not_found_global_row_indices.end(), map_it.second.begin(), map_it.second.end());
                 not_found_points_senders.push_back(mpi_rank);
             }
             // if the fine point collides with a coarse cell owned by the current processor,
             // find the coarse cell the fine point lives in
             else if (found_ranks[0] == mpi_rank)
             {
                 // find the coarse cell where the fine point lies
                 id = treec->compute_first_entity_collision(curr_point);

                 // Safety control: if no cell is found on the current processor
                 // mark the point as not_found
                 if (id == std::numeric_limits<unsigned int>::max())
                 {
                     not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
                     not_found_global_row_indices.insert(not_found_global_row_indices.end(), map_it.second.begin(), map_it.second.end());
                     not_found_points_senders.push_back(mpi_rank);
                 }
                 else
                 {
                     // if a cell is found on the current processor, add the point
                     // and relative information to the various vectors
                     found_ids.push_back(id);
                     found_points.insert(found_points.end(), _x.begin(), _x.end());
                     global_row_indices.insert(global_row_indices.end(), map_it.second.begin(), map_it.second.end());
                     found_points_senders.push_back(mpi_rank);
                 }
             }
             // if found elsewhere, store the process where it was found
             else
             {
                 found_elsewhere.insert(found_elsewhere.end(),_x.begin(), _x.end());
                 which_processor.push_back(found_ranks[0]);
                 found_elsewhere_global_row_indices.insert(found_elsewhere_global_row_indices.end(), map_it.second.begin(), map_it.second.end());
             }
        } // end for loop

        // We now need to communicate various information to all the processors:
        // processor column and row ownership
        std::vector<std::vector<std::size_t>> global_n_range_recv(mpi_size, std::vector<std::size_t>(2));
        std::vector<std::size_t> global_m(mpi_size);
        std::vector<std::size_t> global_row_offset(mpi_size);
        MPI::all_gather(mpi_comm, global_n_range, global_n_range_recv);
        MPI::all_gather(mpi_comm, m, global_m);
        MPI::all_gather(mpi_comm, mbegin, global_row_offset);

        // Ok, now we need to handle the points which have been found elsewhere
        // We need to communicate these points to the other workers
        // as well as the relative information
        std::vector<std::vector<double>> found_elsewhere_recv(mpi_size);
        std::vector<std::vector<int>> found_elsewhere_global_row_indices_recv(mpi_size);
        std::vector<std::vector<unsigned int>> which_processor_recv(mpi_size);
        MPI::all_gather(mpi_comm, found_elsewhere, found_elsewhere_recv);
        MPI::all_gather(mpi_comm, found_elsewhere_global_row_indices, found_elsewhere_global_row_indices_recv);
        MPI::all_gather(mpi_comm, which_processor, which_processor_recv);

        // First, handle the points that were found elsewhere
        unsigned int how_many = 0;
        unsigned int receiver = mpi_rank;

        // we loop over the processors that own the fine points that need to be found
        // we call them senders here.
        for (unsigned sender=0;sender<mpi_size;sender++)
        {
            // We already searched on the current processor
            if (sender == receiver)
                continue;

            // how many fine points do we need to check?
            how_many = found_elsewhere_recv[sender].size()/dim;
            if (how_many == 0)
                continue;

            // for each fine point, create a Point variable and try to find the
            // coarse cell it lives in. If we cannot, mark the fine point as not found
            // for robustness.
            for (unsigned i=0; i<how_many;i++)
            {
                if (receiver == which_processor_recv[sender][i])
                {
                    if (dim == 3)
                    {
                        _x[0] = found_elsewhere_recv[sender][i*dim];
                        _x[1] = found_elsewhere_recv[sender][i*dim + 1];
                        _x[2] = found_elsewhere_recv[sender][i*dim + 2];
                        curr_point = Point(_x[0], _x[1], _x[2]);
                    }
                    else if (dim == 2)
                    {
                        _x[0] = found_elsewhere_recv[sender][i*dim];
                        _x[1] = found_elsewhere_recv[sender][i*dim + 1];
                        curr_point = Point(_x[0], _x[1]);
                    }
                    else
                    {
                        _x[0] = found_elsewhere_recv[sender][i*dim];
                        curr_point = Point(_x[0]);
                    }

                    id = treec->compute_first_entity_collision(curr_point);
                    // if the point is not found on the current processor
                    // mark it as not found and leave it for later
                    if (id == std::numeric_limits<unsigned int>::max())
                    {
                        not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
                        not_found_global_row_indices.insert(not_found_global_row_indices.end(), &found_elsewhere_global_row_indices_recv[sender][data_size*i], &found_elsewhere_global_row_indices_recv[sender][data_size*i + data_size]);
                        not_found_points_senders.push_back(sender);
                    }
                    else
                    {
                        // if found, store information
                        found_ids.push_back(id);
                        found_points.insert(found_points.end(), _x.begin(), _x.end());
                        global_row_indices.insert(global_row_indices.end(), &found_elsewhere_global_row_indices_recv[sender][data_size*i], &found_elsewhere_global_row_indices_recv[sender][data_size*i + data_size]);
                        found_points_senders.push_back(sender);
                    }
                }
            }
        }

        // communicate the not found list to all the processors
        std::vector<std::vector<double>> not_found_points_recv(mpi_size);
        std::vector<std::vector<int>> not_found_global_row_indices_recv(mpi_size);
        std::vector<std::vector<unsigned int>> not_found_points_senders_recv(mpi_size);
        MPI::all_gather(mpi_comm, not_found_points, not_found_points_recv);
        MPI::all_gather(mpi_comm, not_found_global_row_indices, not_found_global_row_indices_recv);
        MPI::all_gather(mpi_comm, not_found_points_senders, not_found_points_senders_recv);

        // handle not_found points:
        // we need to compute their distances from the closest owned coarse cell
        // and the index/id of that cell.
        std::vector<double> not_found_distances;
        std::vector<unsigned int> not_found_cell_indices;
        // we also need to store the fine point coordinates
        // in case the current processor owns the closest cell
        std::vector<double> found_not_found_points;
        // We need to flatten some vectors for further use
        std::vector<int> not_found_global_row_indices_flattened;
        std::vector<unsigned int> not_found_points_senders_flattened;

        // we loop over all the processors where a fine point was found
        // note that from now on, every processor is doing the same check:
        // compute id and distance of the closest owned coarse cell to the
        // fine point, then send the distances to all the processors, so that
        // each processor can determine which processor owns the closest coarse cell
        for (unsigned int proc=0; proc<mpi_size; proc++)
        {
            how_many = not_found_points_recv[proc].size()/dim;

            if (how_many == 0)
                continue;

            // flattening not_found_global_row_indices_recv one step at a time.
            not_found_global_row_indices_flattened.insert(not_found_global_row_indices_flattened.end(), not_found_global_row_indices_recv[proc].begin(), not_found_global_row_indices_recv[proc].end());
            // updating the std::vector of who owns the fine points
            not_found_points_senders_flattened.insert(not_found_points_senders_flattened.end(), not_found_points_senders_recv[proc].begin(), not_found_points_senders_recv[proc].end());

            // reserve memory for speed
            not_found_cell_indices.reserve(not_found_cell_indices.size() + how_many);
            found_not_found_points.reserve(not_found_points.size() + dim*how_many);
            not_found_distances.reserve(not_found_distances.size() + how_many);

            // same trick as before, store the fine point coordinates into a Point
            // variable, then run compute_closest_entity to find the closest owned
            // cell id and distance from the fine point
            for (unsigned i=0; i<how_many;i++)
            {
                if (dim == 3)
                {
                    _x[0] = not_found_points_recv[proc][i*dim];
                    _x[1] = not_found_points_recv[proc][i*dim + 1];
                    _x[2] = not_found_points_recv[proc][i*dim + 2];
                    curr_point = Point(_x[0], _x[1], _x[2]);
                }
                else if (dim == 2)
                {
                    _x[0] = not_found_points_recv[proc][i*dim];
                    _x[1] = not_found_points_recv[proc][i*dim + 1];
                    curr_point = Point(_x[0], _x[1]);
                }
                else
                {
                    _x[0] = not_found_points_recv[proc][i*dim];
                    curr_point = Point(_x[0]);
                }

                std::pair<unsigned int, double> find_point = treec->compute_closest_entity(curr_point);
                not_found_cell_indices.push_back(find_point.first);
                not_found_distances.push_back(find_point.second);
                // store the (now) found, (previously) not found fine point coordinates in a vector
                found_not_found_points.insert(found_not_found_points.end(), _x.begin(), _x.end());
            }
        }

        // communicate all distances to all processor so that each one can tell
        // which processor owns the closest coarse cell to the not found point
        std::vector<std::vector<double>> not_found_distances_recv(mpi_size);
        MPI::all_gather(mpi_comm, not_found_distances, not_found_distances_recv);

        // now need to find which processor has a cell which is closest to the not_found points

        // initialise some variables
        double min_val; // minimum distance
        unsigned min_proc=0; // processor that owns the minimum distance cell
        unsigned int sender; // processor that asked to search for the not found fine point

        how_many = not_found_cell_indices.size();
        for (unsigned i=0; i<how_many; i++)
        {
            // loop over the distances and find the processor who has
            // the point closest to one of its cells
            min_proc = 0;
            min_val = not_found_distances_recv[min_proc][i];
            for (unsigned proc_it = 1; proc_it<mpi_size; proc_it++)
            {
                if (not_found_distances_recv[proc_it][i] < min_val)
                {
                    min_val = not_found_distances_recv[proc_it][i];
                    min_proc = proc_it;
                }
            }

            // if the current processor is the one which owns the closest cell,
            // add the fine point and closest coarse cell information to the
            // vectors of found points
            if (min_proc == mpi_rank)
            {
                // allocate cell id to current worker if distance is minimum
                id = not_found_cell_indices[i];
                found_ids.push_back(id);
                global_row_indices.insert(global_row_indices.end(), &not_found_global_row_indices_flattened[data_size*i], &not_found_global_row_indices_flattened[data_size*i + data_size]);
                found_points.insert(found_points.end(), found_not_found_points.begin() + dim*i, found_not_found_points.begin() + dim*(i+1));
                sender = not_found_points_senders_flattened[i];
                found_points_senders.push_back(sender);
            }
        }

        // Now every processor should have the information needed to assemble its portion of the matrix.
        // The ids of coarse cell owned by each processor are currently stored in found_ids
        // and their respective global row indices are stored in global_row_indices.
        // The processors that own the matrix rows relative to the fine point are stored in found_points_senders.
        // One last loop and we are ready to go!

        // m_owned is the number of rows the current processor needs to set
        // note that the processor might not own these rows
        std::size_t m_owned = found_ids.size()*data_size;

        // initialise row and column indices and values of the transfer matrix
        int row_indices = 0;
        int** col_indices = new int*[m_owned];
        int*  fine_indices = new int[m_owned];
        memset(fine_indices, 0, m_owned*sizeof(int));
        double** values = new double*[m_owned];
        for(unsigned i = 0; i < m_owned; ++i)
        {
            col_indices[i] = new int[eldim];
            values[i] = new double[eldim];
        }
        // initialise a single chunk of values (needed for later)
        double temp_values[eldim*data_size];

        // initialise column ownership range
        std::size_t n_own_begin;
        std::size_t n_own_end;

        // initialise global sparsity pattern
        std::vector<int> global_d_nnz(M,0);
        std::vector<int> global_o_nnz(M,0);

        // loop over the found coarse cells
        for (unsigned i=0; i<found_ids.size(); i++)
        {
            // get coarse cell id
            id = found_ids[i];

            // save fine point coordinates into a Point variable
            if (dim == 3)
            {
                _x[0] = found_points[i*dim];
                _x[1] = found_points[i*dim + 1];
                _x[2] = found_points[i*dim + 2];
                curr_point = Point(_x[0], _x[1], _x[2]);
            }
            else if (dim == 2)
            {
                _x[0] = found_points[i*dim];
                _x[1] = found_points[i*dim + 1];
                curr_point = Point(_x[0], _x[1]);
            }
            else
            {
                _x[0] = found_points[i*dim];
                curr_point = Point(_x[0]);
            }

            // create coarse cell
            curr_cell = Cell(meshc, static_cast<std::size_t>(id));
            // get dofs coordinates of the coarse cell
            curr_cell.get_coordinate_dofs(coordinate_dofs);
            // save cell information into the ufc cell
            curr_cell.get_cell_data(ufc_cell);
            // evaluate the basis functions of the coarse cells
            // at the fine point and store the values into temp_values
            el->evaluate_basis_all(temp_values,
                                   curr_point.coordinates(),
                                   coordinate_dofs.data(),
                                   ufc_cell.orientation);

            // Loop over dofs of the coarse cell
            ArrayView<const dolfin::la_index> temp_dofs = coarsemap->cell_dofs(id);
            for (unsigned j=0; j < eldim; j++)
            {
                // Loop over the fine dofs this coarse dof contributes to
                for (unsigned k=0; k < eldim; k++)
                {
                    // Get the fine dof <-> coarse_dof
                    int coarse_dof = coarse_local_to_global_dofs[temp_dofs[j]];
                    int fine_dof   = global_row_indices[data_size*i + k];

                    // Get the index into the arrays
                    int fine_index = fine_indices[fine_dof];

                    // Set the column
                    col_indices[fine_dof][fine_index] = coarse_local_to_global_dofs[temp_dofs[j]];
                    // Set the value
                    values[fine_dof][fine_index] = temp_values[data_size*j + k];

                    // Increment the fine index for the next time we find this fine dof.
                    fine_indices[fine_dof]++;

                    // once we have the global column indices,
                    // determine sparsity pattern:
                    // which columns are owned by the process that
                    // owns the fine point?

                    // get the fine point owner processor
                    sender = found_points_senders[i];
                    // get its column ownership range
                    n_own_begin = global_n_range_recv[sender][0];
                    n_own_end = global_n_range_recv[sender][1];
                    // check and allocate sparsity pattern
                    if ((n_own_begin <= col_indices[fine_dof][fine_index]) && (col_indices[fine_dof][fine_index] < n_own_end))
                        global_d_nnz[fine_dof] += 1;
                    else
                        global_o_nnz[fine_dof] += 1;
                }
            } // end loop over coarse dofs
        } // end loop over found points
        delete [] fine_indices;

        // need to send the d_nnz and o_nnz to the correct processor
        // at the moment can only do global communication
        std::vector<std::vector<int>> global_d_nnz_recv(mpi_size);
        std::vector<std::vector<int>> global_o_nnz_recv(mpi_size);
        MPI::all_gather(mpi_comm, global_d_nnz, global_d_nnz_recv);
        MPI::all_gather(mpi_comm, global_o_nnz, global_o_nnz_recv);

        // initialise local sparsity pattern to 0
        // We use new here rather than std::vector because we need to pass this to C.
        int* d_nnz = new int[m];
        int* o_nnz = new int[m];
        memset(d_nnz, 0, m*sizeof(int));
        memset(o_nnz, 0, m*sizeof(int));

        int index = 0;
        for (unsigned i=0; i<M; i++)
        {
            // local row index
            index = i - mbegin;
            // if within local row range, sum the global sparsity pattern
            // into the local sparsity pattern
            if ((index >= 0) && (index < m))
            {
                for (unsigned proc = 0; proc<mpi_size; proc++)
                {
                    d_nnz[index] = d_nnz[index] + global_d_nnz_recv[proc][i];
                    o_nnz[index] = o_nnz[index] + global_o_nnz_recv[proc][i];
                }
            }
        }

        if (mpi_size > 1)
        {
            ierr = MatMPIAIJSetPreallocation(I, PETSC_DEFAULT, d_nnz, PETSC_DEFAULT, o_nnz); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else
        {
            ierr = MatSeqAIJSetPreallocation(I, PETSC_DEFAULT, d_nnz); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }

        delete [] d_nnz;
        delete [] o_nnz;

        // Setting transfer matrix values row by row
        for (unsigned i=0; i < m_owned;i++)
        {
            row_indices = global_row_indices[i];
            ierr = MatSetValues(I, 1, &row_indices, eldim, col_indices[i], values[i], INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);

            delete [] col_indices[i];
            delete [] values[i];
        }

        delete [] col_indices;
        delete [] values;

        // Assemble the transfer matrix
        ierr = MatAssemblyBegin(I, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAssemblyEnd(I, MAT_FINAL_ASSEMBLY); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // create shared pointer and return the pointer to the transfer matrix
        std::shared_ptr<PETScMatrix> ptr = std::make_shared<PETScMatrix>(I);
        return ptr;
        }
    }'''

    # compile C++ code
    create_transfer_matrix =  compile_extension_module(code=create_transfer_matrix_code, cppargs=["-fpermissive", "-g"]).create_transfer_matrix