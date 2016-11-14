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
    #include <petscmat.h>

    namespace dolfin
    {
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
        std::vector<std::size_t> fine_local_to_global_dofs;
        finemap->tabulate_local_to_global_dofs(fine_local_to_global_dofs);

        // Get local dof coordinates
        const std::vector<double> dofsf = fine_space->tabulate_dof_coordinates();

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
        // (we assume it is the same type of finite element for both spaces)
        std::shared_ptr<const FiniteElement> el = coarse_space->element();
        // number of dofs per cell for the finite element.
        std::size_t eldim = el->space_dimension();

        // Miscellaneous initialisations, these will all be needed later
        Point curr_point; // point variable
        Cell curr_cell; // cell variable
        std::vector<double> coordinate_dofs; // cell dofs coordinates vector
        ufc::cell ufc_cell; // ufc cell
        unsigned dofs_pos = 0; // this will be used to mark the first dof in dofsf
        unsigned int id = 0; // cell id

        // The overall idea is: a fine node can be on a coarse cell in the current processor,
        // on a coarse cell in a different processor, or outside the coarse domain.
        // If the point is found on the processor, evaluate basis functions,
        // if found elsewhere, use the other processor to evaluate basis functions,
        // if not found at all, or if found in multiple processors,
        // use compute_closest_entity on all processors and find
        // which coarse cell is the closest entity to the fine node amongst all processors.

        std::vector<double> _x(dim); // vector with point coordinates
        // vector containing the ranks of the processors which might contain a fine node
        std::vector<unsigned int> found_ranks;

        // the next vectors we are defining here contain information relative to the
        // fine nodes for which a corresponding coarse cell owned by the current
        // processor was found.
        // found_ids[i] contains the coarse cell id relative to each fine node i
        std::vector<std::size_t> found_ids;
        found_ids.reserve((std::size_t)M/mpi_size);
        // found_points[dim*i:dim*i + dim] contains the coordinates of the fine node i
        std::vector<double> found_points;
        found_points.reserve((std::size_t)dim*M/mpi_size);
        // global_row indices[i] contains the global row indices of the fine node i
        // i.e. in which row of the matrix the values relative to fine node i need to be stored
        std::vector<int> global_row_indices;
        global_row_indices.reserve((std::size_t)M/mpi_size);
        // found_points_senders[i] holds the rank of the process that owns the fine node
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
        // which processor contains the rank of the processor
        // that owns the coarse cell where the fine point was found
        std::vector<unsigned int> which_processor;

        for (unsigned i=0;i<m;i++)
        {
             dofs_pos = dim*i; // position of fine dofs coordinates in dofsf vector
             // save fine point coordinates in vector _x
             for (unsigned k=0;k<dim;k++)
                 _x[k] = dofsf[dofs_pos+k];

             // get the fine node into a Point variable
             if (dim == 3)
                 curr_point = Point(dofsf[dofs_pos], dofsf[dofs_pos+1], dofsf[dofs_pos+2]);
             else if (dim == 2)
                 curr_point = Point(dofsf[dofs_pos], dofsf[dofs_pos+1]);
             else
                 curr_point = Point(dofsf[dofs_pos]);

             // compute which processors share ownership of the coarse cell
             // that contains the fine node
             found_ranks = treec->compute_process_collisions(curr_point);
             // if the fine node is not in the domain or if more than one
             // processor share it, mark it as not found
             // (not found points will be searched by all the processors and
             // the processor that owns closest coarse cell to that node will be found,
             // so that even if multiple processes share the cell, we find the one that
             // actually owns it)
             if (found_ranks.empty() || found_ranks.size() > 1)
             {
                 // we store fine node coordinates, global row indices and the senders
                 // this information will be sent to all the processors
                 not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
                 not_found_global_row_indices.push_back(fine_local_to_global_dofs[i]);
                 not_found_points_senders.push_back(mpi_rank);
             }
             // if the fine node collides with a coarse cell owned by the current processor,
             // find the coarse cell the fine point lives in
             else if (found_ranks[0] == mpi_rank)
             {
                 // find the coarse cell where the fine node lies
                 id = treec->compute_first_entity_collision(curr_point);

                 // Safety control: if no cell is found on the current processor
                 // mark the point as not_found
                 if (id == std::numeric_limits<unsigned int>::max())
                 {
                     not_found_points.insert(not_found_points.end(), _x.begin(), _x.end());
                     not_found_global_row_indices.push_back(fine_local_to_global_dofs[i]);
                     not_found_points_senders.push_back(mpi_rank);
                 }
                 else
                 {
                     // if a cell is found on the current processor, add the point
                     // and relative information to the various vectors
                     found_ids.push_back(id);
                     found_points.insert(found_points.end(), _x.begin(), _x.end());
                     global_row_indices.push_back(fine_local_to_global_dofs[i]);
                     found_points_senders.push_back(mpi_rank);
                 }
             }
             // if found elsewhere, store the process where it was found
             else
             {
                 found_elsewhere.insert(found_elsewhere.end(),_x.begin(), _x.end());
                 which_processor.push_back(found_ranks[0]);
                 found_elsewhere_global_row_indices.push_back(fine_local_to_global_dofs[i]);
             }
        }
        //std::cout << "Rank: " << mpi_rank << "; We own: " << found_ids.size() << "; Found elsewhere: " << which_processor.size() << "; Not found: " << not_found_global_row_indices.size() << std::endl;
        //std::cout << "Rank: " << mpi_rank << "; Check 1: " << not_found_global_row_indices.size() << " = " << not_found_points_senders.size() << std::endl;
        auto old_found_ids_size = found_ids.size();
        auto old_not_found_size = not_found_global_row_indices.size();

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

        // we loop over the processors that own the fine nodes that need to be found
        // we call them senders here. Basically these are the processors that own
        // the matrix row relative to the found_elsewhere fine nodes
        for (unsigned sender=0;sender<mpi_size;sender++)
        {
            // We already searched on the current processor
            if (sender == receiver)
                continue;

            // how many fine nodes do we need to check?
            how_many = found_elsewhere_recv[sender].size()/dim;
            //std::cout << "Rank: " << mpi_rank << "; how many: " << how_many << std::endl;
            // If there is nothing to do, then just do nothing
            if (how_many == 0)
                continue;

            // for each fine node, create a Point variable and try to find the
            // coarse cell it lives in. If we cannot, mark the fine node as not found
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
                        not_found_global_row_indices.push_back(found_elsewhere_global_row_indices_recv[sender][i]);
                        not_found_points_senders.push_back(sender);
                    }
                    else
                    {
                        // if found, store information
                        found_ids.push_back(id);
                        found_points.insert(found_points.end(), _x.begin(), _x.end());
                        global_row_indices.push_back(found_elsewhere_global_row_indices_recv[sender][i]);
                        found_points_senders.push_back(sender);
                    }
                }
            }
        }
        //std::cout << "Rank: " << mpi_rank << "; We evaluate: " << found_ids.size() << "; Newly responsible for: " << found_ids.size() - old_found_ids_size << std::endl;
        //std::cout << "Rank: " << mpi_rank << "; Not found: " << not_found_global_row_indices.size() << "; New not found: " << not_found_global_row_indices.size() - old_not_found_size << std::endl;
        //std::cout << "Rank: " << mpi_rank << "; Check 2: " << not_found_global_row_indices.size() << " = " << not_found_points_senders.size() << std::endl;


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
        // we also need to store the fine node coordinates
        // in case the current processor owns the closest cell
        std::vector<double> found_not_found_points;
        // We need to flatten some vectors for further use
        std::vector<int> not_found_global_row_indices_flattened;
        std::vector<unsigned int> not_found_points_senders_flattened;

        //std::cout << "GO!!!\n\n";

        // we loop over all the processors were a fine node was found
        // note that from now on, every processor is doing the same check:
        // compute id and distance of the closest owned coarse cell to the
        // fine node, then send the distances to all the processors, so that
        // each processor can determine which processor owns the closest coarse cell
        for (unsigned int proc=0; proc<mpi_size; proc++)
        {
            how_many = not_found_points_recv[proc].size()/dim;
            //std::cout << "Rank: " << mpi_rank << " ; proc: " << proc << " ; Not found, how many: " << how_many << std::endl;

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

            // same trick as before, store the fine node coordinates into a Point
            // variable, then run compute_closest_entity to find the closest owned
            // cell id and distance from the fine node
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
                // store the (now) found, (previously) not found fine node coordinates in a vector
                found_not_found_points.insert(found_not_found_points.end(), _x.begin(), _x.end());
            }
        }

        //std::cout << "ALMOST...\n\n";

        // communicate all distances to all processor so that each one can tell
        // which processor owns the closest coarse cell to the not found point
        std::vector<std::vector<double>> not_found_distances_recv(mpi_size);
        MPI::all_gather(mpi_comm, not_found_distances, not_found_distances_recv);


        // now need to find which processor has a cell which is closest to the not_found points

        // initialise some variables
        double min_val; // minimum distance
        unsigned min_proc=0; // processor that owns the minimum distance cell
        unsigned int sender; // processor that asked to search for the not found fine node

        how_many = not_found_cell_indices.size();
        //std::cout << "Rank: " << mpi_rank << " ; Not found to process, how many: " << how_many << " ; check: " << not_found_points_senders_flattened.size() << " = " << not_found_global_row_indices_flattened.size() << std::endl;
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
            // add the fine node and closest coarse cell information to the
            // vectors of found nodes
            if (min_proc == mpi_rank)
            {
                // allocate cell id to current worker if distance is minimum
                id = not_found_cell_indices[i];
                found_ids.push_back(id);
                global_row_indices.push_back(not_found_global_row_indices_flattened[i]);
                found_points.insert(found_points.end(), found_not_found_points.begin() + dim*i, found_not_found_points.begin() + dim*(i+1));
                sender = not_found_points_senders_flattened[i];
                found_points_senders.push_back(sender);
            }
        }

        //std::cout << "DONE!!!\n\n";
        // Now every processor should have the information needed to assemble its portion of the matrix
        // the ids of coarse cell owned by each processor are currently stored in found_ids
        // and their respective global row indices are stored in global_row_indices.
        // The processors that own the matrix row relative to the fine node are stored in found_points_senders.
        // One last loop and we are ready to go!

        // m_owned is the number of rows the current processor needs to set
        // note that the processor might not own these rows
        std::size_t m_owned = found_ids.size();
        //std::cout << "m_owned: " << m_owned << "; M: " << M << " ; eldim: " << eldim << " ; check: " << found_points.size()/dim << " = " << found_points_senders.size() << std::endl;

        // initialise row and column indices and values of the transfer matrix
        int row_indices = 0;
        int** col_indices = new int*[m_owned];
        double** values = new double*[m_owned];
        for(unsigned i = 0; i < m_owned; ++i)
        {
            col_indices[i] = new int[eldim];
            values[i] = new double[eldim];
        }
        // initialise a single chunk of values (needed for later)
        double temp_values[eldim];

        // initialise column ownership range
        std::size_t n_own_begin;
        std::size_t n_own_end;

        // initialise global sparsity pattern
        std::vector<int> global_d_nnz(M,0);
        std::vector<int> global_o_nnz(M,0);

        // loop over the found coarse cells
        for (unsigned i=0; i<m_owned; i++)
        {
            // get coarse cell id
            id = found_ids[i];

            // save fine node coordinates into a Point variable
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
            // at the fine node and store the values into temp_values
            el->evaluate_basis_all(temp_values,
                                    curr_point.coordinates(),
                                    coordinate_dofs.data(),
                                    ufc_cell.orientation);

            // get dofs of the coarse cell
            ArrayView<const dolfin::la_index> temp_dofs = coarsemap->cell_dofs(id);
            for (unsigned j=0;j<eldim;j++)
            {
                // store temp values into the matrix values 2d vector
                values[i][j] = temp_values[j];
                // store global column indices
                col_indices[i][j] = coarse_local_to_global_dofs[temp_dofs[j]];

                // once we have the global column indices,
                // determine sparsity pattern:
                // which columns are owned by the process that
                // owns the fine node?

                // get the fine node owner processor
                sender = found_points_senders[i];
                // get its column ownership range
                n_own_begin = global_n_range_recv[sender][0];
                n_own_end = global_n_range_recv[sender][1];
                // check and allocate sparsity pattern
                if ((n_own_begin <= col_indices[i][j]) && (col_indices[i][j] < n_own_end))
                    global_d_nnz[global_row_indices[i]] += 1;
                else
                    global_o_nnz[global_row_indices[i]] += 1;
            }
        }

        // need to send the d_nnz and o_nnz to the correct processor
        // at the moment can only do global communication
        std::vector<std::vector<int>> global_d_nnz_recv(mpi_size);
        std::vector<std::vector<int>> global_o_nnz_recv(mpi_size);
        MPI::all_gather(mpi_comm, global_d_nnz, global_d_nnz_recv);
        MPI::all_gather(mpi_comm, global_o_nnz, global_o_nnz_recv);

        // initialise local sparsity pattern to 0
        int d_nnz[m], o_nnz[m];
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

        // Setting transfer matrix values row by row
        for (unsigned i=0;i<m_owned;i++)
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
