
#include "build_tree.hxx"
#include "basis.hxx"
#include "dist.hxx"

#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <cstdio>

using namespace nbd;

void nbd::randomUniformBodies(Body* bodies, int64_t nbodies, const double dmin, const double dmax, int64_t dim, int seed) {
  if (seed > 0)
    srand(seed);

  double range = dmax - dmin;
  for (int64_t i = 0; i < nbodies; i++)
    for (int64_t d = 0; d < dim; d++) {
      double r = ((double)rand() / RAND_MAX) * range + dmin;
      bodies[i].X[d] = r;
    }
}

void nbd::randomSurfaceBodies(Body* bodies, int64_t nbodies, int64_t dim, int seed) {
  if (seed > 0)
    srand(seed);

  double pi = M_PI;
  double pi2 = 2. * pi;

  if (dim == 2)
    for (int64_t i = 0; i < nbodies; i++) {
      double theta = ((double)rand() / RAND_MAX) * pi2;
      bodies[i].X[0] = std::cos(theta);
      bodies[i].X[1] = std::sin(theta);
    }
  
  if (dim == 3)
    for (int64_t i = 0; i < nbodies; i++) {
      double theta = ((double)rand() / RAND_MAX) * pi;
      double phi = ((double)rand() / RAND_MAX) * pi2;
      double cost = std::cos(theta);
      double sint = std::sin(theta);
      double cosp = std::cos(phi);
      double sinp = std::sin(phi);
      bodies[i].X[0] = sint * cosp;
      bodies[i].X[1] = sint * sinp;
      bodies[i].X[2] = cost;
    }
}

void nbd::randomNeutralCharge(Body* bodies, int64_t nbodies, double cmax, int seed) {
  if (seed > 0)
    srand(seed);

  double avg = 0.;
  double cmax2 = cmax * 2;
  for (int64_t i = 0; i < nbodies; i++) {
    double c = ((double)rand() / RAND_MAX) * cmax2 - cmax;
    bodies[i].B = c;
    avg = avg + c;
  }
  avg = avg / nbodies;
  for (int64_t i = 0; i < nbodies; i++)
    bodies[i].B = bodies[i].B - avg;
}

int64_t nbd::partition(Body* bodies, int64_t nbodies, int64_t sdim) {
  auto comp = [sdim](const Body& b1, const Body& b2) -> bool {
    return b1.X[sdim] < b2.X[sdim];
  };
  int64_t loc = nbodies / 2;
  std::nth_element(bodies, bodies + loc, bodies + nbodies, comp);
  return loc;
}

void nbd::getBounds(const Body* bodies, int64_t nbodies, double R[], double C[], int64_t dim) {
  std::vector<double> Xmin(dim), Xmax(dim);
  for (int64_t d = 0; d < dim; d++) 
    Xmin[d] = Xmax[d] = bodies[0].X[d];
  for (int64_t i = 0; i < nbodies; i++) {
    const Body* b = &bodies[i];
    for (int64_t d = 0; d < dim; d++) 
      Xmin[d] = std::fmin(b->X[d], Xmin[d]);
    for (int64_t d = 0; d < dim; d++) 
      Xmax[d] = std::fmax(b->X[d], Xmax[d]);
  }

  for (int64_t d = 0; d < dim; d++) {
    C[d] = (Xmin[d] + Xmax[d]) / 2;
    R[d] = 1.e-7 + std::fabs(Xmin[d] - Xmax[d]) / 2;
  }
}

int64_t nbd::buildTree(Cells& cells, Bodies& bodies, int64_t ncrit, int64_t dim) {
  int64_t nbody = bodies.size();
  int64_t levels = (int64_t)(std::log2(nbody / ncrit));
  int64_t nleaves = (int64_t)1 << levels;
  int64_t ncells = (nleaves << 1) - 1;
  cells.resize(ncells);

  Cell* root = &cells[0];
  root->BODY = &bodies[0];
  root->NBODY = nbody;
  root->ZID = 0;
  root->LEVEL = 0;
  getBounds(root->BODY, root->NBODY, root->R, root->C, dim);

  for (int64_t i = 0; i < ncells; i++) {
    Cell* ci = &cells[i];

    if (ci->LEVEL < levels) {
      int64_t sdim = 0;
      double maxR = ci->R[0];
      for (int64_t d = 1; d < dim; d++) {
        if (ci->R[d] > maxR)
        { sdim = d; maxR = ci->R[d]; }
      }
      int64_t loc = partition(ci->BODY, ci->NBODY, sdim);

      Cell* c0 = &cells[(i << 1) + 1];
      Cell* c1 = &cells[(i << 1) + 2];
      ci->CHILD = c0;
      ci->NCHILD = 2;

      c0->BODY = ci->BODY;
      c0->NBODY = loc;
      c0->ZID = (ci->ZID) << 1;
      c0->LEVEL = ci->LEVEL + 1;

      c1->BODY = ci->BODY + loc;
      c1->NBODY = ci->NBODY - loc;
      c1->ZID = ((ci->ZID) << 1) + 1;
      c1->LEVEL = ci->LEVEL + 1;

      for (int64_t d = 0; d < dim; d++)
        if (d == sdim) {
          double med = (ci->BODY)[loc].X[d];
          double Xmin = ci->C[d] - ci->R[d];
          double Xmax = ci->C[d] + ci->R[d];
          c0->C[d] = (Xmin + med) / 2;
          c0->R[d] = (med - Xmin) / 2;
          c1->C[d] = (med + Xmax) / 2;
          c1->R[d] = (Xmax - med) / 2;
        }
        else {
          c0->C[d] = ci->C[d];
          c0->R[d] = ci->R[d];
          c1->C[d] = ci->C[d];
          c1->R[d] = ci->R[d];
        }
    }
    else {
      ci->CHILD = NULL;
      ci->NCHILD = 0;
    }
  }

  return levels;
}


void nbd::getList(Cell* Ci, Cell* Cj, int64_t dim, int64_t theta) {
  if (Ci->LEVEL < Cj->LEVEL)
    for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
      getList(ci, Cj, dim, theta);
  else if (Cj->LEVEL < Ci->LEVEL)
    for (Cell* cj = Cj->CHILD; cj != Cj->CHILD + Cj->NCHILD; cj++)
      getList(Ci, cj, dim, theta);
  else {
    double dC = 0.;
    double dRi = 0.;
    double dRj = 0.;
    for (int64_t d = 0; d < dim; d++) {
      double diff = Ci->C[d] - Cj->C[d];
      dC = dC + diff * diff;
      dRi = dRi + Ci->R[d] * Ci->R[d];
      dRj = dRj + Cj->R[d] * Cj->R[d];
    }
    double dR = (dRi + dRj) * theta;

    if (dC > dR)
      Ci->listFar.push_back(Cj);
    else {
      Ci->listNear.push_back(Cj);

      if (Ci->NCHILD > 0)
        for (Cell* ci = Ci->CHILD; ci != Ci->CHILD + Ci->NCHILD; ci++)
          getList(ci, Cj, dim, theta);
    }
  }
}

void nbd::findCellsAtLevel(const Cell* cells[], int64_t* len, const Cell* cell, int64_t level) {
  if (level == cell->LEVEL) {
    int64_t i = *len;
    cells[i] = cell;
    *len = i + 1;
  }
  else if (level > cell->LEVEL && cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      findCellsAtLevel(cells, len, cell->CHILD + i, level);
}

void nbd::findCellsAtLevelModify(Cell* cells[], int64_t* len, Cell* cell, int64_t level) {
  if (level == cell->LEVEL) {
    int64_t i = *len;
    cells[i] = cell;
    *len = i + 1;
  }
  else if (level > cell->LEVEL && cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      findCellsAtLevelModify(cells, len, cell->CHILD + i, level);
}

const Cell* nbd::findLocalAtLevel(const Cell* cell, int64_t level) {
  const Cell* iter = cell;
  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, NULL, &mpi_levels);
  int64_t iters = level < mpi_levels ? level : mpi_levels;

  for (int64_t i = iter->LEVEL + 1; i <= iters; i++) {
    int64_t lvl_diff = mpi_levels - i;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t nchild = iter->NCHILD;
    Cell* child = iter->CHILD;
    for (int64_t n = 0; n < nchild; n++)
      if (child[n].ZID == my_rank)
        iter = child + n;
  }

  int64_t my_rank = mpi_rank >> (mpi_levels - iters);
  if (iter->ZID == my_rank)
    return iter;
  else
    return nullptr;
}

Cell* nbd::findLocalAtLevelModify(Cell* cell, int64_t level) {
  Cell* iter = cell;
  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, NULL, &mpi_levels);
  int64_t iters = level < mpi_levels ? level : mpi_levels;

  for (int64_t i = iter->LEVEL + 1; i <= iters; i++) {
    int64_t lvl_diff = mpi_levels - i;
    int64_t my_rank = mpi_rank >> lvl_diff;
    int64_t nchild = iter->NCHILD;
    Cell* child = iter->CHILD;
    for (int64_t n = 0; n < nchild; n++)
      if (child[n].ZID == my_rank)
        iter = child + n;
  }

  int64_t my_rank = mpi_rank >> (mpi_levels - iters);
  if (iter->ZID == my_rank)
    return iter;
  else
    return nullptr;
}


void nbd::traverse(Cells& cells, int64_t levels, int64_t dim, int64_t theta) {
  getList(&cells[0], &cells[0], dim, theta);
  int64_t mpi_levels;
  commRank(NULL, NULL, &mpi_levels);

  const Cell* local = &cells[0];
  for (int64_t i = 0; i <= mpi_levels; i++) {
    local = findLocalAtLevel(local, i);
    if (local != nullptr) {
      int64_t nlen = local->listNear.size();
      std::vector<int64_t> ngbs(nlen);
      for (int64_t n = 0; n < nlen; n++) {
        Cell* c = (local->listNear)[n];
        ngbs[n] = c->ZID;
      }

      configureComm(i, &ngbs[0], nlen);
    }
  }
}

void nbd::remoteBodies(Bodies& remote, int64_t size, const Cell& cell, const Bodies& bodies, int64_t dim) {
  int64_t avail = bodies.size();
  int64_t len = cell.listNear.size();
  std::vector<int64_t> offsets(len);
  std::vector<int64_t> lens(len);

  const Body* begin = &bodies[0];
  for (int64_t i = 0; i < len; i++) {
    const Cell* c = cell.listNear[i];
    avail = avail - c->NBODY;
    offsets[i] = c->BODY - begin;
    lens[i] = c->NBODY;
  }

  size = size > avail ? avail : size;
  remote.resize(size);

  for (int64_t i = 0; i < size; i++) {
    int64_t loc = (int64_t)((double)(avail * i) / size);
    for (int64_t j = 0; j < len; j++)
      if (loc >= offsets[j])
        loc = loc + lens[j];

    for (int64_t d = 0; d < dim; d++)
      remote[i].X[d] = bodies[loc].X[d];
    remote[i].B = bodies[loc].B;
  }
}

void nbd::collectChildMultipoles(const Cell& cell, int64_t multipoles[]) {
  if (cell.NCHILD > 0) {
    int64_t count = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++) {
      const Cell& c = cell.CHILD[i];
      int64_t loc = c.BODY - cell.BODY;
      int64_t len = c.Multipole.size();
      for (int64_t n = 0; n < len; n++) {
        int64_t nloc = loc + c.Multipole[n];
        multipoles[count] = nloc;
        count += 1;
      }
    }
  }
  else {
    int64_t len = cell.NBODY;
    std::iota(multipoles, multipoles + len, 0);
  }
}

void nbd::writeChildMultipoles(Cell& cell, const int64_t multipoles[], int64_t mlen) {
  if (cell.NCHILD > 0) {
    int64_t max = 0;
    int64_t count = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++) {
      Cell& c = cell.CHILD[i];
      int64_t begin = count;
      max = max + c.NBODY;
      while (count < mlen && multipoles[count] < max)
        count++;
      int64_t len = count - begin;
      if (c.Multipole.size() != len)
        c.Multipole.resize(len);
    }

    count = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++) {
      Cell& c = cell.CHILD[i];
      int64_t loc = c.BODY - cell.BODY;
      int64_t len = c.Multipole.size();
      for (int64_t n = 0; n < len; n++) {
        int64_t nloc = multipoles[count] - loc;
        c.Multipole[n] = nloc;
        count += 1;
      }
    }
  }
}

void nbd::childMultipoleSize(int64_t* size, const Cell& cell) {
  if (cell.NCHILD > 0) {
    int64_t s = 0;
    for (int64_t i = 0; i < cell.NCHILD; i++)
      s += cell.CHILD[i].Multipole.size();
    *size = s;
  }
  else
    *size = cell.NBODY;
}


void nbd::evaluateBasis(EvalFunc ef, Matrix& Base, Matrix& Biv, Cell* cell, const Bodies& bodies, double epi, int64_t mrank, int64_t sp_pts, int64_t dim) {
  int64_t m;
  childMultipoleSize(&m, *cell);

  Bodies remote;
  remoteBodies(remote, sp_pts, *cell, bodies, dim);
  int64_t n = remote.size();

  if (m > 0 && n > 0) {
    std::vector<int64_t> cellm(m);
    collectChildMultipoles(*cell, cellm.data());

    int64_t rank = mrank;
    rank = std::min(m, rank);
    rank = std::min(n, rank);
    Matrix a;
    std::vector<int64_t> pa(rank);
    cMatrix(a, m, n);
    cMatrix(Base, m, rank);

    M2Lmat_bodies(ef, m, n, cellm.data(), nullptr, cell->BODY, remote.data(), dim, a);

    int64_t iters;
    lraID(epi, rank, a, Base, pa.data(), &iters);

    if (cell->Multipole.size() != iters)
      cell->Multipole.resize(iters);
    for (int64_t i = 0; i < iters; i++) {
      int64_t ai = pa[i];
      cell->Multipole[i] = cellm[ai];
    }

    if (iters != rank)
      cMatrix(Base, m, iters);
    cMatrix(Biv, iters, m);
    invBasis(Base, Biv);
  }
}


void nbd::relationsNear(CSC rels[], const Cells& cells) {
  int64_t levels = 0;
  int64_t len = cells.size();
  for (int64_t i = 0; i < len; i++)
    levels = levels > cells[i].LEVEL ? levels : cells[i].LEVEL;

  int64_t mpi_rank;
  int64_t mpi_levels;
  commRank(&mpi_rank, NULL, &mpi_levels);

  for (int64_t i = 0; i <= levels; i++) {
    int64_t mpi_boxes = i > mpi_levels ? (int64_t)1 << (i - mpi_levels) : 1;
    int64_t mpi_dups = i < mpi_levels ? (mpi_levels - i) : 0;
    CSC& csc = rels[i];

    csc.M = (int64_t)1 << i;
    csc.N = mpi_boxes;
    csc.CSC_COLS.resize(mpi_boxes + 1);
    std::fill(csc.CSC_COLS.begin(), csc.CSC_COLS.end(), 0);
    csc.CSC_ROWS.clear();
    csc.NNZ = 0;
    csc.CBGN = (mpi_rank >> mpi_dups) * mpi_boxes;
  }

  for (int64_t i = 0; i < len; i++) {
    const Cell& c = cells[i];
    int64_t l = c.LEVEL;
    CSC& csc = rels[l];
    int64_t n = c.ZID - csc.CBGN;
    if (n >= 0 && n < csc.N) {
      int64_t ent = c.listNear.size();
      csc.CSC_COLS[n] = ent;
      for (int64_t j = 0; j < ent; j++)
        csc.CSC_ROWS.emplace_back((c.listNear[j])->ZID);
      csc.NNZ = csc.NNZ + ent;
    }
  }

  for (int64_t i = 0; i <= levels; i++) {
    CSC& csc = rels[i];
    int64_t count = 0;
    for (int64_t j = 0; j <= csc.N; j++) {
      int64_t ent = csc.CSC_COLS[j];
      csc.CSC_COLS[j] = count;
      count = count + ent;
    }
  }
}

void nbd::evaluateLeafNear(Matrices& d, EvalFunc ef, const Cell* cell, int64_t dim, const CSC& csc) {
  if (cell->NCHILD > 0)
    for (int64_t i = 0; i < cell->NCHILD; i++)
      evaluateLeafNear(d, ef, cell->CHILD + i, dim, csc);
  else {
    int64_t n = cell->ZID - csc.CBGN;
    if (n >= 0 && n < csc.N) {
      int64_t len = cell->listNear.size();
      int64_t off = csc.CSC_COLS[n];
      for (int64_t i = 0; i < len; i++)
        P2Pmat(ef, cell->listNear[i], cell, dim, d[off + i]);
    }
  }
}

void nbd::lookupIJ(int64_t& ij, const CSC& rels, int64_t i, int64_t j) {
  int64_t lj = j - rels.CBGN;
  if (lj < 0 || lj >= rels.N)
  { ij = -1; return; }
  int64_t k = std::distance(rels.CSC_ROWS.data(), 
    std::find(rels.CSC_ROWS.data() + rels.CSC_COLS[lj], rels.CSC_ROWS.data() + rels.CSC_COLS[lj + 1], i));
  ij = (k < rels.CSC_COLS[lj + 1]) ? k : -1;
}


void writeIntermediate(Matrix& d, EvalFunc ef, const Cell* ci, const Cell* cj, int64_t dim, const Matrices& dc, const Matrices& uc, const CSC& cscc) {
  int64_t m, n;
  childMultipoleSize(&m, *ci);
  childMultipoleSize(&n, *cj);
  if (m > 0 && n > 0) {
    cMatrix(d, m, n);
    zeroMatrix(d);

    int64_t y_off = 0;
    for (int64_t i = 0; i < ci->NCHILD; i++) {
      const Cell* cii = ci->CHILD + i;
      int64_t mi = cii->Multipole.size();
      const std::vector<Cell*>& cii_m2l = cii->listFar;
      int64_t box_i = cii->ZID;
      neighborsILocal(box_i, cii->ZID, cii->LEVEL);
      const Matrix* u = box_i >= 0 ? &uc[box_i] : nullptr;

      int64_t x_off = 0;
      for (int64_t j = 0; j < cj->NCHILD; j++) {
        const Cell* cjj = cj->CHILD + j;
        int64_t mj = cjj->Multipole.size();
        int64_t box_j = cjj->ZID;
        neighborsILocal(box_j, cjj->ZID, cjj->LEVEL);
        const Matrix* vt = box_j >= 0 ? &uc[box_j] : nullptr;

        Matrix m2l;
        cMatrix(m2l, mi, mj);

        if (std::find(cii_m2l.begin(), cii_m2l.end(), cjj) != cii_m2l.end()) {
          M2Lmat(ef, cii, cjj, dim, m2l);
          cpyMatToMat(mi, mj, m2l, d, 0, 0, y_off, x_off);
        }
        else {
          int64_t zii = cii->ZID;
          int64_t zjj = cjj->ZID;
          int64_t ij;
          lookupIJ(ij, cscc, zii, zjj);
          if (ij >= 0 && u != nullptr && vt != nullptr) {
            utav('T', *u, dc[ij], *vt, m2l);
            cpyMatToMat(mi, mj, m2l, d, 0, 0, y_off, x_off);
          }
        }
        x_off = x_off + mj;
      }
      y_off = y_off + mi;
    }
  }
}

void evaluateIntermediate(EvalFunc ef, const Cell* c, int64_t dim, const CSC* csc, const Base* base, Matrices* d, int64_t level) {
  if (c->LEVEL < level)
    for (int64_t i = 0; i < c->NCHILD; i++)
      evaluateIntermediate(ef, c->CHILD + i, dim, csc, base, d, level);

  if (c->LEVEL == level) {
    int64_t zj = c->ZID;
    int64_t lnear = c->listNear.size();
    for (int64_t i = 0; i < lnear; i++) {
      const Cell* ci = c->listNear[i];
      int64_t zi = ci->ZID;
      int64_t ij;
      lookupIJ(ij, *csc, zi, zj);
      if (ij >= 0)
        writeIntermediate((*d)[ij], ef, ci, c, dim, d[1], base[1].Uc, csc[1]);
    }
  }
}

void nbd::evaluateNear(Matrices d[], EvalFunc ef, const Cells& cells, int64_t dim, const CSC rels[], const Base base[], int64_t levels) {
  Matrices& dleaf = d[levels];
  const CSC& cleaf = rels[levels];
  for (int64_t i = 0; i <= levels; i++)
    d[i].resize(rels[i].NNZ);
  evaluateLeafNear(dleaf, ef, &cells[0], dim, cleaf);
  for (int64_t i = levels - 1; i >= 0; i--) {
    evaluateIntermediate(ef, &cells[0], dim, &rels[i], &base[i], &d[i], i);
    if (rels[i].N == rels[i + 1].N)
      butterflySumA(d[i], i + 1);
  }
}

void nbd::loadX(Vectors& X, const Cell* cell, int64_t level) {
  int64_t xlen = (int64_t)1 << level;
  neighborContentLength(xlen, level);
  X.resize(xlen);

  int64_t ibegin = 0;
  int64_t iend = (int64_t)1 << level;
  selfLocalRange(ibegin, iend, level);
  int64_t nodes = iend - ibegin;

  int64_t len = 0;
  std::vector<const Cell*> cells(nodes);
  std::vector<int64_t> dims(xlen);
  findCellsAtLevel(&cells[0], &len, cell, level);

  for (int64_t i = 0; i < len; i++) {
    const Cell* ci = cells[i];
    int64_t li = ci->ZID;
    neighborsILocal(li, ci->ZID, level);
    Vector& Xi = X[li];
    cVector(Xi, ci->NBODY);
    dims[li] = ci->NBODY;

    for (int64_t n = 0; n < ci->NBODY; n++)
      Xi.X[n] = ci->BODY[n].B;
  }

  DistributeDims(&dims[0], level);

  for (int64_t i = 0; i < xlen; i++)
    if (X[i].N != dims[i])
      cVector(X[i], dims[i]);
  DistributeVectorsList(X, level);
}
