/-- A Young diagram is a finite collection of cells on the `ℕ × ℕ` grid such that whenever
a cell is present, so are all the ones above and to the left of it. Like matrices, an `(i, j)` cell
is a cell in row `i` and column `j`, where rows are enumerated downward and columns rightward.

Young diagrams are modeled as finite sets in `ℕ × ℕ` that are lower sets with respect to the
standard order on products. -/
@[ext]
structure YoungDiagram where
  /-- A finite set which represents a finite collection of cells on the `ℕ × ℕ` grid. -/
  cells : Finset (ℕ × ℕ)
  /-- Cells are up-left justified, witnessed by the fact that `cells` is a lower set in `ℕ × ℕ`. -/
  isLowerSet : IsLowerSet (cells : Set (ℕ × ℕ))


instance : SetLike YoungDiagram (ℕ × ℕ) where
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: figure out how to do this correctly
  coe y := y.cells
                             /-
                               μ ν : YoungDiagram
                               h : Eq ((fun y => ↑y.cells) μ) ((fun y => ↑y.cells) ν)
                               ⊢ Eq μ ν
                             -/
  coe_injective' μ ν h := by rwa [YoungDiagram.ext_iff, ← Finset.coe_inj]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem mem_cells {μ : YoungDiagram} (c : ℕ × ℕ) : c ∈ μ.cells ↔ c ∈ μ :=
  Iff.rfl


@[simp]
theorem mem_mk (c : ℕ × ℕ) (cells) (isLowerSet) :
    c ∈ YoungDiagram.mk cells isLowerSet ↔ c ∈ cells :=
  Iff.rfl


instance decidableMem (μ : YoungDiagram) : DecidablePred (· ∈ μ) :=
  inferInstanceAs (DecidablePred (· ∈ μ.cells))


/-- In "English notation", a Young diagram is drawn so that (i1, j1) ≤ (i2, j2)
    means (i1, j1) is weakly up-and-left of (i2, j2). -/
theorem up_left_mem (μ : YoungDiagram) {i1 i2 j1 j2 : ℕ} (hi : i1 ≤ i2) (hj : j1 ≤ j2)
    (hcell : (i2, j2) ∈ μ) : (i1, j1) ∈ μ :=
  μ.isLowerSet (Prod.mk_le_mk.mpr ⟨hi, hj⟩) hcell


@[simp]
theorem cells_subset_iff {μ ν : YoungDiagram} : μ.cells ⊆ ν.cells ↔ μ ≤ ν :=
  Iff.rfl


@[simp]
theorem cells_ssubset_iff {μ ν : YoungDiagram} : μ.cells ⊂ ν.cells ↔ μ < ν :=
  Iff.rfl


instance : Max YoungDiagram where
  max μ ν :=
    { cells := μ.cells ∪ ν.cells
      isLowerSet := by
        /-
          μ ν : YoungDiagram
          ⊢ IsLowerSet ↑(Union.union μ.cells ν.cells)
        -/
        rw [Finset.coe_union]
        /-
          μ ν : YoungDiagram
          ⊢ IsLowerSet (Union.union ↑μ.cells ↑ν.cells)
        -/
        exact μ.isLowerSet.union ν.isLowerSet }
        /-
          🎉 no goals
        -/


@[simp]
theorem cells_sup (μ ν : YoungDiagram) : (μ ⊔ ν).cells = μ.cells ∪ ν.cells :=
  rfl


@[simp, norm_cast]
theorem coe_sup (μ ν : YoungDiagram) : ↑(μ ⊔ ν) = (μ ∪ ν : Set (ℕ × ℕ)) :=
  Finset.coe_union _ _


@[simp]
theorem mem_sup {μ ν : YoungDiagram} {x : ℕ × ℕ} : x ∈ μ ⊔ ν ↔ x ∈ μ ∨ x ∈ ν :=
  Finset.mem_union


instance : Min YoungDiagram where
  min μ ν :=
    { cells := μ.cells ∩ ν.cells
      isLowerSet := by
        /-
          μ ν : YoungDiagram
          ⊢ IsLowerSet ↑(Inter.inter μ.cells ν.cells)
        -/
        rw [Finset.coe_inter]
        /-
          μ ν : YoungDiagram
          ⊢ IsLowerSet (Inter.inter ↑μ.cells ↑ν.cells)
        -/
        exact μ.isLowerSet.inter ν.isLowerSet }
        /-
          🎉 no goals
        -/


@[simp]
theorem cells_inf (μ ν : YoungDiagram) : (μ ⊓ ν).cells = μ.cells ∩ ν.cells :=
  rfl


@[simp, norm_cast]
theorem coe_inf (μ ν : YoungDiagram) : ↑(μ ⊓ ν) = (μ ∩ ν : Set (ℕ × ℕ)) :=
  Finset.coe_inter _ _


@[simp]
theorem mem_inf {μ ν : YoungDiagram} {x : ℕ × ℕ} : x ∈ μ ⊓ ν ↔ x ∈ μ ∧ x ∈ ν :=
  Finset.mem_inter


/-- The empty Young diagram is (⊥ : young_diagram). -/
instance : OrderBot YoungDiagram where
  bot :=
    { cells := ∅
      isLowerSet := by
        /-
          ⊢ IsLowerSet ↑EmptyCollection.emptyCollection
        -/
        intros a b _ h
        /-
          a b : Prod Nat Nat
          a✝ : LE.le b a
          h : Membership.mem (↑EmptyCollection.emptyCollection) a
          ⊢ Membership.mem (↑EmptyCollection.emptyCollection) b
        -/
        simp only [Finset.coe_empty, Set.mem_empty_iff_false]
        /-
          a b : Prod Nat Nat
          a✝ : LE.le b a
          h : Membership.mem (↑EmptyCollection.emptyCollection) a
          ⊢ False
        -/
        simp only [Finset.coe_empty, Set.mem_empty_iff_false] at h }
        /-
          🎉 no goals
        -/
  bot_le _ _ := by
    /-
      x✝¹ : YoungDiagram
      x✝ : Prod Nat Nat
      ⊢ Membership.mem Bot.bot x✝ → Membership.mem x✝¹ x✝
    -/
    intro y
    /-
      x✝¹ : YoungDiagram
      x✝ : Prod Nat Nat
      y : Membership.mem Bot.bot x✝
      ⊢ Membership.mem x✝¹ x✝
    -/
    simp only [mem_mk, Finset.not_mem_empty] at y
    /-
      🎉 no goals
    -/


@[simp]
theorem cells_bot : (⊥ : YoungDiagram).cells = ∅ :=
  rfl


@[simp]
theorem not_mem_bot (x : ℕ × ℕ) : x ∉ (⊥ : YoungDiagram) :=
  Finset.not_mem_empty x


@[norm_cast]
theorem coe_bot : (⊥ : YoungDiagram) = (∅ : Set (ℕ × ℕ)) := by
  /-
    ⊢ Eq (↑Bot.bot) EmptyCollection.emptyCollection
  -/
  ext; simp
       /-
         🎉 no goals
       -/


instance : Inhabited YoungDiagram :=
  ⟨⊥⟩


instance : DistribLattice YoungDiagram :=
                                                                        /-
                                                                          μ ν : YoungDiagram
                                                                          h : Eq μ.cells ν.cells
                                                                          ⊢ Eq μ ν
                                                                        -/
  Function.Injective.distribLattice YoungDiagram.cells (fun μ ν h => by rwa [YoungDiagram.ext_iff])
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
    (fun _ _ => rfl) fun _ _ => rfl


/-- Cardinality of a Young diagram -/
protected abbrev card (μ : YoungDiagram) : ℕ :=
  μ.cells.card


/-- The `transpose` of a Young diagram is obtained by swapping i's with j's. -/
def transpose (μ : YoungDiagram) : YoungDiagram where
  cells := (Equiv.prodComm _ _).finsetCongr μ.cells
  isLowerSet _ _ h := by
    /-
      μ : YoungDiagram
      x✝¹ x✝ : Prod Nat Nat
      h : LE.le x✝ x✝¹
      ⊢ Membership.mem (↑((Equiv.prodComm Nat Nat).finsetCongr μ.cells)) x✝¹ → Membe …
    -/
    simp only [Finset.mem_coe, Equiv.finsetCongr_apply, Finset.mem_map_equiv]
    /-
      μ : YoungDiagram
      x✝¹ x✝ : Prod Nat Nat
      h : LE.le x✝ x✝¹
      ⊢ Membership.mem μ.cells ((Equiv.prodComm Nat Nat).symm x✝¹) → Membership.mem  …
    -/
    intro hcell
    /-
      μ : YoungDiagram
      x✝¹ x✝ : Prod Nat Nat
      h : LE.le x✝ x✝¹
      hcell : Membership.mem μ.cells ((Equiv.prodComm Nat Nat).symm x✝¹)
      ⊢ Membership.mem μ.cells ((Equiv.prodComm Nat Nat).symm x✝)
    -/
    apply μ.isLowerSet _ hcell
    /-
      μ : YoungDiagram
      x✝¹ x✝ : Prod Nat Nat
      h : LE.le x✝ x✝¹
      hcell : Membership.mem μ.cells ((Equiv.prodComm Nat Nat).symm x✝¹)
      ⊢ LE.le ((Equiv.prodComm Nat Nat).symm x✝) ((Equiv.prodComm Nat Nat).symm x✝¹)
    -/
    simp [h]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_transpose {μ : YoungDiagram} {c : ℕ × ℕ} : c ∈ μ.transpose ↔ c.swap ∈ μ := by
  /-
    μ : YoungDiagram
    c : Prod Nat Nat
    ⊢ Iff (Membership.mem μ.transpose c) (Membership.mem μ c.swap)
  -/
  simp [transpose]
  /-
    🎉 no goals
  -/


@[simp]
theorem transpose_transpose (μ : YoungDiagram) : μ.transpose.transpose = μ := by
  /-
    μ : YoungDiagram
    ⊢ Eq μ.transpose.transpose μ
  -/
  ext x
  /-
    case cells.h
    μ : YoungDiagram
    x : Prod Nat Nat
    ⊢ Iff (Membership.mem μ.transpose.transpose.cells x) (Membership.mem μ.cells x)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem transpose_eq_iff_eq_transpose {μ ν : YoungDiagram} : μ.transpose = ν ↔ μ = ν.transpose := by
  /-
    μ ν : YoungDiagram
    ⊢ Iff (Eq μ.transpose ν) (Eq μ ν.transpose)
  -/
  constructor <;>
      /-
        case mp
        μ ν : YoungDiagram
        ⊢ Eq μ.transpose ν → Eq μ ν.transpose
      -/
      /-
        case mp
        μ : YoungDiagram
        ⊢ Eq μ μ.transpose.transpose
      -/
      /-
        🎉 no goals
      -/
      /-
        case mpr
        ν : YoungDiagram
        ⊢ Eq ν.transpose.transpose ν
      -/
      simp
      /-
        🎉 no goals
      -/


@[simp]
theorem transpose_eq_iff {μ ν : YoungDiagram} : μ.transpose = ν.transpose ↔ μ = ν := by
  /-
    μ ν : YoungDiagram
    ⊢ Iff (Eq μ.transpose ν.transpose) (Eq μ ν)
  -/
  rw [transpose_eq_iff_eq_transpose]
  /-
    μ ν : YoungDiagram
    ⊢ Iff (Eq μ ν.transpose.transpose) (Eq μ ν)
  -/
  simp
  /-
    🎉 no goals
  -/

-- This is effectively both directions of `transpose_le_iff` below.

protected theorem le_of_transpose_le {μ ν : YoungDiagram} (h_le : μ.transpose ≤ ν) :
    μ ≤ ν.transpose := fun c hc => by
  /-
    μ ν : YoungDiagram
    h_le : LE.le μ.transpose ν
    c : Prod Nat Nat
    hc : Membership.mem μ.cells c
    ⊢ Membership.mem ν.transpose.cells c
  -/
  simp only [mem_cells, mem_transpose]
  /-
    μ ν : YoungDiagram
    h_le : LE.le μ.transpose ν
    c : Prod Nat Nat
    hc : Membership.mem μ.cells c
    ⊢ Membership.mem ν c.swap
  -/
  apply h_le
  /-
    case a
    μ ν : YoungDiagram
    h_le : LE.le μ.transpose ν
    c : Prod Nat Nat
    hc : Membership.mem μ.cells c
    ⊢ Membership.mem μ.transpose.cells c.swap
  -/
  simpa
  /-
    🎉 no goals
  -/


@[simp]
theorem transpose_le_iff {μ ν : YoungDiagram} : μ.transpose ≤ ν.transpose ↔ μ ≤ ν :=
  ⟨fun h => by
    /-
      μ ν : YoungDiagram
      h : LE.le μ.transpose ν.transpose
      ⊢ LE.le μ ν
    -/
    convert YoungDiagram.le_of_transpose_le h
    /-
      case h.e'_4
      μ ν : YoungDiagram
      h : LE.le μ.transpose ν.transpose
      ⊢ Eq ν ν.transpose.transpose
    -/
    simp, fun h => by
    /-
      🎉 no goals
    -/
    /-
      μ ν : YoungDiagram
      h : LE.le μ ν
      ⊢ LE.le μ.transpose ν.transpose
    -/
    rw [← transpose_transpose μ] at h
    /-
      μ ν : YoungDiagram
      h : LE.le μ.transpose.transpose ν
      ⊢ LE.le μ.transpose ν.transpose
    -/
    exact YoungDiagram.le_of_transpose_le h ⟩
    /-
      🎉 no goals
    -/


@[mono]
protected theorem transpose_mono {μ ν : YoungDiagram} (h_le : μ ≤ ν) : μ.transpose ≤ ν.transpose :=
  transpose_le_iff.mpr h_le


/-- Transposing Young diagrams is an `OrderIso`. -/
@[simps]
def transposeOrderIso : YoungDiagram ≃o YoungDiagram :=
                                      /-
                                        x✝ : YoungDiagram
                                        ⊢ Eq x✝.transpose.transpose x✝
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  ⟨⟨transpose, transpose, fun _ => by simp, fun _ => by simp⟩, by simp⟩
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- The `i`-th row of a Young diagram consists of the cells whose first coordinate is `i`. -/
def row (μ : YoungDiagram) (i : ℕ) : Finset (ℕ × ℕ) :=
  μ.cells.filter fun c => c.fst = i


theorem mem_row_iff {μ : YoungDiagram} {i : ℕ} {c : ℕ × ℕ} : c ∈ μ.row i ↔ c ∈ μ ∧ c.fst = i := by
  /-
    μ : YoungDiagram
    i : Nat
    c : Prod Nat Nat
    ⊢ Iff (Membership.mem (μ.row i) c) (And (Membership.mem μ c) (Eq c.1 i))
  -/
  simp [row]
  /-
    🎉 no goals
  -/


                                                                                          /-
                                                                                            μ : YoungDiagram
                                                                                            i j : Nat
                                                                                            ⊢ Iff (Membership.mem (μ.row i) { fst := i, snd := j }) (Membership.mem μ { fs …
                                                                                          -/
theorem mk_mem_row_iff {μ : YoungDiagram} {i j : ℕ} : (i, j) ∈ μ.row i ↔ (i, j) ∈ μ := by simp [row]
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


protected theorem exists_not_mem_row (μ : YoungDiagram) (i : ℕ) : ∃ j, (i, j) ∉ μ := by
  obtain ⟨j, hj⟩ :=
    Infinite.exists_not_mem_finset
      (μ.cells.preimage (Prod.mk i) fun _ _ _ _ h => by
        cases h
        rfl)
  /-
    case intro
    μ : YoungDiagram
    i j : Nat
    hj : Not (Membership.mem (μ.cells.preimage (Prod.mk i) ⋯) j)
    ⊢ Exists fun j => Not (Membership.mem μ { fst := i, snd := j })
  -/
  rw [Finset.mem_preimage] at hj
  /-
    case intro
    μ : YoungDiagram
    i j : Nat
    hj : Not (Membership.mem μ.cells { fst := i, snd := j })
    ⊢ Exists fun j => Not (Membership.mem μ { fst := i, snd := j })
  -/
  exact ⟨j, hj⟩
  /-
    🎉 no goals
  -/


/-- Length of a row of a Young diagram -/
def rowLen (μ : YoungDiagram) (i : ℕ) : ℕ :=
  Nat.find <| μ.exists_not_mem_row i


theorem mem_iff_lt_rowLen {μ : YoungDiagram} {i j : ℕ} : (i, j) ∈ μ ↔ j < μ.rowLen i := by
  /-
    μ : YoungDiagram
    i j : Nat
    ⊢ Iff (Membership.mem μ { fst := i, snd := j }) (LT.lt j (μ.rowLen i))
  -/
  rw [rowLen, Nat.lt_find_iff]
  /-
    μ : YoungDiagram
    i j : Nat
    ⊢ Iff (Membership.mem μ { fst := i, snd := j }) (∀ (m : Nat), LE.le m j → Not  …
  -/
  push_neg
  /-
    μ : YoungDiagram
    i j : Nat
    ⊢ Iff (Membership.mem μ { fst := i, snd := j }) (∀ (m : Nat), LE.le m j → Memb …
  -/
  exact ⟨fun h _ hmj => μ.up_left_mem (by rfl) hmj h, fun h => h _ (by rfl)⟩
  /-
    🎉 no goals
  -/


theorem row_eq_prod {μ : YoungDiagram} {i : ℕ} : μ.row i = {i} ×ˢ Finset.range (μ.rowLen i) := by
  /-
    μ : YoungDiagram
    i : Nat
    ⊢ Eq (μ.row i) (SProd.sprod (Singleton.singleton i) (Finset.range (μ.rowLen i)))
  -/
  ext ⟨a, b⟩
  simp only [Finset.mem_product, Finset.mem_singleton, Finset.mem_range, mem_row_iff,
    mem_iff_lt_rowLen, and_comm, and_congr_right_iff]
  /-
    case h.mk
    μ : YoungDiagram
    i a b : Nat
    ⊢ Eq a i → Iff (LT.lt b (μ.rowLen a)) (LT.lt b (μ.rowLen i))
  -/
  rintro rfl
  /-
    case h.mk
    μ : YoungDiagram
    a b : Nat
    ⊢ Iff (LT.lt b (μ.rowLen a)) (LT.lt b (μ.rowLen a))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem rowLen_eq_card (μ : YoungDiagram) {i : ℕ} : μ.rowLen i = (μ.row i).card := by
  /-
    μ : YoungDiagram
    i : Nat
    ⊢ Eq (μ.rowLen i) (μ.row i).card
  -/
  simp [row_eq_prod]
  /-
    🎉 no goals
  -/


@[mono]
theorem rowLen_anti (μ : YoungDiagram) (i1 i2 : ℕ) (hi : i1 ≤ i2) : μ.rowLen i2 ≤ μ.rowLen i1 := by
  /-
    μ : YoungDiagram
    i1 i2 : Nat
    hi : LE.le i1 i2
    ⊢ LE.le (μ.rowLen i2) (μ.rowLen i1)
  -/
  by_contra! h_lt
  /-
    μ : YoungDiagram
    i1 i2 : Nat
    hi : LE.le i1 i2
    h_lt : LT.lt (μ.rowLen i1) (μ.rowLen i2)
    ⊢ False
  -/
  rw [← lt_self_iff_false (μ.rowLen i1)]
  /-
    μ : YoungDiagram
    i1 i2 : Nat
    hi : LE.le i1 i2
    h_lt : LT.lt (μ.rowLen i1) (μ.rowLen i2)
    ⊢ LT.lt (μ.rowLen i1) (μ.rowLen i1)
  -/
  rw [← mem_iff_lt_rowLen] at h_lt ⊢
  /-
    μ : YoungDiagram
    i1 i2 : Nat
    hi : LE.le i1 i2
    h_lt : Membership.mem μ { fst := i2, snd := μ.rowLen i1 }
    ⊢ Membership.mem μ { fst := i1, snd := μ.rowLen i1 }
  -/
  exact μ.up_left_mem hi (by rfl) h_lt
  /-
    🎉 no goals
  -/


/-- The `j`-th column of a Young diagram consists of the cells whose second coordinate is `j`. -/
def col (μ : YoungDiagram) (j : ℕ) : Finset (ℕ × ℕ) :=
  μ.cells.filter fun c => c.snd = j


theorem mem_col_iff {μ : YoungDiagram} {j : ℕ} {c : ℕ × ℕ} : c ∈ μ.col j ↔ c ∈ μ ∧ c.snd = j := by
  /-
    μ : YoungDiagram
    j : Nat
    c : Prod Nat Nat
    ⊢ Iff (Membership.mem (μ.col j) c) (And (Membership.mem μ c) (Eq c.2 j))
  -/
  simp [col]
  /-
    🎉 no goals
  -/


                                                                                          /-
                                                                                            μ : YoungDiagram
                                                                                            i j : Nat
                                                                                            ⊢ Iff (Membership.mem (μ.col j) { fst := i, snd := j }) (Membership.mem μ { fs …
                                                                                          -/
theorem mk_mem_col_iff {μ : YoungDiagram} {i j : ℕ} : (i, j) ∈ μ.col j ↔ (i, j) ∈ μ := by simp [col]
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


protected theorem exists_not_mem_col (μ : YoungDiagram) (j : ℕ) : ∃ i, (i, j) ∉ μ.cells := by
  /-
    μ : YoungDiagram
    j : Nat
    ⊢ Exists fun i => Not (Membership.mem μ.cells { fst := i, snd := j })
  -/
  convert μ.transpose.exists_not_mem_row j using 1
  /-
    case h.e'_2
    μ : YoungDiagram
    j : Nat
    ⊢ Eq (fun i => Not (Membership.mem μ.cells { fst := i, snd := j })) fun j_1 => …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Length of a column of a Young diagram -/
def colLen (μ : YoungDiagram) (j : ℕ) : ℕ :=
  Nat.find <| μ.exists_not_mem_col j


@[simp]
theorem colLen_transpose (μ : YoungDiagram) (j : ℕ) : μ.transpose.colLen j = μ.rowLen j := by
  /-
    μ : YoungDiagram
    j : Nat
    ⊢ Eq (μ.transpose.colLen j) (μ.rowLen j)
  -/
  simp [rowLen, colLen]
  /-
    🎉 no goals
  -/


@[simp]
theorem rowLen_transpose (μ : YoungDiagram) (i : ℕ) : μ.transpose.rowLen i = μ.colLen i := by
  /-
    μ : YoungDiagram
    i : Nat
    ⊢ Eq (μ.transpose.rowLen i) (μ.colLen i)
  -/
  simp [rowLen, colLen]
  /-
    🎉 no goals
  -/


theorem mem_iff_lt_colLen {μ : YoungDiagram} {i j : ℕ} : (i, j) ∈ μ ↔ i < μ.colLen j := by
  /-
    μ : YoungDiagram
    i j : Nat
    ⊢ Iff (Membership.mem μ { fst := i, snd := j }) (LT.lt i (μ.colLen j))
  -/
  rw [← rowLen_transpose, ← mem_iff_lt_rowLen]
  /-
    μ : YoungDiagram
    i j : Nat
    ⊢ Iff (Membership.mem μ { fst := i, snd := j }) (Membership.mem μ.transpose {  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem col_eq_prod {μ : YoungDiagram} {j : ℕ} : μ.col j = Finset.range (μ.colLen j) ×ˢ {j} := by
  /-
    μ : YoungDiagram
    j : Nat
    ⊢ Eq (μ.col j) (SProd.sprod (Finset.range (μ.colLen j)) (Singleton.singleton j))
  -/
  ext ⟨a, b⟩
  simp only [Finset.mem_product, Finset.mem_singleton, Finset.mem_range, mem_col_iff,
    mem_iff_lt_colLen, and_comm, and_congr_right_iff]
  /-
    case h.mk
    μ : YoungDiagram
    j a b : Nat
    ⊢ Eq b j → Iff (LT.lt a (μ.colLen b)) (LT.lt a (μ.colLen j))
  -/
  rintro rfl
  /-
    case h.mk
    μ : YoungDiagram
    a b : Nat
    ⊢ Iff (LT.lt a (μ.colLen b)) (LT.lt a (μ.colLen b))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem colLen_eq_card (μ : YoungDiagram) {j : ℕ} : μ.colLen j = (μ.col j).card := by
  /-
    μ : YoungDiagram
    j : Nat
    ⊢ Eq (μ.colLen j) (μ.col j).card
  -/
  simp [col_eq_prod]
  /-
    🎉 no goals
  -/


@[mono]
theorem colLen_anti (μ : YoungDiagram) (j1 j2 : ℕ) (hj : j1 ≤ j2) : μ.colLen j2 ≤ μ.colLen j1 := by
  /-
    μ : YoungDiagram
    j1 j2 : Nat
    hj : LE.le j1 j2
    ⊢ LE.le (μ.colLen j2) (μ.colLen j1)
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  convert μ.transpose.rowLen_anti j1 j2 hj using 1 <;> simp
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- List of row lengths of a Young diagram -/
def rowLens (μ : YoungDiagram) : List ℕ :=
  (List.range <| μ.colLen 0).map μ.rowLen


@[simp]
theorem get_rowLens {μ : YoungDiagram} {i : Nat} {h : i < μ.rowLens.length} :
                                    /-
                                      μ : YoungDiagram
                                      i : Nat
                                      h : LT.lt i μ.rowLens.length
                                      ⊢ Eq (GetElem.getElem μ.rowLens i h) (μ.rowLen i)
                                    -/
    μ.rowLens[i] = μ.rowLen i := by simp only [rowLens, List.getElem_range, List.getElem_map]
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem length_rowLens {μ : YoungDiagram} : μ.rowLens.length = μ.colLen 0 := by
  /-
    μ : YoungDiagram
    ⊢ Eq μ.rowLens.length (μ.colLen 0)
  -/
  simp only [rowLens, List.length_map, List.length_range]
  /-
    🎉 no goals
  -/


theorem rowLens_sorted (μ : YoungDiagram) : μ.rowLens.Sorted (· ≥ ·) :=
  (List.pairwise_le_range _).map _ μ.rowLen_anti


theorem pos_of_mem_rowLens (μ : YoungDiagram) (x : ℕ) (hx : x ∈ μ.rowLens) : 0 < x := by
  /-
    μ : YoungDiagram
    x : Nat
    hx : Membership.mem μ.rowLens x
    ⊢ LT.lt 0 x
  -/
  rw [rowLens, List.mem_map] at hx
  /-
    μ : YoungDiagram
    x : Nat
    hx : Exists fun a => And (Membership.mem (List.range (μ.colLen 0)) a) (Eq (μ.r …
    ⊢ LT.lt 0 x
  -/
  obtain ⟨i, hi, rfl : μ.rowLen i = x⟩ := hx
  /-
    case intro.intro
    μ : YoungDiagram
    i : Nat
    hi : Membership.mem (List.range (μ.colLen 0)) i
    ⊢ LT.lt 0 (μ.rowLen i)
  -/
  rwa [List.mem_range, ← mem_iff_lt_colLen, mem_iff_lt_rowLen] at hi
  /-
    🎉 no goals
  -/


/-- The cells making up a `YoungDiagram` from a list of row lengths -/
protected def cellsOfRowLens : List ℕ → Finset (ℕ × ℕ)
  | [] => ∅
  | w::ws =>
    ({0} : Finset ℕ) ×ˢ Finset.range w ∪
      (YoungDiagram.cellsOfRowLens ws).map
        (Embedding.prodMap ⟨_, Nat.succ_injective⟩ (Embedding.refl ℕ))


protected theorem mem_cellsOfRowLens {w : List ℕ} {c : ℕ × ℕ} :
    c ∈ YoungDiagram.cellsOfRowLens w ↔ ∃ h : c.fst < w.length, c.snd < w[c.fst] := by
  /-
    w : List Nat
    c : Prod Nat Nat
    ⊢ Iff (Membership.mem (YoungDiagram.cellsOfRowLens w) c) (Exists fun h => LT.l …
  -/
  induction' w with w_hd w_tl w_ih generalizing c <;> rw [YoungDiagram.cellsOfRowLens]
    /-
      case nil
      c : Prod Nat Nat
      ⊢ Iff (Membership.mem EmptyCollection.emptyCollection c) (Exists fun h => LT.l …
    -/
  · simp [YoungDiagram.cellsOfRowLens]
    /-
      🎉 no goals
    -/
    /-
      case cons
      w_hd : Nat
      w_tl : List Nat
      w_ih : ∀ {c : Prod Nat Nat}, Iff (Membership.mem (YoungDiagram.cellsOfRowLens  …
      c : Prod Nat Nat
      ⊢ Iff (Membership.mem (Union.union (SProd.sprod (Singleton.singleton 0) (Finse …
    -/
  · rcases c with ⟨⟨_, _⟩, _⟩
      /-
        case cons.mk.zero
        w_hd : Nat
        w_tl : List Nat
        w_ih : ∀ {c : Prod Nat Nat}, Iff (Membership.mem (YoungDiagram.cellsOfRowLens  …
        snd✝ : Nat
        ⊢ Iff (Membership.mem (Union.union (SProd.sprod (Singleton.singleton 0) (Finse …
      -/
    · simp
      /-
        🎉 no goals
      -/
    -- Porting note: was `simpa`
      /-
        case cons.mk.succ
        w_hd : Nat
        w_tl : List Nat
        w_ih : ∀ {c : Prod Nat Nat}, Iff (Membership.mem (YoungDiagram.cellsOfRowLens  …
        snd✝ n✝ : Nat
        ⊢ Iff (Membership.mem (Union.union (SProd.sprod (Singleton.singleton 0) (Finse …
      -/
    · simp [w_ih, -Finset.singleton_product, Nat.succ_lt_succ_iff]
      /-
        🎉 no goals
      -/


/-- Young diagram from a sorted list -/
def ofRowLens (w : List ℕ) (hw : w.Sorted (· ≥ ·)) : YoungDiagram where
  cells := YoungDiagram.cellsOfRowLens w
  isLowerSet := by
    /-
      w : List Nat
      hw : List.Sorted (fun x1 x2 => GE.ge x1 x2) w
      ⊢ IsLowerSet ↑(YoungDiagram.cellsOfRowLens w)
    -/
    rintro ⟨i2, j2⟩ ⟨i1, j1⟩ ⟨hi : i1 ≤ i2, hj : j1 ≤ j2⟩ hcell
    /-
      case mk.mk.intro
      w : List Nat
      hw : List.Sorted (fun x1 x2 => GE.ge x1 x2) w
      i2 j2 i1 j1 : Nat
      hi : LE.le i1 i2
      hj : LE.le j1 j2
      hcell : Membership.mem ↑(YoungDiagram.cellsOfRowLens w) { fst := i2, snd := j2 }
      ⊢ Membership.mem ↑(YoungDiagram.cellsOfRowLens w) { fst := i1, snd := j1 }
    -/
    rw [Finset.mem_coe, YoungDiagram.mem_cellsOfRowLens] at hcell ⊢
    /-
      case mk.mk.intro
      w : List Nat
      hw : List.Sorted (fun x1 x2 => GE.ge x1 x2) w
      i2 j2 i1 j1 : Nat
      hi : LE.le i1 i2
      hj : LE.le j1 j2
      hcell : Exists fun h => LT.lt { fst := i2, snd := j2 }.2 (GetElem.getElem w {  …
      ⊢ Exists fun h => LT.lt { fst := i1, snd := j1 }.2 (GetElem.getElem w { fst := …
    -/
    obtain ⟨h1, h2⟩ := hcell
    /-
      case mk.mk.intro.intro
      w : List Nat
      hw : List.Sorted (fun x1 x2 => GE.ge x1 x2) w
      i2 j2 i1 j1 : Nat
      hi : LE.le i1 i2
      hj : LE.le j1 j2
      h1 : LT.lt { fst := i2, snd := j2 }.1 w.length
      h2 : LT.lt { fst := i2, snd := j2 }.2 (GetElem.getElem w { fst := i2, snd := j …
      ⊢ Exists fun h => LT.lt { fst := i1, snd := j1 }.2 (GetElem.getElem w { fst := …
    -/
    refine ⟨hi.trans_lt h1, ?_⟩
    calc
      j1 ≤ j2 := hj
      _ < w[i2]  := h2
      _ ≤ w[i1] := by
        obtain rfl | h := eq_or_lt_of_le hi
        · rfl
        · exact List.pairwise_iff_get.mp hw _ _ h


theorem mem_ofRowLens {w : List ℕ} {hw : w.Sorted (· ≥ ·)} {c : ℕ × ℕ} :
    c ∈ ofRowLens w hw ↔ ∃ h : c.fst < w.length, c.snd < w[c.fst] :=
  YoungDiagram.mem_cellsOfRowLens


/-- The number of rows in `ofRowLens w hw` is the length of `w` -/
theorem rowLens_length_ofRowLens {w : List ℕ} {hw : w.Sorted (· ≥ ·)} (hpos : ∀ x ∈ w, 0 < x) :
    (ofRowLens w hw).rowLens.length = w.length := by
  simp only [length_rowLens, colLen, Nat.find_eq_iff, mem_cells, mem_ofRowLens,
    lt_self_iff_false, IsEmpty.exists_iff, Classical.not_not]
  /-
    w : List Nat
    hw : List.Sorted (fun x1 x2 => GE.ge x1 x2) w
    hpos : ∀ (x : Nat), Membership.mem w x → LT.lt 0 x
    ⊢ And (Not False) (∀ (n : Nat), LT.lt n w.length → Exists fun h => LT.lt 0 (Ge …
  -/
  exact ⟨not_false, fun n hn => ⟨hn, hpos _ (List.getElem_mem hn)⟩⟩
  /-
    🎉 no goals
  -/


/-- The length of the `i`th row in `ofRowLens w hw` is the `i`th entry of `w` -/
theorem rowLen_ofRowLens {w : List ℕ} {hw : w.Sorted (· ≥ ·)} (i : Fin w.length) :
    (ofRowLens w hw).rowLen i = w[i] := by
  /-
    w : List Nat
    hw : List.Sorted (fun x1 x2 => GE.ge x1 x2) w
    i : Fin w.length
    ⊢ Eq ((YoungDiagram.ofRowLens w hw).rowLen ↑i) (GetElem.getElem w i ⋯)
  -/
  simp [rowLen, Nat.find_eq_iff, mem_ofRowLens]
  /-
    🎉 no goals
  -/


/-- The left_inv direction of the equivalence -/
theorem ofRowLens_to_rowLens_eq_self {μ : YoungDiagram} : ofRowLens _ (rowLens_sorted μ) = μ := by
  /-
    μ : YoungDiagram
    ⊢ Eq (YoungDiagram.ofRowLens μ.rowLens ⋯) μ
  -/
  ext ⟨i, j⟩
  /-
    case cells.h.mk
    μ : YoungDiagram
    i j : Nat
    ⊢ Iff (Membership.mem (YoungDiagram.ofRowLens μ.rowLens ⋯).cells { fst := i, s …
  -/
  simp only [mem_cells, mem_ofRowLens, length_rowLens, get_rowLens]
  /-
    case cells.h.mk
    μ : YoungDiagram
    i j : Nat
    ⊢ Iff (Exists fun h => LT.lt j (μ.rowLen i)) (Membership.mem μ { fst := i, snd …
  -/
  simpa [← mem_iff_lt_colLen, mem_iff_lt_rowLen] using j.zero_le.trans_lt
  /-
    🎉 no goals
  -/


/-- The right_inv direction of the equivalence -/
theorem rowLens_ofRowLens_eq_self {w : List ℕ} {hw : w.Sorted (· ≥ ·)} (hpos : ∀ x ∈ w, 0 < x) :
    (ofRowLens w hw).rowLens = w :=
  List.ext_get (rowLens_length_ofRowLens hpos) fun i h₁ h₂ =>
    (get_rowLens (h := h₁)).trans <| rowLen_ofRowLens ⟨i, h₂⟩


/-- Equivalence between Young diagrams and weakly decreasing lists of positive natural numbers.
A Young diagram `μ` is equivalent to a list of row lengths. -/
@[simps]
def equivListRowLens : YoungDiagram ≃ { w : List ℕ // w.Sorted (· ≥ ·) ∧ ∀ x ∈ w, 0 < x } where
  toFun μ := ⟨μ.rowLens, μ.rowLens_sorted, μ.pos_of_mem_rowLens⟩
  invFun ww := ofRowLens ww.1 ww.2.1
  left_inv _ := ofRowLens_to_rowLens_eq_self
  right_inv := fun ⟨_, hw⟩ => Subtype.mk_eq_mk.mpr (rowLens_ofRowLens_eq_self hw.2)


