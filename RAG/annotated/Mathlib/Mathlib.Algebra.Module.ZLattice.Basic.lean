                                                                   /-
                                                                     E : Type u_1
                                                                     ι : Type u_2
                                                                     K : Type u_3
                                                                     inst✝² : NormedLinearOrderedField K
                                                                     inst✝¹ : NormedAddCommGroup E
                                                                     inst✝ : NormedSpace K E
                                                                     b : Basis ι K E
                                                                     ⊢ Eq (Submodule.span K ↑(Submodule.span Int (Set.range ⇑b))) Top.top
                                                                   -/
theorem span_top : span K (span ℤ (Set.range b) : Set E) = ⊤ := by simp [span_span_of_tower]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem map {F : Type*} [NormedAddCommGroup F] [NormedSpace K F] (f : E ≃ₗ[K] F) :
    Submodule.map (f.restrictScalars ℤ) (span ℤ (Set.range b)) = span ℤ (Set.range (b.map f)) := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    F : Type u_4
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace K F
    f : LinearEquiv (RingHom.id K) E F
    ⊢ Eq (Submodule.map (LinearEquiv.restrictScalars Int f) (Submodule.span Int (S …
  -/
  simp_rw [Submodule.map_span, LinearEquiv.restrictScalars_apply, Basis.coe_map, Set.range_comp]
  /-
    🎉 no goals
  -/


open scoped Pointwise in
theorem smul {c : K} (hc : c ≠ 0) :
    c • span ℤ (Set.range b) = span ℤ (Set.range (b.isUnitSMul (fun _ ↦ hc.isUnit))) := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝² : NormedLinearOrderedField K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    b : Basis ι K E
    c : K
    hc : Ne c 0
    ⊢ Eq (HSMul.hSMul c (Submodule.span Int (Set.range ⇑b))) (Submodule.span Int ( …
  -/
  rw [smul_span, Set.smul_set_range]
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝² : NormedLinearOrderedField K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    b : Basis ι K E
    c : K
    hc : Ne c 0
    ⊢ Eq (Submodule.span Int (Set.range fun i => HSMul.hSMul c (b i))) (Submodule. …
  -/
  congr!
  /-
    case h.e'_6.h.e'_3.h
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝² : NormedLinearOrderedField K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    b : Basis ι K E
    c : K
    hc : Ne c 0
    x✝ : ι
    ⊢ Eq (HSMul.hSMul c (b x✝)) ((b.isUnitSMul ⋯) x✝)
  -/
  rw [Basis.isUnitSMul_apply]
  /-
    🎉 no goals
  -/


/-- The fundamental domain of the ℤ-lattice spanned by `b`. See `ZSpan.isAddFundamentalDomain`
for the proof that it is a fundamental domain. -/
def fundamentalDomain : Set E := {m | ∀ i, b.repr m i ∈ Set.Ico (0 : K) 1}


@[simp]
theorem mem_fundamentalDomain {m : E} :
    m ∈ fundamentalDomain b ↔ ∀ i, b.repr m i ∈ Set.Ico (0 : K) 1 := Iff.rfl


theorem map_fundamentalDomain {F : Type*} [NormedAddCommGroup F] [NormedSpace K F] (f : E ≃ₗ[K] F) :
    f '' (fundamentalDomain b) = fundamentalDomain (b.map f) := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    F : Type u_4
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace K F
    f : LinearEquiv (RingHom.id K) E F
    ⊢ Eq (Set.image (⇑f) (ZSpan.fundamentalDomain b)) (ZSpan.fundamentalDomain (b. …
  -/
  ext x
  rw [mem_fundamentalDomain, Basis.map_repr, LinearEquiv.trans_apply, ← mem_fundamentalDomain,
    show f.symm x = f.toEquiv.symm x by rfl, ← Set.mem_image_equiv]
  /-
    case h
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    F : Type u_4
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace K F
    f : LinearEquiv (RingHom.id K) E F
    x : F
    ⊢ Iff (Membership.mem (Set.image (⇑f) (ZSpan.fundamentalDomain b)) x) (Members …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem fundamentalDomain_reindex {ι' : Type*} (e : ι ≃ ι') :
    fundamentalDomain (b.reindex e) = fundamentalDomain b := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝² : NormedLinearOrderedField K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    b : Basis ι K E
    ι' : Type u_4
    e : Equiv ι ι'
    ⊢ Eq (ZSpan.fundamentalDomain (b.reindex e)) (ZSpan.fundamentalDomain b)
  -/
  ext
  /-
    case h
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝² : NormedLinearOrderedField K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    b : Basis ι K E
    ι' : Type u_4
    e : Equiv ι ι'
    x✝ : E
    ⊢ Iff (Membership.mem (ZSpan.fundamentalDomain (b.reindex e)) x✝) (Membership. …
  -/
  simp_rw [mem_fundamentalDomain, Basis.repr_reindex_apply]
  /-
    case h
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝² : NormedLinearOrderedField K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    b : Basis ι K E
    ι' : Type u_4
    e : Equiv ι ι'
    x✝ : E
    ⊢ Iff (∀ (i : ι'), Membership.mem (Set.Ico 0 1) ((b.repr x✝) (e.symm i))) (∀ ( …
  -/
  rw [Equiv.forall_congr' e]
  /-
    case h
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝² : NormedLinearOrderedField K
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace K E
    b : Basis ι K E
    ι' : Type u_4
    e : Equiv ι ι'
    x✝ : E
    ⊢ ∀ (b_1 : ι'), Iff (Membership.mem (Set.Ico 0 1) ((b.repr x✝) (e.symm b_1)))  …
  -/
  simp_rw [implies_true]
  /-
    🎉 no goals
  -/


lemma fundamentalDomain_pi_basisFun [Fintype ι] :
    fundamentalDomain (Pi.basisFun ℝ ι) = Set.pi Set.univ fun _ : ι ↦ Set.Ico (0 : ℝ) 1 := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    ⊢ Eq (ZSpan.fundamentalDomain (Pi.basisFun Real ι)) (Set.univ.pi fun x => Set. …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- The map that sends a vector of `E` to the element of the ℤ-lattice spanned by `b` obtained
by rounding down its coordinates on the basis `b`. -/
def floor (m : E) : span ℤ (Set.range b) := ∑ i, ⌊b.repr m i⌋ • b.restrictScalars ℤ i


/-- The map that sends a vector of `E` to the element of the ℤ-lattice spanned by `b` obtained
by rounding up its coordinates on the basis `b`. -/
def ceil (m : E) : span ℤ (Set.range b) := ∑ i, ⌈b.repr m i⌉ • b.restrictScalars ℤ i


@[simp]
theorem repr_floor_apply (m : E) (i : ι) : b.repr (floor b m) i = ⌊b.repr m i⌋ := by
  classical simp only [floor, ← Int.cast_smul_eq_zsmul K, b.repr.map_smul, Finsupp.single_apply,
    Finset.sum_apply', Basis.repr_self, Finsupp.smul_single', mul_one, Finset.sum_ite_eq', coe_sum,
    Finset.mem_univ, if_true, coe_smul_of_tower, Basis.restrictScalars_apply, map_sum]


@[simp]
theorem repr_ceil_apply (m : E) (i : ι) : b.repr (ceil b m) i = ⌈b.repr m i⌉ := by
  classical simp only [ceil, ← Int.cast_smul_eq_zsmul K, b.repr.map_smul, Finsupp.single_apply,
    Finset.sum_apply', Basis.repr_self, Finsupp.smul_single', mul_one, Finset.sum_ite_eq', coe_sum,
    Finset.mem_univ, if_true, coe_smul_of_tower, Basis.restrictScalars_apply, map_sum]


@[simp]
theorem floor_eq_self_of_mem (m : E) (h : m ∈ span ℤ (Set.range b)) : (floor b m : E) = m := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    ⊢ Eq (↑(ZSpan.floor b m)) m
  -/
  apply b.ext_elem
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    ⊢ ∀ (i : ι), Eq ((b.repr ↑(ZSpan.floor b m)) i) ((b.repr m) i)
  -/
  simp_rw [repr_floor_apply b]
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    ⊢ ∀ (i : ι), Eq (↑(Int.floor ((b.repr m) i))) ((b.repr m) i)
  -/
  intro i
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    i : ι
    ⊢ Eq (↑(Int.floor ((b.repr m) i))) ((b.repr m) i)
  -/
  obtain ⟨z, hz⟩ := (b.mem_span_iff_repr_mem ℤ _).mp h i
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    i : ι
    z : Int
    hz : Eq ((algebraMap Int K) z) ((b.repr m) i)
    ⊢ Eq (↑(Int.floor ((b.repr m) i))) ((b.repr m) i)
  -/
  rw [← hz]
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    i : ι
    z : Int
    hz : Eq ((algebraMap Int K) z) ((b.repr m) i)
    ⊢ Eq (↑(Int.floor ((algebraMap Int K) z))) ((algebraMap Int K) z)
  -/
  exact congr_arg (Int.cast : ℤ → K) (Int.floor_intCast z)
  /-
    🎉 no goals
  -/


@[simp]
theorem ceil_eq_self_of_mem (m : E) (h : m ∈ span ℤ (Set.range b)) : (ceil b m : E) = m := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    ⊢ Eq (↑(ZSpan.ceil b m)) m
  -/
  apply b.ext_elem
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    ⊢ ∀ (i : ι), Eq ((b.repr ↑(ZSpan.ceil b m)) i) ((b.repr m) i)
  -/
  simp_rw [repr_ceil_apply b]
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    ⊢ ∀ (i : ι), Eq (↑(Int.ceil ((b.repr m) i))) ((b.repr m) i)
  -/
  intro i
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    i : ι
    ⊢ Eq (↑(Int.ceil ((b.repr m) i))) ((b.repr m) i)
  -/
  obtain ⟨z, hz⟩ := (b.mem_span_iff_repr_mem ℤ _).mp h i
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    i : ι
    z : Int
    hz : Eq ((algebraMap Int K) z) ((b.repr m) i)
    ⊢ Eq (↑(Int.ceil ((b.repr m) i))) ((b.repr m) i)
  -/
  rw [← hz]
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    h : Membership.mem (Submodule.span Int (Set.range ⇑b)) m
    i : ι
    z : Int
    hz : Eq ((algebraMap Int K) z) ((b.repr m) i)
    ⊢ Eq (↑(Int.ceil ((algebraMap Int K) z))) ((algebraMap Int K) z)
  -/
  exact congr_arg (Int.cast : ℤ → K) (Int.ceil_intCast z)
  /-
    🎉 no goals
  -/


/-- The map that sends a vector `E` to the `fundamentalDomain` of the lattice,
see `ZSpan.fract_mem_fundamentalDomain`, and `fractRestrict` for the map with the codomain
restricted to `fundamentalDomain`. -/
def fract (m : E) : E := m - floor b m


theorem fract_apply (m : E) : fract b m = m - floor b m := rfl


@[simp]
theorem repr_fract_apply (m : E) (i : ι) : b.repr (fract b m) i = Int.fract (b.repr m i) := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    m : E
    i : ι
    ⊢ Eq ((b.repr (ZSpan.fract b m)) i) (Int.fract ((b.repr m) i))
  -/
  rw [fract, map_sub, Finsupp.coe_sub, Pi.sub_apply, repr_floor_apply, Int.fract]
  /-
    🎉 no goals
  -/


@[simp]
theorem fract_fract (m : E) : fract b (fract b m) = fract b m :=
                               /-
                                 E : Type u_1
                                 ι : Type u_2
                                 K : Type u_3
                                 inst✝⁴ : NormedLinearOrderedField K
                                 inst✝³ : NormedAddCommGroup E
                                 inst✝² : NormedSpace K E
                                 b : Basis ι K E
                                 inst✝¹ : FloorRing K
                                 inst✝ : Fintype ι
                                 m : E
                                 x✝ : ι
                                 ⊢ Eq ((b.repr (ZSpan.fract b (ZSpan.fract b m))) x✝) ((b.repr (ZSpan.fract b m …
                               -/
  Basis.ext_elem b fun _ => by classical simp only [repr_fract_apply, Int.fract_fract]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem fract_zSpan_add (m : E) {v : E} (h : v ∈ span ℤ (Set.range b)) :
    fract b (v + m) = fract b m := by
  classical
  refine (Basis.ext_elem_iff b).mpr fun i => ?_
  simp_rw [repr_fract_apply, Int.fract_eq_fract]
  use (b.restrictScalars ℤ).repr ⟨v, h⟩ i
  rw [map_add, Finsupp.coe_add, Pi.add_apply, add_tsub_cancel_right,
    ← eq_intCast (algebraMap ℤ K) _, Basis.restrictScalars_repr_apply, coe_mk]


@[simp]
theorem fract_add_ZSpan (m : E) {v : E} (h : v ∈ span ℤ (Set.range b)) :
                                      /-
                                        E : Type u_1
                                        ι : Type u_2
                                        K : Type u_3
                                        inst✝⁴ : NormedLinearOrderedField K
                                        inst✝³ : NormedAddCommGroup E
                                        inst✝² : NormedSpace K E
                                        b : Basis ι K E
                                        inst✝¹ : FloorRing K
                                        inst✝ : Fintype ι
                                        m v : E
                                        h : Membership.mem (Submodule.span Int (Set.range ⇑b)) v
                                        ⊢ Eq (ZSpan.fract b (HAdd.hAdd m v)) (ZSpan.fract b m)
                                      -/
    fract b (m + v) = fract b m := by rw [add_comm, fract_zSpan_add b m h]
                                      /-
                                        🎉 no goals
                                      -/


theorem fract_eq_self {x : E} : fract b x = x ↔ x ∈ fundamentalDomain b := by
  classical simp only [Basis.ext_elem_iff b, repr_fract_apply, Int.fract_eq_self,
    mem_fundamentalDomain, Set.mem_Ico]


theorem fract_mem_fundamentalDomain (x : E) : fract b x ∈ fundamentalDomain b :=
  fract_eq_self.mp (fract_fract b _)


/-- The map `fract` with codomain restricted to `fundamentalDomain`. -/
def fractRestrict (x : E) : fundamentalDomain b := ⟨fract b x, fract_mem_fundamentalDomain b x⟩


theorem fractRestrict_surjective : Function.Surjective (fractRestrict b) :=
  fun x => ⟨↑x, Subtype.eq (fract_eq_self.mpr (Subtype.mem x))⟩


@[simp]
theorem fractRestrict_apply (x : E) : (fractRestrict b x : E) = fract b x := rfl


theorem fract_eq_fract (m n : E) : fract b m = fract b n ↔ -m + n ∈ span ℤ (Set.range b) := by
  classical
  rw [eq_comm, Basis.ext_elem_iff b]
  simp_rw [repr_fract_apply, Int.fract_eq_fract, eq_comm, Basis.mem_span_iff_repr_mem,
    sub_eq_neg_add, map_add, map_neg, Finsupp.coe_add, Finsupp.coe_neg, Pi.add_apply,
    Pi.neg_apply, ← eq_intCast (algebraMap ℤ K) _, Set.mem_range]


theorem norm_fract_le [HasSolidNorm K] (m : E) : ‖fract b m‖ ≤ ∑ i, ‖b i‖ := by
  classical
  calc
    ‖fract b m‖ = ‖∑ i, b.repr (fract b m) i • b i‖ := by rw [b.sum_repr]
    _ = ‖∑ i, Int.fract (b.repr m i) • b i‖ := by simp_rw [repr_fract_apply]
    _ ≤ ∑ i, ‖Int.fract (b.repr m i) • b i‖ := norm_sum_le _ _
    _ = ∑ i, ‖Int.fract (b.repr m i)‖ * ‖b i‖ := by simp_rw [norm_smul]
    _ ≤ ∑ i, ‖b i‖ := Finset.sum_le_sum fun i _ => ?_
  suffices ‖Int.fract ((b.repr m) i)‖ ≤ 1 by
    convert mul_le_mul_of_nonneg_right this (norm_nonneg _ : 0 ≤ ‖b i‖)
    exact (one_mul _).symm
  rw [(norm_one.symm : 1 = ‖(1 : K)‖)]
  apply norm_le_norm_of_abs_le_abs
  rw [abs_one, Int.abs_fract]
  exact le_of_lt (Int.fract_lt_one _)


@[simp]
theorem coe_floor_self (k : K) : (floor (Basis.singleton ι K) k : K) = ⌊k⌋ :=
  Basis.ext_elem (Basis.singleton ι K) fun _ => by
    /-
      ι : Type u_2
      K : Type u_3
      inst✝³ : NormedLinearOrderedField K
      inst✝² : FloorRing K
      inst✝¹ : Fintype ι
      inst✝ : Unique ι
      k : K
      x✝ : ι
      ⊢ Eq (((Basis.singleton ι K).repr ↑(ZSpan.floor (Basis.singleton ι K) k)) x✝)  …
    -/
    rw [repr_floor_apply, Basis.singleton_repr, Basis.singleton_repr]
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_fract_self (k : K) : (fract (Basis.singleton ι K) k : K) = Int.fract k :=
  Basis.ext_elem (Basis.singleton ι K) fun _ => by
    /-
      ι : Type u_2
      K : Type u_3
      inst✝³ : NormedLinearOrderedField K
      inst✝² : FloorRing K
      inst✝¹ : Fintype ι
      inst✝ : Unique ι
      k : K
      x✝ : ι
      ⊢ Eq (((Basis.singleton ι K).repr (ZSpan.fract (Basis.singleton ι K) k)) x✝) ( …
    -/
    rw [repr_fract_apply, Basis.singleton_repr, Basis.singleton_repr]
    /-
      🎉 no goals
    -/


theorem fundamentalDomain_isBounded [Finite ι] [HasSolidNorm K] :
    IsBounded (fundamentalDomain b) := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁵ : NormedLinearOrderedField K
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    b : Basis ι K E
    inst✝² : FloorRing K
    inst✝¹ : Finite ι
    inst✝ : HasSolidNorm K
    ⊢ Bornology.IsBounded (ZSpan.fundamentalDomain b)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁵ : NormedLinearOrderedField K
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    b : Basis ι K E
    inst✝² : FloorRing K
    inst✝¹ : Finite ι
    inst✝ : HasSolidNorm K
    val✝ : Fintype ι
    ⊢ Bornology.IsBounded (ZSpan.fundamentalDomain b)
  -/
  refine isBounded_iff_forall_norm_le.2 ⟨∑ j, ‖b j‖, fun x hx ↦ ?_⟩
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁵ : NormedLinearOrderedField K
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    b : Basis ι K E
    inst✝² : FloorRing K
    inst✝¹ : Finite ι
    inst✝ : HasSolidNorm K
    val✝ : Fintype ι
    x : E
    hx : Membership.mem (ZSpan.fundamentalDomain b) x
    ⊢ LE.le (Norm.norm x) (Finset.univ.sum fun j => Norm.norm (b j))
  -/
  rw [← fract_eq_self.mpr hx]
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁵ : NormedLinearOrderedField K
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    b : Basis ι K E
    inst✝² : FloorRing K
    inst✝¹ : Finite ι
    inst✝ : HasSolidNorm K
    val✝ : Fintype ι
    x : E
    hx : Membership.mem (ZSpan.fundamentalDomain b) x
    ⊢ LE.le (Norm.norm (ZSpan.fract b x)) (Finset.univ.sum fun j => Norm.norm (b j))
  -/
  apply norm_fract_le
  /-
    🎉 no goals
  -/


theorem vadd_mem_fundamentalDomain [Fintype ι] (y : span ℤ (Set.range b)) (x : E) :
    y +ᵥ x ∈ fundamentalDomain b ↔ y = -floor b x := by
  rw [Subtype.ext_iff, ← add_right_inj x, NegMemClass.coe_neg, ← sub_eq_add_neg, ← fract_apply,
    ← fract_zSpan_add b _ (Subtype.mem y), add_comm, ← vadd_eq_add, ← vadd_def, eq_comm, ←
    fract_eq_self]


theorem exist_unique_vadd_mem_fundamentalDomain [Finite ι] (x : E) :
    ∃! v : span ℤ (Set.range b), v +ᵥ x ∈ fundamentalDomain b := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Finite ι
    x : E
    ⊢ ExistsUnique fun v => Membership.mem (ZSpan.fundamentalDomain b) (HVAdd.hVAd …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Finite ι
    x : E
    val✝ : Fintype ι
    ⊢ ExistsUnique fun v => Membership.mem (ZSpan.fundamentalDomain b) (HVAdd.hVAd …
  -/
  refine ⟨-floor b x, ?_, fun y h => ?_⟩
    /-
      case intro.refine_1
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Finite ι
      x : E
      val✝ : Fintype ι
      ⊢ (fun v => Membership.mem (ZSpan.fundamentalDomain b) (HVAdd.hVAdd v x)) (Neg …
    -/
  · exact (vadd_mem_fundamentalDomain b (-floor b x) x).mpr rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Finite ι
      x : E
      val✝ : Fintype ι
      y : Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑b)) x
      h : (fun v => Membership.mem (ZSpan.fundamentalDomain b) (HVAdd.hVAdd v x)) y
      ⊢ Eq y (Neg.neg (ZSpan.floor b x))
    -/
  · exact (vadd_mem_fundamentalDomain b y x).mp h
    /-
      🎉 no goals
    -/


/-- The map `ZSpan.fractRestrict` defines an equiv map between `E ⧸ span ℤ (Set.range b)`
and `ZSpan.fundamentalDomain b`. -/
def quotientEquiv [Fintype ι] :
    E ⧸ span ℤ (Set.range b) ≃ (fundamentalDomain b) := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    ⊢ Equiv (HasQuotient.Quotient E (Submodule.span Int (Set.range ⇑b))) ↑(ZSpan.f …
  -/
  refine Equiv.ofBijective ?_ ⟨fun x y => ?_, fun x => ?_⟩
    /-
      case refine_1
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      ⊢ HasQuotient.Quotient E (Submodule.span Int (Set.range ⇑b)) → ↑(ZSpan.fundame …
    -/
  · refine fun q => Quotient.liftOn q (fractRestrict b) (fun _ _ h => ?_)
    /-
      case refine_1
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      q : HasQuotient.Quotient E (Submodule.span Int (Set.range ⇑b))
      x✝¹ x✝ : E
      h : HasEquiv.Equiv x✝¹ x✝
      ⊢ Eq (ZSpan.fractRestrict b x✝¹) (ZSpan.fractRestrict b x✝)
    -/
    rw [Subtype.mk.injEq, fractRestrict_apply, fractRestrict_apply, fract_eq_fract]
    /-
      case refine_1
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      q : HasQuotient.Quotient E (Submodule.span Int (Set.range ⇑b))
      x✝¹ x✝ : E
      h : HasEquiv.Equiv x✝¹ x✝
      ⊢ Membership.mem (Submodule.span Int (Set.range ⇑b)) (HAdd.hAdd (Neg.neg x✝¹)  …
    -/
    exact QuotientAddGroup.leftRel_apply.mp h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      x y : HasQuotient.Quotient E (Submodule.span Int (Set.range ⇑b))
      ⊢ Eq (Quotient.liftOn x (ZSpan.fractRestrict b) ⋯) (Quotient.liftOn y (ZSpan.f …
    -/
  · refine Quotient.inductionOn₂ x y (fun _ _ hxy => ?_)
    rw [Quotient.liftOn_mk (s := quotientRel (span ℤ (Set.range b))), fractRestrict,
      Quotient.liftOn_mk (s := quotientRel (span ℤ (Set.range b))),  fractRestrict,
      Subtype.mk.injEq] at hxy
    /-
      case refine_2
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      x y : HasQuotient.Quotient E (Submodule.span Int (Set.range ⇑b))
      x✝¹ x✝ : E
      hxy : Eq (ZSpan.fract b x✝¹) (ZSpan.fract b x✝)
      ⊢ Eq (Quotient.mk (Submodule.span Int (Set.range ⇑b)).quotientRel x✝¹) (Quotie …
    -/
    apply Quotient.sound'
    /-
      case refine_2.a
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      x y : HasQuotient.Quotient E (Submodule.span Int (Set.range ⇑b))
      x✝¹ x✝ : E
      hxy : Eq (ZSpan.fract b x✝¹) (ZSpan.fract b x✝)
      ⊢ (Submodule.span Int (Set.range ⇑b)).quotientRel x✝¹ x✝
    -/
    rwa [QuotientAddGroup.leftRel_apply, mem_toAddSubgroup, ← fract_eq_fract]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      x : ↑(ZSpan.fundamentalDomain b)
      ⊢ Exists fun a => Eq (Quotient.liftOn a (ZSpan.fractRestrict b) ⋯) x
    -/
  · obtain ⟨a, rfl⟩ := fractRestrict_surjective b x
    /-
      case refine_3.intro
      E : Type u_1
      ι : Type u_2
      K : Type u_3
      inst✝⁴ : NormedLinearOrderedField K
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace K E
      b : Basis ι K E
      inst✝¹ : FloorRing K
      inst✝ : Fintype ι
      a : E
      ⊢ Exists fun a_1 => Eq (Quotient.liftOn a_1 (ZSpan.fractRestrict b) ⋯) (ZSpan. …
    -/
    exact ⟨Quotient.mk'' a, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem quotientEquiv_apply_mk [Fintype ι] (x : E) :
    quotientEquiv b (Submodule.Quotient.mk x) = fractRestrict b x := rfl


@[simp]
theorem quotientEquiv.symm_apply [Fintype ι] (x : fundamentalDomain b) :
    (quotientEquiv b).symm x = Submodule.Quotient.mk ↑x := by
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    x : ↑(ZSpan.fundamentalDomain b)
    ⊢ Eq ((ZSpan.quotientEquiv b).symm x) (Submodule.Quotient.mk ↑x)
  -/
  rw [Equiv.symm_apply_eq, quotientEquiv_apply_mk b ↑x, Subtype.ext_iff, fractRestrict_apply]
  /-
    E : Type u_1
    ι : Type u_2
    K : Type u_3
    inst✝⁴ : NormedLinearOrderedField K
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    b : Basis ι K E
    inst✝¹ : FloorRing K
    inst✝ : Fintype ι
    x : ↑(ZSpan.fundamentalDomain b)
    ⊢ Eq (↑x) (ZSpan.fract b ↑x)
  -/
  exact (fract_eq_self.mpr x.prop).symm
  /-
    🎉 no goals
  -/


theorem discreteTopology_pi_basisFun [Finite ι] :
    DiscreteTopology (span ℤ (Set.range (Pi.basisFun ℝ ι))) := by
  /-
    ι : Type u_2
    inst✝ : Finite ι
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (Submodule.span Int (Set.r …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    val✝ : Fintype ι
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (Submodule.span Int (Set.r …
  -/
  refine discreteTopology_iff_isOpen_singleton_zero.mpr ⟨Metric.ball 0 1, Metric.isOpen_ball, ?_⟩
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    val✝ : Fintype ι
    ⊢ Eq (Set.preimage Subtype.val (Metric.ball 0 1)) (Singleton.singleton 0)
  -/
  ext x
  /-
    case intro.h
    ι : Type u_2
    inst✝ : Finite ι
    val✝ : Fintype ι
    x : Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑(Pi.basisF …
    ⊢ Iff (Membership.mem (Set.preimage Subtype.val (Metric.ball 0 1)) x) (Members …
  -/
  rw [Set.mem_preimage, mem_ball_zero_iff, pi_norm_lt_iff zero_lt_one, Set.mem_singleton_iff]
  /-
    case intro.h
    ι : Type u_2
    inst✝ : Finite ι
    val✝ : Fintype ι
    x : Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑(Pi.basisF …
    ⊢ Iff (∀ (i : ι), LT.lt (Norm.norm (↑x i)) 1) (Eq x 0)
  -/
  simp_rw [← coe_eq_zero, funext_iff, Pi.zero_apply, Real.norm_eq_abs]
  /-
    case intro.h
    ι : Type u_2
    inst✝ : Finite ι
    val✝ : Fintype ι
    x : Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑(Pi.basisF …
    ⊢ Iff (∀ (i : ι), LT.lt (abs (↑x i)) 1) (∀ (x_1 : ι), Eq (↑x x_1) 0)
  -/
  refine forall_congr' (fun i => ?_)
  /-
    case intro.h
    ι : Type u_2
    inst✝ : Finite ι
    val✝ : Fintype ι
    x : Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑(Pi.basisF …
    i : ι
    ⊢ Iff (LT.lt (abs (↑x i)) 1) (Eq (↑x i) 0)
  -/
  rsuffices ⟨y, hy⟩ : ∃ (y : ℤ), (y : ℝ) = (x : ι → ℝ) i
    /-
      case intro.h.intro
      ι : Type u_2
      inst✝ : Finite ι
      val✝ : Fintype ι
      x : Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑(Pi.basisF …
      i : ι
      y : Int
      hy : Eq (↑y) (↑x i)
      ⊢ Iff (LT.lt (abs (↑x i)) 1) (Eq (↑x i) 0)
    -/
  · rw [← hy, ← Int.cast_abs, ← Int.cast_one,  Int.cast_lt, Int.abs_lt_one_iff, Int.cast_eq_zero]
    /-
      🎉 no goals
    -/
  /-
    ι : Type u_2
    inst✝ : Finite ι
    val✝ : Fintype ι
    x : Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑(Pi.basisF …
    i : ι
    ⊢ Exists fun y => Eq (↑y) (↑x i)
  -/
  exact ((Pi.basisFun ℝ ι).mem_span_iff_repr_mem ℤ x).mp (SetLike.coe_mem x) i
  /-
    🎉 no goals
  -/


theorem fundamentalDomain_subset_parallelepiped [Fintype ι] :
    fundamentalDomain b ⊆ parallelepiped b := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    b : Basis ι Real E
    inst✝ : Fintype ι
    ⊢ HasSubset.Subset (ZSpan.fundamentalDomain b) (parallelepiped ⇑b)
  -/
  rw [fundamentalDomain, parallelepiped_basis_eq, Set.setOf_subset_setOf]
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    b : Basis ι Real E
    inst✝ : Fintype ι
    ⊢ ∀ (a : E), (∀ (i : ι), Membership.mem (Set.Ico 0 1) ((b.repr a) i)) → ∀ (i : …
  -/
  exact fun _ h i ↦ Set.Ico_subset_Icc_self (h i)
  /-
    🎉 no goals
  -/


instance [Finite ι] : DiscreteTopology (span ℤ (Set.range b)) := by
  have h : Set.MapsTo b.equivFun (span ℤ (Set.range b)) (span ℤ (Set.range (Pi.basisFun ℝ ι))) := by
    intro _ hx
    rwa [SetLike.mem_coe, Basis.mem_span_iff_repr_mem] at hx ⊢
  /-
    E : Type u_1
    ι : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    b : Basis ι Real E
    inst✝ : Finite ι
    h : Set.MapsTo ⇑b.equivFun ↑(Submodule.span Int (Set.range ⇑b)) ↑(Submodule.sp …
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (Submodule.span Int (Set.r …
  -/
  convert DiscreteTopology.of_continuous_injective ((continuous_equivFun_basis b).restrict h) ?_
    /-
      case convert_1
      E : Type u_1
      ι : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      b : Basis ι Real E
      inst✝ : Finite ι
      h : Set.MapsTo ⇑b.equivFun ↑(Submodule.span Int (Set.range ⇑b)) ↑(Submodule.sp …
      ⊢ DiscreteTopology ↑↑(Submodule.span Int (Set.range ⇑(Pi.basisFun Real ι)))
    -/
  · exact discreteTopology_pi_basisFun
    /-
      🎉 no goals
    -/
    /-
      case convert_2
      E : Type u_1
      ι : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      b : Basis ι Real E
      inst✝ : Finite ι
      h : Set.MapsTo ⇑b.equivFun ↑(Submodule.span Int (Set.range ⇑b)) ↑(Submodule.sp …
      ⊢ Function.Injective (Set.MapsTo.restrict (⇑b.equivFun) (↑(Submodule.span Int  …
    -/
  · refine Subtype.map_injective _ (Basis.equivFun b).injective
    /-
      🎉 no goals
    -/


instance [Finite ι] : DiscreteTopology (span ℤ (Set.range b)).toAddSubgroup :=
  inferInstanceAs <| DiscreteTopology (span ℤ (Set.range b))


theorem setFinite_inter [ProperSpace E] [Finite ι] {s : Set E} (hs : Bornology.IsBounded s) :
    Set.Finite (s ∩ span ℤ (Set.range b)) := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    b : Basis ι Real E
    inst✝¹ : ProperSpace E
    inst✝ : Finite ι
    s : Set E
    hs : Bornology.IsBounded s
    ⊢ (Inter.inter s ↑(Submodule.span Int (Set.range ⇑b))).Finite
  -/
  have : DiscreteTopology (span ℤ (Set.range b)) := inferInstance
  /-
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    b : Basis ι Real E
    inst✝¹ : ProperSpace E
    inst✝ : Finite ι
    s : Set E
    hs : Bornology.IsBounded s
    this : DiscreteTopology (Subtype fun x => Membership.mem (Submodule.span Int ( …
    ⊢ (Inter.inter s ↑(Submodule.span Int (Set.range ⇑b))).Finite
  -/
  refine Metric.finite_isBounded_inter_isClosed hs ?_
  /-
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    b : Basis ι Real E
    inst✝¹ : ProperSpace E
    inst✝ : Finite ι
    s : Set E
    hs : Bornology.IsBounded s
    this : DiscreteTopology (Subtype fun x => Membership.mem (Submodule.span Int ( …
    ⊢ IsClosed ↑(Submodule.span Int (Set.range ⇑b))
  -/
  change IsClosed ((span ℤ (Set.range b)).toAddSubgroup : Set E)
  /-
    E : Type u_1
    ι : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    b : Basis ι Real E
    inst✝¹ : ProperSpace E
    inst✝ : Finite ι
    s : Set E
    hs : Bornology.IsBounded s
    this : DiscreteTopology (Subtype fun x => Membership.mem (Submodule.span Int ( …
    ⊢ IsClosed ↑(Submodule.span Int (Set.range ⇑b)).toAddSubgroup
  -/
  exact AddSubgroup.isClosed_of_discrete
  /-
    🎉 no goals
  -/


@[measurability]
theorem fundamentalDomain_measurableSet [MeasurableSpace E] [OpensMeasurableSpace E] [Finite ι] :
    MeasurableSet (fundamentalDomain b) := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    b : Basis ι Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : Finite ι
    ⊢ MeasurableSet (ZSpan.fundamentalDomain b)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    b : Basis ι Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : Finite ι
    val✝ : Fintype ι
    ⊢ MeasurableSet (ZSpan.fundamentalDomain b)
  -/
  haveI : FiniteDimensional ℝ E := FiniteDimensional.of_fintype_basis b
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    b : Basis ι Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : Finite ι
    val✝ : Fintype ι
    this : FiniteDimensional Real E
    ⊢ MeasurableSet (ZSpan.fundamentalDomain b)
  -/
  let D : Set (ι → ℝ) := Set.pi Set.univ fun _ : ι => Set.Ico (0 : ℝ) 1
  /-
    case intro
    E : Type u_1
    ι : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    b : Basis ι Real E
    inst✝² : MeasurableSpace E
    inst✝¹ : OpensMeasurableSpace E
    inst✝ : Finite ι
    val✝ : Fintype ι
    this : FiniteDimensional Real E
    D : Set (ι → Real) := Set.univ.pi fun x => Set.Ico 0 1
    ⊢ MeasurableSet (ZSpan.fundamentalDomain b)
  -/
  rw [(_ : fundamentalDomain b = b.equivFun.toLinearMap ⁻¹' D)]
    /-
      case intro
      E : Type u_1
      ι : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      b : Basis ι Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : OpensMeasurableSpace E
      inst✝ : Finite ι
      val✝ : Fintype ι
      this : FiniteDimensional Real E
      D : Set (ι → Real) := Set.univ.pi fun x => Set.Ico 0 1
      ⊢ MeasurableSet (Set.preimage (⇑↑b.equivFun) D)
    -/
  · refine measurableSet_preimage (LinearMap.continuous_of_finiteDimensional _).measurable ?_
    /-
      case intro
      E : Type u_1
      ι : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      b : Basis ι Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : OpensMeasurableSpace E
      inst✝ : Finite ι
      val✝ : Fintype ι
      this : FiniteDimensional Real E
      D : Set (ι → Real) := Set.univ.pi fun x => Set.Ico 0 1
      ⊢ MeasurableSet D
    -/
    exact MeasurableSet.pi Set.countable_univ fun _ _ => measurableSet_Ico
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      ι : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      b : Basis ι Real E
      inst✝² : MeasurableSpace E
      inst✝¹ : OpensMeasurableSpace E
      inst✝ : Finite ι
      val✝ : Fintype ι
      this : FiniteDimensional Real E
      D : Set (ι → Real) := Set.univ.pi fun x => Set.Ico 0 1
      ⊢ Eq (ZSpan.fundamentalDomain b) (Set.preimage (⇑↑b.equivFun) D)
    -/
  · ext
    simp only [D, fundamentalDomain, Set.mem_Ico, Set.mem_setOf_eq, LinearEquiv.coe_coe,
      Set.mem_preimage, Basis.equivFun_apply, Set.mem_pi, Set.mem_univ, forall_true_left]


/-- For a ℤ-lattice `Submodule.span ℤ (Set.range b)`, proves that the set defined
by `ZSpan.fundamentalDomain` is a fundamental domain. -/
protected theorem isAddFundamentalDomain [Finite ι] [MeasurableSpace E] [OpensMeasurableSpace E]
    (μ : Measure E) :
    IsAddFundamentalDomain (span ℤ (Set.range b)) (fundamentalDomain b) μ := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    b : Basis ι Real E
    inst✝² : Finite ι
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    μ : MeasureTheory.Measure E
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (Submo …
  -/
  cases nonempty_fintype ι
  exact IsAddFundamentalDomain.mk' (nullMeasurableSet (fundamentalDomain_measurableSet b))
    fun x => exist_unique_vadd_mem_fundamentalDomain b x


/-- A version of `ZSpan.isAddFundamentalDomain` for `AddSubgroup`. -/
protected theorem isAddFundamentalDomain' [Finite ι] [MeasurableSpace E] [OpensMeasurableSpace E]
    (μ : Measure E) :
    IsAddFundamentalDomain (span ℤ (Set.range b)).toAddSubgroup (fundamentalDomain b) μ :=
  ZSpan.isAddFundamentalDomain b μ


theorem measure_fundamentalDomain_ne_zero [Finite ι] [MeasurableSpace E] [BorelSpace E]
    {μ : Measure E} [Measure.IsAddHaarMeasure μ] :
    μ (fundamentalDomain b) ≠ 0 := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    b : Basis ι Real E
    inst✝³ : Finite ι
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    ⊢ Ne (μ (ZSpan.fundamentalDomain b)) 0
  -/
  convert (ZSpan.isAddFundamentalDomain b μ).measure_ne_zero (NeZero.ne μ)
  /-
    E : Type u_1
    ι : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    b : Basis ι Real E
    inst✝³ : Finite ι
    inst✝² : MeasurableSpace E
    inst✝¹ : BorelSpace E
    μ : MeasureTheory.Measure E
    inst✝ : μ.IsAddHaarMeasure
    ⊢ MeasureTheory.VAddInvariantMeasure (Subtype fun x => Membership.mem (Submodu …
  -/
  exact inferInstanceAs <| VAddInvariantMeasure (span ℤ (Set.range b)).toAddSubgroup E μ
  /-
    🎉 no goals
  -/


theorem measure_fundamentalDomain [Fintype ι] [DecidableEq ι] [MeasurableSpace E] (μ : Measure E)
    [BorelSpace E] [Measure.IsAddHaarMeasure μ] (b₀ : Basis ι ℝ E) :
    μ (fundamentalDomain b) = ENNReal.ofReal |b₀.det b| * μ (fundamentalDomain b₀) := by
  /-
    E : Type u_1
    ι : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    b : Basis ι Real E
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝¹ : BorelSpace E
    inst✝ : μ.IsAddHaarMeasure
    b₀ : Basis ι Real E
    ⊢ Eq (μ (ZSpan.fundamentalDomain b)) (HMul.hMul (ENNReal.ofReal (abs (b₀.det ⇑ …
  -/
  have : FiniteDimensional ℝ E := FiniteDimensional.of_fintype_basis b
  /-
    E : Type u_1
    ι : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    b : Basis ι Real E
    inst✝⁴ : Fintype ι
    inst✝³ : DecidableEq ι
    inst✝² : MeasurableSpace E
    μ : MeasureTheory.Measure E
    inst✝¹ : BorelSpace E
    inst✝ : μ.IsAddHaarMeasure
    b₀ : Basis ι Real E
    this : FiniteDimensional Real E
    ⊢ Eq (μ (ZSpan.fundamentalDomain b)) (HMul.hMul (ENNReal.ofReal (abs (b₀.det ⇑ …
  -/
  convert μ.addHaar_preimage_linearEquiv (b.equiv b₀ (Equiv.refl ι)) (fundamentalDomain b₀)
  · rw [Set.eq_preimage_iff_image_eq (LinearEquiv.bijective _), map_fundamentalDomain,
      Basis.map_equiv, Equiv.refl_symm, Basis.reindex_refl]
    /-
      case h.e'_3.h.e'_5.h.e'_1.h.e'_4
      E : Type u_1
      ι : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      b : Basis ι Real E
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝¹ : BorelSpace E
      inst✝ : μ.IsAddHaarMeasure
      b₀ : Basis ι Real E
      this : FiniteDimensional Real E
      ⊢ Eq (b₀.det ⇑b) (LinearMap.det ↑(b.equiv b₀ (Equiv.refl ι)).symm)
    -/
  · rw [← LinearMap.det_toMatrix b₀, Basis.equiv_symm, Equiv.refl_symm, Basis.det_apply]
    /-
      case h.e'_3.h.e'_5.h.e'_1.h.e'_4
      E : Type u_1
      ι : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      b : Basis ι Real E
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝¹ : BorelSpace E
      inst✝ : μ.IsAddHaarMeasure
      b₀ : Basis ι Real E
      this : FiniteDimensional Real E
      ⊢ Eq (b₀.toMatrix ⇑b).det ((LinearMap.toMatrix b₀ b₀) ↑(b₀.equiv b (Equiv.refl …
    -/
    congr
    /-
      case h.e'_3.h.e'_5.h.e'_1.h.e'_4.e_M
      E : Type u_1
      ι : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      b : Basis ι Real E
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝¹ : BorelSpace E
      inst✝ : μ.IsAddHaarMeasure
      b₀ : Basis ι Real E
      this : FiniteDimensional Real E
      ⊢ Eq (b₀.toMatrix ⇑b) ((LinearMap.toMatrix b₀ b₀) ↑(b₀.equiv b (Equiv.refl ι)))
    -/
    ext
    /-
      case h.e'_3.h.e'_5.h.e'_1.h.e'_4.e_M.a
      E : Type u_1
      ι : Type u_2
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      b : Basis ι Real E
      inst✝⁴ : Fintype ι
      inst✝³ : DecidableEq ι
      inst✝² : MeasurableSpace E
      μ : MeasureTheory.Measure E
      inst✝¹ : BorelSpace E
      inst✝ : μ.IsAddHaarMeasure
      b₀ : Basis ι Real E
      this : FiniteDimensional Real E
      i✝ j✝ : ι
      ⊢ Eq (b₀.toMatrix (⇑b) i✝ j✝) ((LinearMap.toMatrix b₀ b₀) (↑(b₀.equiv b (Equiv …
    -/
    simp [Basis.toMatrix_apply, LinearMap.toMatrix_apply, LinearEquiv.coe_coe, Basis.equiv_apply]
    /-
      🎉 no goals
    -/


@[simp]
theorem volume_fundamentalDomain [Fintype ι] [DecidableEq ι] (b : Basis ι ℝ (ι → ℝ)) :
    volume (fundamentalDomain b) = ENNReal.ofReal |(Matrix.of b).det| := by
  rw [measure_fundamentalDomain b volume (b₀ := Pi.basisFun ℝ ι), fundamentalDomain_pi_basisFun,
    volume_pi, Measure.pi_pi, Real.volume_Ico, sub_zero, ENNReal.ofReal_one, Finset.prod_const_one,
    mul_one, ← Matrix.det_transpose]
  /-
    ι : Type u_2
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι Real (ι → Real)
    ⊢ Eq (ENNReal.ofReal (abs ((Pi.basisFun Real ι).det ⇑b))) (ENNReal.ofReal (abs …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem fundamentalDomain_ae_parallelepiped [Fintype ι] [MeasurableSpace E] (μ : Measure E)
    [BorelSpace E] [Measure.IsAddHaarMeasure μ] :
    fundamentalDomain b =ᵐ[μ] parallelepiped b := by
  classical
  have : FiniteDimensional ℝ E := FiniteDimensional.of_fintype_basis b
  rw [← measure_symmDiff_eq_zero_iff, symmDiff_of_le (fundamentalDomain_subset_parallelepiped b)]
  suffices (parallelepiped b \ fundamentalDomain b) ⊆ ⋃ i,
      AffineSubspace.mk' (b i) (span ℝ (b '' (Set.univ \ {i}))) by
    refine measure_mono_null this
      (measure_iUnion_null_iff.mpr fun i ↦ Measure.addHaar_affineSubspace μ _ ?_)
    refine (ne_of_mem_of_not_mem' (AffineSubspace.mem_top _ _ 0)
      (AffineSubspace.mem_mk'_iff_vsub_mem.not.mpr ?_)).symm
    simp_rw [vsub_eq_sub, zero_sub, neg_mem_iff]
    exact linearIndependent_iff_not_mem_span.mp b.linearIndependent i
  intro x hx
  simp_rw [parallelepiped_basis_eq, Set.mem_Icc, Set.mem_diff, Set.mem_setOf_eq,
    mem_fundamentalDomain, Set.mem_Ico, not_forall, not_and, not_lt] at hx
  obtain ⟨i, hi⟩ := hx.2
  have : b.repr x i = 1 := le_antisymm (hx.1 i).2 (hi (hx.1 i).1)
  rw [← b.sum_repr x, ← Finset.sum_erase_add _ _ (Finset.mem_univ i), this, one_smul, ← vadd_eq_add]
  refine Set.mem_iUnion.mpr ⟨i, AffineSubspace.vadd_mem_mk' _
    (sum_smul_mem _ _ (fun i hi ↦ Submodule.subset_span ?_))⟩
  exact ⟨i, Set.mem_diff_singleton.mpr ⟨trivial, Finset.ne_of_mem_erase hi⟩, rfl⟩


/-- `L : Submodule ℤ E` where `E` is a vector space over a normed field `K` is a `ℤ`-lattice if
it is discrete and spans `E` over `K`. -/
class IsZLattice (K : Type*) [NormedField K] {E : Type*} [NormedAddCommGroup E] [NormedSpace K E]
    (L : Submodule ℤ E) [DiscreteTopology L] : Prop where
  /-- `L` spans the full space `E` over `K`. -/
  span_top : span K (L : Set E) = ⊤


theorem _root_.ZSpan.isZLattice {E ι : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [Finite ι] (b : Basis ι ℝ E) :
    IsZLattice ℝ (span ℤ (Set.range b)) where
  span_top := ZSpan.span_top b


theorem Zlattice.FG [hs : IsZLattice K L] : L.FG := by
  /-
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    hs : IsZLattice K L
    ⊢ L.FG
  -/
  obtain ⟨s, ⟨h_incl, ⟨h_span, h_lind⟩⟩⟩ := exists_linearIndependent K (L : Set E)
  -- Let `s` be a maximal `K`-linear independent family of elements of `L`. We show that
  -- `L` is finitely generated (as a ℤ-module) because it fits in the exact sequence
  -- `0 → span ℤ s → L → L ⧸ span ℤ s → 0` with `span ℤ s` and `L ⧸ span ℤ s` finitely generated.
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    hs : IsZLattice K L
    s : Set E
    h_incl : HasSubset.Subset s ↑L
    h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
    h_lind : LinearIndependent K Subtype.val
    ⊢ L.FG
  -/
  refine fg_of_fg_map_of_fg_inf_ker (span ℤ s).mkQ ?_ ?_
  · -- Let `b` be the `K`-basis of `E` formed by the vectors in `s`. The elements of
    -- `L ⧸ span ℤ s = L ⧸ span ℤ b` are in bijection with elements of `L ∩ fundamentalDomain b`
    -- so there are finitely many since `fundamentalDomain b` is bounded.
    /-
      case intro.intro.intro.refine_1
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      ⊢ (Submodule.map (Submodule.span Int s).mkQ L).FG
    -/
    refine fg_def.mpr ⟨map (span ℤ s).mkQ L, ?_, span_eq _⟩
    let b := Basis.mk h_lind (by
      rw [← hs.span_top, ← h_span]
      exact span_mono (by simp only [Subtype.range_coe_subtype, Set.setOf_mem_eq, subset_rfl]))
    /-
      case intro.intro.intro.refine_1
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
      ⊢ (↑(Submodule.map (Submodule.span Int s).mkQ L)).Finite
    -/
    rw [show span ℤ s = span ℤ (Set.range b) by simp [b, Basis.coe_mk, Subtype.range_coe_subtype]]
    /-
      case intro.intro.intro.refine_1
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
      ⊢ (↑(Submodule.map (Submodule.span Int (Set.range ⇑b)).mkQ L)).Finite
    -/
    have : Fintype s := h_lind.setFinite.fintype
    refine Set.Finite.of_finite_image (f := ((↑) : _ →  E) ∘ quotientEquiv b) ?_
      (Function.Injective.injOn (Subtype.coe_injective.comp (quotientEquiv b).injective))
    have : ((fundamentalDomain b) ∩ L).Finite := by
      change ((fundamentalDomain b) ∩ L.toAddSubgroup).Finite
      have : DiscreteTopology L.toAddSubgroup := (inferInstance : DiscreteTopology L)
      exact Metric.finite_isBounded_inter_isClosed (fundamentalDomain_isBounded b) inferInstance
    /-
      case intro.intro.intro.refine_1
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
      this✝ : Fintype ↑s
      this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
      ⊢ (Set.image (Function.comp Subtype.val ⇑(ZSpan.quotientEquiv b)) ↑(Submodule. …
    -/
    refine Set.Finite.subset this ?_
    /-
      case intro.intro.intro.refine_1
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
      this✝ : Fintype ↑s
      this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
      ⊢ HasSubset.Subset (Set.image (Function.comp Subtype.val ⇑(ZSpan.quotientEquiv …
    -/
    rintro _ ⟨_, ⟨⟨x, ⟨h_mem, rfl⟩⟩, rfl⟩⟩
    /-
      case intro.intro.intro.refine_1.intro.intro.intro.intro
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
      this✝ : Fintype ↑s
      this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
      x : E
      h_mem : Membership.mem (↑L) x
      ⊢ Membership.mem (Inter.inter (ZSpan.fundamentalDomain b) ↑L) (Function.comp S …
    -/
    rw [Function.comp_apply, mkQ_apply, quotientEquiv_apply_mk, fractRestrict_apply]
    /-
      case intro.intro.intro.refine_1.intro.intro.intro.intro
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
      this✝ : Fintype ↑s
      this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
      x : E
      h_mem : Membership.mem (↑L) x
      ⊢ Membership.mem (Inter.inter (ZSpan.fundamentalDomain b) ↑L) (ZSpan.fract b x)
    -/
    refine ⟨?_, ?_⟩
      /-
        case intro.intro.intro.refine_1.intro.intro.intro.intro.refine_1
        K : Type u_1
        inst✝⁷ : NormedLinearOrderedField K
        inst✝⁶ : HasSolidNorm K
        inst✝⁵ : FloorRing K
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace K E
        inst✝² : FiniteDimensional K E
        inst✝¹ : ProperSpace E
        L : Submodule Int E
        inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
        hs : IsZLattice K L
        s : Set E
        h_incl : HasSubset.Subset s ↑L
        h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
        h_lind : LinearIndependent K Subtype.val
        b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
        this✝ : Fintype ↑s
        this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
        x : E
        h_mem : Membership.mem (↑L) x
        ⊢ Membership.mem (ZSpan.fundamentalDomain b) (ZSpan.fract b x)
      -/
    · exact fract_mem_fundamentalDomain b x
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_1.intro.intro.intro.intro.refine_2
        K : Type u_1
        inst✝⁷ : NormedLinearOrderedField K
        inst✝⁶ : HasSolidNorm K
        inst✝⁵ : FloorRing K
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace K E
        inst✝² : FiniteDimensional K E
        inst✝¹ : ProperSpace E
        L : Submodule Int E
        inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
        hs : IsZLattice K L
        s : Set E
        h_incl : HasSubset.Subset s ↑L
        h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
        h_lind : LinearIndependent K Subtype.val
        b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
        this✝ : Fintype ↑s
        this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
        x : E
        h_mem : Membership.mem (↑L) x
        ⊢ Membership.mem (↑L) (ZSpan.fract b x)
      -/
    · rw [fract, SetLike.mem_coe, sub_eq_add_neg]
      refine Submodule.add_mem _ h_mem
        (neg_mem (Set.mem_of_subset_of_mem ?_ (Subtype.mem (floor b x))))
      /-
        case intro.intro.intro.refine_1.intro.intro.intro.intro.refine_2
        K : Type u_1
        inst✝⁷ : NormedLinearOrderedField K
        inst✝⁶ : HasSolidNorm K
        inst✝⁵ : FloorRing K
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace K E
        inst✝² : FiniteDimensional K E
        inst✝¹ : ProperSpace E
        L : Submodule Int E
        inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
        hs : IsZLattice K L
        s : Set E
        h_incl : HasSubset.Subset s ↑L
        h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
        h_lind : LinearIndependent K Subtype.val
        b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
        this✝ : Fintype ↑s
        this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
        x : E
        h_mem : Membership.mem (↑L) x
        ⊢ HasSubset.Subset ↑(Submodule.span Int (Set.range ⇑b)) ↑L
      -/
      rw [SetLike.coe_subset_coe, Basis.coe_mk, Subtype.range_coe_subtype, Set.setOf_mem_eq]
      /-
        case intro.intro.intro.refine_1.intro.intro.intro.intro.refine_2
        K : Type u_1
        inst✝⁷ : NormedLinearOrderedField K
        inst✝⁶ : HasSolidNorm K
        inst✝⁵ : FloorRing K
        E : Type u_2
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace K E
        inst✝² : FiniteDimensional K E
        inst✝¹ : ProperSpace E
        L : Submodule Int E
        inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
        hs : IsZLattice K L
        s : Set E
        h_incl : HasSubset.Subset s ↑L
        h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
        h_lind : LinearIndependent K Subtype.val
        b : Basis (Subtype fun x => Membership.mem s x) K E := Basis.mk h_lind ⋯
        this✝ : Fintype ↑s
        this : (Inter.inter (ZSpan.fundamentalDomain b) ↑L).Finite
        x : E
        h_mem : Membership.mem (↑L) x
        ⊢ LE.le (Submodule.span Int s) L
      -/
      exact span_le.mpr h_incl
      /-
        🎉 no goals
      -/
  · -- `span ℤ s` is finitely generated because `s` is finite
    /-
      case intro.intro.intro.refine_2
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      ⊢ (Min.min L (LinearMap.ker (Submodule.span Int s).mkQ)).FG
    -/
    rw [ker_mkQ, inf_of_le_right (span_le.mpr h_incl)]
    /-
      case intro.intro.intro.refine_2
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      hs : IsZLattice K L
      s : Set E
      h_incl : HasSubset.Subset s ↑L
      h_span : Eq (Submodule.span K s) (Submodule.span K ↑L)
      h_lind : LinearIndependent K Subtype.val
      ⊢ (Submodule.span Int s).FG
    -/
    exact fg_span (LinearIndependent.setFinite h_lind)
    /-
      🎉 no goals
    -/


theorem ZLattice.module_finite [IsZLattice K L] : Module.Finite ℤ L :=
  Module.Finite.iff_fg.mpr (Zlattice.FG K L)


instance instModuleFinite_of_discrete_submodule {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] (L : Submodule ℤ E) [DiscreteTopology L] :
    Module.Finite ℤ L := by
  /-
    K : Type u_1
    inst✝¹¹ : NormedLinearOrderedField K
    inst✝¹⁰ : HasSolidNorm K
    inst✝⁹ : FloorRing K
    E✝ : Type u_2
    inst✝⁸ : NormedAddCommGroup E✝
    inst✝⁷ : NormedSpace K E✝
    inst✝⁶ : FiniteDimensional K E✝
    inst✝⁵ : ProperSpace E✝
    L✝ : Submodule Int E✝
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : FiniteDimensional Real E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ⊢ Module.Finite Int (Subtype fun x => Membership.mem L x)
  -/
  let f := (span ℝ (L : Set E)).subtype
  /-
    K : Type u_1
    inst✝¹¹ : NormedLinearOrderedField K
    inst✝¹⁰ : HasSolidNorm K
    inst✝⁹ : FloorRing K
    E✝ : Type u_2
    inst✝⁸ : NormedAddCommGroup E✝
    inst✝⁷ : NormedSpace K E✝
    inst✝⁶ : FiniteDimensional K E✝
    inst✝⁵ : ProperSpace E✝
    L✝ : Submodule Int E✝
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : FiniteDimensional Real E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    f : LinearMap (RingHom.id Real) (Subtype fun x => Membership.mem (Submodule.sp …
    ⊢ Module.Finite Int (Subtype fun x => Membership.mem L x)
  -/
  let L₀ := L.comap (f.restrictScalars ℤ)
  have h_img : f '' L₀ = L := by
    rw [← LinearMap.coe_restrictScalars ℤ f, ← Submodule.map_coe (f.restrictScalars ℤ),
      Submodule.map_comap_eq_self]
    exact fun x hx ↦ LinearMap.mem_range.mpr ⟨⟨x, Submodule.subset_span hx⟩, rfl⟩
  suffices Module.Finite ℤ L₀ by
    have : L₀.map (f.restrictScalars ℤ) = L :=
      SetLike.ext'_iff.mpr h_img
    convert this ▸ Module.Finite.map L₀ (f.restrictScalars ℤ)
  have : DiscreteTopology L₀ := by
    refine DiscreteTopology.preimage_of_continuous_injective (L : Set E) ?_ (injective_subtype _)
    exact LinearMap.continuous_of_finiteDimensional f
  have : IsZLattice ℝ L₀ := ⟨by
    rw [← (Submodule.map_injective_of_injective (injective_subtype _)).eq_iff, Submodule.map_span,
      Submodule.map_top, range_subtype, h_img]⟩
  /-
    K : Type u_1
    inst✝¹¹ : NormedLinearOrderedField K
    inst✝¹⁰ : HasSolidNorm K
    inst✝⁹ : FloorRing K
    E✝ : Type u_2
    inst✝⁸ : NormedAddCommGroup E✝
    inst✝⁷ : NormedSpace K E✝
    inst✝⁶ : FiniteDimensional K E✝
    inst✝⁵ : ProperSpace E✝
    L✝ : Submodule Int E✝
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : FiniteDimensional Real E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    f : LinearMap (RingHom.id Real) (Subtype fun x => Membership.mem (Submodule.sp …
    L₀ : Submodule Int (Subtype fun x => Membership.mem (Submodule.span Real ↑L) x …
    h_img : Eq (Set.image ⇑f ↑L₀) ↑L
    this✝ : DiscreteTopology (Subtype fun x => Membership.mem L₀ x)
    this : IsZLattice Real L₀
    ⊢ Module.Finite Int (Subtype fun x => Membership.mem L₀ x)
  -/
  exact ZLattice.module_finite ℝ L₀
  /-
    🎉 no goals
  -/


theorem ZLattice.module_free [IsZLattice K L] : Module.Free ℤ L := by
  /-
    K : Type u_1
    inst✝⁸ : NormedLinearOrderedField K
    inst✝⁷ : HasSolidNorm K
    inst✝⁶ : FloorRing K
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace K E
    inst✝³ : FiniteDimensional K E
    inst✝² : ProperSpace E
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice K L
    ⊢ Module.Free Int (Subtype fun x => Membership.mem L x)
  -/
  have : Module.Finite ℤ L := module_finite K L
  /-
    K : Type u_1
    inst✝⁸ : NormedLinearOrderedField K
    inst✝⁷ : HasSolidNorm K
    inst✝⁶ : FloorRing K
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace K E
    inst✝³ : FiniteDimensional K E
    inst✝² : ProperSpace E
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice K L
    this : Module.Finite Int (Subtype fun x => Membership.mem L x)
    ⊢ Module.Free Int (Subtype fun x => Membership.mem L x)
  -/
  have : Module ℚ E := Module.compHom E (algebraMap ℚ K)
  /-
    K : Type u_1
    inst✝⁸ : NormedLinearOrderedField K
    inst✝⁷ : HasSolidNorm K
    inst✝⁶ : FloorRing K
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace K E
    inst✝³ : FiniteDimensional K E
    inst✝² : ProperSpace E
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice K L
    this✝ : Module.Finite Int (Subtype fun x => Membership.mem L x)
    this : Module Rat E
    ⊢ Module.Free Int (Subtype fun x => Membership.mem L x)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance instModuleFree_of_discrete_submodule {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] (L : Submodule ℤ E) [DiscreteTopology L] :
    Module.Free ℤ L := by
  /-
    K : Type u_1
    inst✝¹¹ : NormedLinearOrderedField K
    inst✝¹⁰ : HasSolidNorm K
    inst✝⁹ : FloorRing K
    E✝ : Type u_2
    inst✝⁸ : NormedAddCommGroup E✝
    inst✝⁷ : NormedSpace K E✝
    inst✝⁶ : FiniteDimensional K E✝
    inst✝⁵ : ProperSpace E✝
    L✝ : Submodule Int E✝
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : FiniteDimensional Real E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ⊢ Module.Free Int (Subtype fun x => Membership.mem L x)
  -/
  have : Module ℚ E := Module.compHom E (algebraMap ℚ ℝ)
  /-
    K : Type u_1
    inst✝¹¹ : NormedLinearOrderedField K
    inst✝¹⁰ : HasSolidNorm K
    inst✝⁹ : FloorRing K
    E✝ : Type u_2
    inst✝⁸ : NormedAddCommGroup E✝
    inst✝⁷ : NormedSpace K E✝
    inst✝⁶ : FiniteDimensional K E✝
    inst✝⁵ : ProperSpace E✝
    L✝ : Submodule Int E✝
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : FiniteDimensional Real E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    this : Module Rat E
    ⊢ Module.Free Int (Subtype fun x => Membership.mem L x)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem ZLattice.rank [hs : IsZLattice K L] : finrank ℤ L = finrank K E := by
  classical
  have : Module.Finite ℤ L := module_finite K L
  have : Module.Free ℤ L := module_free K L
  have : Module ℚ E := Module.compHom E (algebraMap ℚ K)
  let b₀ := Module.Free.chooseBasis ℤ L
  -- Let `b` be a `ℤ`-basis of `L` formed of vectors of `E`
  let b := Subtype.val ∘ b₀
  have : LinearIndependent ℤ b :=
    LinearIndependent.map' b₀.linearIndependent (L.subtype) (ker_subtype _)
  -- We prove some assertions that will be useful later on
  have h_spanL : span ℤ (Set.range b) = L := by
    convert congrArg (map (Submodule.subtype L)) b₀.span_eq
    · rw [map_span, Set.range_comp]
      rfl
    · exact (map_subtype_top _).symm
  have h_spanE : span K (Set.range b) = ⊤ := by
    rw [← span_span_of_tower (R := ℤ), h_spanL]
    exact hs.span_top
  have h_card : Fintype.card (Module.Free.ChooseBasisIndex ℤ L) =
      (Set.range b).toFinset.card := by
    rw [Set.toFinset_range, Finset.univ.card_image_of_injective]
    · rfl
    · exact Subtype.coe_injective.comp (Basis.injective _)
  rw [finrank_eq_card_chooseBasisIndex]
    -- We prove that `finrank ℤ L ≤ finrank K E` and `finrank K E ≤ finrank ℤ L`
  refine le_antisymm ?_ ?_
  · -- To prove that `finrank ℤ L ≤ finrank K E`, we proceed by contradiction and prove that, in
    -- this case, there is a ℤ-relation between the vectors of `b`
    obtain ⟨t, ⟨ht_inc, ⟨ht_span, ht_lin⟩⟩⟩ := exists_linearIndependent K (Set.range b)
    -- `e` is a `K`-basis of `E` formed of vectors of `b`
    let e : Basis t K E := Basis.mk ht_lin (by simp [ht_span, h_spanE])
    have : Fintype t := Set.Finite.fintype ((Set.range b).toFinite.subset ht_inc)
    have h : LinearIndependent ℤ (fun x : (Set.range b) => (x : E)) := by
      rwa [linearIndependent_subtype_range (Subtype.coe_injective.comp b₀.injective)]
    contrapose! h
    -- Since `finrank ℤ L > finrank K E`, there exists a vector `v ∈ b` with `v ∉ e`
    obtain ⟨v, hv⟩ : (Set.range b \ Set.range e).Nonempty := by
      rw [Basis.coe_mk, Subtype.range_coe_subtype, Set.setOf_mem_eq, ← Set.toFinset_nonempty]
      contrapose h
      rw [Finset.not_nonempty_iff_eq_empty, Set.toFinset_diff,
        Finset.sdiff_eq_empty_iff_subset] at h
      replace h := Finset.card_le_card h
      rwa [not_lt, h_card, ← topEquiv.finrank_eq, ← h_spanE, ← ht_span,
        finrank_span_set_eq_card ht_lin]
    -- Assume that `e ∪ {v}` is not `ℤ`-linear independent then we get the contradiction
    suffices ¬ LinearIndependent ℤ (fun x : ↥(insert v (Set.range e)) => (x : E)) by
      contrapose! this
      refine LinearIndependent.mono ?_ this
      exact Set.insert_subset (Set.mem_of_mem_diff hv) (by simp [e, ht_inc])
    -- We prove finally that `e ∪ {v}` is not ℤ-linear independent or, equivalently,
    -- not ℚ-linear independent by showing that `v ∈ span ℚ e`.
    rw [LinearIndependent.iff_fractionRing ℤ ℚ,
      linearIndependent_insert (Set.not_mem_of_mem_diff hv),  not_and, not_not]
    intro _
    -- But that follows from the fact that there exist `n, m : ℕ`, `n ≠ m`
    -- such that `(n - m) • v ∈ span ℤ e` which is true since `n ↦ ZSpan.fract e (n • v)`
    -- takes value into the finite set `fundamentalDomain e ∩ L`
    have h_mapsto : Set.MapsTo (fun n : ℤ => fract e (n • v)) Set.univ
        (Metric.closedBall 0 (∑ i, ‖e i‖) ∩ (L : Set E)) := by
      rw [Set.mapsTo_inter, Set.mapsTo_univ_iff, Set.mapsTo_univ_iff]
      refine ⟨fun _ ↦ mem_closedBall_zero_iff.mpr (norm_fract_le e _), fun _ => ?_⟩
      · rw [← h_spanL]
        refine sub_mem ?_ ?_
        · exact zsmul_mem (subset_span (Set.diff_subset hv)) _
        · exact span_mono (by simp [e, ht_inc]) (coe_mem _)
    have h_finite : Set.Finite (Metric.closedBall 0 (∑ i, ‖e i‖) ∩ (L : Set E)) := by
      change ((_ : Set E) ∩ L.toAddSubgroup).Finite
      have : DiscreteTopology L.toAddSubgroup := (inferInstance : DiscreteTopology L)
      exact Metric.finite_isBounded_inter_isClosed  Metric.isBounded_closedBall inferInstance
    obtain ⟨n, -, m, -, h_neq, h_eq⟩ := Set.Infinite.exists_ne_map_eq_of_mapsTo
      Set.infinite_univ h_mapsto h_finite
    have h_nz : (-n + m : ℚ) ≠ 0 := by
      rwa [Ne, add_eq_zero_iff_eq_neg.not, neg_inj, Rat.coe_int_inj, ← Ne]
    apply (smul_mem_iff _ h_nz).mp
    refine span_subset_span ℤ ℚ _ ?_
    rwa [add_smul, neg_smul, SetLike.mem_coe, ← fract_eq_fract, Int.cast_smul_eq_zsmul ℚ,
      Int.cast_smul_eq_zsmul ℚ]
  · -- To prove that `finrank K E ≤ finrank ℤ L`, we use the fact `b` generates `E` over `K`
    -- and thus `finrank K E ≤ card b = finrank ℤ L`
    rw [← topEquiv.finrank_eq, ← h_spanE]
    convert finrank_span_le_card (R := K) (Set.range b)


/-- Any `ℤ`-basis of `L` is also a `K`-basis of `E`. -/
def Basis.ofZLatticeBasis :
    Basis ι K E := by
  /-
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ι : Type u_3
    hs : IsZLattice K L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    ⊢ Basis ι K E
  -/
  have : Module.Finite ℤ L := ZLattice.module_finite K L
  /-
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ι : Type u_3
    hs : IsZLattice K L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    this : Module.Finite Int (Subtype fun x => Membership.mem L x)
    ⊢ Basis ι K E
  -/
  have : Free ℤ L := ZLattice.module_free K L
  /-
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ι : Type u_3
    hs : IsZLattice K L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    this✝ : Module.Finite Int (Subtype fun x => Membership.mem L x)
    this : Module.Free Int (Subtype fun x => Membership.mem L x)
    ⊢ Basis ι K E
  -/
  let e :=  Basis.indexEquiv (Free.chooseBasis ℤ L) b
  /-
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ι : Type u_3
    hs : IsZLattice K L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    this✝ : Module.Finite Int (Subtype fun x => Membership.mem L x)
    this : Module.Free Int (Subtype fun x => Membership.mem L x)
    e : Equiv (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
    ⊢ Basis ι K E
  -/
  have : Fintype ι := Fintype.ofEquiv _ e
  /-
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ι : Type u_3
    hs : IsZLattice K L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    this✝¹ : Module.Finite Int (Subtype fun x => Membership.mem L x)
    this✝ : Module.Free Int (Subtype fun x => Membership.mem L x)
    e : Equiv (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
    this : Fintype ι
    ⊢ Basis ι K E
  -/
  refine basisOfTopLeSpanOfCardEqFinrank (L.subtype ∘ b) ?_ ?_
  · rw [← span_span_of_tower ℤ, Set.range_comp, ← map_span, Basis.span_eq, Submodule.map_top,
      range_subtype, top_le_iff, hs.span_top]
    /-
      case refine_2
      K : Type u_1
      inst✝⁷ : NormedLinearOrderedField K
      inst✝⁶ : HasSolidNorm K
      inst✝⁵ : FloorRing K
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace K E
      inst✝² : FiniteDimensional K E
      inst✝¹ : ProperSpace E
      L : Submodule Int E
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
      ι : Type u_3
      hs : IsZLattice K L
      b : Basis ι Int (Subtype fun x => Membership.mem L x)
      this✝¹ : Module.Finite Int (Subtype fun x => Membership.mem L x)
      this✝ : Module.Free Int (Subtype fun x => Membership.mem L x)
      e : Equiv (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem L …
      this : Fintype ι
      ⊢ Eq (Fintype.card ι) (Module.finrank K E)
    -/
  · rw [← Fintype.card_congr e, ← finrank_eq_card_chooseBasisIndex, ZLattice.rank K L]
    /-
      🎉 no goals
    -/


@[simp]
theorem Basis.ofZLatticeBasis_apply (i : ι) :
                                        /-
                                          K : Type u_1
                                          inst✝⁷ : NormedLinearOrderedField K
                                          inst✝⁶ : HasSolidNorm K
                                          inst✝⁵ : FloorRing K
                                          E : Type u_2
                                          inst✝⁴ : NormedAddCommGroup E
                                          inst✝³ : NormedSpace K E
                                          inst✝² : FiniteDimensional K E
                                          inst✝¹ : ProperSpace E
                                          L : Submodule Int E
                                          inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
                                          ι : Type u_3
                                          hs : IsZLattice K L
                                          b : Basis ι Int (Subtype fun x => Membership.mem L x)
                                          i : ι
                                          ⊢ Eq ((Basis.ofZLatticeBasis K L b) i) ↑(b i)
                                        -/
    b.ofZLatticeBasis K L i = b i := by simp [Basis.ofZLatticeBasis]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem Basis.ofZLatticeBasis_repr_apply (x : L) (i : ι) :
    (b.ofZLatticeBasis K L).repr x i = b.repr x i := by
  suffices ((b.ofZLatticeBasis K L).repr.toLinearMap.restrictScalars ℤ) ∘ₗ L.subtype
      = Finsupp.mapRange.linearMap (Algebra.linearMap ℤ K) ∘ₗ b.repr.toLinearMap by
    exact DFunLike.congr_fun (LinearMap.congr_fun this x) i
  /-
    K : Type u_1
    inst✝⁷ : NormedLinearOrderedField K
    inst✝⁶ : HasSolidNorm K
    inst✝⁵ : FloorRing K
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace K E
    inst✝² : FiniteDimensional K E
    inst✝¹ : ProperSpace E
    L : Submodule Int E
    inst✝ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    ι : Type u_3
    hs : IsZLattice K L
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    x : Subtype fun x => Membership.mem L x
    i : ι
    ⊢ Eq ((↑Int ↑(Basis.ofZLatticeBasis K L b).repr).comp L.subtype) ((Finsupp.map …
  -/
  refine Basis.ext b fun i ↦ ?_
  simp_rw [LinearMap.coe_comp, Function.comp_apply, LinearMap.coe_restrictScalars,
    LinearEquiv.coe_coe, coe_subtype, ← b.ofZLatticeBasis_apply K, repr_self,
    Finsupp.mapRange.linearMap_apply, Finsupp.mapRange_single, Algebra.linearMap_apply, map_one]


theorem Basis.ofZLatticeBasis_span :
    (span ℤ (Set.range (b.ofZLatticeBasis K))) = L := by
  calc (span ℤ (Set.range (b.ofZLatticeBasis K)))
    _ = (span ℤ (L.subtype '' (Set.range b))) := by congr; ext; simp
    _ = (map L.subtype (span ℤ (Set.range b))) := by rw [Submodule.map_span]
    _ = L := by simp [b.span_eq]


open MeasureTheory in
theorem ZLattice.isAddFundamentalDomain {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {L : Submodule ℤ E} [DiscreteTopology L] [IsZLattice ℝ L] [Finite ι]
    (b : Basis ι ℤ L) [MeasurableSpace E] [OpensMeasurableSpace E] (μ : Measure E) :
    IsAddFundamentalDomain L (fundamentalDomain (b.ofZLatticeBasis ℝ)) μ := by
  /-
    ι : Type u_3
    E : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    L : Submodule Int E
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝³ : IsZLattice Real L
    inst✝² : Finite ι
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    μ : MeasureTheory.Measure E
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem L x) ( …
  -/
  convert ZSpan.isAddFundamentalDomain (b.ofZLatticeBasis ℝ) μ
  /-
    case h.e'_1.h.e'_2.h.h.e'_4
    ι : Type u_3
    E : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    inst✝⁵ : FiniteDimensional Real E
    L : Submodule Int E
    inst✝⁴ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝³ : IsZLattice Real L
    inst✝² : Finite ι
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    inst✝¹ : MeasurableSpace E
    inst✝ : OpensMeasurableSpace E
    μ : MeasureTheory.Measure E
    x✝ : E
    ⊢ Eq L (Submodule.span Int (Set.range ⇑(Basis.ofZLatticeBasis Real L b)))
  -/
  all_goals exact (b.ofZLatticeBasis_span ℝ).symm
  /-
    🎉 no goals
  -/


instance instCountable_of_discrete_submodule {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] (L : Submodule ℤ E) [DiscreteTopology L] [IsZLattice ℝ L] :
    Countable L := by
  /-
    K : Type u_1
    inst✝¹² : NormedLinearOrderedField K
    inst✝¹¹ : HasSolidNorm K
    inst✝¹⁰ : FloorRing K
    E✝ : Type u_2
    inst✝⁹ : NormedAddCommGroup E✝
    inst✝⁸ : NormedSpace K E✝
    inst✝⁷ : FiniteDimensional K E✝
    inst✝⁶ : ProperSpace E✝
    L✝ : Submodule Int E✝
    inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
    ι : Type u_3
    hs : IsZLattice K L✝
    b : Basis ι Int (Subtype fun x => Membership.mem L✝ x)
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    ⊢ Countable (Subtype fun x => Membership.mem L x)
  -/
  simp_rw [← (Module.Free.chooseBasis ℤ L).ofZLatticeBasis_span ℝ]
  /-
    K : Type u_1
    inst✝¹² : NormedLinearOrderedField K
    inst✝¹¹ : HasSolidNorm K
    inst✝¹⁰ : FloorRing K
    E✝ : Type u_2
    inst✝⁹ : NormedAddCommGroup E✝
    inst✝⁸ : NormedSpace K E✝
    inst✝⁷ : FiniteDimensional K E✝
    inst✝⁶ : ProperSpace E✝
    L✝ : Submodule Int E✝
    inst✝⁵ : DiscreteTopology (Subtype fun x => Membership.mem L✝ x)
    ι : Type u_3
    hs : IsZLattice K L✝
    b : Basis ι Int (Subtype fun x => Membership.mem L✝ x)
    E : Type u_4
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : FiniteDimensional Real E
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice Real L
    ⊢ Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.range ⇑( …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Let `e : E → F` a linear map, the map that sends a `L : Submodule ℤ E` to the
`Submodule ℤ F` that is the pullback of `L` by `e`. If `IsZLattice L` and `e` is a continuous
linear equiv, then it is a `IsZLattice` of `E`, see `instIsZLatticeComap`. -/
protected def ZLattice.comap (e : F →ₗ[K] E) := L.comap (e.restrictScalars ℤ)


@[simp]
theorem ZLattice.coe_comap (e : F →ₗ[K] E) :
    (ZLattice.comap K L e : Set F) = e⁻¹' L := rfl


theorem ZLattice.comap_refl :
    ZLattice.comap K L (1 : E →ₗ[K] E)= L := Submodule.comap_id L


theorem ZLattice.comap_discreteTopology [hL : DiscreteTopology L] {e : F →ₗ[K] E}
    (he₁ : Continuous e) (he₂ : Function.Injective e) :
    DiscreteTopology (ZLattice.comap K L e) := by
  /-
    K : Type u_1
    inst✝⁴ : NormedField K
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace K F
    L : Submodule Int E
    hL : DiscreteTopology (Subtype fun x => Membership.mem L x)
    e : LinearMap (RingHom.id K) F E
    he₁ : Continuous ⇑e
    he₂ : Function.Injective ⇑e
    ⊢ DiscreteTopology (Subtype fun x => Membership.mem (ZLattice.comap K L e) x)
  -/
  exact DiscreteTopology.preimage_of_continuous_injective L he₁ he₂
  /-
    🎉 no goals
  -/


instance [DiscreteTopology L] (e : F ≃L[K] E) :
    DiscreteTopology (ZLattice.comap K L e.toLinearMap) :=
  ZLattice.comap_discreteTopology K L e.continuous e.injective


theorem ZLattice.comap_span_top (hL : span K (L : Set E) = ⊤) {e : F →ₗ[K] E}
    (he : (L : Set E) ⊆ LinearMap.range e) :
    span K (ZLattice.comap K L e : Set F) = ⊤ := by
  /-
    K : Type u_1
    inst✝⁴ : NormedField K
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace K F
    L : Submodule Int E
    hL : Eq (Submodule.span K ↑L) Top.top
    e : LinearMap (RingHom.id K) F E
    he : HasSubset.Subset ↑L ↑(LinearMap.range e)
    ⊢ Eq (Submodule.span K ↑(ZLattice.comap K L e)) Top.top
  -/
  rw [ZLattice.coe_comap, Submodule.span_preimage_eq (Submodule.nonempty L) he, hL, comap_top]
  /-
    🎉 no goals
  -/


instance instIsZLatticeComap [DiscreteTopology L] [IsZLattice K L] (e : F ≃L[K] E) :
    IsZLattice K (ZLattice.comap K L e.toLinearMap) where
  span_top := by
    rw [ZLattice.coe_comap, LinearEquiv.coe_coe, e.coe_toLinearEquiv, ← e.image_symm_eq_preimage,
      ← Submodule.map_span, IsZLattice.span_top, Submodule.map_top, LinearEquivClass.range]


theorem ZLattice.comap_comp {G : Type*} [NormedAddCommGroup G] [NormedSpace K G]
    (e : F →ₗ[K] E) (e' : G →ₗ[K] F) :
    (ZLattice.comap K (ZLattice.comap K L e) e') = ZLattice.comap K L (e ∘ₗ e') :=
  (Submodule.comap_comp _ _ L).symm


/-- If `e` is a linear equivalence, it induces a `ℤ`-linear equivalence between
`L` and `ZLattice.comap K L e`. -/
def ZLattice.comap_equiv (e : F ≃ₗ[K] E) :
    L ≃ₗ[ℤ] (ZLattice.comap K L e.toLinearMap) :=
  LinearEquiv.ofBijective
    ((e.symm.toLinearMap.restrictScalars ℤ).restrict
                    /-
                      K : Type u_1
                      inst✝⁴ : NormedField K
                      E : Type u_2
                      F : Type u_3
                      inst✝³ : NormedAddCommGroup E
                      inst✝² : NormedSpace K E
                      inst✝¹ : NormedAddCommGroup F
                      inst✝ : NormedSpace K F
                      L : Submodule Int E
                      e : LinearEquiv (RingHom.id K) F E
                      x✝ : E
                      h : Membership.mem L x✝
                      ⊢ Membership.mem (ZLattice.comap K L ↑e) ((↑Int ↑e.symm) x✝)
                    -/
      (fun _ h ↦ by simpa [← SetLike.mem_coe] using h))
                    /-
                      🎉 no goals
                    -/
    ⟨fun _ _ h ↦ Subtype.ext_iff_val.mpr (e.symm.injective (congr_arg Subtype.val h)),
                            /-
                              K : Type u_1
                              inst✝⁴ : NormedField K
                              E : Type u_2
                              F : Type u_3
                              inst✝³ : NormedAddCommGroup E
                              inst✝² : NormedSpace K E
                              inst✝¹ : NormedAddCommGroup F
                              inst✝ : NormedSpace K F
                              L : Submodule Int E
                              e : LinearEquiv (RingHom.id K) F E
                              x✝ : Subtype fun x => Membership.mem (ZLattice.comap K L ↑e) x
                              x : F
                              hx : Membership.mem (ZLattice.comap K L ↑e) x
                              ⊢ Membership.mem L (e x)
                            -/
    fun ⟨x, hx⟩ ↦ ⟨⟨e x, by rwa [← SetLike.mem_coe, ZLattice.coe_comap] at hx⟩,
                            /-
                              🎉 no goals
                            -/
         /-
           K : Type u_1
           inst✝⁴ : NormedField K
           E : Type u_2
           F : Type u_3
           inst✝³ : NormedAddCommGroup E
           inst✝² : NormedSpace K E
           inst✝¹ : NormedAddCommGroup F
           inst✝ : NormedSpace K F
           L : Submodule Int E
           e : LinearEquiv (RingHom.id K) F E
           x✝ : Subtype fun x => Membership.mem (ZLattice.comap K L ↑e) x
           x : F
           hx : Membership.mem (ZLattice.comap K L ↑e) x
           ⊢ Eq (((↑Int ↑e.symm).restrict ⋯) ⟨e x, ⋯⟩) ⟨x, hx⟩
         -/
      by simp [Subtype.ext_iff_val]⟩⟩
         /-
           🎉 no goals
         -/


@[simp]
theorem ZLattice.comap_equiv_apply (e : F ≃ₗ[K] E) (x : L) :
    ZLattice.comap_equiv K L e x = e.symm x := rfl


/-- The basis of `ZLattice.comap K L e` given by the image of a basis `b` of `L` by `e.symm`. -/
def Basis.ofZLatticeComap (e : F ≃ₗ[K] E) {ι : Type*}
    (b : Basis ι ℤ L) :
    Basis ι ℤ (ZLattice.comap K L e.toLinearMap) := b.map (ZLattice.comap_equiv K L e)


@[simp]
theorem Basis.ofZLatticeComap_apply (e : F ≃ₗ[K] E) {ι : Type*}
    (b : Basis ι ℤ L) (i : ι) :
                                                   /-
                                                     K : Type u_1
                                                     inst✝⁴ : NormedField K
                                                     E : Type u_2
                                                     F : Type u_3
                                                     inst✝³ : NormedAddCommGroup E
                                                     inst✝² : NormedSpace K E
                                                     inst✝¹ : NormedAddCommGroup F
                                                     inst✝ : NormedSpace K F
                                                     L : Submodule Int E
                                                     e : LinearEquiv (RingHom.id K) F E
                                                     ι : Type u_4
                                                     b : Basis ι Int (Subtype fun x => Membership.mem L x)
                                                     i : ι
                                                     ⊢ Eq (↑((Basis.ofZLatticeComap K L e b) i)) (e.symm ↑(b i))
                                                   -/
    b.ofZLatticeComap K L e i = e.symm (b i) := by simp [Basis.ofZLatticeComap]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem Basis.ofZLatticeComap_repr_apply (e : F ≃ₗ[K] E) {ι : Type*} (b : Basis ι ℤ L) (x : L)
    (i : ι) :
    (b.ofZLatticeComap K L e).repr (ZLattice.comap_equiv K L e x) i = b.repr x i := by
  /-
    K : Type u_1
    inst✝⁴ : NormedField K
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace K E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace K F
    L : Submodule Int E
    e : LinearEquiv (RingHom.id K) F E
    ι : Type u_4
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    x : Subtype fun x => Membership.mem L x
    i : ι
    ⊢ Eq (((Basis.ofZLatticeComap K L e b).repr ((ZLattice.comap_equiv K L e) x))  …
  -/
  simp [Basis.ofZLatticeComap]
  /-
    🎉 no goals
  -/


theorem Basis.ofZLatticeBasis_comap (e : F ≃L[K] E) {ι : Type*} (b : Basis ι ℤ L) :
    (b.ofZLatticeComap K L e.toLinearEquiv).ofZLatticeBasis K (ZLattice.comap K L e.toLinearMap) =
    (b.ofZLatticeBasis K L).map e.symm.toLinearEquiv := by
  /-
    K : Type u_1
    inst✝¹² : NormedLinearOrderedField K
    inst✝¹¹ : HasSolidNorm K
    inst✝¹⁰ : FloorRing K
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace K E
    inst✝⁷ : FiniteDimensional K E
    inst✝⁶ : ProperSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace K F
    inst✝³ : FiniteDimensional K F
    inst✝² : ProperSpace F
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice K L
    e : ContinuousLinearEquiv (RingHom.id K) F E
    ι : Type u_4
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    ⊢ Eq (Basis.ofZLatticeBasis K (ZLattice.comap K L ↑e.toLinearEquiv) (Basis.ofZ …
  -/
  ext
  /-
    case a
    K : Type u_1
    inst✝¹² : NormedLinearOrderedField K
    inst✝¹¹ : HasSolidNorm K
    inst✝¹⁰ : FloorRing K
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace K E
    inst✝⁷ : FiniteDimensional K E
    inst✝⁶ : ProperSpace E
    F : Type u_3
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace K F
    inst✝³ : FiniteDimensional K F
    inst✝² : ProperSpace F
    L : Submodule Int E
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem L x)
    inst✝ : IsZLattice K L
    e : ContinuousLinearEquiv (RingHom.id K) F E
    ι : Type u_4
    b : Basis ι Int (Subtype fun x => Membership.mem L x)
    i✝ : ι
    ⊢ Eq ((Basis.ofZLatticeBasis K (ZLattice.comap K L ↑e.toLinearEquiv) (Basis.of …
  -/
  simp
  /-
    🎉 no goals
  -/


