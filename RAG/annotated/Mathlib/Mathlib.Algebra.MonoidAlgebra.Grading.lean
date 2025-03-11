/-- The submodule corresponding to each grade given by the degree function `f`. -/
abbrev gradeBy (f : M → ι) (i : ι) : Submodule R R[M] where
  carrier := { a | ∀ m, m ∈ a.support → f m = i }
                      /-
                        M : Type u_1
                        ι : Type u_2
                        R : Type u_3
                        inst✝ : CommSemiring R
                        f : M → ι
                        i : ι
                        m : M
                        h : Membership.mem (Finsupp.support 0) m
                        ⊢ Eq (f m) i
                      -/
  zero_mem' m h := by cases h
    /-
      M : Type u_1
      ι : Type u_2
      R : Type u_3
      inst✝ : CommSemiring R
      f : M → ι
      i : ι
      a b : AddMonoidAlgebra R M
      ha : Membership.mem (setOf fun a => ∀ (m : M), Membership.mem a.support m → Eq …
      hb : Membership.mem (setOf fun a => ∀ (m : M), Membership.mem a.support m → Eq …
      m : M
      h : Membership.mem (HAdd.hAdd a b).support m
      ⊢ Eq (f m) i
    -/
                      /-
                        🎉 no goals
                      -/
    /-
      🎉 no goals
    -/
  add_mem' {a b} ha hb m h := by
    classical exact (Finset.mem_union.mp (Finsupp.support_add h)).elim (ha m) (hb m)
  smul_mem' _ _ h := Set.Subset.trans Finsupp.support_smul h


/-- The submodule corresponding to each grade. -/
abbrev grade (m : M) : Submodule R R[M] :=
  gradeBy R id m


theorem gradeBy_id : gradeBy R (id : M → M) = grade R := rfl


theorem mem_gradeBy_iff (f : M → ι) (i : ι) (a : R[M]) :
                                                              /-
                                                                M : Type u_1
                                                                ι : Type u_2
                                                                R : Type u_3
                                                                inst✝ : CommSemiring R
                                                                f : M → ι
                                                                i : ι
                                                                a : AddMonoidAlgebra R M
                                                                ⊢ Iff (Membership.mem (AddMonoidAlgebra.gradeBy R f i) a) (HasSubset.Subset (↑ …
                                                              -/
    a ∈ gradeBy R f i ↔ (a.support : Set M) ⊆ f ⁻¹' {i} := by rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem mem_grade_iff (m : M) (a : R[M]) : a ∈ grade R m ↔ a.support ⊆ {m} := by
  /-
    M : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    m : M
    a : AddMonoidAlgebra R M
    ⊢ Iff (Membership.mem (AddMonoidAlgebra.grade R m) a) (HasSubset.Subset a.supp …
  -/
  rw [← Finset.coe_subset, Finset.coe_singleton]
  /-
    M : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    m : M
    a : AddMonoidAlgebra R M
    ⊢ Iff (Membership.mem (AddMonoidAlgebra.grade R m) a) (HasSubset.Subset (↑a.su …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_grade_iff' (m : M) (a : R[M]) :
    a ∈ grade R m ↔ a ∈ (LinearMap.range (Finsupp.lsingle m : R →ₗ[R] M →₀ R) :
      Submodule R R[M]) := by
  /-
    M : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    m : M
    a : AddMonoidAlgebra R M
    ⊢ Iff (Membership.mem (AddMonoidAlgebra.grade R m) a) (Membership.mem (LinearM …
  -/
  rw [mem_grade_iff, Finsupp.support_subset_singleton']
  /-
    M : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    m : M
    a : AddMonoidAlgebra R M
    ⊢ Iff (Exists fun b => Eq a (Finsupp.single m b)) (Membership.mem (LinearMap.r …
  -/
  apply exists_congr
  /-
    case h
    M : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    m : M
    a : AddMonoidAlgebra R M
    ⊢ ∀ (a_1 : R), Iff (Eq a (Finsupp.single m a_1)) (Eq ((Finsupp.lsingle m) a_1) …
  -/
  intro r
  /-
    case h
    M : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    m : M
    a : AddMonoidAlgebra R M
    r : R
    ⊢ Iff (Eq a (Finsupp.single m r)) (Eq ((Finsupp.lsingle m) r) a)
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> exact Eq.symm
                  /-
                    🎉 no goals
                  -/


theorem grade_eq_lsingle_range (m : M) :
    grade R m = LinearMap.range (Finsupp.lsingle m : R →ₗ[R] M →₀ R) :=
  Submodule.ext (mem_grade_iff' R m)


theorem single_mem_gradeBy {R} [CommSemiring R] (f : M → ι) (m : M) (r : R) :
    Finsupp.single m r ∈ gradeBy R f (f m) := by
  /-
    M : Type u_1
    ι : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : M → ι
    m : M
    r : R
    ⊢ Membership.mem (AddMonoidAlgebra.gradeBy R f (f m)) (Finsupp.single m r)
  -/
  intro x hx
  /-
    M : Type u_1
    ι : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : M → ι
    m : M
    r : R
    x : M
    hx : Membership.mem (Finsupp.single m r).support x
    ⊢ Eq (f x) (f m)
  -/
  rw [Finset.mem_singleton.mp (Finsupp.support_single_subset hx)]
  /-
    🎉 no goals
  -/


theorem single_mem_grade {R} [CommSemiring R] (i : M) (r : R) : Finsupp.single i r ∈ grade R i :=
  single_mem_gradeBy _ _ _


instance gradeBy.gradedMonoid [AddMonoid M] [AddMonoid ι] [CommSemiring R] (f : M →+ ι) :
    SetLike.GradedMonoid (gradeBy R f : ι → Submodule R R[M]) where
  one_mem m h := by
    /-
      M : Type u_1
      ι : Type u_2
      R : Type u_3
      inst✝² : AddMonoid M
      inst✝¹ : AddMonoid ι
      inst✝ : CommSemiring R
      f : AddMonoidHom M ι
      m : M
      h : Membership.mem (Finsupp.support 1) m
      ⊢ Eq (f m) 0
    -/
    rw [one_def] at h
    /-
      M : Type u_1
      ι : Type u_2
      R : Type u_3
      inst✝² : AddMonoid M
      inst✝¹ : AddMonoid ι
      inst✝ : CommSemiring R
      f : AddMonoidHom M ι
      m : M
      h : Membership.mem (AddMonoidAlgebra.single 0 1).support m
      ⊢ Eq (f m) 0
    -/
    obtain rfl : m = 0 := Finset.mem_singleton.1 <| Finsupp.support_single_subset h
    /-
      M : Type u_1
      ι : Type u_2
      R : Type u_3
      inst✝² : AddMonoid M
      inst✝¹ : AddMonoid ι
      inst✝ : CommSemiring R
      f : AddMonoidHom M ι
      h : Membership.mem (AddMonoidAlgebra.single 0 1).support 0
      ⊢ Eq (f 0) 0
    -/
    apply map_zero
    /-
      🎉 no goals
    -/
  mul_mem i j a b ha hb c hc := by
    classical
    obtain ⟨ma, hma, mb, hmb, rfl⟩ : ∃ y ∈ a.support, ∃ z ∈ b.support, y + z = c :=
      Finset.mem_add.1 <| support_mul a b hc
    rw [map_add, ha ma hma, hb mb hmb]


instance grade.gradedMonoid [AddMonoid M] [CommSemiring R] :
    SetLike.GradedMonoid (grade R : M → Submodule R R[M]) := by
  /-
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝¹ : AddMonoid M
    inst✝ : CommSemiring R
    ⊢ SetLike.GradedMonoid (AddMonoidAlgebra.grade R)
  -/
  apply gradeBy.gradedMonoid (AddMonoidHom.id _)
  /-
    🎉 no goals
  -/


/-- Auxiliary definition; the canonical grade decomposition, used to provide
`DirectSum.decompose`. -/
def decomposeAux : R[M] →ₐ[R] ⨁ i : ι, gradeBy R f i :=
  AddMonoidAlgebra.lift R M _
    { toFun := fun m =>
        DirectSum.of (fun i : ι => gradeBy R f i) (f m.toAdd)
          ⟨Finsupp.single m.toAdd 1, single_mem_gradeBy _ _ _⟩
      map_one' :=
        DirectSum.of_eq_of_gradedMonoid_eq
              /-
                M : Type u_1
                ι : Type u_2
                R : Type u_3
                inst✝³ : AddMonoid M
                inst✝² : DecidableEq ι
                inst✝¹ : AddMonoid ι
                inst✝ : CommSemiring R
                f : AddMonoidHom M ι
                ⊢ Eq (GradedMonoid.mk (f (Multiplicative.toAdd 1)) ⟨Finsupp.single (Multiplica …
              -/
                          /-
                            🎉 no goals
                          -/
                          /-
                            🎉 no goals
                          -/
          (by congr 2 <;> simp)
                          /-
                            🎉 no goals
                          -/
      map_mul' := fun i j => by
        /-
          M : Type u_1
          ι : Type u_2
          R : Type u_3
          inst✝³ : AddMonoid M
          inst✝² : DecidableEq ι
          inst✝¹ : AddMonoid ι
          inst✝ : CommSemiring R
          f : AddMonoidHom M ι
          i j : Multiplicative M
          ⊢ Eq ({ toFun := fun m => (DirectSum.of (fun i => Subtype fun x => Membership. …
        -/
        symm
        dsimp only [toAdd_one, Eq.ndrec, Set.mem_setOf_eq, ne_eq, OneHom.toFun_eq_coe,
          OneHom.coe_mk, toAdd_mul]
        /-
          M : Type u_1
          ι : Type u_2
          R : Type u_3
          inst✝³ : AddMonoid M
          inst✝² : DecidableEq ι
          inst✝¹ : AddMonoid ι
          inst✝ : CommSemiring R
          f : AddMonoidHom M ι
          i j : Multiplicative M
          ⊢ Eq (HMul.hMul ((DirectSum.of (fun i => Subtype fun x => Membership.mem (AddM …
        -/
        convert DirectSum.of_mul_of (A := (fun i : ι => gradeBy R f i)) _ _
        /-
          case h.e'_3.h.e'_1.h.e'_1.h.e'_2.h.h.e'_4.h.e'_6
          M : Type u_1
          ι : Type u_2
          R : Type u_3
          inst✝³ : AddMonoid M
          inst✝² : DecidableEq ι
          inst✝¹ : AddMonoid ι
          inst✝ : CommSemiring R
          f : AddMonoidHom M ι
          i j : Multiplicative M
          x✝ : AddMonoidAlgebra R M
          ⊢ Eq (f (HAdd.hAdd (Multiplicative.toAdd i) (Multiplicative.toAdd j))) (HAdd.h …
        -/
        repeat { rw [AddMonoidHom.map_add] }
        /-
          case h.e'_3.h.e'_6.e'_3
          M : Type u_1
          ι : Type u_2
          R : Type u_3
          inst✝³ : AddMonoid M
          inst✝² : DecidableEq ι
          inst✝¹ : AddMonoid ι
          inst✝ : CommSemiring R
          f : AddMonoidHom M ι
          i j : Multiplicative M
          e_2✝ : Eq (Subtype fun x => Membership.mem (AddMonoidAlgebra.gradeBy R (⇑f) (f …
          ⊢ Eq (Finsupp.single (HAdd.hAdd (Multiplicative.toAdd i) (Multiplicative.toAdd …
        -/
        simp only [SetLike.coe_gMul]
        /-
          case h.e'_3.h.e'_6.e'_3
          M : Type u_1
          ι : Type u_2
          R : Type u_3
          inst✝³ : AddMonoid M
          inst✝² : DecidableEq ι
          inst✝¹ : AddMonoid ι
          inst✝ : CommSemiring R
          f : AddMonoidHom M ι
          i j : Multiplicative M
          e_2✝ : Eq (Subtype fun x => Membership.mem (AddMonoidAlgebra.gradeBy R (⇑f) (f …
          ⊢ Eq (Finsupp.single (HAdd.hAdd (Multiplicative.toAdd i) (Multiplicative.toAdd …
        -/
        exact Eq.trans (by rw [one_mul]) single_mul_single.symm }
        /-
          🎉 no goals
        -/


theorem decomposeAux_single (m : M) (r : R) :
    decomposeAux f (Finsupp.single m r) =
      DirectSum.of (fun i : ι => gradeBy R f i) (f m)
        ⟨Finsupp.single m r, single_mem_gradeBy _ _ _⟩ := by
  /-
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    m : M
    r : R
    ⊢ Eq ((AddMonoidAlgebra.decomposeAux f) (Finsupp.single m r)) ((DirectSum.of ( …
  -/
  refine (lift_single _ _ _).trans ?_
  /-
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    m : M
    r : R
    ⊢ Eq (HSMul.hSMul r ({ toFun := fun m => (DirectSum.of (fun i => Subtype fun x …
  -/
  refine (DirectSum.of_smul R _ _ _).symm.trans ?_
  /-
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    m : M
    r : R
    ⊢ Eq ((DirectSum.of (fun i => Subtype fun x => Membership.mem (AddMonoidAlgebr …
  -/
  apply DirectSum.of_eq_of_gradedMonoid_eq
  /-
    case h
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    m : M
    r : R
    ⊢ Eq (GradedMonoid.mk (f (Multiplicative.toAdd (Multiplicative.ofAdd m))) (HSM …
  -/
  refine Sigma.subtype_ext rfl ?_
  /-
    case h
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    m : M
    r : R
    ⊢ Eq ↑(GradedMonoid.mk (f (Multiplicative.toAdd (Multiplicative.ofAdd m))) (HS …
  -/
  refine (smul_single' _ _ _).trans ?_
  /-
    case h
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    m : M
    r : R
    ⊢ Eq (AddMonoidAlgebra.single (Multiplicative.toAdd (Multiplicative.ofAdd m))  …
  -/
  rw [mul_one]
  /-
    case h
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    m : M
    r : R
    ⊢ Eq (AddMonoidAlgebra.single (Multiplicative.toAdd (Multiplicative.ofAdd m))  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem decomposeAux_coe {i : ι} (x : gradeBy R f i) :
    decomposeAux f ↑x = DirectSum.of (fun i => gradeBy R f i) i x := by
  classical
  obtain ⟨x, hx⟩ := x
  revert hx
  refine Finsupp.induction x ?_ ?_
  · intro hx
    symm
    exact AddMonoidHom.map_zero _
  · intro m b y hmy hb ih hmby
    have : Disjoint (Finsupp.single m b).support y.support := by
      simpa only [Finsupp.support_single_ne_zero _ hb, Finset.disjoint_singleton_left]
    rw [mem_gradeBy_iff, Finsupp.support_add_eq this, Finset.coe_union, Set.union_subset_iff]
      at hmby
    cases' hmby with h1 h2
    have : f m = i := by
      rwa [Finsupp.support_single_ne_zero _ hb, Finset.coe_singleton, Set.singleton_subset_iff]
        at h1
    subst this
    simp only [map_add, Submodule.coe_mk, decomposeAux_single f m]
    let ih' := ih h2
    dsimp at ih'
    rw [ih', ← AddMonoidHom.map_add]
    apply DirectSum.of_eq_of_gradedMonoid_eq
    congr 2


instance gradeBy.gradedAlgebra : GradedAlgebra (gradeBy R f) :=
  GradedAlgebra.ofAlgHom _ (decomposeAux f)
    (by
      /-
        M : Type u_1
        ι : Type u_2
        R : Type u_3
        inst✝³ : AddMonoid M
        inst✝² : DecidableEq ι
        inst✝¹ : AddMonoid ι
        inst✝ : CommSemiring R
        f : AddMonoidHom M ι
        ⊢ Eq ((DirectSum.coeAlgHom (AddMonoidAlgebra.gradeBy R ⇑f)).comp (AddMonoidAlg …
      -/
      ext : 2
      simp only [MonoidHom.coe_comp, MonoidHom.coe_coe, AlgHom.coe_comp, Function.comp_apply,
        of_apply, AlgHom.coe_id, id_eq]
      /-
        case h.h
        M : Type u_1
        ι : Type u_2
        R : Type u_3
        inst✝³ : AddMonoid M
        inst✝² : DecidableEq ι
        inst✝¹ : AddMonoid ι
        inst✝ : CommSemiring R
        f : AddMonoidHom M ι
        x✝ : Multiplicative M
        ⊢ Eq ((DirectSum.coeAlgHom (AddMonoidAlgebra.gradeBy R ⇑f)) ((AddMonoidAlgebra …
      -/
      rw [decomposeAux_single, DirectSum.coeAlgHom_of, Subtype.coe_mk])
      /-
        🎉 no goals
      -/
                  /-
                    M : Type u_1
                    ι : Type u_2
                    R : Type u_3
                    inst✝³ : AddMonoid M
                    inst✝² : DecidableEq ι
                    inst✝¹ : AddMonoid ι
                    inst✝ : CommSemiring R
                    f : AddMonoidHom M ι
                    i : ι
                    x : Subtype fun x => Membership.mem (AddMonoidAlgebra.gradeBy R (⇑f) i) x
                    ⊢ Eq ((AddMonoidAlgebra.decomposeAux f) ↑x) ((DirectSum.of (fun i => Subtype f …
                  -/
    fun i x => by rw [decomposeAux_coe f x]
                  /-
                    🎉 no goals
                  -/

-- Lean can't find this later without us repeating it

                                                                             /-
                                                                               M : Type u_1
                                                                               ι : Type u_2
                                                                               R : Type u_3
                                                                               inst✝³ : AddMonoid M
                                                                               inst✝² : DecidableEq ι
                                                                               inst✝¹ : AddMonoid ι
                                                                               inst✝ : CommSemiring R
                                                                               f : AddMonoidHom M ι
                                                                               ⊢ DirectSum.Decomposition (AddMonoidAlgebra.gradeBy R ⇑f)
                                                                             -/
instance gradeBy.decomposition : DirectSum.Decomposition (gradeBy R f) := by infer_instance
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem decomposeAux_eq_decompose :
    ⇑(decomposeAux f : R[M] →ₐ[R] ⨁ i : ι, gradeBy R f i) =
      DirectSum.decompose (gradeBy R f) :=
  rfl


theorem GradesBy.decompose_single (m : M) (r : R) :
    DirectSum.decompose (gradeBy R f) (Finsupp.single m r : R[M]) =
      DirectSum.of (fun i : ι => gradeBy R f i) (f m)
        ⟨Finsupp.single m r, single_mem_gradeBy _ _ _⟩ :=
  decomposeAux_single _ _ _


instance grade.gradedAlgebra : GradedAlgebra (grade R : ι → Submodule _ _) :=
  AddMonoidAlgebra.gradeBy.gradedAlgebra (AddMonoidHom.id _)

-- Lean can't find this later without us repeating it

instance grade.decomposition : DirectSum.Decomposition (grade R : ι → Submodule _ _) := by
  /-
    M : Type u_1
    ι : Type u_2
    R : Type u_3
    inst✝³ : AddMonoid M
    inst✝² : DecidableEq ι
    inst✝¹ : AddMonoid ι
    inst✝ : CommSemiring R
    f : AddMonoidHom M ι
    ⊢ DirectSum.Decomposition (AddMonoidAlgebra.grade R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem grade.decompose_single (i : ι) (r : R) :
    DirectSum.decompose (grade R : ι → Submodule _ _) (Finsupp.single i r : AddMonoidAlgebra _ _) =
      DirectSum.of (fun i : ι => grade R i) i ⟨Finsupp.single i r, single_mem_grade _ _⟩ :=
  decomposeAux_single _ _ _


/-- `AddMonoidAlgebra.gradeBy` describe an internally graded algebra. -/
theorem gradeBy.isInternal : DirectSum.IsInternal (gradeBy R f) :=
  DirectSum.Decomposition.isInternal _


/-- `AddMonoidAlgebra.grade` describe an internally graded algebra. -/
theorem grade.isInternal : DirectSum.IsInternal (grade R : ι → Submodule R _) :=
  DirectSum.Decomposition.isInternal _


