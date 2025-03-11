theorem adic_basis (I : Ideal R) : SubmodulesRingBasis fun n : ℕ => (I ^ n • ⊤ : Ideal R) :=
  { inter := by
      suffices ∀ i j : ℕ, ∃ k, I ^ k ≤ I ^ i ∧ I ^ k ≤ I ^ j by
        simpa only [smul_eq_mul, mul_top, Algebra.id.map_eq_id, map_id, le_inf_iff] using this
      /-
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        ⊢ ∀ (i j : Nat), Exists fun k => And (LE.le (HPow.hPow I k) (HPow.hPow I i)) ( …
      -/
      intro i j
      /-
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        i j : Nat
        ⊢ Exists fun k => And (LE.le (HPow.hPow I k) (HPow.hPow I i)) (LE.le (HPow.hPo …
      -/
      exact ⟨max i j, pow_le_pow_right (le_max_left i j), pow_le_pow_right (le_max_right i j)⟩
      /-
        🎉 no goals
      -/
    leftMul := by
      suffices ∀ (a : R) (i : ℕ), ∃ j : ℕ, a • I ^ j ≤ I ^ i by
        simpa only [smul_top_eq_map, Algebra.id.map_eq_id, map_id] using this
      /-
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        ⊢ ∀ (a : R) (i : Nat), Exists fun j => LE.le (HSMul.hSMul a (HPow.hPow I j)) ( …
      -/
      intro r n
      /-
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        r : R
        n : Nat
        ⊢ Exists fun j => LE.le (HSMul.hSMul r (HPow.hPow I j)) (HPow.hPow I n)
      -/
      use n
      /-
        case h
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        r : R
        n : Nat
        ⊢ LE.le (HSMul.hSMul r (HPow.hPow I n)) (HPow.hPow I n)
      -/
      rintro a ⟨x, hx, rfl⟩
      /-
        case h.intro.intro
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        r : R
        n : Nat
        x : R
        hx : Membership.mem (↑(HPow.hPow I n)) x
        ⊢ Membership.mem (HPow.hPow I n) ((DistribMulAction.toLinearMap R R r) x)
      -/
      exact (I ^ n).smul_mem r hx
      /-
        🎉 no goals
      -/
    mul := by
      suffices ∀ i : ℕ, ∃ j : ℕ, (↑(I ^ j) * ↑(I ^ j) : Set R) ⊆ (↑(I ^ i) : Set R) by
        simpa only [smul_top_eq_map, Algebra.id.map_eq_id, map_id] using this
      /-
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        ⊢ ∀ (i : Nat), Exists fun j => HasSubset.Subset (HMul.hMul ↑(HPow.hPow I j) ↑( …
      -/
      intro n
      /-
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        ⊢ Exists fun j => HasSubset.Subset (HMul.hMul ↑(HPow.hPow I j) ↑(HPow.hPow I j …
      -/
      use n
      /-
        case h
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        ⊢ HasSubset.Subset (HMul.hMul ↑(HPow.hPow I n) ↑(HPow.hPow I n)) ↑(HPow.hPow I …
      -/
      rintro a ⟨x, _hx, b, hb, rfl⟩
      /-
        case h.intro.intro.intro.intro
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        n : Nat
        x : R
        _hx : Membership.mem (↑(HPow.hPow I n)) x
        b : R
        hb : Membership.mem (↑(HPow.hPow I n)) b
        ⊢ Membership.mem (↑(HPow.hPow I n)) ((fun x1 x2 => HMul.hMul x1 x2) x b)
      -/
      exact (I ^ n).smul_mem x hb }
      /-
        🎉 no goals
      -/


/-- The adic ring filter basis associated to an ideal `I` is made of powers of `I`. -/
def ringFilterBasis (I : Ideal R) :=
  I.adic_basis.toRing_subgroups_basis.toRingFilterBasis


/-- The adic topology associated to an ideal `I`. This topology admits powers of `I` as a basis of
neighborhoods of zero. It is compatible with the ring structure and is non-archimedean. -/
def adicTopology (I : Ideal R) : TopologicalSpace R :=
  (adic_basis I).topology


theorem nonarchimedean (I : Ideal R) : @NonarchimedeanRing R _ I.adicTopology :=
  I.adic_basis.toRing_subgroups_basis.nonarchimedean


/-- For the `I`-adic topology, the neighborhoods of zero has basis given by the powers of `I`. -/
theorem hasBasis_nhds_zero_adic (I : Ideal R) :
    HasBasis (@nhds R I.adicTopology (0 : R)) (fun _n : ℕ => True) fun n =>
      ((I ^ n : Ideal R) : Set R) :=
  ⟨by
    /-
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      ⊢ ∀ (t : Set R), Iff (Membership.mem (nhds 0) t) (Exists fun i => And True (Ha …
    -/
    intro U
    /-
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      U : Set R
      ⊢ Iff (Membership.mem (nhds 0) U) (Exists fun i => And True (HasSubset.Subset  …
    -/
    rw [I.ringFilterBasis.toAddGroupFilterBasis.nhds_zero_hasBasis.mem_iff]
    /-
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      U : Set R
      ⊢ Iff (Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBas …
    -/
    constructor
      /-
        case mp
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        U : Set R
        ⊢ (Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBasis i …
      -/
    · rintro ⟨-, ⟨i, rfl⟩, h⟩
      /-
        case mp.intro.intro.intro
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        U : Set R
        i : Nat
        h : HasSubset.Subset (id ↑((fun i => Submodule.toAddSubgroup (HSMul.hSMul (HPo …
        ⊢ Exists fun i => And True (HasSubset.Subset (↑(HPow.hPow I i)) U)
      -/
      replace h : ↑(I ^ i) ⊆ U := by simpa using h
      /-
        case mp.intro.intro.intro
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        U : Set R
        i : Nat
        h : HasSubset.Subset (↑(HPow.hPow I i)) U
        ⊢ Exists fun i => And True (HasSubset.Subset (↑(HPow.hPow I i)) U)
      -/
      exact ⟨i, trivial, h⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        U : Set R
        ⊢ (Exists fun i => And True (HasSubset.Subset (↑(HPow.hPow I i)) U)) → Exists  …
      -/
    · rintro ⟨i, -, h⟩
      /-
        case mpr.intro.intro
        R : Type u_1
        inst✝ : CommRing R
        I : Ideal R
        U : Set R
        i : Nat
        h : HasSubset.Subset (↑(HPow.hPow I i)) U
        ⊢ Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBasis i) …
      -/
      exact ⟨(I ^ i : Ideal R), ⟨i, by simp⟩, h⟩⟩
      /-
        🎉 no goals
      -/


theorem hasBasis_nhds_adic (I : Ideal R) (x : R) :
    HasBasis (@nhds R I.adicTopology x) (fun _n : ℕ => True) fun n =>
      (fun y => x + y) '' (I ^ n : Ideal R) := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    x : R
    ⊢ (nhds x).HasBasis (fun _n => True) fun n => Set.image (fun y => HAdd.hAdd x  …
  -/
  letI := I.adicTopology
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    x : R
    this : TopologicalSpace R := I.adicTopology
    ⊢ (nhds x).HasBasis (fun _n => True) fun n => Set.image (fun y => HAdd.hAdd x  …
  -/
  have := I.hasBasis_nhds_zero_adic.map fun y => x + y
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    x : R
    this✝ : TopologicalSpace R := I.adicTopology
    this : (Filter.map (fun y => HAdd.hAdd x y) (nhds 0)).HasBasis (fun _n => True …
    ⊢ (nhds x).HasBasis (fun _n => True) fun n => Set.image (fun y => HAdd.hAdd x  …
  -/
  rwa [map_add_left_nhds_zero x] at this
  /-
    🎉 no goals
  -/


theorem adic_module_basis :
    I.ringFilterBasis.SubmodulesBasis fun n : ℕ => I ^ n • (⊤ : Submodule R M) :=
  { inter := fun i j =>
      ⟨max i j,
        le_inf_iff.mpr
          ⟨smul_mono_left <| pow_le_pow_right (le_max_left i j),
            smul_mono_left <| pow_le_pow_right (le_max_right i j)⟩⟩
    smul := fun m i =>
                                     /-
                                       R : Type u_1
                                       inst✝² : CommRing R
                                       I : Ideal R
                                       M : Type u_2
                                       inst✝¹ : AddCommGroup M
                                       inst✝ : Module R M
                                       m : M
                                       i : Nat
                                       ⊢ Eq ↑(HSMul.hSMul (HPow.hPow I i) Top.top) ↑((fun i => Submodule.toAddSubgrou …
                                     -/
      ⟨(I ^ i • ⊤ : Ideal R), ⟨i, by simp⟩, fun a a_in => by
                                     /-
                                       🎉 no goals
                                     -/
        /-
          R : Type u_1
          inst✝² : CommRing R
          I : Ideal R
          M : Type u_2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          m : M
          i : Nat
          a : R
          a_in : Membership.mem (↑(HSMul.hSMul (HPow.hPow I i) Top.top)) a
          ⊢ Membership.mem (Set.preimage (fun x => HSMul.hSMul x m) ↑(HSMul.hSMul (HPow. …
        -/
        replace a_in : a ∈ I ^ i := by simpa [(I ^ i).mul_top] using a_in
        /-
          R : Type u_1
          inst✝² : CommRing R
          I : Ideal R
          M : Type u_2
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          m : M
          i : Nat
          a : R
          a_in : Membership.mem (HPow.hPow I i) a
          ⊢ Membership.mem (Set.preimage (fun x => HSMul.hSMul x m) ↑(HSMul.hSMul (HPow. …
        -/
        exact smul_mem_smul a_in mem_top⟩ }
        /-
          🎉 no goals
        -/


/-- The topology on an `R`-module `M` associated to an ideal `M`. Submodules $I^n M$,
written `I^n • ⊤` form a basis of neighborhoods of zero. -/
def adicModuleTopology : TopologicalSpace M :=
  @ModuleFilterBasis.topology R M _ I.adic_basis.topology _ _
    (I.ringFilterBasis.moduleFilterBasis (I.adic_module_basis M))


/-- The elements of the basis of neighborhoods of zero for the `I`-adic topology
on an `R`-module `M`, seen as open additive subgroups of `M`. -/
def openAddSubgroup (n : ℕ) : @OpenAddSubgroup R _ I.adicTopology := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    ⊢ OpenAddSubgroup R
  -/
  letI := I.adicTopology
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    this : TopologicalSpace R := I.adicTopology
    ⊢ OpenAddSubgroup R
  -/
  refine ⟨(I ^ n).toAddSubgroup, ?_⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    this : TopologicalSpace R := I.adicTopology
    ⊢ IsOpen (Submodule.toAddSubgroup (HPow.hPow I n)).carrier
  -/
  convert (I.adic_basis.toRing_subgroups_basis.openAddSubgroup n).isOpen
  /-
    case h.e'_3
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    this : TopologicalSpace R := I.adicTopology
    ⊢ Eq (Submodule.toAddSubgroup (HPow.hPow I n)).carrier ↑(⋯.openAddSubgroup n)
  -/
  change (↑(I ^ n) : Set R) = ↑(I ^ n • (⊤ : Ideal R))
  /-
    case h.e'_3
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    this : TopologicalSpace R := I.adicTopology
    ⊢ Eq ↑(HPow.hPow I n) ↑(HSMul.hSMul (HPow.hPow I n) Top.top)
  -/
  simp [smul_top_eq_map, Algebra.id.map_eq_id, map_id, restrictScalars_self]
  /-
    🎉 no goals
  -/


/-- Given a topology on a ring `R` and an ideal `J`, `IsAdic J` means the topology is the
`J`-adic one. -/
def IsAdic [H : TopologicalSpace R] (J : Ideal R) : Prop :=
  H = J.adicTopology


/-- A topological ring is `J`-adic if and only if it admits the powers of `J` as a basis of
open neighborhoods of zero. -/
theorem isAdic_iff [top : TopologicalSpace R] [TopologicalRing R] {J : Ideal R} :
    IsAdic J ↔
      (∀ n : ℕ, IsOpen ((J ^ n : Ideal R) : Set R)) ∧
        ∀ s ∈ 𝓝 (0 : R), ∃ n : ℕ, ((J ^ n : Ideal R) : Set R) ⊆ s := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    top : TopologicalSpace R
    inst✝ : TopologicalRing R
    J : Ideal R
    ⊢ Iff (IsAdic J) (And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), M …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      top : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      ⊢ IsAdic J → And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Member …
    -/
  · intro H
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      top : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      H : IsAdic J
      ⊢ And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem (n …
    -/
    change _ = _ at H
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      top : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      H : Eq top J.adicTopology
      ⊢ And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem (n …
    -/
    rw [H]
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      top : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      H : Eq top J.adicTopology
      ⊢ And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem (n …
    -/
    letI := J.adicTopology
    /-
      case mp
      R : Type u_1
      inst✝¹ : CommRing R
      top : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      H : Eq top J.adicTopology
      this : TopologicalSpace R := J.adicTopology
      ⊢ And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem (n …
    -/
    constructor
      /-
        case mp.left
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H : Eq top J.adicTopology
        this : TopologicalSpace R := J.adicTopology
        ⊢ ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
      -/
    · intro n
      /-
        case mp.left
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H : Eq top J.adicTopology
        this : TopologicalSpace R := J.adicTopology
        n : Nat
        ⊢ IsOpen ↑(HPow.hPow J n)
      -/
      exact (J.openAddSubgroup n).isOpen'
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H : Eq top J.adicTopology
        this : TopologicalSpace R := J.adicTopology
        ⊢ ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subset  …
      -/
    · intro s hs
      /-
        case mp.right
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H : Eq top J.adicTopology
        this : TopologicalSpace R := J.adicTopology
        s : Set R
        hs : Membership.mem (nhds 0) s
        ⊢ Exists fun n => HasSubset.Subset (↑(HPow.hPow J n)) s
      -/
      simpa using J.hasBasis_nhds_zero_adic.mem_iff.mp hs
      /-
        🎉 no goals
      -/
    /-
      case mpr
      R : Type u_1
      inst✝¹ : CommRing R
      top : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      ⊢ And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem (n …
    -/
  · rintro ⟨H₁, H₂⟩
    /-
      case mpr.intro
      R : Type u_1
      inst✝¹ : CommRing R
      top : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
      H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
      ⊢ IsAdic J
    -/
    apply TopologicalAddGroup.ext
      /-
        case mpr.intro.tg
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
        H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
        ⊢ TopologicalAddGroup R
      -/
    · apply @TopologicalRing.to_topologicalAddGroup
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.tg'
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
        H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
        ⊢ TopologicalAddGroup R
      -/
    · apply (RingSubgroupsBasis.toRingFilterBasis _).toAddGroupFilterBasis.isTopologicalAddGroup
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.h
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
        H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
        ⊢ Eq (nhds 0) (nhds 0)
      -/
    · ext s
      /-
        case mpr.intro.h.h
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
        H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
        s : Set R
        ⊢ Iff (Membership.mem (nhds 0) s) (Membership.mem (nhds 0) s)
      -/
      letI := Ideal.adic_basis J
      /-
        case mpr.intro.h.h
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
        H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
        s : Set R
        this : SubmodulesRingBasis fun n => HSMul.hSMul (HPow.hPow J n) Top.top := Ide …
        ⊢ Iff (Membership.mem (nhds 0) s) (Membership.mem (nhds 0) s)
      -/
      rw [J.hasBasis_nhds_zero_adic.mem_iff]
      /-
        case mpr.intro.h.h
        R : Type u_1
        inst✝¹ : CommRing R
        top : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
        H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
        s : Set R
        this : SubmodulesRingBasis fun n => HSMul.hSMul (HPow.hPow J n) Top.top := Ide …
        ⊢ Iff (Membership.mem (nhds 0) s) (Exists fun i => And True (HasSubset.Subset  …
      -/
      constructor <;> intro H
        /-
          case mpr.intro.h.h.mp
          R : Type u_1
          inst✝¹ : CommRing R
          top : TopologicalSpace R
          inst✝ : TopologicalRing R
          J : Ideal R
          H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
          H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
          s : Set R
          this : SubmodulesRingBasis fun n => HSMul.hSMul (HPow.hPow J n) Top.top := Ide …
          H : Membership.mem (nhds 0) s
          ⊢ Exists fun i => And True (HasSubset.Subset (↑(HPow.hPow J i)) s)
        -/
      · rcases H₂ s H with ⟨n, h⟩
        /-
          case mpr.intro.h.h.mp.intro
          R : Type u_1
          inst✝¹ : CommRing R
          top : TopologicalSpace R
          inst✝ : TopologicalRing R
          J : Ideal R
          H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
          H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
          s : Set R
          this : SubmodulesRingBasis fun n => HSMul.hSMul (HPow.hPow J n) Top.top := Ide …
          H : Membership.mem (nhds 0) s
          n : Nat
          h : HasSubset.Subset (↑(HPow.hPow J n)) s
          ⊢ Exists fun i => And True (HasSubset.Subset (↑(HPow.hPow J i)) s)
        -/
        exact ⟨n, trivial, h⟩
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.h.h.mpr
          R : Type u_1
          inst✝¹ : CommRing R
          top : TopologicalSpace R
          inst✝ : TopologicalRing R
          J : Ideal R
          H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
          H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
          s : Set R
          this : SubmodulesRingBasis fun n => HSMul.hSMul (HPow.hPow J n) Top.top := Ide …
          H : Exists fun i => And True (HasSubset.Subset (↑(HPow.hPow J i)) s)
          ⊢ Membership.mem (nhds 0) s
        -/
      · rcases H with ⟨n, -, hn⟩
        /-
          case mpr.intro.h.h.mpr.intro.intro
          R : Type u_1
          inst✝¹ : CommRing R
          top : TopologicalSpace R
          inst✝ : TopologicalRing R
          J : Ideal R
          H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
          H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
          s : Set R
          this : SubmodulesRingBasis fun n => HSMul.hSMul (HPow.hPow J n) Top.top := Ide …
          n : Nat
          hn : HasSubset.Subset (↑(HPow.hPow J n)) s
          ⊢ Membership.mem (nhds 0) s
        -/
        rw [mem_nhds_iff]
        /-
          case mpr.intro.h.h.mpr.intro.intro
          R : Type u_1
          inst✝¹ : CommRing R
          top : TopologicalSpace R
          inst✝ : TopologicalRing R
          J : Ideal R
          H₁ : ∀ (n : Nat), IsOpen ↑(HPow.hPow J n)
          H₂ : ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subs …
          s : Set R
          this : SubmodulesRingBasis fun n => HSMul.hSMul (HPow.hPow J n) Top.top := Ide …
          n : Nat
          hn : HasSubset.Subset (↑(HPow.hPow J n)) s
          ⊢ Exists fun t => And (HasSubset.Subset t s) (And (IsOpen t) (Membership.mem t …
        -/
        exact ⟨_, hn, H₁ n, (J ^ n).zero_mem⟩
        /-
          🎉 no goals
        -/


theorem is_ideal_adic_pow {J : Ideal R} (h : IsAdic J) {n : ℕ} (hn : 0 < n) : IsAdic (J ^ n) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalRing R
    J : Ideal R
    h : IsAdic J
    n : Nat
    hn : LT.lt 0 n
    ⊢ IsAdic (HPow.hPow J n)
  -/
  rw [isAdic_iff] at h ⊢
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalRing R
    J : Ideal R
    h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
    n : Nat
    hn : LT.lt 0 n
    ⊢ And (∀ (n_1 : Nat), IsOpen ↑(HPow.hPow (HPow.hPow J n) n_1)) (∀ (s : Set R), …
  -/
  constructor
    /-
      case left
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      ⊢ ∀ (n_1 : Nat), IsOpen ↑(HPow.hPow (HPow.hPow J n) n_1)
    -/
  · intro m
    /-
      case left
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      m : Nat
      ⊢ IsOpen ↑(HPow.hPow (HPow.hPow J n) m)
    -/
    rw [← pow_mul]
    /-
      case left
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      m : Nat
      ⊢ IsOpen ↑(HPow.hPow J (HMul.hMul n m))
    -/
    apply h.left
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      ⊢ ∀ (s : Set R), Membership.mem (nhds 0) s → Exists fun n_1 => HasSubset.Subse …
    -/
  · intro V hV
    /-
      case right
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      V : Set R
      hV : Membership.mem (nhds 0) V
      ⊢ Exists fun n_1 => HasSubset.Subset (↑(HPow.hPow (HPow.hPow J n) n_1)) V
    -/
    cases' h.right V hV with m hm
    /-
      case right.intro
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      V : Set R
      hV : Membership.mem (nhds 0) V
      m : Nat
      hm : HasSubset.Subset (↑(HPow.hPow J m)) V
      ⊢ Exists fun n_1 => HasSubset.Subset (↑(HPow.hPow (HPow.hPow J n) n_1)) V
    -/
    use m
    /-
      case h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      V : Set R
      hV : Membership.mem (nhds 0) V
      m : Nat
      hm : HasSubset.Subset (↑(HPow.hPow J m)) V
      ⊢ HasSubset.Subset (↑(HPow.hPow (HPow.hPow J n) m)) V
    -/
    refine Set.Subset.trans ?_ hm
    /-
      case h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      n : Nat
      hn : LT.lt 0 n
      V : Set R
      hV : Membership.mem (nhds 0) V
      m : Nat
      hm : HasSubset.Subset (↑(HPow.hPow J m)) V
      ⊢ HasSubset.Subset ↑(HPow.hPow (HPow.hPow J n) m) ↑(HPow.hPow J m)
    -/
    cases n
      /-
        case h.zero
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
        V : Set R
        hV : Membership.mem (nhds 0) V
        m : Nat
        hm : HasSubset.Subset (↑(HPow.hPow J m)) V
        hn : LT.lt 0 0
        ⊢ HasSubset.Subset ↑(HPow.hPow (HPow.hPow J 0) m) ↑(HPow.hPow J m)
      -/
    · exfalso
      /-
        case h.zero
        R : Type u_1
        inst✝² : CommRing R
        inst✝¹ : TopologicalSpace R
        inst✝ : TopologicalRing R
        J : Ideal R
        h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
        V : Set R
        hV : Membership.mem (nhds 0) V
        m : Nat
        hm : HasSubset.Subset (↑(HPow.hPow J m)) V
        hn : LT.lt 0 0
        ⊢ False
      -/
      exact Nat.not_succ_le_zero 0 hn
      /-
        🎉 no goals
      -/
    /-
      case h.succ
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      V : Set R
      hV : Membership.mem (nhds 0) V
      m : Nat
      hm : HasSubset.Subset (↑(HPow.hPow J m)) V
      n✝ : Nat
      hn : LT.lt 0 (HAdd.hAdd n✝ 1)
      ⊢ HasSubset.Subset ↑(HPow.hPow (HPow.hPow J (HAdd.hAdd n✝ 1)) m) ↑(HPow.hPow J …
    -/
    rw [← pow_mul, Nat.succ_mul]
    /-
      case h.succ
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      V : Set R
      hV : Membership.mem (nhds 0) V
      m : Nat
      hm : HasSubset.Subset (↑(HPow.hPow J m)) V
      n✝ : Nat
      hn : LT.lt 0 (HAdd.hAdd n✝ 1)
      ⊢ HasSubset.Subset ↑(HPow.hPow J (HAdd.hAdd (HMul.hMul n✝ m) m)) ↑(HPow.hPow J …
    -/
    apply Ideal.pow_le_pow_right
    /-
      case h.succ.h
      R : Type u_1
      inst✝² : CommRing R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalRing R
      J : Ideal R
      h : And (∀ (n : Nat), IsOpen ↑(HPow.hPow J n)) (∀ (s : Set R), Membership.mem  …
      V : Set R
      hV : Membership.mem (nhds 0) V
      m : Nat
      hm : HasSubset.Subset (↑(HPow.hPow J m)) V
      n✝ : Nat
      hn : LT.lt 0 (HAdd.hAdd n✝ 1)
      ⊢ LE.le m (HAdd.hAdd (HMul.hMul n✝ m) m)
    -/
    apply Nat.le_add_left
    /-
      🎉 no goals
    -/


theorem is_bot_adic_iff {A : Type*} [CommRing A] [TopologicalSpace A] [TopologicalRing A] :
    IsAdic (⊥ : Ideal A) ↔ DiscreteTopology A := by
  /-
    A : Type u_2
    inst✝² : CommRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalRing A
    ⊢ Iff (IsAdic Bot.bot) (DiscreteTopology A)
  -/
  rw [isAdic_iff]
  /-
    A : Type u_2
    inst✝² : CommRing A
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalRing A
    ⊢ Iff (And (∀ (n : Nat), IsOpen ↑(HPow.hPow Bot.bot n)) (∀ (s : Set A), Member …
  -/
  constructor
    /-
      case mp
      A : Type u_2
      inst✝² : CommRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      ⊢ And (∀ (n : Nat), IsOpen ↑(HPow.hPow Bot.bot n)) (∀ (s : Set A), Membership. …
    -/
  · rintro ⟨h, _h'⟩
    /-
      case mp.intro
      A : Type u_2
      inst✝² : CommRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      h : ∀ (n : Nat), IsOpen ↑(HPow.hPow Bot.bot n)
      _h' : ∀ (s : Set A), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Sub …
      ⊢ DiscreteTopology A
    -/
    rw [discreteTopology_iff_isOpen_singleton_zero]
    /-
      case mp.intro
      A : Type u_2
      inst✝² : CommRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      h : ∀ (n : Nat), IsOpen ↑(HPow.hPow Bot.bot n)
      _h' : ∀ (s : Set A), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Sub …
      ⊢ IsOpen (Singleton.singleton 0)
    -/
    simpa using h 1
    /-
      🎉 no goals
    -/
    /-
      case mpr
      A : Type u_2
      inst✝² : CommRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      ⊢ DiscreteTopology A → And (∀ (n : Nat), IsOpen ↑(HPow.hPow Bot.bot n)) (∀ (s  …
    -/
  · intros
    /-
      case mpr
      A : Type u_2
      inst✝² : CommRing A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      a✝ : DiscreteTopology A
      ⊢ And (∀ (n : Nat), IsOpen ↑(HPow.hPow Bot.bot n)) (∀ (s : Set A), Membership. …
    -/
    constructor
      /-
        case mpr.left
        A : Type u_2
        inst✝² : CommRing A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        a✝ : DiscreteTopology A
        ⊢ ∀ (n : Nat), IsOpen ↑(HPow.hPow Bot.bot n)
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mpr.right
        A : Type u_2
        inst✝² : CommRing A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        a✝ : DiscreteTopology A
        ⊢ ∀ (s : Set A), Membership.mem (nhds 0) s → Exists fun n => HasSubset.Subset  …
      -/
    · intro U U_nhds
      /-
        case mpr.right
        A : Type u_2
        inst✝² : CommRing A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        a✝ : DiscreteTopology A
        U : Set A
        U_nhds : Membership.mem (nhds 0) U
        ⊢ Exists fun n => HasSubset.Subset (↑(HPow.hPow Bot.bot n)) U
      -/
      use 1
      /-
        case h
        A : Type u_2
        inst✝² : CommRing A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        a✝ : DiscreteTopology A
        U : Set A
        U_nhds : Membership.mem (nhds 0) U
        ⊢ HasSubset.Subset (↑(HPow.hPow Bot.bot 1)) U
      -/
      simp [mem_of_mem_nhds U_nhds]
      /-
        🎉 no goals
      -/


/-- The ring `R` is equipped with a preferred ideal. -/
class WithIdeal (R : Type*) [CommRing R] where
  i : Ideal R


instance (priority := 100) : TopologicalSpace R :=
  i.adicTopology


instance (priority := 100) : NonarchimedeanRing R :=
  RingSubgroupsBasis.nonarchimedean _


instance (priority := 100) : UniformSpace R :=
  TopologicalAddGroup.toUniformSpace R


instance (priority := 100) : UniformAddGroup R :=
  comm_topologicalAddGroup_is_uniform


/-- The adic topology on an `R` module coming from the ideal `WithIdeal.I`.
This cannot be an instance because `R` cannot be inferred from `M`. -/
def topologicalSpaceModule (M : Type*) [AddCommGroup M] [Module R M] : TopologicalSpace M :=
  (i : Ideal R).adicModuleTopology M

/-
The next examples are kept to make sure potential future refactors won't break the instance
chaining.
-/

