/-- `pi` construction for linear functions. From a family of linear functions it produces a linear
function into a family of modules. -/
def pi (f : (i : ι) → M₂ →ₗ[R] φ i) : M₂ →ₗ[R] (i : ι) → φ i :=
  { Pi.addHom fun i => (f i).toAddHom with
    toFun := fun c i => f i c
    map_smul' := fun _ _ => funext fun i => (f i).map_smul _ _ }


@[simp]
theorem pi_apply (f : (i : ι) → M₂ →ₗ[R] φ i) (c : M₂) (i : ι) : pi f c i = f i c :=
  rfl


theorem ker_pi (f : (i : ι) → M₂ →ₗ[R] φ i) : ker (pi f) = ⨅ i : ι, ker (f i) := by
  /-
    R : Type u
    M₂ : Type w
    ι : Type x
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    φ : ι → Type i
    inst✝¹ : (i : ι) → AddCommMonoid (φ i)
    inst✝ : (i : ι) → Module R (φ i)
    f : (i : ι) → LinearMap (RingHom.id R) M₂ (φ i)
    ⊢ Eq (LinearMap.ker (LinearMap.pi f)) (iInf fun i => LinearMap.ker (f i))
  -/
  ext c; simp [funext_iff]
         /-
           🎉 no goals
         -/


theorem pi_eq_zero (f : (i : ι) → M₂ →ₗ[R] φ i) : pi f = 0 ↔ ∀ i, f i = 0 := by
  /-
    R : Type u
    M₂ : Type w
    ι : Type x
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    φ : ι → Type i
    inst✝¹ : (i : ι) → AddCommMonoid (φ i)
    inst✝ : (i : ι) → Module R (φ i)
    f : (i : ι) → LinearMap (RingHom.id R) M₂ (φ i)
    ⊢ Iff (Eq (LinearMap.pi f) 0) (∀ (i : ι), Eq (f i) 0)
  -/
  simp only [LinearMap.ext_iff, pi_apply, funext_iff]
  /-
    R : Type u
    M₂ : Type w
    ι : Type x
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    φ : ι → Type i
    inst✝¹ : (i : ι) → AddCommMonoid (φ i)
    inst✝ : (i : ι) → Module R (φ i)
    f : (i : ι) → LinearMap (RingHom.id R) M₂ (φ i)
    ⊢ Iff (∀ (x : M₂) (x_1 : ι), Eq ((f x_1) x) (0 x x_1)) (∀ (i : ι) (x : M₂), Eq …
  -/
  exact ⟨fun h a b => h b a, fun h a b => h b a⟩
  /-
    🎉 no goals
  -/


                                                                     /-
                                                                       R : Type u
                                                                       M₂ : Type w
                                                                       ι : Type x
                                                                       inst✝⁴ : Semiring R
                                                                       inst✝³ : AddCommMonoid M₂
                                                                       inst✝² : Module R M₂
                                                                       φ : ι → Type i
                                                                       inst✝¹ : (i : ι) → AddCommMonoid (φ i)
                                                                       inst✝ : (i : ι) → Module R (φ i)
                                                                       ⊢ Eq (LinearMap.pi fun x => 0) 0
                                                                     -/
theorem pi_zero : pi (fun _ => 0 : (i : ι) → M₂ →ₗ[R] φ i) = 0 := by ext; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem pi_comp (f : (i : ι) → M₂ →ₗ[R] φ i) (g : M₃ →ₗ[R] M₂) :
    (pi f).comp g = pi fun i => (f i).comp g :=
  rfl


/-- The projections from a family of modules are linear maps.

Note:  known here as `LinearMap.proj`, this construction is in other categories called `eval`, for
example `Pi.evalMonoidHom`, `Pi.evalRingHom`. -/
def proj (i : ι) : ((i : ι) → φ i) →ₗ[R] φ i where
  toFun := Function.eval i
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
theorem coe_proj (i : ι) : ⇑(proj i : ((i : ι) → φ i) →ₗ[R] φ i) = Function.eval i :=
  rfl


theorem proj_apply (i : ι) (b : (i : ι) → φ i) : (proj i : ((i : ι) → φ i) →ₗ[R] φ i) b = b i :=
  rfl


theorem proj_pi (f : (i : ι) → M₂ →ₗ[R] φ i) (i : ι) : (proj i).comp (pi f) = f i :=
  ext fun _ => rfl


theorem iInf_ker_proj : (⨅ i, ker (proj i : ((i : ι) → φ i) →ₗ[R] φ i) :
    Submodule R ((i : ι) → φ i)) = ⊥ :=
  bot_unique <|
    SetLike.le_def.2 fun a h => by
      /-
        R : Type u
        ι : Type x
        inst✝² : Semiring R
        φ : ι → Type i
        inst✝¹ : (i : ι) → AddCommMonoid (φ i)
        inst✝ : (i : ι) → Module R (φ i)
        a : (i : ι) → φ i
        h : Membership.mem (iInf fun i => LinearMap.ker (LinearMap.proj i)) a
        ⊢ Membership.mem Bot.bot a
      -/
      simp only [mem_iInf, mem_ker, proj_apply] at h
      /-
        R : Type u
        ι : Type x
        inst✝² : Semiring R
        φ : ι → Type i
        inst✝¹ : (i : ι) → AddCommMonoid (φ i)
        inst✝ : (i : ι) → Module R (φ i)
        a : (i : ι) → φ i
        h : ∀ (i : ι), Eq (a i) 0
        ⊢ Membership.mem Bot.bot a
      -/
      exact (mem_bot _).2 (funext fun i => h i)
      /-
        🎉 no goals
      -/


instance CompatibleSMul.pi (R S M N ι : Type*) [Semiring S]
    [AddCommMonoid M] [AddCommMonoid N] [SMul R M] [SMul R N] [Module S M] [Module S N]
    [LinearMap.CompatibleSMul M N R S] : LinearMap.CompatibleSMul M (ι → N) R S where
                       /-
                         R✝ : Type u
                         K : Type u'
                         M✝ : Type v
                         V : Type v'
                         M₂ : Type w
                         V₂ : Type w'
                         M₃ : Type y
                         V₃ : Type y'
                         M₄ : Type z
                         ι✝ : Type x
                         ι' : Type x'
                         inst✝¹⁴ : Semiring R✝
                         inst✝¹³ : AddCommMonoid M₂
                         inst✝¹² : Module R✝ M₂
                         inst✝¹¹ : AddCommMonoid M₃
                         inst✝¹⁰ : Module R✝ M₃
                         φ : ι✝ → Type i
                         inst✝⁹ : (i : ι✝) → AddCommMonoid (φ i)
                         inst✝⁸ : (i : ι✝) → Module R✝ (φ i)
                         R : Type u_1
                         S : Type u_2
                         M : Type u_3
                         N : Type u_4
                         ι : Type u_5
                         inst✝⁷ : Semiring S
                         inst✝⁶ : AddCommMonoid M
                         inst✝⁵ : AddCommMonoid N
                         inst✝⁴ : SMul R M
                         inst✝³ : SMul R N
                         inst✝² : Module S M
                         inst✝¹ : Module S N
                         inst✝ : LinearMap.CompatibleSMul M N R S
                         f : LinearMap (RingHom.id S) M (ι → N)
                         r : R
                         m : M
                         ⊢ Eq (f (HSMul.hSMul r m)) (HSMul.hSMul r (f m))
                       -/
  map_smul f r m := by ext i; apply ((LinearMap.proj i).comp f).map_smul_of_tower
                              /-
                                🎉 no goals
                              -/


/-- Linear map between the function spaces `I → M₂` and `I → M₃`, induced by a linear map `f`
between `M₂` and `M₃`. -/
@[simps]
protected def compLeft (f : M₂ →ₗ[R] M₃) (I : Type*) : (I → M₂) →ₗ[R] I → M₃ :=
  { f.toAddMonoidHom.compLeft I with
    toFun := fun h => f ∘ h
    map_smul' := fun c h => by
      /-
        R : Type u
        K : Type u'
        M : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommMonoid M₂
        inst✝⁴ : Module R M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : Module R M₃
        φ : ι → Type i
        inst✝¹ : (i : ι) → AddCommMonoid (φ i)
        inst✝ : (i : ι) → Module R (φ i)
        f : LinearMap (RingHom.id R) M₂ M₃
        I : Type u_1
        c : R
        h : I → M₂
        ⊢ Eq ({ toFun := fun h => Function.comp (⇑f) h, map_add' := ⋯ }.toFun (HSMul.h …
      -/
      ext x
      /-
        case h
        R : Type u
        K : Type u'
        M : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommMonoid M₂
        inst✝⁴ : Module R M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : Module R M₃
        φ : ι → Type i
        inst✝¹ : (i : ι) → AddCommMonoid (φ i)
        inst✝ : (i : ι) → Module R (φ i)
        f : LinearMap (RingHom.id R) M₂ M₃
        I : Type u_1
        c : R
        h : I → M₂
        x : I
        ⊢ Eq ({ toFun := fun h => Function.comp (⇑f) h, map_add' := ⋯ }.toFun (HSMul.h …
      -/
      exact f.map_smul' c (h x) }
      /-
        🎉 no goals
      -/


theorem apply_single [AddCommMonoid M] [Module R M] [DecidableEq ι] (f : (i : ι) → φ i →ₗ[R] M)
    (i j : ι) (x : φ i) : f j (Pi.single i x j) = (Pi.single i (f i x) : ι → M) j :=
  Pi.apply_single (fun i => f i) (fun i => (f i).map_zero) _ _ _


/-- The `LinearMap` version of `AddMonoidHom.single` and `Pi.single`. -/
def single [DecidableEq ι] (i : ι) : φ i →ₗ[R] (i : ι) → φ i :=
  { AddMonoidHom.single φ i with
    toFun := Pi.single i
    map_smul' := Pi.single_smul i }


lemma single_apply [DecidableEq ι] {i : ι} (v : φ i) :
    single R φ i v = Pi.single i v :=
  rfl


@[simp]
theorem coe_single [DecidableEq ι] (i : ι) :
    ⇑(single R φ i : φ i →ₗ[R] (i : ι) → φ i) = Pi.single i :=
  rfl


theorem proj_comp_single_same (i : ι) : (proj i).comp (single R φ i) = id :=
  LinearMap.ext <| Pi.single_eq_same i


theorem proj_comp_single_ne (i j : ι) (h : i ≠ j) : (proj i).comp (single R φ j) = 0 :=
  LinearMap.ext <| Pi.single_eq_of_ne h


theorem iSup_range_single_le_iInf_ker_proj (I J : Set ι) (h : Disjoint I J) :
    ⨆ i ∈ I, range (single R φ i) ≤ ⨅ i ∈ J, ker (proj i : (∀ i, φ i) →ₗ[R] φ i) := by
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    h : Disjoint I J
    ⊢ LE.le (iSup fun i => iSup fun h => LinearMap.range (LinearMap.single R φ i)) …
  -/
  refine iSup_le fun i => iSup_le fun hi => range_le_iff_comap.2 ?_
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    h : Disjoint I J
    i : ι
    hi : Membership.mem I i
    ⊢ Eq (Submodule.comap (LinearMap.single R φ i) (iInf fun i => iInf fun h => Li …
  -/
  simp only [← ker_comp, eq_top_iff, SetLike.le_def, mem_ker, comap_iInf, mem_iInf]
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    h : Disjoint I J
    i : ι
    hi : Membership.mem I i
    ⊢ ∀ ⦃x : φ i⦄, Membership.mem Top.top x → ∀ (i_1 : ι), Membership.mem J i_1 →  …
  -/
  rintro b - j hj
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    h : Disjoint I J
    i : ι
    hi : Membership.mem I i
    b : φ i
    j : ι
    hj : Membership.mem J j
    ⊢ Eq (((LinearMap.proj j).comp (LinearMap.single R φ i)) b) 0
  -/
  rw [proj_comp_single_ne R φ j i, zero_apply]
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    h : Disjoint I J
    i : ι
    hi : Membership.mem I i
    b : φ i
    j : ι
    hj : Membership.mem J j
    ⊢ Ne j i
  -/
  rintro rfl
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    h : Disjoint I J
    j : ι
    hj : Membership.mem J j
    hi : Membership.mem I j
    b : φ j
    ⊢ False
  -/
  exact h.le_bot ⟨hi, hj⟩
  /-
    🎉 no goals
  -/


theorem iInf_ker_proj_le_iSup_range_single {I : Finset ι} {J : Set ι} (hu : Set.univ ⊆ ↑I ∪ J) :
    ⨅ i ∈ J, ker (proj i : (∀ i, φ i) →ₗ[R] φ i) ≤ ⨆ i ∈ I, range (single R φ i) :=
  SetLike.le_def.2
    (by
      /-
        R : Type u
        ι : Type x
        inst✝³ : Semiring R
        φ : ι → Type i
        inst✝² : (i : ι) → AddCommMonoid (φ i)
        inst✝¹ : (i : ι) → Module R (φ i)
        inst✝ : DecidableEq ι
        I : Finset ι
        J : Set ι
        hu : HasSubset.Subset Set.univ (Union.union (↑I) J)
        ⊢ ∀ ⦃x : (i : ι) → φ i⦄, Membership.mem (iInf fun i => iInf fun h => LinearMap …
      -/
      intro b hb
      /-
        R : Type u
        ι : Type x
        inst✝³ : Semiring R
        φ : ι → Type i
        inst✝² : (i : ι) → AddCommMonoid (φ i)
        inst✝¹ : (i : ι) → Module R (φ i)
        inst✝ : DecidableEq ι
        I : Finset ι
        J : Set ι
        hu : HasSubset.Subset Set.univ (Union.union (↑I) J)
        b : (i : ι) → φ i
        hb : Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.proj …
        ⊢ Membership.mem (iSup fun i => iSup fun h => LinearMap.range (LinearMap.singl …
      -/
      simp only [mem_iInf, mem_ker, proj_apply] at hb
      rw [←
        show (∑ i ∈ I, Pi.single i (b i)) = b by
          ext i
          rw [Finset.sum_apply, ← Pi.single_eq_same i (b i)]
          refine Finset.sum_eq_single i (fun j _ ne => Pi.single_eq_of_ne ne.symm _) ?_
          intro hiI
          rw [Pi.single_eq_same]
          exact hb _ ((hu trivial).resolve_left hiI)]
      /-
        R : Type u
        ι : Type x
        inst✝³ : Semiring R
        φ : ι → Type i
        inst✝² : (i : ι) → AddCommMonoid (φ i)
        inst✝¹ : (i : ι) → Module R (φ i)
        inst✝ : DecidableEq ι
        I : Finset ι
        J : Set ι
        hu : HasSubset.Subset Set.univ (Union.union (↑I) J)
        b : (i : ι) → φ i
        hb : ∀ (i : ι), Membership.mem J i → Eq (b i) 0
        ⊢ Membership.mem (iSup fun i => iSup fun h => LinearMap.range (LinearMap.singl …
      -/
      exact sum_mem_biSup fun i _ => mem_range_self (single R φ i) (b i))
      /-
        🎉 no goals
      -/


theorem iSup_range_single_eq_iInf_ker_proj {I J : Set ι} (hd : Disjoint I J)
    (hu : Set.univ ⊆ I ∪ J) (hI : Set.Finite I) :
    ⨆ i ∈ I, range (single R φ i) = ⨅ i ∈ J, ker (proj i : (∀ i, φ i) →ₗ[R] φ i) := by
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    hd : Disjoint I J
    hu : HasSubset.Subset Set.univ (Union.union I J)
    hI : I.Finite
    ⊢ Eq (iSup fun i => iSup fun h => LinearMap.range (LinearMap.single R φ i)) (i …
  -/
  refine le_antisymm (iSup_range_single_le_iInf_ker_proj _ _ _ _ hd) ?_
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    hd : Disjoint I J
    hu : HasSubset.Subset Set.univ (Union.union I J)
    hI : I.Finite
    ⊢ LE.le (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.proj i)) (iSup f …
  -/
  have : Set.univ ⊆ ↑hI.toFinset ∪ J := by rwa [hI.coe_toFinset]
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    hd : Disjoint I J
    hu : HasSubset.Subset Set.univ (Union.union I J)
    hI : I.Finite
    this : HasSubset.Subset Set.univ (Union.union (↑hI.toFinset) J)
    ⊢ LE.le (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.proj i)) (iSup f …
  -/
  refine le_trans (iInf_ker_proj_le_iSup_range_single R φ this) (iSup_mono fun i => ?_)
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    hd : Disjoint I J
    hu : HasSubset.Subset Set.univ (Union.union I J)
    hI : I.Finite
    this : HasSubset.Subset Set.univ (Union.union (↑hI.toFinset) J)
    i : ι
    ⊢ LE.le (iSup fun h => LinearMap.range (LinearMap.single R φ i)) (iSup fun h = …
  -/
  rw [Set.Finite.mem_toFinset]
  /-
    🎉 no goals
  -/


theorem iSup_range_single [Finite ι] : ⨆ i, range (single R φ i) = ⊤ := by
  /-
    R : Type u
    ι : Type x
    inst✝⁴ : Semiring R
    φ : ι → Type i
    inst✝³ : (i : ι) → AddCommMonoid (φ i)
    inst✝² : (i : ι) → Module R (φ i)
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι
    ⊢ Eq (iSup fun i => LinearMap.range (LinearMap.single R φ i)) Top.top
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u
    ι : Type x
    inst✝⁴ : Semiring R
    φ : ι → Type i
    inst✝³ : (i : ι) → AddCommMonoid (φ i)
    inst✝² : (i : ι) → Module R (φ i)
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι
    val✝ : Fintype ι
    ⊢ Eq (iSup fun i => LinearMap.range (LinearMap.single R φ i)) Top.top
  -/
  convert top_unique (iInf_emptyset.ge.trans <| iInf_ker_proj_le_iSup_range_single R φ _)
    /-
      case h.e'_2.h.e'_4.h
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      x✝ : ι
      ⊢ Eq (LinearMap.range (LinearMap.single R φ x✝)) (iSup fun h => LinearMap.rang …
    -/
  · rename_i i
    /-
      case h.e'_2.h.e'_4.h
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      i : ι
      ⊢ Eq (LinearMap.range (LinearMap.single R φ i)) (iSup fun h => LinearMap.range …
    -/
    exact ((@iSup_pos _ _ _ fun _ => range <| single R φ i) <| Finset.mem_univ i).symm
    /-
      🎉 no goals
    -/
    /-
      case intro.convert_2
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      ⊢ HasSubset.Subset Set.univ (Union.union (↑Finset.univ) EmptyCollection.emptyC …
    -/
  · rw [Finset.coe_univ, Set.union_empty]
    /-
      🎉 no goals
    -/


theorem disjoint_single_single (I J : Set ι) (h : Disjoint I J) :
    Disjoint (⨆ i ∈ I, range (single R φ i)) (⨆ i ∈ J, range (single R φ i)) := by
  refine
    Disjoint.mono (iSup_range_single_le_iInf_ker_proj _ _ _ _ <| disjoint_compl_right)
      (iSup_range_single_le_iInf_ker_proj _ _ _ _ <| disjoint_compl_right) ?_
  simp only [disjoint_iff_inf_le, SetLike.le_def, mem_iInf, mem_inf, mem_ker, mem_bot, proj_apply,
    funext_iff]
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    I J : Set ι
    h : Disjoint I J
    ⊢ ∀ ⦃x : (i : ι) → φ i⦄, And (∀ (i : ι), Membership.mem (HasCompl.compl I) i → …
  -/
  rintro b ⟨hI, hJ⟩ i
  classical
    by_cases hiI : i ∈ I
    · by_cases hiJ : i ∈ J
      · exact (h.le_bot ⟨hiI, hiJ⟩).elim
      · exact hJ i hiJ
    · exact hI i hiI


/-- The linear equivalence between linear functions on a finite product of modules and
families of functions on these modules. See note [bundled maps over different rings]. -/
@[simps symm_apply]
def lsum (S) [AddCommMonoid M] [Module R M] [Fintype ι] [Semiring S] [Module S M]
    [SMulCommClass R S M] : ((i : ι) → φ i →ₗ[R] M) ≃ₗ[S] ((i : ι) → φ i) →ₗ[R] M where
  toFun f := ∑ i : ι, (f i).comp (proj i)
  invFun f i := f.comp (single R φ i)
                     /-
                       R : Type u
                       K : Type u'
                       M : Type v
                       V : Type v'
                       M₂ : Type w
                       V₂ : Type w'
                       M₃ : Type y
                       V₃ : Type y'
                       M₄ : Type z
                       ι : Type x
                       ι' : Type x'
                       inst✝¹³ : Semiring R
                       inst✝¹² : AddCommMonoid M₂
                       inst✝¹¹ : Module R M₂
                       inst✝¹⁰ : AddCommMonoid M₃
                       inst✝⁹ : Module R M₃
                       φ : ι → Type i
                       inst✝⁸ : (i : ι) → AddCommMonoid (φ i)
                       inst✝⁷ : (i : ι) → Module R (φ i)
                       inst✝⁶ : DecidableEq ι
                       S : Type ?u.73122
                       inst✝⁵ : AddCommMonoid M
                       inst✝⁴ : Module R M
                       inst✝³ : Fintype ι
                       inst✝² : Semiring S
                       inst✝¹ : Module S M
                       inst✝ : SMulCommClass R S M
                       f g : (i : ι) → LinearMap (RingHom.id R) (φ i) M
                       ⊢ Eq ((fun f => Finset.univ.sum fun i => (f i).comp (LinearMap.proj i)) (HAdd. …
                     -/
  map_add' f g := by simp only [Pi.add_apply, add_comp, Finset.sum_add_distrib]
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u
                        K : Type u'
                        M : Type v
                        V : Type v'
                        M₂ : Type w
                        V₂ : Type w'
                        M₃ : Type y
                        V₃ : Type y'
                        M₄ : Type z
                        ι : Type x
                        ι' : Type x'
                        inst✝¹³ : Semiring R
                        inst✝¹² : AddCommMonoid M₂
                        inst✝¹¹ : Module R M₂
                        inst✝¹⁰ : AddCommMonoid M₃
                        inst✝⁹ : Module R M₃
                        φ : ι → Type i
                        inst✝⁸ : (i : ι) → AddCommMonoid (φ i)
                        inst✝⁷ : (i : ι) → Module R (φ i)
                        inst✝⁶ : DecidableEq ι
                        S : Type ?u.73122
                        inst✝⁵ : AddCommMonoid M
                        inst✝⁴ : Module R M
                        inst✝³ : Fintype ι
                        inst✝² : Semiring S
                        inst✝¹ : Module S M
                        inst✝ : SMulCommClass R S M
                        c : S
                        f : (i : ι) → LinearMap (RingHom.id R) (φ i) M
                        ⊢ Eq ({ toFun := fun f => Finset.univ.sum fun i => (f i).comp (LinearMap.proj  …
                      -/
  map_smul' c f := by simp only [Pi.smul_apply, smul_comp, Finset.smul_sum, RingHom.id_apply]
                      /-
                        🎉 no goals
                      -/
  left_inv f := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝¹³ : Semiring R
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : AddCommMonoid M₃
      inst✝⁹ : Module R M₃
      φ : ι → Type i
      inst✝⁸ : (i : ι) → AddCommMonoid (φ i)
      inst✝⁷ : (i : ι) → Module R (φ i)
      inst✝⁶ : DecidableEq ι
      S : Type ?u.73122
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : Fintype ι
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      f : (i : ι) → LinearMap (RingHom.id R) (φ i) M
      ⊢ Eq ((fun f i => f.comp (LinearMap.single R φ i)) ({ toFun := fun f => Finset …
    -/
    ext i x
    /-
      case h.h
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝¹³ : Semiring R
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : AddCommMonoid M₃
      inst✝⁹ : Module R M₃
      φ : ι → Type i
      inst✝⁸ : (i : ι) → AddCommMonoid (φ i)
      inst✝⁷ : (i : ι) → Module R (φ i)
      inst✝⁶ : DecidableEq ι
      S : Type ?u.73122
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : Fintype ι
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      f : (i : ι) → LinearMap (RingHom.id R) (φ i) M
      i : ι
      x : φ i
      ⊢ Eq (((fun f i => f.comp (LinearMap.single R φ i)) ({ toFun := fun f => Finse …
    -/
    simp [apply_single]
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝¹³ : Semiring R
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : AddCommMonoid M₃
      inst✝⁹ : Module R M₃
      φ : ι → Type i
      inst✝⁸ : (i : ι) → AddCommMonoid (φ i)
      inst✝⁷ : (i : ι) → Module R (φ i)
      inst✝⁶ : DecidableEq ι
      S : Type ?u.73122
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : Fintype ι
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      f : LinearMap (RingHom.id R) ((i : ι) → φ i) M
      ⊢ Eq ({ toFun := fun f => Finset.univ.sum fun i => (f i).comp (LinearMap.proj  …
    -/
    ext x
    /-
      case h
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝¹³ : Semiring R
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : AddCommMonoid M₃
      inst✝⁹ : Module R M₃
      φ : ι → Type i
      inst✝⁸ : (i : ι) → AddCommMonoid (φ i)
      inst✝⁷ : (i : ι) → Module R (φ i)
      inst✝⁶ : DecidableEq ι
      S : Type ?u.73122
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : Fintype ι
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      f : LinearMap (RingHom.id R) ((i : ι) → φ i) M
      x : (i : ι) → φ i
      ⊢ Eq (({ toFun := fun f => Finset.univ.sum fun i => (f i).comp (LinearMap.proj …
    -/
    suffices f (∑ j, Pi.single j (x j)) = f x by simpa [apply_single]
    /-
      case h
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝¹³ : Semiring R
      inst✝¹² : AddCommMonoid M₂
      inst✝¹¹ : Module R M₂
      inst✝¹⁰ : AddCommMonoid M₃
      inst✝⁹ : Module R M₃
      φ : ι → Type i
      inst✝⁸ : (i : ι) → AddCommMonoid (φ i)
      inst✝⁷ : (i : ι) → Module R (φ i)
      inst✝⁶ : DecidableEq ι
      S : Type ?u.73122
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : Fintype ι
      inst✝² : Semiring S
      inst✝¹ : Module S M
      inst✝ : SMulCommClass R S M
      f : LinearMap (RingHom.id R) ((i : ι) → φ i) M
      x : (i : ι) → φ i
      ⊢ Eq (f (Finset.univ.sum fun j => Pi.single j (x j))) (f x)
    -/
    rw [Finset.univ_sum_single]
    /-
      🎉 no goals
    -/


@[simp]
theorem lsum_apply (S) [AddCommMonoid M] [Module R M] [Fintype ι] [Semiring S]
    [Module S M] [SMulCommClass R S M] (f : (i : ι) → φ i →ₗ[R] M) :
    lsum R φ S f = ∑ i : ι, (f i).comp (proj i) := rfl


@[simp high]
theorem lsum_single {ι R : Type*} [Fintype ι] [DecidableEq ι] [CommRing R] {M : ι → Type*}
    [(i : ι) → AddCommGroup (M i)] [(i : ι) → Module R (M i)] :
    LinearMap.lsum R M R (LinearMap.single R M) = LinearMap.id :=
                            /-
                              ι : Type u_1
                              R : Type u_2
                              inst✝⁴ : Fintype ι
                              inst✝³ : DecidableEq ι
                              inst✝² : CommRing R
                              M : ι → Type u_3
                              inst✝¹ : (i : ι) → AddCommGroup (M i)
                              inst✝ : (i : ι) → Module R (M i)
                              x : (i : ι) → M i
                              ⊢ Eq (((LinearMap.lsum R M R) (LinearMap.single R M)) x) (LinearMap.id x)
                            -/
  LinearMap.ext fun x => by simp [Finset.univ_sum_single]
                            /-
                              🎉 no goals
                            -/


theorem pi_ext (h : ∀ i x, f (Pi.single i x) = g (Pi.single i x)) : f = g :=
  toAddMonoidHom_injective <| AddMonoidHom.functions_ext _ _ _ h


theorem pi_ext_iff : f = g ↔ ∀ i x, f (Pi.single i x) = g (Pi.single i x) :=
  ⟨fun h _ _ => h ▸ rfl, pi_ext⟩


/-- This is used as the ext lemma instead of `LinearMap.pi_ext` for reasons explained in
note [partially-applied ext lemmas]. -/
@[ext]
theorem pi_ext' (h : ∀ i, f.comp (single R φ i) = g.comp (single R φ i)) : f = g := by
  /-
    R : Type u
    M : Type v
    ι : Type x
    inst✝⁶ : Semiring R
    φ : ι → Type i
    inst✝⁵ : (i : ι) → AddCommMonoid (φ i)
    inst✝⁴ : (i : ι) → Module R (φ i)
    inst✝³ : DecidableEq ι
    inst✝² : Finite ι
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) ((i : ι) → φ i) M
    h : ∀ (i : ι), Eq (f.comp (LinearMap.single R φ i)) (g.comp (LinearMap.single  …
    ⊢ Eq f g
  -/
  refine pi_ext fun i x => ?_
  /-
    R : Type u
    M : Type v
    ι : Type x
    inst✝⁶ : Semiring R
    φ : ι → Type i
    inst✝⁵ : (i : ι) → AddCommMonoid (φ i)
    inst✝⁴ : (i : ι) → Module R (φ i)
    inst✝³ : DecidableEq ι
    inst✝² : Finite ι
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) ((i : ι) → φ i) M
    h : ∀ (i : ι), Eq (f.comp (LinearMap.single R φ i)) (g.comp (LinearMap.single  …
    i : ι
    x : φ i
    ⊢ Eq (f (Pi.single i x)) (g (Pi.single i x))
  -/
  convert LinearMap.congr_fun (h i) x
  /-
    🎉 no goals
  -/


/-- If `I` and `J` are disjoint index sets, the product of the kernels of the `J`th projections of
`φ` is linearly equivalent to the product over `I`. -/
def iInfKerProjEquiv {I J : Set ι} [DecidablePred fun i => i ∈ I] (hd : Disjoint I J)
    (hu : Set.univ ⊆ I ∪ J) :
    (⨅ i ∈ J, ker (proj i : ((i : ι) → φ i) →ₗ[R] φ i) :
    Submodule R ((i : ι) → φ i)) ≃ₗ[R] (i : I) → φ i := by
  refine
    LinearEquiv.ofLinear (pi fun i => (proj (i : ι)).comp (Submodule.subtype _))
      (codRestrict _ (pi fun i => if h : i ∈ I then proj (⟨i, h⟩ : I) else 0) ?_) ?_ ?_
    /-
      case refine_1
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      ⊢ ∀ (c : (i : ↑I) → φ ↑i), Membership.mem (iInf fun i => iInf fun h => LinearM …
    -/
  · intro b
    /-
      case refine_1
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ↑I) → φ ↑i
      ⊢ Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.proj i) …
    -/
    simp only [mem_iInf, mem_ker, funext_iff, proj_apply, pi_apply]
    /-
      case refine_1
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ↑I) → φ ↑i
      ⊢ ∀ (i : ι), Membership.mem J i → Eq ((dite (Membership.mem I i) (fun h => Lin …
    -/
    intro j hjJ
    /-
      case refine_1
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ↑I) → φ ↑i
      j : ι
      hjJ : Membership.mem J j
      ⊢ Eq ((dite (Membership.mem I j) (fun h => LinearMap.proj ⟨j, h⟩) fun h => 0)  …
    -/
    have : j ∉ I := fun hjI => hd.le_bot ⟨hjI, hjJ⟩
    /-
      case refine_1
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ↑I) → φ ↑i
      j : ι
      hjJ : Membership.mem J j
      this : Not (Membership.mem I j)
      ⊢ Eq ((dite (Membership.mem I j) (fun h => LinearMap.proj ⟨j, h⟩) fun h => 0)  …
    -/
    rw [dif_neg this, zero_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      ⊢ Eq ((LinearMap.pi fun i => (LinearMap.proj ↑i).comp (iInf fun i => iInf fun  …
    -/
  · simp only [pi_comp, comp_assoc, subtype_comp_codRestrict, proj_pi, Subtype.coe_prop]
    /-
      case refine_2
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      ⊢ Eq (LinearMap.pi fun i => dite True (fun h => LinearMap.proj ⟨↑i, ⋯⟩) fun h  …
    -/
    ext b ⟨j, hj⟩
    simp only [dif_pos, Function.comp_apply, Function.eval_apply, LinearMap.codRestrict_apply,
      LinearMap.coe_comp, LinearMap.coe_proj, LinearMap.pi_apply, Submodule.subtype_apply,
      Subtype.coe_prop]
    /-
      case refine_2.h.h.mk
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ↑I) → φ ↑i
      j : ι
      hj : Membership.mem I j
      ⊢ Eq (Function.eval ⟨j, ⋯⟩ b) (LinearMap.id b ⟨j, hj⟩)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      ⊢ Eq ((LinearMap.codRestrict (iInf fun i => iInf fun h => LinearMap.ker (Linea …
    -/
  · ext1 ⟨b, hb⟩
    /-
      case refine_3.h.mk
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ι) → φ i
      hb : Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.proj …
      ⊢ Eq (((LinearMap.codRestrict (iInf fun i => iInf fun h => LinearMap.ker (Line …
    -/
    apply Subtype.ext
    /-
      case refine_3.h.mk.a
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ι) → φ i
      hb : Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.proj …
      ⊢ Eq ↑(((LinearMap.codRestrict (iInf fun i => iInf fun h => LinearMap.ker (Lin …
    -/
    ext j
    have hb : ∀ i ∈ J, b i = 0 := by
      simpa only [mem_iInf, mem_ker, proj_apply] using (mem_iInf _).1 hb
    /-
      case refine_3.h.mk.a.h
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ι) → φ i
      hb✝ : Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.pro …
      j : ι
      hb : ∀ (i : ι), Membership.mem J i → Eq (b i) 0
      ⊢ Eq (↑(((LinearMap.codRestrict (iInf fun i => iInf fun h => LinearMap.ker (Li …
    -/
    simp only [comp_apply, pi_apply, id_apply, proj_apply, subtype_apply, codRestrict_apply]
    /-
      case refine_3.h.mk.a.h
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁸ : Semiring R
      inst✝⁷ : AddCommMonoid M₂
      inst✝⁶ : Module R M₂
      inst✝⁵ : AddCommMonoid M₃
      inst✝⁴ : Module R M₃
      φ : ι → Type i
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      inst✝¹ : DecidableEq ι
      I J : Set ι
      inst✝ : DecidablePred fun i => Membership.mem I i
      hd : Disjoint I J
      hu : HasSubset.Subset Set.univ (Union.union I J)
      b : (i : ι) → φ i
      hb✝ : Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.pro …
      j : ι
      hb : ∀ (i : ι), Membership.mem J i → Eq (b i) 0
      ⊢ Eq ((dite (Membership.mem I j) (fun h => LinearMap.proj ⟨j, h⟩) fun h => 0)  …
    -/
    split_ifs with h
      /-
        case pos
        R : Type u
        K : Type u'
        M : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝⁸ : Semiring R
        inst✝⁷ : AddCommMonoid M₂
        inst✝⁶ : Module R M₂
        inst✝⁵ : AddCommMonoid M₃
        inst✝⁴ : Module R M₃
        φ : ι → Type i
        inst✝³ : (i : ι) → AddCommMonoid (φ i)
        inst✝² : (i : ι) → Module R (φ i)
        inst✝¹ : DecidableEq ι
        I J : Set ι
        inst✝ : DecidablePred fun i => Membership.mem I i
        hd : Disjoint I J
        hu : HasSubset.Subset Set.univ (Union.union I J)
        b : (i : ι) → φ i
        hb✝ : Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.pro …
        j : ι
        hb : ∀ (i : ι), Membership.mem J i → Eq (b i) 0
        h : Membership.mem I j
        ⊢ Eq ((LinearMap.proj ⟨j, h⟩) ((LinearMap.pi fun i => (LinearMap.proj ↑i).comp …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u
        K : Type u'
        M : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝⁸ : Semiring R
        inst✝⁷ : AddCommMonoid M₂
        inst✝⁶ : Module R M₂
        inst✝⁵ : AddCommMonoid M₃
        inst✝⁴ : Module R M₃
        φ : ι → Type i
        inst✝³ : (i : ι) → AddCommMonoid (φ i)
        inst✝² : (i : ι) → Module R (φ i)
        inst✝¹ : DecidableEq ι
        I J : Set ι
        inst✝ : DecidablePred fun i => Membership.mem I i
        hd : Disjoint I J
        hu : HasSubset.Subset Set.univ (Union.union I J)
        b : (i : ι) → φ i
        hb✝ : Membership.mem (iInf fun i => iInf fun h => LinearMap.ker (LinearMap.pro …
        j : ι
        hb : ∀ (i : ι), Membership.mem J i → Eq (b i) 0
        h : Not (Membership.mem I j)
        ⊢ Eq (0 ((LinearMap.pi fun i => (LinearMap.proj ↑i).comp (iInf fun i => iInf f …
      -/
    · exact (hb _ <| (hu trivial).resolve_left h).symm
      /-
        🎉 no goals
      -/


/-- `diag i j` is the identity map if `i = j`. Otherwise it is the constant 0 map. -/
def diag (i j : ι) : φ i →ₗ[R] φ j :=
  @Function.update ι (fun j => φ i →ₗ[R] φ j) _ 0 i id j


theorem update_apply (f : (i : ι) → M₂ →ₗ[R] φ i) (c : M₂) (i j : ι) (b : M₂ →ₗ[R] φ i) :
    (update f i b j) c = update (fun i => f i c) i (b c) j := by
  /-
    R : Type u
    M₂ : Type w
    ι : Type x
    inst✝⁵ : Semiring R
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : Module R M₂
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    f : (i : ι) → LinearMap (RingHom.id R) M₂ (φ i)
    c : M₂
    i j : ι
    b : LinearMap (RingHom.id R) M₂ (φ i)
    ⊢ Eq ((Function.update f i b j) c) (Function.update (fun i => (f i) c) i (b c) …
  -/
  by_cases h : j = i
    /-
      case pos
      R : Type u
      M₂ : Type w
      ι : Type x
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₂
      φ : ι → Type i
      inst✝² : (i : ι) → AddCommMonoid (φ i)
      inst✝¹ : (i : ι) → Module R (φ i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) M₂ (φ i)
      c : M₂
      i j : ι
      b : LinearMap (RingHom.id R) M₂ (φ i)
      h : Eq j i
      ⊢ Eq ((Function.update f i b j) c) (Function.update (fun i => (f i) c) i (b c) …
    -/
  · rw [h, update_self, update_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      M₂ : Type w
      ι : Type x
      inst✝⁵ : Semiring R
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : Module R M₂
      φ : ι → Type i
      inst✝² : (i : ι) → AddCommMonoid (φ i)
      inst✝¹ : (i : ι) → Module R (φ i)
      inst✝ : DecidableEq ι
      f : (i : ι) → LinearMap (RingHom.id R) M₂ (φ i)
      c : M₂
      i j : ι
      b : LinearMap (RingHom.id R) M₂ (φ i)
      h : Not (Eq j i)
      ⊢ Eq ((Function.update f i b j) c) (Function.update (fun i => (f i) c) i (b c) …
    -/
  · rw [update_of_ne h, update_of_ne h]
    /-
      🎉 no goals
    -/


theorem single_eq_pi_diag (i : ι) : single R φ i = pi (diag i) := by
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    i : ι
    ⊢ Eq (LinearMap.single R φ i) (LinearMap.pi (LinearMap.diag i))
  -/
  ext x j
  -- Porting note: made types explicit
  /-
    case h.h
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    i : ι
    x : φ i
    j : ι
    ⊢ Eq ((LinearMap.single R φ i) x j) ((LinearMap.pi (LinearMap.diag i)) x j)
  -/
  convert (update_apply (R := R) (φ := φ) (ι := ι) 0 x i j _).symm
  /-
    case h.e'_2.h.e'_1
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    i : ι
    x : φ i
    j : ι
    ⊢ Eq x (LinearMap.id x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ker_single (i : ι) : ker (single R φ i) = ⊥ :=
  ker_eq_bot_of_injective <| Pi.single_injective _ _


theorem proj_comp_single (i j : ι) : (proj i).comp (single R φ j) = diag j i := by
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type i
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    i j : ι
    ⊢ Eq ((LinearMap.proj i).comp (LinearMap.single R φ j)) (LinearMap.diag j i)
  -/
  rw [single_eq_pi_diag, proj_pi]
  /-
    🎉 no goals
  -/


/-- A linear map `f` applied to `x : ι → R` can be computed using the image under `f` of elements
of the canonical basis. -/
theorem pi_apply_eq_sum_univ [Fintype ι] (f : (ι → R) →ₗ[R] M₂) (x : ι → R) :
    f x = ∑ i, x i • f fun j => if i = j then 1 else 0 := by
  /-
    R : Type u
    M₂ : Type w
    ι : Type x
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id R) (ι → R) M₂
    x : ι → R
    ⊢ Eq (f x) (Finset.univ.sum fun i => HSMul.hSMul (x i) (f fun j => ite (Eq i j …
  -/
  conv_lhs => rw [pi_eq_sum_univ x, map_sum]
  /-
    R : Type u
    M₂ : Type w
    ι : Type x
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id R) (ι → R) M₂
    x : ι → R
    ⊢ Eq (Finset.univ.sum fun x_1 => f (HSMul.hSMul (x x_1) fun j => ite (Eq x_1 j …
  -/
  refine Finset.sum_congr rfl (fun _ _ => ?_)
  /-
    R : Type u
    M₂ : Type w
    ι : Type x
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M₂
    inst✝² : Module R M₂
    inst✝¹ : DecidableEq ι
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id R) (ι → R) M₂
    x : ι → R
    x✝¹ : ι
    x✝ : Membership.mem Finset.univ x✝¹
    ⊢ Eq (f (HSMul.hSMul (x x✝¹) fun j => ite (Eq x✝¹ j) 1 0)) (HSMul.hSMul (x x✝¹ …
  -/
  rw [map_smul]
  /-
    🎉 no goals
  -/


/-- A version of `Set.pi` for submodules. Given an index set `I` and a family of submodules
`p : (i : ι) → Submodule R (φ i)`, `pi I s` is the submodule of dependent functions
`f : (i : ι) → φ i` such that `f i` belongs to `p a` whenever `i ∈ I`. -/
def pi (I : Set ι) (p : (i : ι) → Submodule R (φ i)) : Submodule R ((i : ι) → φ i) where
  carrier := Set.pi I fun i => p i
  zero_mem' i _ := (p i).zero_mem
  add_mem' {_ _} hx hy i hi := (p i).add_mem (hx i hi) (hy i hi)
  smul_mem' c _ hx i hi := (p i).smul_mem c (hx i hi)


@[simp]
theorem mem_pi : x ∈ pi I p ↔ ∀ i ∈ I, x i ∈ p i :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_pi : (pi I p : Set ((i : ι) → φ i)) = Set.pi I fun i => p i :=
  rfl


@[simp]
theorem pi_empty (p : (i : ι) → Submodule R (φ i)) : pi ∅ p = ⊤ :=
  SetLike.coe_injective <| Set.empty_pi _


@[simp]
theorem pi_top (s : Set ι) : (pi s fun i : ι => (⊤ : Submodule R (φ i))) = ⊤ :=
  SetLike.coe_injective <| Set.pi_univ _


theorem pi_mono {s : Set ι} (h : ∀ i ∈ s, p i ≤ q i) : pi s p ≤ pi s q :=
  Set.pi_mono h


theorem biInf_comap_proj :
    ⨅ i ∈ I, comap (proj i : ((i : ι) → φ i) →ₗ[R] φ i) (p i) = pi I p := by
  /-
    R : Type u
    ι : Type x
    inst✝² : Semiring R
    φ : ι → Type u_1
    inst✝¹ : (i : ι) → AddCommMonoid (φ i)
    inst✝ : (i : ι) → Module R (φ i)
    I : Set ι
    p : (i : ι) → Submodule R (φ i)
    ⊢ Eq (iInf fun i => iInf fun h => Submodule.comap (LinearMap.proj i) (p i)) (S …
  -/
  ext x
  /-
    case h
    R : Type u
    ι : Type x
    inst✝² : Semiring R
    φ : ι → Type u_1
    inst✝¹ : (i : ι) → AddCommMonoid (φ i)
    inst✝ : (i : ι) → Module R (φ i)
    I : Set ι
    p : (i : ι) → Submodule R (φ i)
    x : (i : ι) → φ i
    ⊢ Iff (Membership.mem (iInf fun i => iInf fun h => Submodule.comap (LinearMap. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iInf_comap_proj :
    ⨅ i, comap (proj i : ((i : ι) → φ i) →ₗ[R] φ i) (p i) = pi Set.univ p := by
  /-
    R : Type u
    ι : Type x
    inst✝² : Semiring R
    φ : ι → Type u_1
    inst✝¹ : (i : ι) → AddCommMonoid (φ i)
    inst✝ : (i : ι) → Module R (φ i)
    p : (i : ι) → Submodule R (φ i)
    ⊢ Eq (iInf fun i => Submodule.comap (LinearMap.proj i) (p i)) (Submodule.pi Se …
  -/
  ext x
  /-
    case h
    R : Type u
    ι : Type x
    inst✝² : Semiring R
    φ : ι → Type u_1
    inst✝¹ : (i : ι) → AddCommMonoid (φ i)
    inst✝ : (i : ι) → Module R (φ i)
    p : (i : ι) → Submodule R (φ i)
    x : (i : ι) → φ i
    ⊢ Iff (Membership.mem (iInf fun i => Submodule.comap (LinearMap.proj i) (p i)) …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iSup_map_single [DecidableEq ι] [Finite ι] :
    ⨆ i, map (LinearMap.single R φ i : φ i →ₗ[R] (i : ι) → φ i) (p i) = pi Set.univ p := by
  /-
    R : Type u
    ι : Type x
    inst✝⁴ : Semiring R
    φ : ι → Type u_1
    inst✝³ : (i : ι) → AddCommMonoid (φ i)
    inst✝² : (i : ι) → Module R (φ i)
    p : (i : ι) → Submodule R (φ i)
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι
    ⊢ Eq (iSup fun i => Submodule.map (LinearMap.single R φ i) (p i)) (Submodule.p …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u
    ι : Type x
    inst✝⁴ : Semiring R
    φ : ι → Type u_1
    inst✝³ : (i : ι) → AddCommMonoid (φ i)
    inst✝² : (i : ι) → Module R (φ i)
    p : (i : ι) → Submodule R (φ i)
    inst✝¹ : DecidableEq ι
    inst✝ : Finite ι
    val✝ : Fintype ι
    ⊢ Eq (iSup fun i => Submodule.map (LinearMap.single R φ i) (p i)) (Submodule.p …
  -/
  refine (iSup_le fun i => ?_).antisymm ?_
    /-
      case intro.refine_1
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type u_1
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      p : (i : ι) → Submodule R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      i : ι
      ⊢ LE.le (Submodule.map (LinearMap.single R φ i) (p i)) (Submodule.pi Set.univ p)
    -/
  · rintro _ ⟨x, hx : x ∈ p i, rfl⟩ j -
    /-
      case intro.refine_1.intro.intro
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type u_1
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      p : (i : ι) → Submodule R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      i : ι
      x : φ i
      hx : Membership.mem (p i) x
      j : ι
      ⊢ Membership.mem ((fun i => ↑(p i)) j) ((LinearMap.single R φ i) x j)
    -/
                                          /-
                                            🎉 no goals
                                          -/
    rcases em (j = i) with (rfl | hj) <;> simp [*]
                                          /-
                                            🎉 no goals
                                          -/
    /-
      case intro.refine_2
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type u_1
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      p : (i : ι) → Submodule R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      ⊢ LE.le (Submodule.pi Set.univ p) (iSup fun i => Submodule.map (LinearMap.sing …
    -/
  · intro x hx
    /-
      case intro.refine_2
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type u_1
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      p : (i : ι) → Submodule R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      x : (i : ι) → φ i
      hx : Membership.mem (Submodule.pi Set.univ p) x
      ⊢ Membership.mem (iSup fun i => Submodule.map (LinearMap.single R φ i) (p i)) x
    -/
    rw [← Finset.univ_sum_single x]
    /-
      case intro.refine_2
      R : Type u
      ι : Type x
      inst✝⁴ : Semiring R
      φ : ι → Type u_1
      inst✝³ : (i : ι) → AddCommMonoid (φ i)
      inst✝² : (i : ι) → Module R (φ i)
      p : (i : ι) → Submodule R (φ i)
      inst✝¹ : DecidableEq ι
      inst✝ : Finite ι
      val✝ : Fintype ι
      x : (i : ι) → φ i
      hx : Membership.mem (Submodule.pi Set.univ p) x
      ⊢ Membership.mem (iSup fun i => Submodule.map (LinearMap.single R φ i) (p i))  …
    -/
    exact sum_mem_iSup fun i => mem_map_of_mem (hx i trivial)
    /-
      🎉 no goals
    -/


theorem le_comap_single_pi [DecidableEq ι] (p : (i : ι) → Submodule R (φ i)) {i} :
    p i ≤ Submodule.comap (LinearMap.single R φ i : φ i →ₗ[R] _) (Submodule.pi Set.univ p) := by
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type u_1
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (φ i)
    i : ι
    ⊢ LE.le (p i) (Submodule.comap (LinearMap.single R φ i) (Submodule.pi Set.univ …
  -/
  intro x hx
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type u_1
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (φ i)
    i : ι
    x : φ i
    hx : Membership.mem (p i) x
    ⊢ Membership.mem (Submodule.comap (LinearMap.single R φ i) (Submodule.pi Set.u …
  -/
  rw [Submodule.mem_comap, Submodule.mem_pi]
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type u_1
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (φ i)
    i : ι
    x : φ i
    hx : Membership.mem (p i) x
    ⊢ ∀ (i_1 : ι), Membership.mem Set.univ i_1 → Membership.mem (p i_1) ((LinearMa …
  -/
  rintro j -
  /-
    R : Type u
    ι : Type x
    inst✝³ : Semiring R
    φ : ι → Type u_1
    inst✝² : (i : ι) → AddCommMonoid (φ i)
    inst✝¹ : (i : ι) → Module R (φ i)
    inst✝ : DecidableEq ι
    p : (i : ι) → Submodule R (φ i)
    i : ι
    x : φ i
    hx : Membership.mem (p i) x
    j : ι
    ⊢ Membership.mem (p j) ((LinearMap.single R φ i) x j)
  -/
  by_cases h : j = i
    /-
      case pos
      R : Type u
      ι : Type x
      inst✝³ : Semiring R
      φ : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (φ i)
      inst✝¹ : (i : ι) → Module R (φ i)
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (φ i)
      i : ι
      x : φ i
      hx : Membership.mem (p i) x
      j : ι
      h : Eq j i
      ⊢ Membership.mem (p j) ((LinearMap.single R φ i) x j)
    -/
  · rwa [h, LinearMap.coe_single, Pi.single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      ι : Type x
      inst✝³ : Semiring R
      φ : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (φ i)
      inst✝¹ : (i : ι) → Module R (φ i)
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (φ i)
      i : ι
      x : φ i
      hx : Membership.mem (p i) x
      j : ι
      h : Not (Eq j i)
      ⊢ Membership.mem (p j) ((LinearMap.single R φ i) x j)
    -/
  · rw [LinearMap.coe_single, Pi.single_eq_of_ne h]
    /-
      case neg
      R : Type u
      ι : Type x
      inst✝³ : Semiring R
      φ : ι → Type u_1
      inst✝² : (i : ι) → AddCommMonoid (φ i)
      inst✝¹ : (i : ι) → Module R (φ i)
      inst✝ : DecidableEq ι
      p : (i : ι) → Submodule R (φ i)
      i : ι
      x : φ i
      hx : Membership.mem (p i) x
      j : ι
      h : Not (Eq j i)
      ⊢ Membership.mem (p j) 0
    -/
    exact (p j).zero_mem
    /-
      🎉 no goals
    -/


/-- Combine a family of linear equivalences into a linear equivalence of `pi`-types.

This is `Equiv.piCongrRight` as a `LinearEquiv` -/
def piCongrRight (e : (i : ι) → φ i ≃ₗ[R] ψ i) : ((i : ι) → φ i) ≃ₗ[R] (i : ι) → ψ i :=
  { AddEquiv.piCongrRight fun j => (e j).toAddEquiv with
    toFun := fun f i => e i (f i)
    invFun := fun f i => (e i).symm (f i)
                               /-
                                 R : Type u
                                 K : Type u'
                                 M : Type v
                                 V : Type v'
                                 M₂ : Type w
                                 V₂ : Type w'
                                 M₃ : Type y
                                 V₃ : Type y'
                                 M₄ : Type z
                                 ι : Type x
                                 ι' : Type x'
                                 inst✝⁶ : Semiring R
                                 φ : ι → Type u_1
                                 ψ : ι → Type u_2
                                 χ : ι → Type u_3
                                 inst✝⁵ : (i : ι) → AddCommMonoid (φ i)
                                 inst✝⁴ : (i : ι) → Module R (φ i)
                                 inst✝³ : (i : ι) → AddCommMonoid (ψ i)
                                 inst✝² : (i : ι) → Module R (ψ i)
                                 inst✝¹ : (i : ι) → AddCommMonoid (χ i)
                                 inst✝ : (i : ι) → Module R (χ i)
                                 e : (i : ι) → LinearEquiv (RingHom.id R) (φ i) (ψ i)
                                 c : R
                                 f : (i : ι) → φ i
                                 ⊢ Eq ({ toFun := fun f i => (e i) (f i), map_add' := ⋯ }.toFun (HSMul.hSMul c  …
                               -/
    map_smul' := fun c f => by ext; simp }
                                    /-
                                      🎉 no goals
                                    -/


@[simp]
theorem piCongrRight_apply (e : (i : ι) → φ i ≃ₗ[R] ψ i) (f i) :
    piCongrRight e f i = e i (f i) := rfl


@[simp]
theorem piCongrRight_refl : (piCongrRight fun j => refl R (φ j)) = refl _ _ :=
  rfl


@[simp]
theorem piCongrRight_symm (e : (i : ι) → φ i ≃ₗ[R] ψ i) :
    (piCongrRight e).symm = piCongrRight fun i => (e i).symm :=
  rfl


@[simp]
theorem piCongrRight_trans (e : (i : ι) → φ i ≃ₗ[R] ψ i) (f : (i : ι) → ψ i ≃ₗ[R] χ i) :
    (piCongrRight e).trans (piCongrRight f) = piCongrRight fun i => (e i).trans (f i) :=
  rfl


/-- Transport dependent functions through an equivalence of the base space.

This is `Equiv.piCongrLeft'` as a `LinearEquiv`. -/
@[simps (config := { simpRhs := true })]
def piCongrLeft' (e : ι ≃ ι') : ((i' : ι) → φ i') ≃ₗ[R] (i : ι') → φ <| e.symm i :=
  { Equiv.piCongrLeft' φ e with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


/-- Transporting dependent functions through an equivalence of the base,
expressed as a "simplification".

This is `Equiv.piCongrLeft` as a `LinearEquiv` -/
def piCongrLeft (e : ι' ≃ ι) : ((i' : ι') → φ (e i')) ≃ₗ[R] (i : ι) → φ i :=
  (piCongrLeft' R φ e.symm).symm


/-- `Equiv.piCurry` as a `LinearEquiv`. -/
def piCurry {ι : Type*} {κ : ι → Type*} (α : ∀ i, κ i → Type*)
    [∀ i k, AddCommMonoid (α i k)] [∀ i k, Module R (α i k)] :
    (Π i : Sigma κ, α i.1 i.2) ≃ₗ[R] Π i j, α i j where
  __ := Equiv.piCurry α
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp] theorem piCurry_apply {ι : Type*} {κ : ι → Type*} (α : ∀ i, κ i → Type*)
    [∀ i k, AddCommMonoid (α i k)] [∀ i k, Module R (α i k)]
    (f : ∀ x : Σ i, κ i, α x.1 x.2) :
    piCurry R α f = Sigma.curry f :=
  rfl


@[simp] theorem piCurry_symm_apply {ι : Type*} {κ : ι → Type*} (α : ∀ i, κ i → Type*)
    [∀ i k, AddCommMonoid (α i k)] [∀ i k, Module R (α i k)]
    (f : ∀ a b, α a b) :
    (piCurry R α).symm f = Sigma.uncurry f :=
  rfl


/-- This is `Equiv.piOptionEquivProd` as a `LinearEquiv` -/
def piOptionEquivProd {ι : Type*} {M : Option ι → Type*} [(i : Option ι) → AddCommGroup (M i)]
    [(i : Option ι) → Module R (M i)] :
    ((i : Option ι) → M i) ≃ₗ[R] M none × ((i : ι) → M (some i)) :=
  { Equiv.piOptionEquivProd with
                   /-
                     R : Type u
                     K : Type u'
                     M✝ : Type v
                     V : Type v'
                     M₂ : Type w
                     V₂ : Type w'
                     M₃ : Type y
                     V₃ : Type y'
                     M₄ : Type z
                     ι✝ : Type x
                     ι' : Type x'
                     inst✝⁸ : Semiring R
                     φ : ι✝ → Type u_1
                     ψ : ι✝ → Type u_2
                     χ : ι✝ → Type u_3
                     inst✝⁷ : (i : ι✝) → AddCommMonoid (φ i)
                     inst✝⁶ : (i : ι✝) → Module R (φ i)
                     inst✝⁵ : (i : ι✝) → AddCommMonoid (ψ i)
                     inst✝⁴ : (i : ι✝) → Module R (ψ i)
                     inst✝³ : (i : ι✝) → AddCommMonoid (χ i)
                     inst✝² : (i : ι✝) → Module R (χ i)
                     ι : Type u_4
                     M : Option ι → Type u_5
                     inst✝¹ : (i : Option ι) → AddCommGroup (M i)
                     inst✝ : (i : Option ι) → Module R (M i)
                     ⊢ ∀ (x y : (i : Option ι) → M i), Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd …
                   -/
    map_add' := by simp [funext_iff]
                   /-
                     🎉 no goals
                   -/
                    /-
                      R : Type u
                      K : Type u'
                      M✝ : Type v
                      V : Type v'
                      M₂ : Type w
                      V₂ : Type w'
                      M₃ : Type y
                      V₃ : Type y'
                      M₄ : Type z
                      ι✝ : Type x
                      ι' : Type x'
                      inst✝⁸ : Semiring R
                      φ : ι✝ → Type u_1
                      ψ : ι✝ → Type u_2
                      χ : ι✝ → Type u_3
                      inst✝⁷ : (i : ι✝) → AddCommMonoid (φ i)
                      inst✝⁶ : (i : ι✝) → Module R (φ i)
                      inst✝⁵ : (i : ι✝) → AddCommMonoid (ψ i)
                      inst✝⁴ : (i : ι✝) → Module R (ψ i)
                      inst✝³ : (i : ι✝) → AddCommMonoid (χ i)
                      inst✝² : (i : ι✝) → Module R (χ i)
                      ι : Type u_4
                      M : Option ι → Type u_5
                      inst✝¹ : (i : Option ι) → AddCommGroup (M i)
                      inst✝ : (i : Option ι) → Module R (M i)
                      ⊢ ∀ (m : R) (x : (i : Option ι) → M i), Eq ({ toFun := __src✝.toFun, map_add'  …
                    -/
    map_smul' := by simp [funext_iff] }
                    /-
                      🎉 no goals
                    -/


/-- Linear equivalence between linear functions `Rⁿ → M` and `Mⁿ`. The spaces `Rⁿ` and `Mⁿ`
are represented as `ι → R` and `ι → M`, respectively, where `ι` is a finite type.

This as an `S`-linear equivalence, under the assumption that `S` acts on `M` commuting with `R`.
When `R` is commutative, we can take this to be the usual action with `S = R`.
Otherwise, `S = ℕ` shows that the equivalence is additive.
See note [bundled maps over different rings]. -/
def piRing : ((ι → R) →ₗ[R] M) ≃ₗ[S] ι → M :=
  (LinearMap.lsum R (fun _ : ι => R) S).symm.trans
    (piCongrRight fun _ => LinearMap.ringLmapEquivSelf R S M)


@[simp]
theorem piRing_apply (f : (ι → R) →ₗ[R] M) (i : ι) : piRing R M ι S f i = f (Pi.single i 1) :=
  rfl


@[simp]
theorem piRing_symm_apply (f : ι → M) (g : ι → R) : (piRing R M ι S).symm f g = ∑ i, g i • f i := by
  /-
    R : Type u
    M : Type v
    ι : Type x
    inst✝⁷ : Semiring R
    S : Type u_4
    inst✝⁶ : Fintype ι
    inst✝⁵ : DecidableEq ι
    inst✝⁴ : Semiring S
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : SMulCommClass R S M
    f : ι → M
    g : ι → R
    ⊢ Eq (((LinearEquiv.piRing R M ι S).symm f) g) (Finset.univ.sum fun i => HSMul …
  -/
  simp [piRing, LinearMap.lsum_apply]
  /-
    🎉 no goals
  -/

-- TODO additive version?

/-- `Equiv.sumArrowEquivProdArrow` as a linear equivalence.
-/
def sumArrowLequivProdArrow (α β R M : Type*) [Semiring R] [AddCommMonoid M] [Module R M] :
    (α ⊕ β → M) ≃ₗ[R] (α → M) × (β → M) :=
  { Equiv.sumArrowEquivProdArrow α β
      M with
    map_add' := by
      /-
        R✝ : Type u
        K : Type u'
        M✝ : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝¹⁶ : Semiring R✝
        φ : ι → Type u_1
        ψ : ι → Type u_2
        χ : ι → Type u_3
        inst✝¹⁵ : (i : ι) → AddCommMonoid (φ i)
        inst✝¹⁴ : (i : ι) → Module R✝ (φ i)
        inst✝¹³ : (i : ι) → AddCommMonoid (ψ i)
        inst✝¹² : (i : ι) → Module R✝ (ψ i)
        inst✝¹¹ : (i : ι) → AddCommMonoid (χ i)
        inst✝¹⁰ : (i : ι) → Module R✝ (χ i)
        S : Type u_4
        inst✝⁹ : Fintype ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : Semiring S
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R✝ M✝
        inst✝⁴ : Module S M✝
        inst✝³ : SMulCommClass R✝ S M✝
        α : Type u_5
        β : Type u_6
        R : Type u_7
        M : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ⊢ ∀ (x y : Sum α β → M), Eq (__src✝.toFun (HAdd.hAdd x y)) (HAdd.hAdd (__src✝. …
      -/
      intro f g
      /-
        R✝ : Type u
        K : Type u'
        M✝ : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝¹⁶ : Semiring R✝
        φ : ι → Type u_1
        ψ : ι → Type u_2
        χ : ι → Type u_3
        inst✝¹⁵ : (i : ι) → AddCommMonoid (φ i)
        inst✝¹⁴ : (i : ι) → Module R✝ (φ i)
        inst✝¹³ : (i : ι) → AddCommMonoid (ψ i)
        inst✝¹² : (i : ι) → Module R✝ (ψ i)
        inst✝¹¹ : (i : ι) → AddCommMonoid (χ i)
        inst✝¹⁰ : (i : ι) → Module R✝ (χ i)
        S : Type u_4
        inst✝⁹ : Fintype ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : Semiring S
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R✝ M✝
        inst✝⁴ : Module S M✝
        inst✝³ : SMulCommClass R✝ S M✝
        α : Type u_5
        β : Type u_6
        R : Type u_7
        M : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        f g : Sum α β → M
        ⊢ Eq (__src✝.toFun (HAdd.hAdd f g)) (HAdd.hAdd (__src✝.toFun f) (__src✝.toFun  …
      -/
              /-
                🎉 no goals
              -/
      ext <;> rfl
              /-
                🎉 no goals
              -/
    map_smul' := by
      /-
        R✝ : Type u
        K : Type u'
        M✝ : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝¹⁶ : Semiring R✝
        φ : ι → Type u_1
        ψ : ι → Type u_2
        χ : ι → Type u_3
        inst✝¹⁵ : (i : ι) → AddCommMonoid (φ i)
        inst✝¹⁴ : (i : ι) → Module R✝ (φ i)
        inst✝¹³ : (i : ι) → AddCommMonoid (ψ i)
        inst✝¹² : (i : ι) → Module R✝ (ψ i)
        inst✝¹¹ : (i : ι) → AddCommMonoid (χ i)
        inst✝¹⁰ : (i : ι) → Module R✝ (χ i)
        S : Type u_4
        inst✝⁹ : Fintype ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : Semiring S
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R✝ M✝
        inst✝⁴ : Module S M✝
        inst✝³ : SMulCommClass R✝ S M✝
        α : Type u_5
        β : Type u_6
        R : Type u_7
        M : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        ⊢ ∀ (m : R) (x : Sum α β → M), Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.to …
      -/
      intro r f
      /-
        R✝ : Type u
        K : Type u'
        M✝ : Type v
        V : Type v'
        M₂ : Type w
        V₂ : Type w'
        M₃ : Type y
        V₃ : Type y'
        M₄ : Type z
        ι : Type x
        ι' : Type x'
        inst✝¹⁶ : Semiring R✝
        φ : ι → Type u_1
        ψ : ι → Type u_2
        χ : ι → Type u_3
        inst✝¹⁵ : (i : ι) → AddCommMonoid (φ i)
        inst✝¹⁴ : (i : ι) → Module R✝ (φ i)
        inst✝¹³ : (i : ι) → AddCommMonoid (ψ i)
        inst✝¹² : (i : ι) → Module R✝ (ψ i)
        inst✝¹¹ : (i : ι) → AddCommMonoid (χ i)
        inst✝¹⁰ : (i : ι) → Module R✝ (χ i)
        S : Type u_4
        inst✝⁹ : Fintype ι
        inst✝⁸ : DecidableEq ι
        inst✝⁷ : Semiring S
        inst✝⁶ : AddCommMonoid M✝
        inst✝⁵ : Module R✝ M✝
        inst✝⁴ : Module S M✝
        inst✝³ : SMulCommClass R✝ S M✝
        α : Type u_5
        β : Type u_6
        R : Type u_7
        M : Type u_8
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        r : R
        f : Sum α β → M
        ⊢ Eq ({ toFun := __src✝.toFun, map_add' := ⋯ }.toFun (HSMul.hSMul r f)) (HSMul …
      -/
              /-
                🎉 no goals
              -/
      ext <;> rfl }
              /-
                🎉 no goals
              -/


@[simp]
theorem sumArrowLequivProdArrow_apply_fst {α β} (f : α ⊕ β → M) (a : α) :
    (sumArrowLequivProdArrow α β R M f).1 a = f (Sum.inl a) :=
  rfl


@[simp]
theorem sumArrowLequivProdArrow_apply_snd {α β} (f : α ⊕ β → M) (b : β) :
    (sumArrowLequivProdArrow α β R M f).2 b = f (Sum.inr b) :=
  rfl


@[simp]
theorem sumArrowLequivProdArrow_symm_apply_inl {α β} (f : α → M) (g : β → M) (a : α) :
    ((sumArrowLequivProdArrow α β R M).symm (f, g)) (Sum.inl a) = f a :=
  rfl


@[simp]
theorem sumArrowLequivProdArrow_symm_apply_inr {α β} (f : α → M) (g : β → M) (b : β) :
    ((sumArrowLequivProdArrow α β R M).symm (f, g)) (Sum.inr b) = g b :=
  rfl


/-- If `ι` has a unique element, then `ι → M` is linearly equivalent to `M`. -/
@[simps (config := { simpRhs := true, fullyApplied := false }) symm_apply]
def funUnique (ι R M : Type*) [Unique ι] [Semiring R] [AddCommMonoid M] [Module R M] :
    (ι → M) ≃ₗ[R] M :=
  { Equiv.funUnique ι M with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


@[simp]
theorem funUnique_apply (ι R M : Type*) [Unique ι] [Semiring R] [AddCommMonoid M] [Module R M] :
    (funUnique ι R M : (ι → M) → M) = eval default := rfl


/-- Linear equivalence between dependent functions `(i : Fin 2) → M i` and `M 0 × M 1`. -/
@[simps (config := { simpRhs := true, fullyApplied := false }) symm_apply]
def piFinTwo (M : Fin 2 → Type v)
    [(i : Fin 2) → AddCommMonoid (M i)] [(i : Fin 2) → Module R (M i)] :
    ((i : Fin 2) → M i) ≃ₗ[R] M 0 × M 1 :=
  { piFinTwoEquiv M with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


@[simp]
theorem piFinTwo_apply (M : Fin 2 → Type v)
    [(i : Fin 2) → AddCommMonoid (M i)] [(i : Fin 2) → Module R (M i)] :
    (piFinTwo R M : ((i : Fin 2) → M i) → M 0 × M 1) = fun f => (f 0, f 1) := rfl


/-- Linear equivalence between vectors in `M² = Fin 2 → M` and `M × M`. -/
@[simps! (config := .asFn)]
def finTwoArrow : (Fin 2 → M) ≃ₗ[R] M × M :=
  { finTwoArrowEquiv M, piFinTwo R fun _ => M with }


/-- `Function.extend s f 0` as a bundled linear map. -/
@[simps]
noncomputable def Function.ExtendByZero.linearMap : (ι → R) →ₗ[R] η → R :=
  { Function.ExtendByZero.hom R s with
    toFun := fun f => Function.extend s f 0
                               /-
                                 R : Type u
                                 K : Type u'
                                 M : Type v
                                 V : Type v'
                                 M₂ : Type w
                                 V₂ : Type w'
                                 M₃ : Type y
                                 V₃ : Type y'
                                 M₄ : Type z
                                 ι : Type x
                                 ι' : Type x'
                                 η : Type x
                                 inst✝ : Semiring R
                                 s : ι → η
                                 r : R
                                 f : ι → R
                                 ⊢ Eq ({ toFun := fun f => Function.extend s f 0, map_add' := ⋯ }.toFun (HSMul. …
                               -/
    map_smul' := fun r f => by simpa using Function.extend_smul r s f 0 }
                               /-
                                 🎉 no goals
                               -/


/-- The linear map defeq to `Matrix.vecEmpty` -/
def LinearMap.vecEmpty : M →ₗ[R] Fin 0 → M₃ where
  toFun _ := Matrix.vecEmpty
  map_add' _ _ := Subsingleton.elim _ _
  map_smul' _ _ := Subsingleton.elim _ _


@[simp]
theorem LinearMap.vecEmpty_apply (m : M) : (LinearMap.vecEmpty : M →ₗ[R] Fin 0 → M₃) m = ![] :=
  rfl


/-- A linear map into `Fin n.succ → M₃` can be built out of a map into `M₃` and a map into
`Fin n → M₃`. -/
def LinearMap.vecCons {n} (f : M →ₗ[R] M₂) (g : M →ₗ[R] Fin n → M₂) : M →ₗ[R] Fin n.succ → M₂ where
  toFun m := Matrix.vecCons (f m) (g m)
  map_add' x y := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      n : Nat
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M (Fin n → M₂)
      x y : M
      ⊢ Eq ((fun m => Matrix.vecCons (f m) (g m)) (HAdd.hAdd x y)) (HAdd.hAdd ((fun  …
    -/
    simp only []
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      n : Nat
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M (Fin n → M₂)
      x y : M
      ⊢ Eq (Matrix.vecCons (f (HAdd.hAdd x y)) (g (HAdd.hAdd x y))) (HAdd.hAdd (Matr …
    -/
    rw [f.map_add, g.map_add, Matrix.cons_add_cons (f x)]
    /-
      🎉 no goals
    -/
  map_smul' c x := by
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      n : Nat
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M (Fin n → M₂)
      c : R
      x : M
      ⊢ Eq ({ toFun := fun m => Matrix.vecCons (f m) (g m), map_add' := ⋯ }.toFun (H …
    -/
    simp only []
    /-
      R : Type u
      K : Type u'
      M : Type v
      V : Type v'
      M₂ : Type w
      V₂ : Type w'
      M₃ : Type y
      V₃ : Type y'
      M₄ : Type z
      ι : Type x
      ι' : Type x'
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : Module R M
      inst✝¹ : Module R M₂
      inst✝ : Module R M₃
      n : Nat
      f : LinearMap (RingHom.id R) M M₂
      g : LinearMap (RingHom.id R) M (Fin n → M₂)
      c : R
      x : M
      ⊢ Eq (Matrix.vecCons (f (HSMul.hSMul c x)) (g (HSMul.hSMul c x))) (HSMul.hSMul …
    -/
    rw [f.map_smul, g.map_smul, RingHom.id_apply, Matrix.smul_cons c (f x)]
    /-
      🎉 no goals
    -/


@[simp]
theorem LinearMap.vecCons_apply {n} (f : M →ₗ[R] M₂) (g : M →ₗ[R] Fin n → M₂) (m : M) :
    f.vecCons g m = Matrix.vecCons (f m) (g m) :=
  rfl


/-- The empty bilinear map defeq to `Matrix.vecEmpty` -/
@[simps]
def LinearMap.vecEmpty₂ : M →ₗ[R] M₂ →ₗ[R] Fin 0 → M₃ where
  toFun _ := LinearMap.vecEmpty
  map_add' _ _ := LinearMap.ext fun _ => Subsingleton.elim _ _
  map_smul' _ _ := LinearMap.ext fun _ => Subsingleton.elim _ _


/-- A bilinear map into `Fin n.succ → M₃` can be built out of a map into `M₃` and a map into
`Fin n → M₃` -/
@[simps]
def LinearMap.vecCons₂ {n} (f : M →ₗ[R] M₂ →ₗ[R] M₃) (g : M →ₗ[R] M₂ →ₗ[R] Fin n → M₃) :
    M →ₗ[R] M₂ →ₗ[R] Fin n.succ → M₃ where
  toFun m := LinearMap.vecCons (f m) (g m)
  map_add' x y :=
    LinearMap.ext fun z => by
      simp only [f.map_add, g.map_add, LinearMap.add_apply, LinearMap.vecCons_apply,
        Matrix.cons_add_cons (f x z)]
                                             /-
                                               R : Type u
                                               K : Type u'
                                               M : Type v
                                               V : Type v'
                                               M₂ : Type w
                                               V₂ : Type w'
                                               M₃ : Type y
                                               V₃ : Type y'
                                               M₄ : Type z
                                               ι : Type x
                                               ι' : Type x'
                                               inst✝⁶ : CommSemiring R
                                               inst✝⁵ : AddCommMonoid M
                                               inst✝⁴ : AddCommMonoid M₂
                                               inst✝³ : AddCommMonoid M₃
                                               inst✝² : Module R M
                                               inst✝¹ : Module R M₂
                                               inst✝ : Module R M₃
                                               n : Nat
                                               f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M₂ M₃)
                                               g : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M₂ (Fin n → M₃))
                                               r : R
                                               x : M
                                               z : M₂
                                               ⊢ Eq (({ toFun := fun m => (f m).vecCons (g m), map_add' := ⋯ }.toFun (HSMul.h …
                                             -/
  map_smul' r x := LinearMap.ext fun z => by simp [Matrix.smul_cons r (f x z)]
                                             /-
                                               🎉 no goals
                                             -/


