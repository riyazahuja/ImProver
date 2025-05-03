/-- Given a topological ring `R` and `s : Set X`, construct the ideal in `C(X, R)` of functions
which vanish on the complement of `s`. -/
def idealOfSet (s : Set X) : Ideal C(X, R) where
  carrier := {f : C(X, R) | ∀ x ∈ sᶜ, f x = 0}
                                  /-
                                    X : Type u_1
                                    R : Type u_2
                                    inst✝³ : TopologicalSpace X
                                    inst✝² : Semiring R
                                    inst✝¹ : TopologicalSpace R
                                    inst✝ : TopologicalSemiring R
                                    s : Set X
                                    f g : ContinuousMap X R
                                    hf : Membership.mem (setOf fun f => ∀ (x : X), Membership.mem (HasCompl.compl  …
                                    hg : Membership.mem (setOf fun f => ∀ (x : X), Membership.mem (HasCompl.compl  …
                                    x : X
                                    hx : Membership.mem (HasCompl.compl s) x
                                    ⊢ Eq ((HAdd.hAdd f g) x) 0
                                  -/
  add_mem' {f g} hf hg x hx := by simp [hf x hx, hg x hx, coe_add, Pi.add_apply, add_zero]
                                  /-
                                    🎉 no goals
                                  -/
  zero_mem' _ _ := rfl
  smul_mem' c _ hf x hx := mul_zero (c x) ▸ congr_arg (fun y => c x * y) (hf x hx)


theorem idealOfSet_closed [T2Space R] (s : Set X) :
    IsClosed (idealOfSet R s : Set C(X, R)) := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Semiring R
    inst✝² : TopologicalSpace R
    inst✝¹ : TopologicalSemiring R
    inst✝ : T2Space R
    s : Set X
    ⊢ IsClosed ↑(ContinuousMap.idealOfSet R s)
  -/
  simp only [idealOfSet, Submodule.coe_set_mk, Set.setOf_forall]
  exact isClosed_iInter fun x => isClosed_iInter fun _ =>
    isClosed_eq (continuous_eval_const x) continuous_const


theorem mem_idealOfSet {s : Set X} {f : C(X, R)} :
    f ∈ idealOfSet R s ↔ ∀ ⦃x : X⦄, x ∈ sᶜ → f x = 0 := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    s : Set X
    f : ContinuousMap X R
    ⊢ Iff (Membership.mem (ContinuousMap.idealOfSet R s) f) (∀ ⦃x : X⦄, Membership …
  -/
  convert Iff.rfl
  /-
    🎉 no goals
  -/


theorem not_mem_idealOfSet {s : Set X} {f : C(X, R)} : f ∉ idealOfSet R s ↔ ∃ x ∈ sᶜ, f x ≠ 0 := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    s : Set X
    f : ContinuousMap X R
    ⊢ Iff (Not (Membership.mem (ContinuousMap.idealOfSet R s) f)) (Exists fun x => …
  -/
  simp_rw [mem_idealOfSet]; push_neg; rfl
                                      /-
                                        🎉 no goals
                                      -/


/-- Given an ideal `I` of `C(X, R)`, construct the set of points for which every function in the
ideal vanishes on the complement. -/
def setOfIdeal (I : Ideal C(X, R)) : Set X :=
  {x : X | ∀ f ∈ I, (f : C(X, R)) x = 0}ᶜ


theorem not_mem_setOfIdeal {I : Ideal C(X, R)} {x : X} :
    x ∉ setOfIdeal I ↔ ∀ ⦃f : C(X, R)⦄, f ∈ I → f x = 0 := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    I : Ideal (ContinuousMap X R)
    x : X
    ⊢ Iff (Not (Membership.mem (ContinuousMap.setOfIdeal I) x)) (∀ ⦃f : Continuous …
  -/
  rw [← Set.mem_compl_iff, setOfIdeal, compl_compl, Set.mem_setOf]
  /-
    🎉 no goals
  -/


theorem mem_setOfIdeal {I : Ideal C(X, R)} {x : X} :
    x ∈ setOfIdeal I ↔ ∃ f ∈ I, (f : C(X, R)) x ≠ 0 := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    I : Ideal (ContinuousMap X R)
    x : X
    ⊢ Iff (Membership.mem (ContinuousMap.setOfIdeal I) x) (Exists fun f => And (Me …
  -/
  simp_rw [setOfIdeal, Set.mem_compl_iff, Set.mem_setOf]; push_neg; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem setOfIdeal_open [T2Space R] (I : Ideal C(X, R)) : IsOpen (setOfIdeal I) := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Semiring R
    inst✝² : TopologicalSpace R
    inst✝¹ : TopologicalSemiring R
    inst✝ : T2Space R
    I : Ideal (ContinuousMap X R)
    ⊢ IsOpen (ContinuousMap.setOfIdeal I)
  -/
  simp only [setOfIdeal, Set.setOf_forall, isOpen_compl_iff]
  exact
    isClosed_iInter fun f =>
      isClosed_iInter fun _ => isClosed_eq (map_continuous f) continuous_const


/-- The open set `ContinuousMap.setOfIdeal I` realized as a term of `opens X`. -/
@[simps]
def opensOfIdeal [T2Space R] (I : Ideal C(X, R)) : Opens X :=
  ⟨setOfIdeal I, setOfIdeal_open I⟩


@[simp]
theorem setOfTop_eq_univ [Nontrivial R] : setOfIdeal (⊤ : Ideal C(X, R)) = Set.univ :=
  Set.univ_subset_iff.mp fun _ _ => mem_setOfIdeal.mpr ⟨1, Submodule.mem_top, one_ne_zero⟩


@[simp]
theorem idealOfEmpty_eq_bot : idealOfSet R (∅ : Set X) = ⊥ :=
  Ideal.ext fun f => by
    simp only [mem_idealOfSet, Set.compl_empty, Set.mem_univ, forall_true_left, Ideal.mem_bot,
      DFunLike.ext_iff, zero_apply]


@[simp]
theorem mem_idealOfSet_compl_singleton (x : X) (f : C(X, R)) :
    f ∈ idealOfSet R ({x}ᶜ : Set X) ↔ f x = 0 := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    x : X
    f : ContinuousMap X R
    ⊢ Iff (Membership.mem (ContinuousMap.idealOfSet R (HasCompl.compl (Singleton.s …
  -/
  simp only [mem_idealOfSet, compl_compl, Set.mem_singleton_iff, forall_eq]
  /-
    🎉 no goals
  -/


theorem ideal_gc : GaloisConnection (setOfIdeal : Ideal C(X, R) → Set X) (idealOfSet R) := by
  /-
    X : Type u_1
    R : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : Semiring R
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalSemiring R
    ⊢ GaloisConnection ContinuousMap.setOfIdeal (ContinuousMap.idealOfSet R)
  -/
  refine fun I s => ⟨fun h f hf => ?_, fun h x hx => ?_⟩
    /-
      case refine_1
      X : Type u_1
      R : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : Semiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      I : Ideal (ContinuousMap X R)
      s : Set X
      h : LE.le (ContinuousMap.setOfIdeal I) s
      f : ContinuousMap X R
      hf : Membership.mem I f
      ⊢ Membership.mem (ContinuousMap.idealOfSet R s) f
    -/
  · by_contra h'
    /-
      case refine_1
      X : Type u_1
      R : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : Semiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      I : Ideal (ContinuousMap X R)
      s : Set X
      h : LE.le (ContinuousMap.setOfIdeal I) s
      f : ContinuousMap X R
      hf : Membership.mem I f
      h' : Not (Membership.mem (ContinuousMap.idealOfSet R s) f)
      ⊢ False
    -/
    rcases not_mem_idealOfSet.mp h' with ⟨x, hx, hfx⟩
    /-
      case refine_1.intro.intro
      X : Type u_1
      R : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : Semiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      I : Ideal (ContinuousMap X R)
      s : Set X
      h : LE.le (ContinuousMap.setOfIdeal I) s
      f : ContinuousMap X R
      hf : Membership.mem I f
      h' : Not (Membership.mem (ContinuousMap.idealOfSet R s) f)
      x : X
      hx : Membership.mem (HasCompl.compl s) x
      hfx : Ne (f x) 0
      ⊢ False
    -/
    exact hfx (not_mem_setOfIdeal.mp (mt (@h x) hx) hf)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      R : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : Semiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      I : Ideal (ContinuousMap X R)
      s : Set X
      h : LE.le I (ContinuousMap.idealOfSet R s)
      x : X
      hx : Membership.mem (ContinuousMap.setOfIdeal I) x
      ⊢ Membership.mem s x
    -/
  · obtain ⟨f, hf, hfx⟩ := mem_setOfIdeal.mp hx
    /-
      case refine_2.intro.intro
      X : Type u_1
      R : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : Semiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      I : Ideal (ContinuousMap X R)
      s : Set X
      h : LE.le I (ContinuousMap.idealOfSet R s)
      x : X
      hx : Membership.mem (ContinuousMap.setOfIdeal I) x
      f : ContinuousMap X R
      hf : Membership.mem I f
      hfx : Ne (f x) 0
      ⊢ Membership.mem s x
    -/
    by_contra hx'
    /-
      case refine_2.intro.intro
      X : Type u_1
      R : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : Semiring R
      inst✝¹ : TopologicalSpace R
      inst✝ : TopologicalSemiring R
      I : Ideal (ContinuousMap X R)
      s : Set X
      h : LE.le I (ContinuousMap.idealOfSet R s)
      x : X
      hx : Membership.mem (ContinuousMap.setOfIdeal I) x
      f : ContinuousMap X R
      hf : Membership.mem I f
      hfx : Ne (f x) 0
      hx' : Not (Membership.mem s x)
      ⊢ False
    -/
    exact not_mem_idealOfSet.mpr ⟨x, hx', hfx⟩ (h hf)
    /-
      🎉 no goals
    -/


/-- An auxiliary lemma used in the proof of `ContinuousMap.idealOfSet_ofIdeal_eq_closure` which may
be useful on its own. -/
theorem exists_mul_le_one_eqOn_ge (f : C(X, ℝ≥0)) {c : ℝ≥0} (hc : 0 < c) :
    ∃ g : C(X, ℝ≥0), (∀ x : X, (g * f) x ≤ 1) ∧ {x : X | c ≤ f x}.EqOn (g * f) 1 :=
  ⟨{  toFun := (f ⊔ const X c)⁻¹
      continuous_toFun :=
        ((map_continuous f).sup <| map_continuous _).inv₀ fun _ => (hc.trans_le le_sup_right).ne' },
    fun x =>
    (inv_mul_le_iff₀ (hc.trans_le le_sup_right)).mpr ((mul_one (f x ⊔ c)).symm ▸ le_sup_left),
    fun x hx => by
    simpa only [coe_const, mul_apply, coe_mk, Pi.inv_apply, Pi.sup_apply,
      Function.const_apply, sup_eq_left.mpr (Set.mem_setOf.mp hx), ne_eq, Pi.one_apply]
      using inv_mul_cancel₀ (hc.trans_le hx).ne' ⟩


@[simp]
theorem idealOfSet_ofIdeal_eq_closure (I : Ideal C(X, 𝕜)) :
    idealOfSet 𝕜 (setOfIdeal I) = I.closure := by
  /- Since `idealOfSet 𝕜 (setOfIdeal I)` is closed and contains `I`, it contains `I.closure`.
    For the reverse inclusion, given `f ∈ idealOfSet 𝕜 (setOfIdeal I)` and `(ε : ℝ≥0) > 0` it
    suffices to show that `f` is within `ε` of `I`. -/
  refine le_antisymm ?_
      ((idealOfSet_closed 𝕜 <| setOfIdeal I).closure_subset_iff.mpr fun f hf x hx =>
        not_mem_setOfIdeal.mp hx hf)
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    ⊢ LE.le (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) I.closure
  -/
  refine (fun f hf => Metric.mem_closure_iff.mpr fun ε hε => ?_)
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun b => And (Membership.mem (↑I) b) (LT.lt (Dist.dist f b) ε)
  -/
  lift ε to ℝ≥0 using hε.lt.le
  /-
    case intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : GT.gt (↑ε) 0
    ⊢ Exists fun b => And (Membership.mem (↑I) b) (LT.lt (Dist.dist f b) ↑ε)
  -/
  replace hε := show (0 : ℝ≥0) < ε from hε
  /-
    case intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun b => And (Membership.mem (↑I) b) (LT.lt (Dist.dist f b) ↑ε)
  -/
  simp_rw [dist_nndist]
  /-
    case intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun b => And (Membership.mem (↑I) b) (LT.lt ↑(NNDist.nndist f b) ↑ε)
  -/
  norm_cast
  -- Let `t := {x : X | ε / 2 ≤ ‖f x‖₊}}` which is closed and disjoint from `set_of_ideal I`.
  /-
    case intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun b => And (Membership.mem (↑I) b) (LT.lt (NNDist.nndist f b) ε)
  -/
  set t := {x : X | ε / 2 ≤ ‖f x‖₊}
  /-
    case intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    t : Set X := setOf fun x => LE.le (HDiv.hDiv ε 2) (NNNorm.nnnorm (f x))
    ⊢ Exists fun b => And (Membership.mem (↑I) b) (LT.lt (NNDist.nndist f b) ε)
  -/
  have ht : IsClosed t := isClosed_le continuous_const (map_continuous f).nnnorm
  have htI : Disjoint t (setOfIdeal I)ᶜ := by
    refine Set.subset_compl_iff_disjoint_left.mp fun x hx => ?_
    simpa only [t, Set.mem_setOf, Set.mem_compl_iff, not_le] using
      (nnnorm_eq_zero.mpr (mem_idealOfSet.mp hf hx)).trans_lt (half_pos hε)
  /- It suffices to produce `g : C(X, ℝ≥0)` which takes values in `[0,1]` and is constantly `1` on
    `t` such that when composed with the natural embedding of `ℝ≥0` into `𝕜` lies in the ideal `I`.
    Indeed, then `‖f - f * ↑g‖ ≤ ‖f * (1 - ↑g)‖ ≤ ⨆ ‖f * (1 - ↑g) x‖`. When `x ∉ t`, `‖f x‖ < ε / 2`
    and `‖(1 - ↑g) x‖ ≤ 1`, and when `x ∈ t`, `(1 - ↑g) x = 0`, and clearly `f * ↑g ∈ I`. -/
  suffices
    ∃ g : C(X, ℝ≥0), (algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g ∈ I ∧ (∀ x, g x ≤ 1) ∧ t.EqOn g 1 by
    obtain ⟨g, hgI, hg, hgt⟩ := this
    refine ⟨f * (algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g, I.mul_mem_left f hgI, ?_⟩
    rw [nndist_eq_nnnorm]
    refine (nnnorm_lt_iff _ hε).2 fun x => ?_
    simp only [coe_sub, coe_mul, Pi.sub_apply, Pi.mul_apply]
    by_cases hx : x ∈ t
    · simpa only [hgt hx, comp_apply, Pi.one_apply, ContinuousMap.coe_coe, algebraMapCLM_apply,
        map_one, mul_one, sub_self, nnnorm_zero] using hε
    · refine lt_of_le_of_lt ?_ (half_lt_self hε)
      have :=
        calc
          ‖((1 - (algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g) x : 𝕜)‖₊ =
              ‖1 - algebraMap ℝ≥0 𝕜 (g x)‖₊ := by
            simp only [coe_sub, coe_one, coe_comp, ContinuousMap.coe_coe, Pi.sub_apply,
              Pi.one_apply, Function.comp_apply, algebraMapCLM_apply]
          _ = ‖algebraMap ℝ≥0 𝕜 (1 - g x)‖₊ := by
            simp only [Algebra.algebraMap_eq_smul_one, NNReal.smul_def, NNReal.coe_sub (hg x),
              NNReal.coe_one, sub_smul, one_smul]
          _ ≤ 1 := (nnnorm_algebraMap_nnreal 𝕜 (1 - g x)).trans_le tsub_le_self
      calc
        ‖f x - f x * (algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g x‖₊ =
            ‖f x * (1 - (algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g) x‖₊ := by
          simp only [mul_sub, coe_sub, coe_one, Pi.sub_apply, Pi.one_apply, mul_one]
        _ ≤ ε / 2 * ‖(1 - (algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g) x‖₊ :=
          ((nnnorm_mul_le _ _).trans
            (mul_le_mul_right' (not_le.mp <| show ¬ε / 2 ≤ ‖f x‖₊ from hx).le _))
        _ ≤ ε / 2 := by simpa only [mul_one] using mul_le_mul_left' this _
  /- There is some `g' : C(X, ℝ≥0)` which is strictly positive on `t` such that the composition
    `↑g` with the natural embedding of `ℝ≥0` into `𝕜` lies in `I`. This follows from compactness of
    `t` and that we can do it in any neighborhood of a point `x ∈ t`. Indeed, since `x ∈ t`, then
    `fₓ x ≠ 0` for some `fₓ ∈ I` and so `fun y ↦ ‖(star fₓ * fₓ) y‖₊` is strictly posiive in a
    neighborhood of `y`. Moreover, `(‖(star fₓ * fₓ) y‖₊ : 𝕜) = (star fₓ * fₓ) y`, so composition of
    this map with the natural embedding is just `star fₓ * fₓ ∈ I`. -/
  have : ∃ g' : C(X, ℝ≥0), (algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g' ∈ I ∧ ∀ x ∈ t, 0 < g' x := by
    refine ht.isCompact.induction_on ?_ ?_ ?_ ?_
    · refine ⟨0, ?_, fun x hx => False.elim hx⟩
      convert I.zero_mem
      ext
      simp only [comp_apply, zero_apply, ContinuousMap.coe_coe, map_zero]
    · rintro s₁ s₂ hs ⟨g, hI, hgt⟩; exact ⟨g, hI, fun x hx => hgt x (hs hx)⟩
    · rintro s₁ s₂ ⟨g₁, hI₁, hgt₁⟩ ⟨g₂, hI₂, hgt₂⟩
      refine ⟨g₁ + g₂, ?_, fun x hx => ?_⟩
      · convert I.add_mem hI₁ hI₂
        ext y
        simp only [coe_add, Pi.add_apply, map_add, coe_comp, Function.comp_apply,
          ContinuousMap.coe_coe]
      · rcases hx with (hx | hx)
        · simpa only [zero_add] using add_lt_add_of_lt_of_le (hgt₁ x hx) zero_le'
        · simpa only [zero_add] using add_lt_add_of_le_of_lt zero_le' (hgt₂ x hx)
    · intro x hx
      replace hx := htI.subset_compl_right hx
      rw [compl_compl, mem_setOfIdeal] at hx
      obtain ⟨g, hI, hgx⟩ := hx
      have := (map_continuous g).continuousAt.eventually_ne hgx
      refine
        ⟨{y : X | g y ≠ 0} ∩ t,
          mem_nhdsWithin_iff_exists_mem_nhds_inter.mpr ⟨_, this, Set.Subset.rfl⟩,
          ⟨⟨fun x => ‖g x‖₊ ^ 2, (map_continuous g).nnnorm.pow 2⟩, ?_, fun x hx =>
            pow_pos (norm_pos_iff.mpr hx.1) 2⟩⟩
      convert I.mul_mem_left (star g) hI
      ext
      simp only [comp_apply, ContinuousMap.coe_coe, coe_mk, algebraMapCLM_apply, map_pow,
        mul_apply, star_apply, star_def]
      simp only [normSq_eq_def', RCLike.conj_mul, ofReal_pow]
      rfl
  /- Get the function `g'` which is guaranteed to exist above. By the extreme value theorem and
    compactness of `t`, there is some `0 < c` such that `c ≤ g' x` for all `x ∈ t`. Then by
    `exists_mul_le_one_eqOn_ge` there is some `g` for which `g * g'` is the desired function. -/
  /-
    case intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    t : Set X := setOf fun x => LE.le (HDiv.hDiv ε 2) (NNNorm.nnnorm (f x))
    ht : IsClosed t
    htI : Disjoint t (HasCompl.compl (ContinuousMap.setOfIdeal I))
    this : Exists fun g' => And (Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).com …
    ⊢ Exists fun g => And (Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g))  …
  -/
  obtain ⟨g', hI', hgt'⟩ := this
  obtain ⟨c, hc, hgc'⟩ : ∃ c > 0, ∀ y : X, y ∈ t → c ≤ g' y :=
    t.eq_empty_or_nonempty.elim
      (fun ht' => ⟨1, zero_lt_one, fun y hy => False.elim (by rwa [ht'] at hy)⟩) fun ht' =>
      let ⟨x, hx, hx'⟩ := ht.isCompact.exists_isMinOn ht' (map_continuous g').continuousOn
      ⟨g' x, hgt' x hx, hx'⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    t : Set X := setOf fun x => LE.le (HDiv.hDiv ε 2) (NNNorm.nnnorm (f x))
    ht : IsClosed t
    htI : Disjoint t (HasCompl.compl (ContinuousMap.setOfIdeal I))
    g' : ContinuousMap X NNReal
    hI' : Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g')
    hgt' : ∀ (x : X), Membership.mem t x → LT.lt 0 (g' x)
    c : NNReal
    hc : GT.gt c 0
    hgc' : ∀ (y : X), Membership.mem t y → LE.le c (g' y)
    ⊢ Exists fun g => And (Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g))  …
  -/
  obtain ⟨g, hg, hgc⟩ := exists_mul_le_one_eqOn_ge g' hc
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    t : Set X := setOf fun x => LE.le (HDiv.hDiv ε 2) (NNNorm.nnnorm (f x))
    ht : IsClosed t
    htI : Disjoint t (HasCompl.compl (ContinuousMap.setOfIdeal I))
    g' : ContinuousMap X NNReal
    hI' : Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g')
    hgt' : ∀ (x : X), Membership.mem t x → LT.lt 0 (g' x)
    c : NNReal
    hc : GT.gt c 0
    hgc' : ∀ (y : X), Membership.mem t y → LE.le c (g' y)
    g : ContinuousMap X NNReal
    hg : ∀ (x : X), LE.le ((HMul.hMul g g') x) 1
    hgc : Set.EqOn (⇑(HMul.hMul g g')) 1 (setOf fun x => LE.le c (g' x))
    ⊢ Exists fun g => And (Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g))  …
  -/
  refine ⟨g * g', ?_, hg, hgc.mono hgc'⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    t : Set X := setOf fun x => LE.le (HDiv.hDiv ε 2) (NNNorm.nnnorm (f x))
    ht : IsClosed t
    htI : Disjoint t (HasCompl.compl (ContinuousMap.setOfIdeal I))
    g' : ContinuousMap X NNReal
    hI' : Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g')
    hgt' : ∀ (x : X), Membership.mem t x → LT.lt 0 (g' x)
    c : NNReal
    hc : GT.gt c 0
    hgc' : ∀ (y : X), Membership.mem t y → LE.le c (g' y)
    g : ContinuousMap X NNReal
    hg : ∀ (x : X), LE.le ((HMul.hMul g g') x) 1
    hgc : Set.EqOn (⇑(HMul.hMul g g')) 1 (setOf fun x => LE.le c (g' x))
    ⊢ Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp (HMul.hMul g g'))
  -/
  convert I.mul_mem_left ((algebraMapCLM ℝ≥0 𝕜 : C(ℝ≥0, 𝕜)).comp g) hI'
  /-
    case h.e'_5
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    t : Set X := setOf fun x => LE.le (HDiv.hDiv ε 2) (NNNorm.nnnorm (f x))
    ht : IsClosed t
    htI : Disjoint t (HasCompl.compl (ContinuousMap.setOfIdeal I))
    g' : ContinuousMap X NNReal
    hI' : Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g')
    hgt' : ∀ (x : X), Membership.mem t x → LT.lt 0 (g' x)
    c : NNReal
    hc : GT.gt c 0
    hgc' : ∀ (y : X), Membership.mem t y → LE.le c (g' y)
    g : ContinuousMap X NNReal
    hg : ∀ (x : X), LE.le ((HMul.hMul g g') x) 1
    hgc : Set.EqOn (⇑(HMul.hMul g g')) 1 (setOf fun x => LE.le c (g' x))
    ⊢ Eq ((↑(algebraMapCLM NNReal 𝕜)).comp (HMul.hMul g g')) (HMul.hMul ((↑(algebr …
  -/
  ext
  /-
    case h.e'_5.h
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    f : ContinuousMap X 𝕜
    hf : Membership.mem (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)) f
    ε : NNReal
    hε : LT.lt 0 ε
    t : Set X := setOf fun x => LE.le (HDiv.hDiv ε 2) (NNNorm.nnnorm (f x))
    ht : IsClosed t
    htI : Disjoint t (HasCompl.compl (ContinuousMap.setOfIdeal I))
    g' : ContinuousMap X NNReal
    hI' : Membership.mem I ((↑(algebraMapCLM NNReal 𝕜)).comp g')
    hgt' : ∀ (x : X), Membership.mem t x → LT.lt 0 (g' x)
    c : NNReal
    hc : GT.gt c 0
    hgc' : ∀ (y : X), Membership.mem t y → LE.le c (g' y)
    g : ContinuousMap X NNReal
    hg : ∀ (x : X), LE.le ((HMul.hMul g g') x) 1
    hgc : Set.EqOn (⇑(HMul.hMul g g')) 1 (setOf fun x => LE.le c (g' x))
    a✝ : X
    ⊢ Eq (((↑(algebraMapCLM NNReal 𝕜)).comp (HMul.hMul g g')) a✝) ((HMul.hMul ((↑( …
  -/
  simp only [algebraMapCLM_coe, comp_apply, mul_apply, ContinuousMap.coe_coe, map_mul]
  /-
    🎉 no goals
  -/


theorem idealOfSet_ofIdeal_isClosed {I : Ideal C(X, 𝕜)} (hI : IsClosed (I : Set C(X, 𝕜))) :
    idealOfSet 𝕜 (setOfIdeal I) = I :=
  (idealOfSet_ofIdeal_eq_closure I).trans (Ideal.ext <| Set.ext_iff.mp hI.closure_eq)


@[simp]
theorem setOfIdeal_ofSet_eq_interior (s : Set X) : setOfIdeal (idealOfSet 𝕜 s) = interior s := by
  refine
    Set.Subset.antisymm
      ((setOfIdeal_open (idealOfSet 𝕜 s)).subset_interior_iff.mpr fun x hx =>
        let ⟨f, hf, hfx⟩ := mem_setOfIdeal.mp hx
        Set.not_mem_compl_iff.mp (mt (@hf x) hfx))
      fun x hx => ?_
  -- If `x ∉ closure sᶜ`, we must produce `f : C(X, 𝕜)` which is zero on `sᶜ` and `f x ≠ 0`.
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    s : Set X
    x : X
    hx : Membership.mem (interior s) x
    ⊢ Membership.mem (ContinuousMap.setOfIdeal (ContinuousMap.idealOfSet 𝕜 s)) x
  -/
  rw [← compl_compl (interior s), ← closure_compl] at hx
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    s : Set X
    x : X
    hx : Membership.mem (HasCompl.compl (closure (HasCompl.compl s))) x
    ⊢ Membership.mem (ContinuousMap.setOfIdeal (ContinuousMap.idealOfSet 𝕜 s)) x
  -/
  simp_rw [mem_setOfIdeal, mem_idealOfSet]
  /- Apply Urysohn's lemma to get `g : C(X, ℝ)` which is zero on `sᶜ` and `g x ≠ 0`, then compose
    with the natural embedding `ℝ ↪ 𝕜` to produce the desired `f`. -/
  obtain ⟨g, hgs, hgx : Set.EqOn g 1 {x}, -⟩ :=
    exists_continuous_zero_one_of_isClosed isClosed_closure isClosed_singleton
      (Set.disjoint_singleton_right.mpr hx)
  exact
    ⟨⟨fun x => g x, continuous_ofReal.comp (map_continuous g)⟩, by
      simpa only [coe_mk, ofReal_eq_zero] using fun x hx => hgs (subset_closure hx), by
      simpa only [coe_mk, hgx (Set.mem_singleton x), Pi.one_apply, RCLike.ofReal_one] using
        one_ne_zero⟩


theorem setOfIdeal_ofSet_of_isOpen {s : Set X} (hs : IsOpen s) : setOfIdeal (idealOfSet 𝕜 s) = s :=
  (setOfIdeal_ofSet_eq_interior 𝕜 s).trans hs.interior_eq


/-- The Galois insertion `ContinuousMap.opensOfIdeal : Ideal C(X, 𝕜) → Opens X` and
`fun s ↦ ContinuousMap.idealOfSet ↑s`. -/
@[simps]
def idealOpensGI :
    GaloisInsertion (opensOfIdeal : Ideal C(X, 𝕜) → Opens X) fun s => idealOfSet 𝕜 s where
  choice I _ := opensOfIdeal I.closure
  gc I s := ideal_gc X 𝕜 I s
  le_l_u s := (setOfIdeal_ofSet_of_isOpen 𝕜 s.isOpen).ge
  choice_eq I hI :=
    congr_arg _ <|
      Ideal.ext
        (Set.ext_iff.mp
          (isClosed_of_closure_subset <|
              (idealOfSet_ofIdeal_eq_closure I ▸ hI : I.closure ≤ I)).closure_eq)


theorem idealOfSet_isMaximal_iff (s : Opens X) :
    (idealOfSet 𝕜 (s : Set X)).IsMaximal ↔ IsCoatom s := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    s : TopologicalSpace.Opens X
    ⊢ Iff (ContinuousMap.idealOfSet 𝕜 ↑s).IsMaximal (IsCoatom s)
  -/
  rw [Ideal.isMaximal_def]
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    s : TopologicalSpace.Opens X
    ⊢ Iff (IsCoatom (ContinuousMap.idealOfSet 𝕜 ↑s)) (IsCoatom s)
  -/
  refine (idealOpensGI X 𝕜).isCoatom_iff (fun I hI => ?_) s
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    s : TopologicalSpace.Opens X
    I : Ideal (ContinuousMap X 𝕜)
    hI : IsCoatom I
    ⊢ Eq (ContinuousMap.idealOfSet 𝕜 ↑(ContinuousMap.opensOfIdeal I)) I
  -/
  rw [← Ideal.isMaximal_def] at hI
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    s : TopologicalSpace.Opens X
    I : Ideal (ContinuousMap X 𝕜)
    hI : I.IsMaximal
    ⊢ Eq (ContinuousMap.idealOfSet 𝕜 ↑(ContinuousMap.opensOfIdeal I)) I
  -/
  exact idealOfSet_ofIdeal_isClosed inferInstance
  /-
    🎉 no goals
  -/


theorem idealOf_compl_singleton_isMaximal (x : X) : (idealOfSet 𝕜 ({x}ᶜ : Set X)).IsMaximal :=
  (idealOfSet_isMaximal_iff 𝕜 (Closeds.singleton x).compl).mpr <| Opens.isCoatom_iff.mpr ⟨x, rfl⟩


theorem setOfIdeal_eq_compl_singleton (I : Ideal C(X, 𝕜)) [hI : I.IsMaximal] :
    ∃ x : X, setOfIdeal I = {x}ᶜ := by
  have h : (idealOfSet 𝕜 (setOfIdeal I)).IsMaximal :=
    (idealOfSet_ofIdeal_isClosed (inferInstance : IsClosed (I : Set C(X, 𝕜)))).symm ▸ hI
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    hI : I.IsMaximal
    h : (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)).IsMaximal
    ⊢ Exists fun x => Eq (ContinuousMap.setOfIdeal I) (HasCompl.compl (Singleton.s …
  -/
  obtain ⟨x, hx⟩ := Opens.isCoatom_iff.1 ((idealOfSet_isMaximal_iff 𝕜 (opensOfIdeal I)).1 h)
  /-
    case intro
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    hI : I.IsMaximal
    h : (ContinuousMap.idealOfSet 𝕜 (ContinuousMap.setOfIdeal I)).IsMaximal
    x : X
    hx : Eq (ContinuousMap.opensOfIdeal I) (TopologicalSpace.Closeds.singleton x). …
    ⊢ Exists fun x => Eq (ContinuousMap.setOfIdeal I) (HasCompl.compl (Singleton.s …
  -/
  exact ⟨x, congr_arg (fun (s : Opens X) => (s : Set X)) hx⟩
  /-
    🎉 no goals
  -/


theorem ideal_isMaximal_iff (I : Ideal C(X, 𝕜)) [hI : IsClosed (I : Set C(X, 𝕜))] :
    I.IsMaximal ↔ ∃ x : X, idealOfSet 𝕜 {x}ᶜ = I := by
  refine
    ⟨?_, fun h =>
      let ⟨x, hx⟩ := h
      hx ▸ idealOf_compl_singleton_isMaximal 𝕜 x⟩
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    hI : IsClosed ↑I
    ⊢ I.IsMaximal → Exists fun x => Eq (ContinuousMap.idealOfSet 𝕜 (HasCompl.compl …
  -/
  intro hI'
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : TopologicalSpace X
    inst✝¹ : CompactSpace X
    inst✝ : T2Space X
    I : Ideal (ContinuousMap X 𝕜)
    hI : IsClosed ↑I
    hI' : I.IsMaximal
    ⊢ Exists fun x => Eq (ContinuousMap.idealOfSet 𝕜 (HasCompl.compl (Singleton.si …
  -/
  obtain ⟨x, hx⟩ := setOfIdeal_eq_compl_singleton I
  exact
    ⟨x, by
      simpa only [idealOfSet_ofIdeal_eq_closure, I.closure_eq_of_isClosed hI] using
        congr_arg (idealOfSet 𝕜) hx.symm⟩


/-- The natural continuous map from a locally compact topological space `X` to the
`WeakDual.characterSpace 𝕜 C(X, 𝕜)` which sends `x : X` to point evaluation at `x`. -/
def continuousMapEval : C(X, characterSpace 𝕜 C(X, 𝕜)) where
  toFun x :=
    ⟨{  toFun := fun f => f x
        map_add' := fun _ _ => rfl
        map_smul' := fun _ _ => rfl
        cont := continuous_eval_const x }, by
        /-
          X : Type u_1
          𝕜 : Type u_2
          inst✝⁵ : TopologicalSpace X
          inst✝⁴ : CommRing 𝕜
          inst✝³ : TopologicalSpace 𝕜
          inst✝² : TopologicalRing 𝕜
          inst✝¹ : Nontrivial 𝕜
          inst✝ : NoZeroDivisors 𝕜
          x : X
          ⊢ Membership.mem (WeakDual.characterSpace 𝕜 (ContinuousMap X 𝕜)) { toFun := fu …
        -/
        rw [CharacterSpace.eq_set_map_one_map_mul]; exact ⟨rfl, fun f g => rfl⟩⟩
                                                    /-
                                                      🎉 no goals
                                                    -/
  continuous_toFun := by
    /-
      X : Type u_1
      𝕜 : Type u_2
      inst✝⁵ : TopologicalSpace X
      inst✝⁴ : CommRing 𝕜
      inst✝³ : TopologicalSpace 𝕜
      inst✝² : TopologicalRing 𝕜
      inst✝¹ : Nontrivial 𝕜
      inst✝ : NoZeroDivisors 𝕜
      ⊢ Continuous fun x => ⟨{ toFun := fun f => f x, map_add' := ⋯, map_smul' := ⋯, …
    -/
    exact Continuous.subtype_mk (continuous_of_continuous_eval map_continuous) _
    /-
      🎉 no goals
    -/


@[simp]
theorem continuousMapEval_apply_apply (x : X) (f : C(X, 𝕜)) : continuousMapEval X 𝕜 x f = f x :=
  rfl


theorem continuousMapEval_bijective : Bijective (continuousMapEval X 𝕜) := by
  /-
    X : Type u_1
    𝕜 : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : CompactSpace X
    inst✝¹ : T2Space X
    inst✝ : RCLike 𝕜
    ⊢ Function.Bijective ⇑(WeakDual.CharacterSpace.continuousMapEval X 𝕜)
  -/
  refine ⟨fun x y hxy => ?_, fun φ => ?_⟩
    /-
      case refine_1
      X : Type u_1
      𝕜 : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : RCLike 𝕜
      x y : X
      hxy : Eq ((WeakDual.CharacterSpace.continuousMapEval X 𝕜) x) ((WeakDual.Charac …
      ⊢ Eq x y
    -/
  · contrapose! hxy
    rcases exists_continuous_zero_one_of_isClosed (isClosed_singleton : _root_.IsClosed {x})
        (isClosed_singleton : _root_.IsClosed {y}) (Set.disjoint_singleton.mpr hxy) with
      ⟨f, fx, fy, -⟩
    /-
      case refine_1.intro.intro.intro
      X : Type u_1
      𝕜 : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : RCLike 𝕜
      x y : X
      hxy : Ne x y
      f : ContinuousMap X Real
      fx : Set.EqOn (⇑f) 0 (Singleton.singleton x)
      fy : Set.EqOn (⇑f) 1 (Singleton.singleton y)
      ⊢ Ne ((WeakDual.CharacterSpace.continuousMapEval X 𝕜) x) ((WeakDual.CharacterS …
    -/
    rw [DFunLike.ne_iff]
    /-
      case refine_1.intro.intro.intro
      X : Type u_1
      𝕜 : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : RCLike 𝕜
      x y : X
      hxy : Ne x y
      f : ContinuousMap X Real
      fx : Set.EqOn (⇑f) 0 (Singleton.singleton x)
      fy : Set.EqOn (⇑f) 1 (Singleton.singleton y)
      ⊢ Exists fun a => Ne (((WeakDual.CharacterSpace.continuousMapEval X 𝕜) x) a) ( …
    -/
    use (⟨fun (x : ℝ) => (x : 𝕜), RCLike.continuous_ofReal⟩ : C(ℝ, 𝕜)).comp f
    simpa only [continuousMapEval_apply_apply, ContinuousMap.comp_apply, coe_mk, Ne,
      RCLike.ofReal_inj] using
      ((fx (Set.mem_singleton x)).symm ▸ (fy (Set.mem_singleton y)).symm ▸ zero_ne_one : f x ≠ f y)
    /-
      case refine_2
      X : Type u_1
      𝕜 : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : RCLike 𝕜
      φ : ↑(WeakDual.characterSpace 𝕜 (ContinuousMap X 𝕜))
      ⊢ Exists fun a => Eq ((WeakDual.CharacterSpace.continuousMapEval X 𝕜) a) φ
    -/
  · obtain ⟨x, hx⟩ := (ideal_isMaximal_iff (RingHom.ker φ)).mp inferInstance
    /-
      case refine_2.intro
      X : Type u_1
      𝕜 : Type u_2
      inst✝³ : TopologicalSpace X
      inst✝² : CompactSpace X
      inst✝¹ : T2Space X
      inst✝ : RCLike 𝕜
      φ : ↑(WeakDual.characterSpace 𝕜 (ContinuousMap X 𝕜))
      x : X
      hx : Eq (ContinuousMap.idealOfSet 𝕜 (HasCompl.compl (Singleton.singleton x)))  …
      ⊢ Exists fun a => Eq ((WeakDual.CharacterSpace.continuousMapEval X 𝕜) a) φ
    -/
    refine ⟨x, CharacterSpace.ext_ker <| Ideal.ext fun f => ?_⟩
    simpa only [RingHom.mem_ker, continuousMapEval_apply_apply, mem_idealOfSet_compl_singleton,
      RingHom.mem_ker] using SetLike.ext_iff.mp hx f


/-- This is the natural homeomorphism between a compact Hausdorff space `X` and the
`WeakDual.characterSpace 𝕜 C(X, 𝕜)`. -/
noncomputable def homeoEval : X ≃ₜ characterSpace 𝕜 C(X, 𝕜) :=
  @Continuous.homeoOfEquivCompactToT2 _ _ _ _ _ _
    { Equiv.ofBijective _ (continuousMapEval_bijective X 𝕜) with toFun := continuousMapEval X 𝕜 }
    (map_continuous (continuousMapEval X 𝕜))


