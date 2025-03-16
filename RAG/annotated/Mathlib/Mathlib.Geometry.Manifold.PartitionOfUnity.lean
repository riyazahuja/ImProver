local notation "∞" => (⊤ : ℕ∞)


/-- We say that a collection of `SmoothBumpFunction`s is a `SmoothBumpCovering` of a set `s` if

* `(f i).c ∈ s` for all `i`;
* the family `fun i ↦ support (f i)` is locally finite;
* for each point `x ∈ s` there exists `i` such that `f i =ᶠ[𝓝 x] 1`;
  in other words, `x` belongs to the interior of `{y | f i y = 1}`;

If `M` is a finite dimensional real manifold which is a `σ`-compact Hausdorff topological space,
then for every covering `U : M → Set M`, `∀ x, U x ∈ 𝓝 x`, there exists a `SmoothBumpCovering`
subordinate to `U`, see `SmoothBumpCovering.exists_isSubordinate`.

This covering can be used, e.g., to construct a partition of unity and to prove the weak
Whitney embedding theorem. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure SmoothBumpCovering [FiniteDimensional ℝ E] (s : Set M := univ) where
  /-- The center point of each bump in the smooth covering. -/
  c : ι → M
  /-- A smooth bump function around `c i`. -/
  toFun : ∀ i, SmoothBumpFunction I (c i)
  /-- All the bump functions in the covering are centered at points in `s`. -/
  c_mem' : ∀ i, c i ∈ s
  /-- Around each point, there are only finitely many nonzero bump functions in the family. -/
  locallyFinite' : LocallyFinite fun i => support (toFun i)
  /-- Around each point in `s`, one of the bump functions is equal to `1`. -/
  eventuallyEq_one' : ∀ x ∈ s, ∃ i, toFun i =ᶠ[𝓝 x] 1


/-- We say that a collection of functions form a smooth partition of unity on a set `s` if

* all functions are infinitely smooth and nonnegative;
* the family `fun i ↦ support (f i)` is locally finite;
* for all `x ∈ s` the sum `∑ᶠ i, f i x` equals one;
* for all `x`, the sum `∑ᶠ i, f i x` is less than or equal to one. -/
structure SmoothPartitionOfUnity (s : Set M := univ) where
  /-- The family of functions forming the partition of unity. -/
  toFun : ι → C^∞⟮I, M; 𝓘(ℝ), ℝ⟯
  /-- Around each point, there are only finitely many nonzero functions in the family. -/
  locallyFinite' : LocallyFinite fun i => support (toFun i)
  /-- All the functions in the partition of unity are nonnegative. -/
  nonneg' : ∀ i x, 0 ≤ toFun i x
  /-- The functions in the partition of unity add up to `1` at any point of `s`. -/
  sum_eq_one' : ∀ x ∈ s, ∑ᶠ i, toFun i x = 1
  /-- The functions in the partition of unity add up to at most `1` everywhere. -/
  sum_le_one' : ∀ x, ∑ᶠ i, toFun i x ≤ 1


instance {s : Set M} : FunLike (SmoothPartitionOfUnity ι I M s) ι C^∞⟮I, M; 𝓘(ℝ), ℝ⟯ where
  coe := toFun
                             /-
                               ι : Type uι
                               E : Type uE
                               inst✝⁶ : NormedAddCommGroup E
                               inst✝⁵ : NormedSpace Real E
                               F : Type uF
                               inst✝⁴ : NormedAddCommGroup F
                               inst✝³ : NormedSpace Real F
                               H : Type uH
                               inst✝² : TopologicalSpace H
                               I : ModelWithCorners Real E H
                               M : Type uM
                               inst✝¹ : TopologicalSpace M
                               inst✝ : ChartedSpace H M
                               s✝ : Set M
                               f✝ : SmoothPartitionOfUnity ι I M s✝
                               n : ENat
                               s : Set M
                               f g : SmoothPartitionOfUnity ι I M s
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


protected theorem locallyFinite : LocallyFinite fun i => support (f i) :=
  f.locallyFinite'


theorem nonneg (i : ι) (x : M) : 0 ≤ f i x :=
  f.nonneg' i x


theorem sum_eq_one {x} (hx : x ∈ s) : ∑ᶠ i, f i x = 1 :=
  f.sum_eq_one' x hx


theorem exists_pos_of_mem {x} (hx : x ∈ s) : ∃ i, 0 < f i x := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type uH
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    f : SmoothPartitionOfUnity ι I M s
    x : M
    hx : Membership.mem s x
    ⊢ Exists fun i => LT.lt 0 ((f i) x)
  -/
  by_contra! h
  /-
    ι : Type uι
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H : Type uH
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    f : SmoothPartitionOfUnity ι I M s
    x : M
    hx : Membership.mem s x
    h : ∀ (i : ι), LE.le ((f i) x) 0
    ⊢ False
  -/
  have H : ∀ i, f i x = 0 := fun i ↦ le_antisymm (h i) (f.nonneg i x)
  /-
    ι : Type uι
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H✝ : Type uH
    inst✝² : TopologicalSpace H✝
    I : ModelWithCorners Real E H✝
    M : Type uM
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H✝ M
    s : Set M
    f : SmoothPartitionOfUnity ι I M s
    x : M
    hx : Membership.mem s x
    h : ∀ (i : ι), LE.le ((f i) x) 0
    H : ∀ (i : ι), Eq ((f i) x) 0
    ⊢ False
  -/
  have := f.sum_eq_one hx
  /-
    ι : Type uι
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H✝ : Type uH
    inst✝² : TopologicalSpace H✝
    I : ModelWithCorners Real E H✝
    M : Type uM
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H✝ M
    s : Set M
    f : SmoothPartitionOfUnity ι I M s
    x : M
    hx : Membership.mem s x
    h : ∀ (i : ι), LE.le ((f i) x) 0
    H : ∀ (i : ι), Eq ((f i) x) 0
    this : Eq (finsum fun i => (f i) x) 1
    ⊢ False
  -/
  simp_rw [H] at this
  /-
    ι : Type uι
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    H✝ : Type uH
    inst✝² : TopologicalSpace H✝
    I : ModelWithCorners Real E H✝
    M : Type uM
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H✝ M
    s : Set M
    f : SmoothPartitionOfUnity ι I M s
    x : M
    hx : Membership.mem s x
    h : ∀ (i : ι), LE.le ((f i) x) 0
    H : ∀ (i : ι), Eq ((f i) x) 0
    this : Eq (finsum fun i => 0) 1
    ⊢ False
  -/
  simpa
  /-
    🎉 no goals
  -/


theorem sum_le_one (x : M) : ∑ᶠ i, f i x ≤ 1 :=
  f.sum_le_one' x


/-- Reinterpret a smooth partition of unity as a continuous partition of unity. -/
@[simps]
def toPartitionOfUnity : PartitionOfUnity ι M s :=
  { f with toFun := fun i => f i }


theorem contMDiff_sum : ContMDiff I 𝓘(ℝ) ⊤ fun x => ∑ᶠ i, f i x :=
  contMDiff_finsum (fun i => (f i).contMDiff) f.locallyFinite


@[deprecated (since := "2024-11-21")] alias smooth_sum := contMDiff_sum


theorem le_one (i : ι) (x : M) : f i x ≤ 1 :=
  f.toPartitionOfUnity.le_one i x


theorem sum_nonneg (x : M) : 0 ≤ ∑ᶠ i, f i x :=
  f.toPartitionOfUnity.sum_nonneg x


theorem finsum_smul_mem_convex {g : ι → M → F} {t : Set F} {x : M} (hx : x ∈ s)
    (hg : ∀ i, f i x ≠ 0 → g i x ∈ t) (ht : Convex ℝ t) : ∑ᶠ i, f i x • g i x ∈ t :=
  ht.finsum_mem (fun _ => f.nonneg _ _) (f.sum_eq_one hx) hg


theorem contMDiff_smul {g : M → F} {i} (hg : ∀ x ∈ tsupport (f i), ContMDiffAt I 𝓘(ℝ, F) n g x) :
    ContMDiff I 𝓘(ℝ, F) n fun x => f i x • g x :=
  contMDiff_of_tsupport fun x hx =>
    ((f i).contMDiff.contMDiffAt.of_le le_top).smul <| hg x <| tsupport_smul_subset_left _ _ hx


@[deprecated (since := "2024-11-21")] alias smooth_smul := contMDiff_smul


/-- If `f` is a smooth partition of unity on a set `s : Set M` and `g : ι → M → F` is a family of
functions such that `g i` is $C^n$ smooth at every point of the topological support of `f i`, then
the sum `fun x ↦ ∑ᶠ i, f i x • g i x` is smooth on the whole manifold. -/
theorem contMDiff_finsum_smul {g : ι → M → F}
    (hg : ∀ (i), ∀ x ∈ tsupport (f i), ContMDiffAt I 𝓘(ℝ, F) n (g i) x) :
    ContMDiff I 𝓘(ℝ, F) n fun x => ∑ᶠ i, f i x • g i x :=
  (contMDiff_finsum fun i => f.contMDiff_smul (hg i)) <|
    f.locallyFinite.subset fun _ => support_smul_subset_left _ _


@[deprecated (since := "2024-11-21")] alias smooth_finsum_smul := contMDiff_finsum_smul


theorem contMDiffAt_finsum {x₀ : M} {g : ι → M → F}
    (hφ : ∀ i, x₀ ∈ tsupport (f i) → ContMDiffAt I 𝓘(ℝ, F) n (g i) x₀) :
    ContMDiffAt I 𝓘(ℝ, F) n (fun x ↦ ∑ᶠ i, f i x • g i x) x₀ := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    F : Type uF
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    H : Type uH
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    f : SmoothPartitionOfUnity ι I M s
    n : ENat
    x₀ : M
    g : ι → M → F
    hφ : ∀ (i : ι), Membership.mem (tsupport ⇑(f i)) x₀ → ContMDiffAt I (modelWith …
    ⊢ ContMDiffAt I (modelWithCornersSelf Real F) n (fun x => finsum fun i => HSMu …
  -/
  refine _root_.contMDiffAt_finsum (f.locallyFinite.smul_left _) fun i ↦ ?_
  /-
    ι : Type uι
    E : Type uE
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    F : Type uF
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace Real F
    H : Type uH
    inst✝² : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    s : Set M
    f : SmoothPartitionOfUnity ι I M s
    n : ENat
    x₀ : M
    g : ι → M → F
    hφ : ∀ (i : ι), Membership.mem (tsupport ⇑(f i)) x₀ → ContMDiffAt I (modelWith …
    i : ι
    ⊢ ContMDiffAt I (modelWithCornersSelf Real F) n (fun x => HSMul.hSMul ((f i) x …
  -/
  by_cases hx : x₀ ∈ tsupport (f i)
    /-
      case pos
      ι : Type uι
      E : Type uE
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Real E
      F : Type uF
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace Real F
      H : Type uH
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      s : Set M
      f : SmoothPartitionOfUnity ι I M s
      n : ENat
      x₀ : M
      g : ι → M → F
      hφ : ∀ (i : ι), Membership.mem (tsupport ⇑(f i)) x₀ → ContMDiffAt I (modelWith …
      i : ι
      hx : Membership.mem (tsupport ⇑(f i)) x₀
      ⊢ ContMDiffAt I (modelWithCornersSelf Real F) n (fun x => HSMul.hSMul ((f i) x …
    -/
  · exact ContMDiffAt.smul ((f i).contMDiff.of_le le_top).contMDiffAt (hφ i hx)
    /-
      🎉 no goals
    -/
  · exact contMDiffAt_of_not_mem (compl_subset_compl.mpr
      (tsupport_smul_subset_left (f i) (g i)) hx) n


theorem contDiffAt_finsum {s : Set E} (f : SmoothPartitionOfUnity ι 𝓘(ℝ, E) E s) {x₀ : E}
    {g : ι → E → F} (hφ : ∀ i, x₀ ∈ tsupport (f i) → ContDiffAt ℝ n (g i) x₀) :
    ContDiffAt ℝ n (fun x ↦ ∑ᶠ i, f i x • g i x) x₀ := by
  /-
    ι : Type uι
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    n : ENat
    s : Set E
    f : SmoothPartitionOfUnity ι (modelWithCornersSelf Real E) E s
    x₀ : E
    g : ι → E → F
    hφ : ∀ (i : ι), Membership.mem (tsupport ⇑(f i)) x₀ → ContDiffAt Real (↑n) (g  …
    ⊢ ContDiffAt Real (↑n) (fun x => finsum fun i => HSMul.hSMul ((f i) x) (g i x) …
  -/
  simp only [← contMDiffAt_iff_contDiffAt] at *
  /-
    ι : Type uι
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    n : ENat
    s : Set E
    f : SmoothPartitionOfUnity ι (modelWithCornersSelf Real E) E s
    x₀ : E
    g : ι → E → F
    hφ : ∀ (i : ι), Membership.mem (tsupport ⇑(f i)) x₀ → ContMDiffAt (modelWithCo …
    ⊢ ContMDiffAt (modelWithCornersSelf Real E) (modelWithCornersSelf Real F) n (f …
  -/
  exact f.contMDiffAt_finsum hφ
  /-
    🎉 no goals
  -/


/-- The support of a smooth partition of unity at a point `x₀` as a `Finset`.
  This is the set of `i : ι` such that `x₀ ∈ support f i`, i.e. `f i ≠ x₀`. -/
def finsupport : Finset ι := ρ.toPartitionOfUnity.finsupport x₀


@[simp]
theorem mem_finsupport {i : ι} : i ∈ ρ.finsupport x₀ ↔ i ∈ support fun i ↦ ρ i x₀ :=
  ρ.toPartitionOfUnity.mem_finsupport x₀


@[simp]
theorem coe_finsupport : (ρ.finsupport x₀ : Set ι) = support fun i ↦ ρ i x₀ :=
  ρ.toPartitionOfUnity.coe_finsupport x₀


theorem sum_finsupport (hx₀ : x₀ ∈ s) : ∑ i ∈ ρ.finsupport x₀, ρ i x₀ = 1 :=
  ρ.toPartitionOfUnity.sum_finsupport hx₀


theorem sum_finsupport' (hx₀ : x₀ ∈ s) {I : Finset ι} (hI : ρ.finsupport x₀ ⊆ I) :
    ∑ i ∈ I, ρ i x₀ = 1 :=
  ρ.toPartitionOfUnity.sum_finsupport' hx₀ hI


theorem sum_finsupport_smul_eq_finsum {A : Type*} [AddCommGroup A] [Module ℝ A] (φ : ι → M → A) :
    ∑ i ∈ ρ.finsupport x₀, ρ i x₀ • φ i x₀ = ∑ᶠ i, ρ i x₀ • φ i x₀ :=
  ρ.toPartitionOfUnity.sum_finsupport_smul_eq_finsum φ


/-- The `tsupport`s of a smooth partition of unity are locally finite. -/
theorem finite_tsupport : {i | x₀ ∈ tsupport (ρ i)}.Finite :=
  ρ.toPartitionOfUnity.finite_tsupport _


/-- The tsupport of a partition of unity at a point `x₀` as a `Finset`.
  This is the set of `i : ι` such that `x₀ ∈ tsupport f i`. -/
def fintsupport (x : M) : Finset ι :=
  (ρ.finite_tsupport x).toFinset


theorem mem_fintsupport_iff (i : ι) : i ∈ ρ.fintsupport x₀ ↔ x₀ ∈ tsupport (ρ i) :=
  Finite.mem_toFinset _


theorem eventually_fintsupport_subset : ∀ᶠ y in 𝓝 x₀, ρ.fintsupport y ⊆ ρ.fintsupport x₀ :=
  ρ.toPartitionOfUnity.eventually_fintsupport_subset _


theorem finsupport_subset_fintsupport : ρ.finsupport x₀ ⊆ ρ.fintsupport x₀ :=
  ρ.toPartitionOfUnity.finsupport_subset_fintsupport x₀


theorem eventually_finsupport_subset : ∀ᶠ y in 𝓝 x₀, ρ.finsupport y ⊆ ρ.fintsupport x₀ :=
  ρ.toPartitionOfUnity.eventually_finsupport_subset x₀


/-- A smooth partition of unity `f i` is subordinate to a family of sets `U i` indexed by the same
type if for each `i` the closure of the support of `f i` is a subset of `U i`. -/
def IsSubordinate (f : SmoothPartitionOfUnity ι I M s) (U : ι → Set M) :=
  ∀ i, tsupport (f i) ⊆ U i


@[simp]
theorem isSubordinate_toPartitionOfUnity :
    f.toPartitionOfUnity.IsSubordinate U ↔ f.IsSubordinate U :=
  Iff.rfl


alias ⟨_, IsSubordinate.toPartitionOfUnity⟩ := isSubordinate_toPartitionOfUnity


/-- If `f` is a smooth partition of unity on a set `s : Set M` subordinate to a family of open sets
`U : ι → Set M` and `g : ι → M → F` is a family of functions such that `g i` is $C^n$ smooth on
`U i`, then the sum `fun x ↦ ∑ᶠ i, f i x • g i x` is $C^n$ smooth on the whole manifold. -/
theorem IsSubordinate.contMDiff_finsum_smul {g : ι → M → F} (hf : f.IsSubordinate U)
    (ho : ∀ i, IsOpen (U i)) (hg : ∀ i, ContMDiffOn I 𝓘(ℝ, F) n (g i) (U i)) :
    ContMDiff I 𝓘(ℝ, F) n fun x => ∑ᶠ i, f i x • g i x :=
  f.contMDiff_finsum_smul fun i _ hx => (hg i).contMDiffAt <| (ho i).mem_nhds (hf i hx)


@[deprecated (since := "2024-11-21")]
alias IsSubordinate.smooth_finsum_smul := IsSubordinate.contMDiff_finsum_smul


theorem contMDiff_toPartitionOfUnity {E : Type uE} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {H : Type uH} [TopologicalSpace H] {I : ModelWithCorners ℝ E H} {M : Type uM}
    [TopologicalSpace M] [ChartedSpace H M] {s : Set M} (f : BumpCovering ι M s)
    (hf : ∀ i, ContMDiff I 𝓘(ℝ) ⊤ (f i)) (i : ι) : ContMDiff I 𝓘(ℝ) ⊤ (f.toPartitionOfUnity i) :=
  (hf i).mul <| (contMDiff_finprod_cond fun j _ => contMDiff_const.sub (hf j)) <| by
    /-
      ι : Type uι
      E : Type uE
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type uH
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      s : Set M
      f : BumpCovering ι M s
      hf : ∀ (i : ι), ContMDiff I (modelWithCornersSelf Real Real) Top.top ⇑(f i)
      i : ι
      ⊢ LocallyFinite fun i => Function.mulSupport fun x => HSub.hSub 1 ((f i) x)
    -/
    simp only [Pi.sub_def, mulSupport_one_sub]
    /-
      ι : Type uι
      E : Type uE
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      H : Type uH
      inst✝² : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      s : Set M
      f : BumpCovering ι M s
      hf : ∀ (i : ι), ContMDiff I (modelWithCornersSelf Real Real) Top.top ⇑(f i)
      i : ι
      ⊢ LocallyFinite fun i => Function.support ⇑(f i)
    -/
    exact f.locallyFinite
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-21")]
alias smooth_toPartitionOfUnity := contMDiff_toPartitionOfUnity


/-- A `BumpCovering` such that all functions in this covering are smooth generates a smooth
partition of unity.

In our formalization, not every `f : BumpCovering ι M s` with smooth functions `f i` is a
`SmoothBumpCovering`; instead, a `SmoothBumpCovering` is a covering by supports of
`SmoothBumpFunction`s. So, we define `BumpCovering.toSmoothPartitionOfUnity`, then reuse it
in `SmoothBumpCovering.toSmoothPartitionOfUnity`. -/
def toSmoothPartitionOfUnity (f : BumpCovering ι M s) (hf : ∀ i, ContMDiff I 𝓘(ℝ) ⊤ (f i)) :
    SmoothPartitionOfUnity ι I M s :=
  { f.toPartitionOfUnity with
    toFun := fun i => ⟨f.toPartitionOfUnity i, f.contMDiff_toPartitionOfUnity hf i⟩ }


@[simp]
theorem toSmoothPartitionOfUnity_toPartitionOfUnity (f : BumpCovering ι M s)
    (hf : ∀ i, ContMDiff I 𝓘(ℝ) ⊤ (f i)) :
    (f.toSmoothPartitionOfUnity hf).toPartitionOfUnity = f.toPartitionOfUnity :=
  rfl


@[simp]
theorem coe_toSmoothPartitionOfUnity (f : BumpCovering ι M s) (hf : ∀ i, ContMDiff I 𝓘(ℝ) ⊤ (f i))
    (i : ι) : ⇑(f.toSmoothPartitionOfUnity hf i) = f.toPartitionOfUnity i :=
  rfl


theorem IsSubordinate.toSmoothPartitionOfUnity {f : BumpCovering ι M s} {U : ι → Set M}
    (h : f.IsSubordinate U) (hf : ∀ i, ContMDiff I 𝓘(ℝ) ⊤ (f i)) :
    (f.toSmoothPartitionOfUnity hf).IsSubordinate U :=
  h.toPartitionOfUnity


instance : CoeFun (SmoothBumpCovering ι I M s) fun x => ∀ i : ι, SmoothBumpFunction I (x.c i) :=
  ⟨toFun⟩


/--
We say that `f : SmoothBumpCovering ι I M s` is *subordinate* to a map `U : M → Set M` if for each
index `i`, we have `tsupport (f i) ⊆ U (f i).c`. This notion is a bit more general than
being subordinate to an open covering of `M`, because we make no assumption about the way `U x`
depends on `x`.
-/
def IsSubordinate {s : Set M} (f : SmoothBumpCovering ι I M s) (U : M → Set M) :=
  ∀ i, tsupport (f i) ⊆ U (f.c i)


theorem IsSubordinate.support_subset {fs : SmoothBumpCovering ι I M s} {U : M → Set M}
    (h : fs.IsSubordinate U) (i : ι) : support (fs i) ⊆ U (fs.c i) :=
  Subset.trans subset_closure (h i)


variable (I) in
/-- Let `M` be a smooth manifold with corners modelled on a finite dimensional real vector space.
Suppose also that `M` is a Hausdorff `σ`-compact topological space. Let `s` be a closed set
in `M` and `U : M → Set M` be a collection of sets such that `U x ∈ 𝓝 x` for every `x ∈ s`.
Then there exists a smooth bump covering of `s` that is subordinate to `U`. -/
theorem exists_isSubordinate [T2Space M] [SigmaCompactSpace M] (hs : IsClosed s)
    (hU : ∀ x ∈ s, U x ∈ 𝓝 x) :
    ∃ (ι : Type uM) (f : SmoothBumpCovering ι I M s), f.IsSubordinate U := by
  -- First we deduce some missing instances
  /-
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : FiniteDimensional Real E
    s : Set M
    U : M → Set M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    hs : IsClosed s
    hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
    ⊢ Exists fun ι => Exists fun f => f.IsSubordinate U
  -/
  haveI : LocallyCompactSpace H := I.locallyCompactSpace
  /-
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : FiniteDimensional Real E
    s : Set M
    U : M → Set M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    hs : IsClosed s
    hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
    this : LocallyCompactSpace H
    ⊢ Exists fun ι => Exists fun f => f.IsSubordinate U
  -/
  haveI : LocallyCompactSpace M := ChartedSpace.locallyCompactSpace H M
  -- Next we choose a covering by supports of smooth bump functions
  /-
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : FiniteDimensional Real E
    s : Set M
    U : M → Set M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    hs : IsClosed s
    hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
    this✝ : LocallyCompactSpace H
    this : LocallyCompactSpace M
    ⊢ Exists fun ι => Exists fun f => f.IsSubordinate U
  -/
  have hB := fun x hx => SmoothBumpFunction.nhds_basis_support (I := I) (hU x hx)
  rcases refinement_of_locallyCompact_sigmaCompact_of_nhds_basis_set hs hB with
    ⟨ι, c, f, hf, hsub', hfin⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : FiniteDimensional Real E
    s : Set M
    U : M → Set M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    hs : IsClosed s
    hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
    this✝ : LocallyCompactSpace H
    this : LocallyCompactSpace M
    hB : ∀ (x : M), Membership.mem s x → (nhds x).HasBasis (fun f => HasSubset.Sub …
    ι : Type uM
    c : ι → M
    f : (a : ι) → SmoothBumpFunction I (c a)
    hf : ∀ (a : ι), And (Membership.mem s (c a)) (HasSubset.Subset (tsupport ↑(f a …
    hsub' : HasSubset.Subset s (Set.iUnion fun a => Function.support ↑(f a))
    hfin : LocallyFinite fun a => Function.support ↑(f a)
    ⊢ Exists fun ι => Exists fun f => f.IsSubordinate U
  -/
  choose hcs hfU using hf
  -- Then we use the shrinking lemma to get a covering by smaller open
  rcases exists_subset_iUnion_closed_subset hs (fun i => (f i).isOpen_support)
    (fun x _ => hfin.point_finite x) hsub' with ⟨V, hsV, hVc, hVf⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : FiniteDimensional Real E
    s : Set M
    U : M → Set M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    hs : IsClosed s
    hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
    this✝ : LocallyCompactSpace H
    this : LocallyCompactSpace M
    hB : ∀ (x : M), Membership.mem s x → (nhds x).HasBasis (fun f => HasSubset.Sub …
    ι : Type uM
    c : ι → M
    f : (a : ι) → SmoothBumpFunction I (c a)
    hsub' : HasSubset.Subset s (Set.iUnion fun a => Function.support ↑(f a))
    hfin : LocallyFinite fun a => Function.support ↑(f a)
    hcs : ∀ (a : ι), Membership.mem s (c a)
    hfU : ∀ (a : ι), HasSubset.Subset (tsupport ↑(f a)) (U (c a))
    V : ι → Set M
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVc : ∀ (i : ι), IsClosed (V i)
    hVf : ∀ (i : ι), HasSubset.Subset (V i) (Function.support ↑(f i))
    ⊢ Exists fun ι => Exists fun f => f.IsSubordinate U
  -/
  choose r hrR hr using fun i => (f i).exists_r_pos_lt_subset_ball (hVc i) (hVf i)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : FiniteDimensional Real E
    s : Set M
    U : M → Set M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    hs : IsClosed s
    hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
    this✝ : LocallyCompactSpace H
    this : LocallyCompactSpace M
    hB : ∀ (x : M), Membership.mem s x → (nhds x).HasBasis (fun f => HasSubset.Sub …
    ι : Type uM
    c : ι → M
    f : (a : ι) → SmoothBumpFunction I (c a)
    hsub' : HasSubset.Subset s (Set.iUnion fun a => Function.support ↑(f a))
    hfin : LocallyFinite fun a => Function.support ↑(f a)
    hcs : ∀ (a : ι), Membership.mem s (c a)
    hfU : ∀ (a : ι), HasSubset.Subset (tsupport ↑(f a)) (U (c a))
    V : ι → Set M
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVc : ∀ (i : ι), IsClosed (V i)
    hVf : ∀ (i : ι), HasSubset.Subset (V i) (Function.support ↑(f i))
    r : ι → Real
    hrR : ∀ (i : ι), Membership.mem (Set.Ioo 0 (f i).rOut) (r i)
    hr : ∀ (i : ι), HasSubset.Subset (V i) (Inter.inter (chartAt H (c i)).source ( …
    ⊢ Exists fun ι => Exists fun f => f.IsSubordinate U
  -/
  refine ⟨ι, ⟨c, fun i => (f i).updateRIn (r i) (hrR i), hcs, ?_, fun x hx => ?_⟩, fun i => ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      E : Type uE
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type uH
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : FiniteDimensional Real E
      s : Set M
      U : M → Set M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      hs : IsClosed s
      hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
      this✝ : LocallyCompactSpace H
      this : LocallyCompactSpace M
      hB : ∀ (x : M), Membership.mem s x → (nhds x).HasBasis (fun f => HasSubset.Sub …
      ι : Type uM
      c : ι → M
      f : (a : ι) → SmoothBumpFunction I (c a)
      hsub' : HasSubset.Subset s (Set.iUnion fun a => Function.support ↑(f a))
      hfin : LocallyFinite fun a => Function.support ↑(f a)
      hcs : ∀ (a : ι), Membership.mem s (c a)
      hfU : ∀ (a : ι), HasSubset.Subset (tsupport ↑(f a)) (U (c a))
      V : ι → Set M
      hsV : HasSubset.Subset s (Set.iUnion V)
      hVc : ∀ (i : ι), IsClosed (V i)
      hVf : ∀ (i : ι), HasSubset.Subset (V i) (Function.support ↑(f i))
      r : ι → Real
      hrR : ∀ (i : ι), Membership.mem (Set.Ioo 0 (f i).rOut) (r i)
      hr : ∀ (i : ι), HasSubset.Subset (V i) (Inter.inter (chartAt H (c i)).source ( …
      ⊢ LocallyFinite fun i => Function.support ↑((fun i => (f i).updateRIn (r i) ⋯) …
    -/
  · simpa only [SmoothBumpFunction.support_updateRIn]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      E : Type uE
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type uH
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : FiniteDimensional Real E
      s : Set M
      U : M → Set M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      hs : IsClosed s
      hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
      this✝ : LocallyCompactSpace H
      this : LocallyCompactSpace M
      hB : ∀ (x : M), Membership.mem s x → (nhds x).HasBasis (fun f => HasSubset.Sub …
      ι : Type uM
      c : ι → M
      f : (a : ι) → SmoothBumpFunction I (c a)
      hsub' : HasSubset.Subset s (Set.iUnion fun a => Function.support ↑(f a))
      hfin : LocallyFinite fun a => Function.support ↑(f a)
      hcs : ∀ (a : ι), Membership.mem s (c a)
      hfU : ∀ (a : ι), HasSubset.Subset (tsupport ↑(f a)) (U (c a))
      V : ι → Set M
      hsV : HasSubset.Subset s (Set.iUnion V)
      hVc : ∀ (i : ι), IsClosed (V i)
      hVf : ∀ (i : ι), HasSubset.Subset (V i) (Function.support ↑(f i))
      r : ι → Real
      hrR : ∀ (i : ι), Membership.mem (Set.Ioo 0 (f i).rOut) (r i)
      hr : ∀ (i : ι), HasSubset.Subset (V i) (Inter.inter (chartAt H (c i)).source ( …
      x : M
      hx : Membership.mem s x
      ⊢ Exists fun i => (nhds x).EventuallyEq (↑((fun i => (f i).updateRIn (r i) ⋯)  …
    -/
  · refine (mem_iUnion.1 <| hsV hx).imp fun i hi => ?_
    exact ((f i).updateRIn _ _).eventuallyEq_one_of_dist_lt
      ((f i).support_subset_source <| hVf _ hi) (hr i hi).2
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_3
      E : Type uE
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type uH
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : FiniteDimensional Real E
      s : Set M
      U : M → Set M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      hs : IsClosed s
      hU : ∀ (x : M), Membership.mem s x → Membership.mem (nhds x) (U x)
      this✝ : LocallyCompactSpace H
      this : LocallyCompactSpace M
      hB : ∀ (x : M), Membership.mem s x → (nhds x).HasBasis (fun f => HasSubset.Sub …
      ι : Type uM
      c : ι → M
      f : (a : ι) → SmoothBumpFunction I (c a)
      hsub' : HasSubset.Subset s (Set.iUnion fun a => Function.support ↑(f a))
      hfin : LocallyFinite fun a => Function.support ↑(f a)
      hcs : ∀ (a : ι), Membership.mem s (c a)
      hfU : ∀ (a : ι), HasSubset.Subset (tsupport ↑(f a)) (U (c a))
      V : ι → Set M
      hsV : HasSubset.Subset s (Set.iUnion V)
      hVc : ∀ (i : ι), IsClosed (V i)
      hVf : ∀ (i : ι), HasSubset.Subset (V i) (Function.support ↑(f i))
      r : ι → Real
      hrR : ∀ (i : ι), Membership.mem (Set.Ioo 0 (f i).rOut) (r i)
      hr : ∀ (i : ι), HasSubset.Subset (V i) (Inter.inter (chartAt H (c i)).source ( …
      i : ι
      ⊢ HasSubset.Subset (tsupport ↑({ c := c, toFun := fun i => (f i).updateRIn (r  …
    -/
  · simpa only [SmoothBumpFunction.support_updateRIn, tsupport] using hfU i
    /-
      🎉 no goals
    -/


protected theorem locallyFinite : LocallyFinite fun i => support (fs i) :=
  fs.locallyFinite'


protected theorem point_finite (x : M) : {i | fs i x ≠ 0}.Finite :=
  fs.locallyFinite.point_finite x


/-- Index of a bump function such that `fs i =ᶠ[𝓝 x] 1`. -/
def ind (x : M) (hx : x ∈ s) : ι :=
  (fs.eventuallyEq_one' x hx).choose


theorem eventuallyEq_one (x : M) (hx : x ∈ s) : fs (fs.ind x hx) =ᶠ[𝓝 x] 1 :=
  (fs.eventuallyEq_one' x hx).choose_spec


theorem apply_ind (x : M) (hx : x ∈ s) : fs (fs.ind x hx) x = 1 :=
  (fs.eventuallyEq_one x hx).eq_of_nhds


theorem mem_support_ind (x : M) (hx : x ∈ s) : x ∈ support (fs <| fs.ind x hx) := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : FiniteDimensional Real E
    s : Set M
    fs : SmoothBumpCovering ι I M s
    x : M
    hx : Membership.mem s x
    ⊢ Membership.mem (Function.support ↑(fs.toFun (fs.ind x hx))) x
  -/
  simp [fs.apply_ind x hx]
  /-
    🎉 no goals
  -/


theorem mem_chartAt_source_of_eq_one {i : ι} {x : M} (h : fs i x = 1) :
    x ∈ (chartAt H (fs.c i)).source :=
                                     /-
                                       ι : Type uι
                                       E : Type uE
                                       inst✝⁵ : NormedAddCommGroup E
                                       inst✝⁴ : NormedSpace Real E
                                       H : Type uH
                                       inst✝³ : TopologicalSpace H
                                       I : ModelWithCorners Real E H
                                       M : Type uM
                                       inst✝² : TopologicalSpace M
                                       inst✝¹ : ChartedSpace H M
                                       inst✝ : FiniteDimensional Real E
                                       s : Set M
                                       fs : SmoothBumpCovering ι I M s
                                       i : ι
                                       x : M
                                       h : Eq (↑(fs.toFun i) x) 1
                                       ⊢ Membership.mem (Function.support ↑(fs.toFun i)) x
                                     -/
  (fs i).support_subset_source <| by simp [h]
                                     /-
                                       🎉 no goals
                                     -/


theorem mem_extChartAt_source_of_eq_one {i : ι} {x : M} (h : fs i x = 1) :
    x ∈ (extChartAt I (fs.c i)).source := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    H : Type uH
    inst✝³ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝² : TopologicalSpace M
    inst✝¹ : ChartedSpace H M
    inst✝ : FiniteDimensional Real E
    s : Set M
    fs : SmoothBumpCovering ι I M s
    i : ι
    x : M
    h : Eq (↑(fs.toFun i) x) 1
    ⊢ Membership.mem (extChartAt I (fs.c i)).source x
  -/
  rw [extChartAt_source]; exact fs.mem_chartAt_source_of_eq_one h
                          /-
                            🎉 no goals
                          -/


theorem mem_chartAt_ind_source (x : M) (hx : x ∈ s) : x ∈ (chartAt H (fs.c (fs.ind x hx))).source :=
  fs.mem_chartAt_source_of_eq_one (fs.apply_ind x hx)


theorem mem_extChartAt_ind_source (x : M) (hx : x ∈ s) :
    x ∈ (extChartAt I (fs.c (fs.ind x hx))).source :=
  fs.mem_extChartAt_source_of_eq_one (fs.apply_ind x hx)


/-- The index type of a `SmoothBumpCovering` of a compact manifold is finite. -/
protected def fintype [CompactSpace M] : Fintype ι :=
  fs.locallyFinite.fintypeOfCompact fun i => (fs i).nonempty_support


/-- Reinterpret a `SmoothBumpCovering` as a continuous `BumpCovering`. Note that not every
`f : BumpCovering ι M s` with smooth functions `f i` is a `SmoothBumpCovering`. -/
def toBumpCovering : BumpCovering ι M s where
  toFun i := ⟨fs i, (fs i).continuous⟩
  locallyFinite' := fs.locallyFinite
  nonneg' i _ := (fs i).nonneg
  le_one' i _ := (fs i).le_one
  eventuallyEq_one' := fs.eventuallyEq_one'

-- Porting note: `simpNF` says that `simp` can't simplify LHS but it can.

@[simp, nolint simpNF]
theorem isSubordinate_toBumpCovering {f : SmoothBumpCovering ι I M s} {U : M → Set M} :
    (f.toBumpCovering.IsSubordinate fun i => U (f.c i)) ↔ f.IsSubordinate U :=
  Iff.rfl


alias ⟨_, IsSubordinate.toBumpCovering⟩ := isSubordinate_toBumpCovering


/-- Every `SmoothBumpCovering` defines a smooth partition of unity. -/
def toSmoothPartitionOfUnity : SmoothPartitionOfUnity ι I M s :=
  fs.toBumpCovering.toSmoothPartitionOfUnity fun i => (fs i).contMDiff


theorem toSmoothPartitionOfUnity_apply (i : ι) (x : M) :
    fs.toSmoothPartitionOfUnity i x = fs i x * ∏ᶠ (j) (_ : WellOrderingRel j i), (1 - fs j x) :=
  rfl


open Classical in
theorem toSmoothPartitionOfUnity_eq_mul_prod (i : ι) (x : M) (t : Finset ι)
    (ht : ∀ j, WellOrderingRel j i → fs j x ≠ 0 → j ∈ t) :
    fs.toSmoothPartitionOfUnity i x =
      fs i x * ∏ j ∈ t.filter fun j => WellOrderingRel j i, (1 - fs j x) :=
  fs.toBumpCovering.toPartitionOfUnity_eq_mul_prod i x t ht


open Classical in
theorem exists_finset_toSmoothPartitionOfUnity_eventuallyEq (i : ι) (x : M) :
    ∃ t : Finset ι,
      fs.toSmoothPartitionOfUnity i =ᶠ[𝓝 x]
        fs i * ∏ j ∈ t.filter fun j => WellOrderingRel j i, ((1 : M → ℝ) - fs j) := by
  -- Porting note: was defeq, now the continuous lemma uses bundled homs
  /-
    ι : Type uι
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : FiniteDimensional Real E
    s : Set M
    fs : SmoothBumpCovering ι I M s
    inst✝¹ : T2Space M
    inst✝ : SmoothManifoldWithCorners I M
    i : ι
    x : M
    ⊢ Exists fun t => (nhds x).EventuallyEq (⇑(fs.toSmoothPartitionOfUnity i)) (HM …
  -/
  simpa using fs.toBumpCovering.exists_finset_toPartitionOfUnity_eventuallyEq i x
  /-
    🎉 no goals
  -/


theorem toSmoothPartitionOfUnity_zero_of_zero {i : ι} {x : M} (h : fs i x = 0) :
    fs.toSmoothPartitionOfUnity i x = 0 :=
  fs.toBumpCovering.toPartitionOfUnity_zero_of_zero h


theorem support_toSmoothPartitionOfUnity_subset (i : ι) :
    support (fs.toSmoothPartitionOfUnity i) ⊆ support (fs i) :=
  fs.toBumpCovering.support_toPartitionOfUnity_subset i


theorem IsSubordinate.toSmoothPartitionOfUnity {f : SmoothBumpCovering ι I M s} {U : M → Set M}
    (h : f.IsSubordinate U) : f.toSmoothPartitionOfUnity.IsSubordinate fun i => U (f.c i) :=
  h.toBumpCovering.toPartitionOfUnity


theorem sum_toSmoothPartitionOfUnity_eq (x : M) :
    ∑ᶠ i, fs.toSmoothPartitionOfUnity i x = 1 - ∏ᶠ i, (1 - fs i x) :=
  fs.toBumpCovering.sum_toPartitionOfUnity_eq x


/-- Given two disjoint closed sets `s, t` in a Hausdorff σ-compact finite dimensional manifold,
there exists an infinitely smooth function that is equal to `0` on `s` and to `1` on `t`.
See also `exists_msmooth_zero_iff_one_iff_of_isClosed`, which ensures additionally that
`f` is equal to `0` exactly on `s` and to `1` exactly on `t`. -/
theorem exists_smooth_zero_one_of_isClosed [T2Space M] [SigmaCompactSpace M] {s t : Set M}
    (hs : IsClosed s) (ht : IsClosed t) (hd : Disjoint s t) :
    ∃ f : C^∞⟮I, M; 𝓘(ℝ), ℝ⟯, EqOn f 0 s ∧ EqOn f 1 t ∧ ∀ x, f x ∈ Icc 0 1 := by
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    ⊢ Exists fun f => And (Set.EqOn (⇑f) 0 s) (And (Set.EqOn (⇑f) 1 t) (∀ (x : M), …
  -/
  have : ∀ x ∈ t, sᶜ ∈ 𝓝 x := fun x hx => hs.isOpen_compl.mem_nhds (disjoint_right.1 hd hx)
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    this : ∀ (x : M), Membership.mem t x → Membership.mem (nhds x) (HasCompl.compl …
    ⊢ Exists fun f => And (Set.EqOn (⇑f) 0 s) (And (Set.EqOn (⇑f) 1 t) (∀ (x : M), …
  -/
  rcases SmoothBumpCovering.exists_isSubordinate I ht this with ⟨ι, f, hf⟩
  /-
    case intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    this : ∀ (x : M), Membership.mem t x → Membership.mem (nhds x) (HasCompl.compl …
    ι : Type uM
    f : SmoothBumpCovering ι I M t
    hf : f.IsSubordinate fun x => HasCompl.compl s
    ⊢ Exists fun f => And (Set.EqOn (⇑f) 0 s) (And (Set.EqOn (⇑f) 1 t) (∀ (x : M), …
  -/
  set g := f.toSmoothPartitionOfUnity
  refine
    ⟨⟨_, g.contMDiff_sum⟩, fun x hx => ?_, fun x => g.sum_eq_one, fun x =>
      ⟨g.sum_nonneg x, g.sum_le_one x⟩⟩
  /-
    case intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    this : ∀ (x : M), Membership.mem t x → Membership.mem (nhds x) (HasCompl.compl …
    ι : Type uM
    f : SmoothBumpCovering ι I M t
    hf : f.IsSubordinate fun x => HasCompl.compl s
    g : SmoothPartitionOfUnity ι I M t := f.toSmoothPartitionOfUnity
    x : M
    hx : Membership.mem s x
    ⊢ Eq (⟨fun x => finsum fun i => (g i) x, ⋯⟩ x) (0 x)
  -/
  suffices ∀ i, g i x = 0 by simp only [this, ContMDiffMap.coeFn_mk, finsum_zero, Pi.zero_apply]
  /-
    case intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    this : ∀ (x : M), Membership.mem t x → Membership.mem (nhds x) (HasCompl.compl …
    ι : Type uM
    f : SmoothBumpCovering ι I M t
    hf : f.IsSubordinate fun x => HasCompl.compl s
    g : SmoothPartitionOfUnity ι I M t := f.toSmoothPartitionOfUnity
    x : M
    hx : Membership.mem s x
    ⊢ ∀ (i : ι), Eq ((g i) x) 0
  -/
  refine fun i => f.toSmoothPartitionOfUnity_zero_of_zero ?_
  /-
    case intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    this : ∀ (x : M), Membership.mem t x → Membership.mem (nhds x) (HasCompl.compl …
    ι : Type uM
    f : SmoothBumpCovering ι I M t
    hf : f.IsSubordinate fun x => HasCompl.compl s
    g : SmoothPartitionOfUnity ι I M t := f.toSmoothPartitionOfUnity
    x : M
    hx : Membership.mem s x
    i : ι
    ⊢ Eq (↑(f.toFun i) x) 0
  -/
  exact nmem_support.1 (subset_compl_comm.1 (hf.support_subset i) hx)
  /-
    🎉 no goals
  -/


/-- Given two disjoint closed sets `s, t` in a Hausdorff normal σ-compact finite dimensional
manifold `M`, there exists a smooth function `f : M → [0,1]` that vanishes in a neighbourhood of `s`
and is equal to `1` in a neighbourhood of `t`. -/
theorem exists_smooth_zero_one_nhds_of_isClosed [T2Space M] [NormalSpace M] [SigmaCompactSpace M]
    {s t : Set M} (hs : IsClosed s) (ht : IsClosed t) (hd : Disjoint s t) :
    ∃ f : C^∞⟮I, M; 𝓘(ℝ), ℝ⟯, (∀ᶠ x in 𝓝ˢ s, f x = 0) ∧ (∀ᶠ x in 𝓝ˢ t, f x = 1) ∧
      ∀ x, f x ∈ Icc 0 1 := by
  obtain ⟨u, u_op, hsu, hut⟩ := normal_exists_closure_subset hs ht.isOpen_compl
    (subset_compl_iff_disjoint_left.mpr hd.symm)
  obtain ⟨v, v_op, htv, hvu⟩ := normal_exists_closure_subset ht isClosed_closure.isOpen_compl
    (subset_compl_comm.mp hut)
  obtain ⟨f, hfu, hfv, hf⟩ := exists_smooth_zero_one_of_isClosed I isClosed_closure isClosed_closure
    (subset_compl_iff_disjoint_left.mp hvu)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    E : Type uE
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    H : Type uH
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : T2Space M
    inst✝¹ : NormalSpace M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    u : Set M
    u_op : IsOpen u
    hsu : HasSubset.Subset s u
    hut : HasSubset.Subset (closure u) (HasCompl.compl t)
    v : Set M
    v_op : IsOpen v
    htv : HasSubset.Subset t v
    hvu : HasSubset.Subset (closure v) (HasCompl.compl (closure u))
    f : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
    hfu : Set.EqOn (⇑f) 0 (closure u)
    hfv : Set.EqOn (⇑f) 1 (closure v)
    hf : ∀ (x : M), Membership.mem (Set.Icc 0 1) (f x)
    ⊢ Exists fun f => And (Filter.Eventually (fun x => Eq (f x) 0) (nhdsSet s)) (A …
  -/
  refine ⟨f, ?_, ?_, hf⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      E : Type uE
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Real E
      H : Type uH
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : SmoothManifoldWithCorners I M
      inst✝² : T2Space M
      inst✝¹ : NormalSpace M
      inst✝ : SigmaCompactSpace M
      s t : Set M
      hs : IsClosed s
      ht : IsClosed t
      hd : Disjoint s t
      u : Set M
      u_op : IsOpen u
      hsu : HasSubset.Subset s u
      hut : HasSubset.Subset (closure u) (HasCompl.compl t)
      v : Set M
      v_op : IsOpen v
      htv : HasSubset.Subset t v
      hvu : HasSubset.Subset (closure v) (HasCompl.compl (closure u))
      f : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
      hfu : Set.EqOn (⇑f) 0 (closure u)
      hfv : Set.EqOn (⇑f) 1 (closure v)
      hf : ∀ (x : M), Membership.mem (Set.Icc 0 1) (f x)
      ⊢ Filter.Eventually (fun x => Eq (f x) 0) (nhdsSet s)
    -/
  · exact eventually_of_mem (mem_of_superset (u_op.mem_nhdsSet.mpr hsu) subset_closure) hfu
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      E : Type uE
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Real E
      H : Type uH
      inst✝⁷ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁶ : TopologicalSpace M
      inst✝⁵ : ChartedSpace H M
      inst✝⁴ : FiniteDimensional Real E
      inst✝³ : SmoothManifoldWithCorners I M
      inst✝² : T2Space M
      inst✝¹ : NormalSpace M
      inst✝ : SigmaCompactSpace M
      s t : Set M
      hs : IsClosed s
      ht : IsClosed t
      hd : Disjoint s t
      u : Set M
      u_op : IsOpen u
      hsu : HasSubset.Subset s u
      hut : HasSubset.Subset (closure u) (HasCompl.compl t)
      v : Set M
      v_op : IsOpen v
      htv : HasSubset.Subset t v
      hvu : HasSubset.Subset (closure v) (HasCompl.compl (closure u))
      f : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
      hfu : Set.EqOn (⇑f) 0 (closure u)
      hfv : Set.EqOn (⇑f) 1 (closure v)
      hf : ∀ (x : M), Membership.mem (Set.Icc 0 1) (f x)
      ⊢ Filter.Eventually (fun x => Eq (f x) 1) (nhdsSet t)
    -/
  · exact eventually_of_mem (mem_of_superset (v_op.mem_nhdsSet.mpr htv) subset_closure) hfv
    /-
      🎉 no goals
    -/


/-- Given two sets `s, t` in a Hausdorff normal σ-compact finite-dimensional manifold `M`
with `s` open and `s ⊆ interior t`, there is a smooth function `f : M → [0,1]` which is equal to `s`
in a neighbourhood of `s` and has support contained in `t`. -/
theorem exists_smooth_one_nhds_of_subset_interior [T2Space M] [NormalSpace M] [SigmaCompactSpace M]
    {s t : Set M} (hs : IsClosed s) (hd : s ⊆ interior t) :
    ∃ f : C^∞⟮I, M; 𝓘(ℝ), ℝ⟯, (∀ᶠ x in 𝓝ˢ s, f x = 1) ∧ (∀ x ∉ t, f x = 0) ∧
      ∀ x, f x ∈ Icc 0 1 := by
  rcases exists_smooth_zero_one_nhds_of_isClosed I isOpen_interior.isClosed_compl hs
    (by rwa [← subset_compl_iff_disjoint_left, compl_compl]) with ⟨f, h0, h1, hf⟩
  /-
    case intro.intro.intro
    E : Type uE
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    H : Type uH
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : T2Space M
    inst✝¹ : NormalSpace M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    hd : HasSubset.Subset s (interior t)
    f : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
    h0 : Filter.Eventually (fun x => Eq (f x) 0) (nhdsSet (HasCompl.compl (interio …
    h1 : Filter.Eventually (fun x => Eq (f x) 1) (nhdsSet s)
    hf : ∀ (x : M), Membership.mem (Set.Icc 0 1) (f x)
    ⊢ Exists fun f => And (Filter.Eventually (fun x => Eq (f x) 1) (nhdsSet s)) (A …
  -/
  refine ⟨f, h1, fun x hx ↦ ?_, hf⟩
  /-
    case intro.intro.intro
    E : Type uE
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Real E
    H : Type uH
    inst✝⁷ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁶ : TopologicalSpace M
    inst✝⁵ : ChartedSpace H M
    inst✝⁴ : FiniteDimensional Real E
    inst✝³ : SmoothManifoldWithCorners I M
    inst✝² : T2Space M
    inst✝¹ : NormalSpace M
    inst✝ : SigmaCompactSpace M
    s t : Set M
    hs : IsClosed s
    hd : HasSubset.Subset s (interior t)
    f : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
    h0 : Filter.Eventually (fun x => Eq (f x) 0) (nhdsSet (HasCompl.compl (interio …
    h1 : Filter.Eventually (fun x => Eq (f x) 1) (nhdsSet s)
    hf : ∀ (x : M), Membership.mem (Set.Icc 0 1) (f x)
    x : M
    hx : Not (Membership.mem t x)
    ⊢ Eq (f x) 0
  -/
  exact h0.self_of_nhdsSet _ fun hx' ↦ hx <| interior_subset hx'
  /-
    🎉 no goals
  -/


/-- A `SmoothPartitionOfUnity` that consists of a single function, uniformly equal to one,
defined as an example for `Inhabited` instance. -/
def single (i : ι) (s : Set M) : SmoothPartitionOfUnity ι I M s :=
  (BumpCovering.single i s).toSmoothPartitionOfUnity fun j => by
    classical
    rcases eq_or_ne j i with (rfl | h)
    · simp only [contMDiff_one, ContinuousMap.coe_one, BumpCovering.coe_single, Pi.single_eq_same]
    · simp only [contMDiff_zero, BumpCovering.coe_single, Pi.single_eq_of_ne h,
        ContinuousMap.coe_zero]


instance [Inhabited ι] (s : Set M) : Inhabited (SmoothPartitionOfUnity ι I M s) :=
  ⟨single I default s⟩


/-- If `X` is a paracompact normal topological space and `U` is an open covering of a closed set
`s`, then there exists a `SmoothPartitionOfUnity ι M s` that is subordinate to `U`. -/
theorem exists_isSubordinate {s : Set M} (hs : IsClosed s) (U : ι → Set M) (ho : ∀ i, IsOpen (U i))
    (hU : s ⊆ ⋃ i, U i) : ∃ f : SmoothPartitionOfUnity ι I M s, f.IsSubordinate U := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s : Set M
    hs : IsClosed s
    U : ι → Set M
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    ⊢ Exists fun f => f.IsSubordinate U
  -/
  haveI : LocallyCompactSpace H := I.locallyCompactSpace
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s : Set M
    hs : IsClosed s
    U : ι → Set M
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    this : LocallyCompactSpace H
    ⊢ Exists fun f => f.IsSubordinate U
  -/
  haveI : LocallyCompactSpace M := ChartedSpace.locallyCompactSpace H M
  -- porting note(https://github.com/leanprover/std4/issues/116):
  -- split `rcases` into `have` + `rcases`
  /-
    ι : Type uι
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s : Set M
    hs : IsClosed s
    U : ι → Set M
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    this✝ : LocallyCompactSpace H
    this : LocallyCompactSpace M
    ⊢ Exists fun f => f.IsSubordinate U
  -/
  have := BumpCovering.exists_isSubordinate_of_prop (ContMDiff I 𝓘(ℝ) ⊤) ?_ hs U ho hU
    /-
      case refine_2
      ι : Type uι
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      s : Set M
      hs : IsClosed s
      U : ι → Set M
      ho : ∀ (i : ι), IsOpen (U i)
      hU : HasSubset.Subset s (Set.iUnion fun i => U i)
      this✝¹ : LocallyCompactSpace H
      this✝ : LocallyCompactSpace M
      this : Exists fun f => And (∀ (i : ι), ContMDiff I (modelWithCornersSelf Real  …
      ⊢ Exists fun f => f.IsSubordinate U
    -/
  · rcases this with ⟨f, hf, hfU⟩
    /-
      case refine_2.intro.intro
      ι : Type uι
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      s : Set M
      hs : IsClosed s
      U : ι → Set M
      ho : ∀ (i : ι), IsOpen (U i)
      hU : HasSubset.Subset s (Set.iUnion fun i => U i)
      this✝ : LocallyCompactSpace H
      this : LocallyCompactSpace M
      f : BumpCovering ι M s
      hf : ∀ (i : ι), ContMDiff I (modelWithCornersSelf Real Real) Top.top ⇑(f i)
      hfU : f.IsSubordinate U
      ⊢ Exists fun f => f.IsSubordinate U
    -/
    exact ⟨f.toSmoothPartitionOfUnity hf, hfU.toSmoothPartitionOfUnity hf⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      ι : Type uι
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      s : Set M
      hs : IsClosed s
      U : ι → Set M
      ho : ∀ (i : ι), IsOpen (U i)
      hU : HasSubset.Subset s (Set.iUnion fun i => U i)
      this✝ : LocallyCompactSpace H
      this : LocallyCompactSpace M
      ⊢ ∀ (s t : Set M), IsClosed s → IsClosed t → Disjoint s t → Exists fun f => An …
    -/
  · intro s t hs ht hd
    /-
      case refine_1
      ι : Type uι
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      s✝ : Set M
      hs✝ : IsClosed s✝
      U : ι → Set M
      ho : ∀ (i : ι), IsOpen (U i)
      hU : HasSubset.Subset s✝ (Set.iUnion fun i => U i)
      this✝ : LocallyCompactSpace H
      this : LocallyCompactSpace M
      s t : Set M
      hs : IsClosed s
      ht : IsClosed t
      hd : Disjoint s t
      ⊢ Exists fun f => And (ContMDiff I (modelWithCornersSelf Real Real) Top.top ⇑f …
    -/
    rcases exists_smooth_zero_one_of_isClosed I hs ht hd with ⟨f, hf⟩
    /-
      case refine_1.intro
      ι : Type uι
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      inst✝ : SigmaCompactSpace M
      s✝ : Set M
      hs✝ : IsClosed s✝
      U : ι → Set M
      ho : ∀ (i : ι), IsOpen (U i)
      hU : HasSubset.Subset s✝ (Set.iUnion fun i => U i)
      this✝ : LocallyCompactSpace H
      this : LocallyCompactSpace M
      s t : Set M
      hs : IsClosed s
      ht : IsClosed t
      hd : Disjoint s t
      f : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
      hf : And (Set.EqOn (⇑f) 0 s) (And (Set.EqOn (⇑f) 1 t) (∀ (x : M), Membership.m …
      ⊢ Exists fun f => And (ContMDiff I (modelWithCornersSelf Real Real) Top.top ⇑f …
    -/
    exact ⟨f, f.contMDiff, hf⟩
    /-
      🎉 no goals
    -/


theorem exists_isSubordinate_chartAt_source_of_isClosed {s : Set M} (hs : IsClosed s) :
    ∃ f : SmoothPartitionOfUnity s I M s,
      f.IsSubordinate (fun x ↦ (chartAt H (x : M)).source) := by
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s : Set M
    hs : IsClosed s
    ⊢ Exists fun f => f.IsSubordinate fun x => (chartAt H ↑x).source
  -/
  apply exists_isSubordinate _ hs _ (fun i ↦ (chartAt H _).open_source) (fun x hx ↦ ?_)
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    s : Set M
    hs : IsClosed s
    x : M
    hx : Membership.mem s x
    ⊢ Membership.mem (Set.iUnion fun i => (chartAt H ↑i).source) x
  -/
  exact mem_iUnion_of_mem ⟨x, hx⟩ (mem_chart_source H x)
  /-
    🎉 no goals
  -/


theorem exists_isSubordinate_chartAt_source :
    ∃ f : SmoothPartitionOfUnity M I M univ, f.IsSubordinate (fun x ↦ (chartAt H x).source) := by
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    ⊢ Exists fun f => f.IsSubordinate fun x => (chartAt H x).source
  -/
  apply exists_isSubordinate _ isClosed_univ _ (fun i ↦ (chartAt H _).open_source) (fun x _ ↦ ?_)
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : SigmaCompactSpace M
    x : M
    x✝ : Membership.mem Set.univ x
    ⊢ Membership.mem (Set.iUnion fun i => (chartAt H i).source) x
  -/
  exact mem_iUnion_of_mem x (mem_chart_source H x)
  /-
    🎉 no goals
  -/


/-- Let `M` be a σ-compact Hausdorff finite dimensional topological manifold. Let `t : M → Set F`
be a family of convex sets. Suppose that for each point `x : M` there exists a neighborhood
`U ∈ 𝓝 x` and a function `g : M → F` such that `g` is $C^n$ smooth on `U` and `g y ∈ t y` for all
`y ∈ U`. Then there exists a $C^n$ smooth function `g : C^∞⟮I, M; 𝓘(ℝ, F), F⟯` such that `g x ∈ t x`
for all `x`. See also `exists_smooth_forall_mem_convex_of_local` and
`exists_smooth_forall_mem_convex_of_local_const`. -/
theorem exists_contMDiffOn_forall_mem_convex_of_local (ht : ∀ x, Convex ℝ (t x))
    (Hloc : ∀ x : M, ∃ U ∈ 𝓝 x, ∃ g : M → F, ContMDiffOn I 𝓘(ℝ, F) n g U ∧ ∀ y ∈ U, g y ∈ t y) :
    ∃ g : C^n⟮I, M; 𝓘(ℝ, F), F⟯, ∀ x, g x ∈ t x := by
  /-
    E : Type uE
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace Real E
    F : Type uF
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace Real F
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    t : M → Set F
    n : ENat
    ht : ∀ (x : M), Convex Real (t x)
    Hloc : ∀ (x : M), Exists fun U => And (Membership.mem (nhds x) U) (Exists fun  …
    ⊢ Exists fun g => ∀ (x : M), Membership.mem (t x) (g x)
  -/
  choose U hU g hgs hgt using Hloc
  obtain ⟨f, hf⟩ :=
    SmoothPartitionOfUnity.exists_isSubordinate I isClosed_univ (fun x => interior (U x))
      (fun x => isOpen_interior) fun x _ => mem_iUnion.2 ⟨x, mem_interior_iff_mem_nhds.2 (hU x)⟩
  refine ⟨⟨fun x => ∑ᶠ i, f i x • g i x,
      hf.contMDiff_finsum_smul (fun i => isOpen_interior) fun i => (hgs i).mono interior_subset⟩,
    fun x => f.finsum_smul_mem_convex (mem_univ x) (fun i hi => hgt _ _ ?_) (ht _)⟩
  /-
    case intro
    E : Type uE
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : NormedSpace Real E
    F : Type uF
    inst✝⁸ : NormedAddCommGroup F
    inst✝⁷ : NormedSpace Real F
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    t : M → Set F
    n : ENat
    ht : ∀ (x : M), Convex Real (t x)
    U : M → Set M
    hU : ∀ (x : M), Membership.mem (nhds x) (U x)
    g : M → M → F
    hgs : ∀ (x : M), ContMDiffOn I (modelWithCornersSelf Real F) n (g x) (U x)
    hgt : ∀ (x y : M), Membership.mem (U x) y → Membership.mem (t y) (g x y)
    f : SmoothPartitionOfUnity M I M
    hf : f.IsSubordinate fun x => interior (U x)
    x i : M
    hi : Ne ((f i) x) 0
    ⊢ Membership.mem (U i) x
  -/
  exact interior_subset (hf _ <| subset_closure hi)
  /-
    🎉 no goals
  -/


/-- Let `M` be a σ-compact Hausdorff finite dimensional topological manifold. Let `t : M → Set F`
be a family of convex sets. Suppose that for each point `x : M` there exists a neighborhood
`U ∈ 𝓝 x` and a function `g : M → F` such that `g` is smooth on `U` and `g y ∈ t y` for all `y ∈ U`.
Then there exists a smooth function `g : C^∞⟮I, M; 𝓘(ℝ, F), F⟯` such that `g x ∈ t x` for all `x`.
See also `exists_contMDiffOn_forall_mem_convex_of_local` and
`exists_smooth_forall_mem_convex_of_local_const`. -/
theorem exists_smooth_forall_mem_convex_of_local (ht : ∀ x, Convex ℝ (t x))
    (Hloc : ∀ x : M, ∃ U ∈ 𝓝 x, ∃ g : M → F, ContMDiffOn I 𝓘(ℝ, F) ⊤ g U ∧ ∀ y ∈ U, g y ∈ t y) :
    ∃ g : C^∞⟮I, M; 𝓘(ℝ, F), F⟯, ∀ x, g x ∈ t x :=
  exists_contMDiffOn_forall_mem_convex_of_local I ht Hloc


/-- Let `M` be a σ-compact Hausdorff finite dimensional topological manifold. Let `t : M → Set F` be
a family of convex sets. Suppose that for each point `x : M` there exists a vector `c : F` such that
for all `y` in a neighborhood of `x` we have `c ∈ t y`. Then there exists a smooth function
`g : C^∞⟮I, M; 𝓘(ℝ, F), F⟯` such that `g x ∈ t x` for all `x`.  See also
`exists_contMDiffOn_forall_mem_convex_of_local` and `exists_smooth_forall_mem_convex_of_local`. -/
theorem exists_smooth_forall_mem_convex_of_local_const (ht : ∀ x, Convex ℝ (t x))
    (Hloc : ∀ x : M, ∃ c : F, ∀ᶠ y in 𝓝 x, c ∈ t y) : ∃ g : C^∞⟮I, M; 𝓘(ℝ, F), F⟯, ∀ x, g x ∈ t x :=
  exists_smooth_forall_mem_convex_of_local I ht fun x =>
    let ⟨c, hc⟩ := Hloc x
    ⟨_, hc, fun _ => c, contMDiffOn_const, fun _ => id⟩


/-- Let `M` be a smooth σ-compact manifold with extended distance. Let `K : ι → Set M` be a locally
finite family of closed sets, let `U : ι → Set M` be a family of open sets such that `K i ⊆ U i` for
all `i`. Then there exists a positive smooth function `δ : M → ℝ≥0` such that for any `i` and
`x ∈ K i`, we have `EMetric.closedBall x (δ x) ⊆ U i`. -/
theorem Emetric.exists_smooth_forall_closedBall_subset {M} [EMetricSpace M] [ChartedSpace H M]
    [SmoothManifoldWithCorners I M] [SigmaCompactSpace M] {K : ι → Set M} {U : ι → Set M}
    (hK : ∀ i, IsClosed (K i)) (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i)
    (hfin : LocallyFinite K) :
    ∃ δ : C^∞⟮I, M; 𝓘(ℝ, ℝ), ℝ⟯,
      (∀ x, 0 < δ x) ∧ ∀ (i), ∀ x ∈ K i, EMetric.closedBall x (ENNReal.ofReal (δ x)) ⊆ U i := by
  simpa only [mem_inter_iff, forall_and, mem_preimage, mem_iInter, @forall_swap ι M]
    using exists_smooth_forall_mem_convex_of_local_const I
      EMetric.exists_forall_closedBall_subset_aux₂
      (EMetric.exists_forall_closedBall_subset_aux₁ hK hU hKU hfin)


/-- Let `M` be a smooth σ-compact manifold with a metric. Let `K : ι → Set M` be a locally finite
family of closed sets, let `U : ι → Set M` be a family of open sets such that `K i ⊆ U i` for all
`i`. Then there exists a positive smooth function `δ : M → ℝ≥0` such that for any `i` and `x ∈ K i`,
we have `Metric.closedBall x (δ x) ⊆ U i`. -/
theorem Metric.exists_smooth_forall_closedBall_subset {M} [MetricSpace M] [ChartedSpace H M]
    [SmoothManifoldWithCorners I M] [SigmaCompactSpace M] {K : ι → Set M} {U : ι → Set M}
    (hK : ∀ i, IsClosed (K i)) (hU : ∀ i, IsOpen (U i)) (hKU : ∀ i, K i ⊆ U i)
    (hfin : LocallyFinite K) :
    ∃ δ : C^∞⟮I, M; 𝓘(ℝ, ℝ), ℝ⟯,
      (∀ x, 0 < δ x) ∧ ∀ (i), ∀ x ∈ K i, Metric.closedBall x (δ x) ⊆ U i := by
  /-
    ι : Type uι
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    inst✝⁴ : FiniteDimensional Real E
    M : Type u_1
    inst✝³ : MetricSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SigmaCompactSpace M
    K U : ι → Set M
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    ⊢ Exists fun δ => And (∀ (x : M), LT.lt 0 (δ x)) (∀ (i : ι) (x : M), Membershi …
  -/
  rcases Emetric.exists_smooth_forall_closedBall_subset I hK hU hKU hfin with ⟨δ, hδ0, hδ⟩
  /-
    case intro.intro
    ι : Type uι
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    inst✝⁴ : FiniteDimensional Real E
    M : Type u_1
    inst✝³ : MetricSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SigmaCompactSpace M
    K U : ι → Set M
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
    hδ0 : ∀ (x : M), LT.lt 0 (δ x)
    hδ : ∀ (i : ι) (x : M), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    ⊢ Exists fun δ => And (∀ (x : M), LT.lt 0 (δ x)) (∀ (i : ι) (x : M), Membershi …
  -/
  refine ⟨δ, hδ0, fun i x hx => ?_⟩
  /-
    case intro.intro
    ι : Type uι
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    inst✝⁴ : FiniteDimensional Real E
    M : Type u_1
    inst✝³ : MetricSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SigmaCompactSpace M
    K U : ι → Set M
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
    hδ0 : ∀ (x : M), LT.lt 0 (δ x)
    hδ : ∀ (i : ι) (x : M), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    i : ι
    x : M
    hx : Membership.mem (K i) x
    ⊢ HasSubset.Subset (Metric.closedBall x (δ x)) (U i)
  -/
  rw [← Metric.emetric_closedBall (hδ0 _).le]
  /-
    case intro.intro
    ι : Type uι
    E : Type uE
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type uH
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    inst✝⁴ : FiniteDimensional Real E
    M : Type u_1
    inst✝³ : MetricSpace M
    inst✝² : ChartedSpace H M
    inst✝¹ : SmoothManifoldWithCorners I M
    inst✝ : SigmaCompactSpace M
    K U : ι → Set M
    hK : ∀ (i : ι), IsClosed (K i)
    hU : ∀ (i : ι), IsOpen (U i)
    hKU : ∀ (i : ι), HasSubset.Subset (K i) (U i)
    hfin : LocallyFinite K
    δ : ContMDiffMap I (modelWithCornersSelf Real Real) M Real Top.top
    hδ0 : ∀ (x : M), LT.lt 0 (δ x)
    hδ : ∀ (i : ι) (x : M), Membership.mem (K i) x → HasSubset.Subset (EMetric.clo …
    i : ι
    x : M
    hx : Membership.mem (K i) x
    ⊢ HasSubset.Subset (EMetric.closedBall x (ENNReal.ofReal (δ x))) (U i)
  -/
  exact hδ i x hx
  /-
    🎉 no goals
  -/


lemma IsOpen.exists_msmooth_support_eq_aux {s : Set H} (hs : IsOpen s) :
    ∃ f : H → ℝ, f.support = s ∧ ContMDiff I 𝓘(ℝ) ⊤ f ∧ Set.range f ⊆ Set.Icc 0 1 := by
  /-
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type uH
    inst✝¹ : TopologicalSpace H
    I : ModelWithCorners Real E H
    inst✝ : FiniteDimensional Real E
    s : Set H
    hs : IsOpen s
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContMDiff I (modelWith …
  -/
  have h's : IsOpen (I.symm ⁻¹' s) := I.continuous_symm.isOpen_preimage _ hs
  /-
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type uH
    inst✝¹ : TopologicalSpace H
    I : ModelWithCorners Real E H
    inst✝ : FiniteDimensional Real E
    s : Set H
    hs : IsOpen s
    h's : IsOpen (Set.preimage (↑I.symm) s)
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContMDiff I (modelWith …
  -/
  rcases h's.exists_smooth_support_eq with ⟨f, f_supp, f_diff, f_range⟩
  /-
    case intro.intro.intro
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    H : Type uH
    inst✝¹ : TopologicalSpace H
    I : ModelWithCorners Real E H
    inst✝ : FiniteDimensional Real E
    s : Set H
    hs : IsOpen s
    h's : IsOpen (Set.preimage (↑I.symm) s)
    f : E → Real
    f_supp : Eq (Function.support f) (Set.preimage (↑I.symm) s)
    f_diff : ContDiff Real (↑Top.top) f
    f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContMDiff I (modelWith …
  -/
  refine ⟨f ∘ I, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      H : Type uH
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners Real E H
      inst✝ : FiniteDimensional Real E
      s : Set H
      hs : IsOpen s
      h's : IsOpen (Set.preimage (↑I.symm) s)
      f : E → Real
      f_supp : Eq (Function.support f) (Set.preimage (↑I.symm) s)
      f_diff : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      ⊢ Eq (Function.support (Function.comp f ↑I)) s
    -/
  · rw [support_comp_eq_preimage, f_supp, ← preimage_comp]
    /-
      case intro.intro.intro.refine_1
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      H : Type uH
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners Real E H
      inst✝ : FiniteDimensional Real E
      s : Set H
      hs : IsOpen s
      h's : IsOpen (Set.preimage (↑I.symm) s)
      f : E → Real
      f_supp : Eq (Function.support f) (Set.preimage (↑I.symm) s)
      f_diff : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      ⊢ Eq (Set.preimage (Function.comp ↑I.symm ↑I) s) s
    -/
    simp only [ModelWithCorners.symm_comp_self, preimage_id_eq, id_eq]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      H : Type uH
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners Real E H
      inst✝ : FiniteDimensional Real E
      s : Set H
      hs : IsOpen s
      h's : IsOpen (Set.preimage (↑I.symm) s)
      f : E → Real
      f_supp : Eq (Function.support f) (Set.preimage (↑I.symm) s)
      f_diff : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      ⊢ ContMDiff I (modelWithCornersSelf Real Real) Top.top (Function.comp f ↑I)
    -/
  · exact f_diff.comp_contMDiff contMDiff_model
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_3
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      H : Type uH
      inst✝¹ : TopologicalSpace H
      I : ModelWithCorners Real E H
      inst✝ : FiniteDimensional Real E
      s : Set H
      hs : IsOpen s
      h's : IsOpen (Set.preimage (↑I.symm) s)
      f : E → Real
      f_supp : Eq (Function.support f) (Set.preimage (↑I.symm) s)
      f_diff : ContDiff Real (↑Top.top) f
      f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
      ⊢ HasSubset.Subset (Set.range (Function.comp f ↑I)) (Set.Icc 0 1)
    -/
  · exact Subset.trans (range_comp_subset_range _ _) f_range
    /-
      🎉 no goals
    -/


/-- Given an open set in a finite-dimensional real manifold, there exists a nonnegative smooth
function with support equal to `s`. -/
theorem IsOpen.exists_msmooth_support_eq {s : Set M} (hs : IsOpen s) :
    ∃ f : M → ℝ, f.support = s ∧ ContMDiff I 𝓘(ℝ) ⊤ f ∧ ∀ x, 0 ≤ f x := by
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s : Set M
    hs : IsOpen s
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContMDiff I (modelWith …
  -/
  rcases SmoothPartitionOfUnity.exists_isSubordinate_chartAt_source I M with ⟨f, hf⟩
  have A : ∀ (c : M), ∃ g : H → ℝ,
      g.support = (chartAt H c).target ∩ (chartAt H c).symm ⁻¹' s ∧
      ContMDiff I 𝓘(ℝ) ⊤ g ∧ Set.range g ⊆ Set.Icc 0 1 := by
    intro i
    apply IsOpen.exists_msmooth_support_eq_aux
    exact PartialHomeomorph.isOpen_inter_preimage_symm _ hs
  /-
    case intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s : Set M
    hs : IsOpen s
    f : SmoothPartitionOfUnity M I M
    hf : f.IsSubordinate fun x => (chartAt H x).source
    A : ∀ (c : M), Exists fun g => And (Eq (Function.support g) (Inter.inter (char …
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContMDiff I (modelWith …
  -/
  choose g g_supp g_diff hg using A
  /-
    case intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s : Set M
    hs : IsOpen s
    f : SmoothPartitionOfUnity M I M
    hf : f.IsSubordinate fun x => (chartAt H x).source
    g : M → H → Real
    g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
    g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
    hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContMDiff I (modelWith …
  -/
  have h'g : ∀ c x, 0 ≤ g c x := fun c x ↦ (hg c (mem_range_self (f := g c) x)).1
  have h''g : ∀ c x, 0 ≤ f c x * g c (chartAt H c x) :=
    fun c x ↦ mul_nonneg (f.nonneg c x) (h'g c _)
  /-
    case intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s : Set M
    hs : IsOpen s
    f : SmoothPartitionOfUnity M I M
    hf : f.IsSubordinate fun x => (chartAt H x).source
    g : M → H → Real
    g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
    g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
    hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
    h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
    h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
    ⊢ Exists fun f => And (Eq (Function.support f) s) (And (ContMDiff I (modelWith …
  -/
  refine ⟨fun x ↦ ∑ᶠ c, f c x * g c (chartAt H c x), ?_, ?_, ?_⟩
    /-
      case intro.refine_1
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s : Set M
      hs : IsOpen s
      f : SmoothPartitionOfUnity M I M
      hf : f.IsSubordinate fun x => (chartAt H x).source
      g : M → H → Real
      g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
      g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
      hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
      h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
      h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
      ⊢ Eq (Function.support fun x => finsum fun c => HMul.hMul ((f c) x) (g c (↑(ch …
    -/
  · refine support_eq_iff.2 ⟨fun x hx ↦ ?_, fun x hx ↦ ?_⟩
      /-
        case intro.refine_1.refine_1
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Membership.mem s x
        ⊢ Ne (finsum fun c => HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))) 0
      -/
    · apply ne_of_gt
      have B : ∃ c, 0 < f c x * g c (chartAt H c x) := by
        obtain ⟨c, hc⟩ : ∃ c, 0 < f c x := f.exists_pos_of_mem (mem_univ x)
        refine ⟨c, mul_pos hc ?_⟩
        apply lt_of_le_of_ne (h'g _ _) (Ne.symm _)
        rw [← mem_support, g_supp, ← mem_preimage, preimage_inter]
        have Hx : x ∈ tsupport (f c) := subset_tsupport _ (ne_of_gt hc)
        simp [(chartAt H c).left_inv (hf c Hx), hx, (chartAt H c).map_source (hf c Hx)]
      /-
        case intro.refine_1.refine_1.h
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Membership.mem s x
        B : Exists fun c => LT.lt 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        ⊢ LT.lt 0 (finsum fun c => HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
      -/
      apply finsum_pos' (fun c ↦ h''g c x) B
      /-
        case intro.refine_1.refine_1.h
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Membership.mem s x
        B : Exists fun c => LT.lt 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        ⊢ (Function.support fun c => HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))).Fin …
      -/
      apply (f.locallyFinite.point_finite x).subset
      /-
        case intro.refine_1.refine_1.h
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Membership.mem s x
        B : Exists fun c => LT.lt 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        ⊢ HasSubset.Subset (Function.support fun c => HMul.hMul ((f c) x) (g c (↑(char …
      -/
      apply compl_subset_compl.2
      /-
        case intro.refine_1.refine_1.h
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Membership.mem s x
        B : Exists fun c => LT.lt 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        ⊢ HasSubset.Subset (fun x_1 => Eq ((f x_1) x) 0) fun x_1 => Eq ((fun c => HMul …
      -/
      rintro c (hc : f c x = 0)
      /-
        case intro.refine_1.refine_1.h
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Membership.mem s x
        B : Exists fun c => LT.lt 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        c : M
        hc : Eq ((f c) x) 0
        ⊢ Membership.mem (fun x_1 => Eq ((fun c => HMul.hMul ((f c) x) (g c (↑(chartAt …
      -/
      simpa only [mul_eq_zero] using Or.inl hc
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_1.refine_2
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Not (Membership.mem s x)
        ⊢ Eq (finsum fun c => HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))) 0
      -/
    · apply finsum_eq_zero_of_forall_eq_zero
      /-
        case intro.refine_1.refine_2.h
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Not (Membership.mem s x)
        ⊢ ∀ (x_1 : M), Eq (HMul.hMul ((f x_1) x) (g x_1 (↑(chartAt H x_1) x))) 0
      -/
      intro c
      /-
        case intro.refine_1.refine_2.h
        E : Type uE
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Real E
        H : Type uH
        inst✝⁶ : TopologicalSpace H
        I : ModelWithCorners Real E H
        M : Type uM
        inst✝⁵ : TopologicalSpace M
        inst✝⁴ : ChartedSpace H M
        inst✝³ : FiniteDimensional Real E
        inst✝² : SmoothManifoldWithCorners I M
        inst✝¹ : SigmaCompactSpace M
        inst✝ : T2Space M
        s : Set M
        hs : IsOpen s
        f : SmoothPartitionOfUnity M I M
        hf : f.IsSubordinate fun x => (chartAt H x).source
        g : M → H → Real
        g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
        g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
        hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
        h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
        h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
        x : M
        hx : Not (Membership.mem s x)
        c : M
        ⊢ Eq (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))) 0
      -/
      by_cases Hx : x ∈ tsupport (f c)
        /-
          case pos
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x : M
          hx : Not (Membership.mem s x)
          c : M
          Hx : Membership.mem (tsupport ⇑(f c)) x
          ⊢ Eq (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))) 0
        -/
      · suffices g c (chartAt H c x) = 0 by simp only [this, mul_zero]
        /-
          case pos
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x : M
          hx : Not (Membership.mem s x)
          c : M
          Hx : Membership.mem (tsupport ⇑(f c)) x
          ⊢ Eq (g c (↑(chartAt H c) x)) 0
        -/
        rw [← nmem_support, g_supp, ← mem_preimage, preimage_inter]
        /-
          case pos
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x : M
          hx : Not (Membership.mem s x)
          c : M
          Hx : Membership.mem (tsupport ⇑(f c)) x
          ⊢ Not (Membership.mem (Inter.inter (Set.preimage (↑(chartAt H c)) (chartAt H c …
        -/
        contrapose! hx
        /-
          case pos
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x c : M
          Hx : Membership.mem (tsupport ⇑(f c)) x
          hx : Membership.mem (Inter.inter (Set.preimage (↑(chartAt H c)) (chartAt H c). …
          ⊢ Membership.mem s x
        -/
        simp only [mem_inter_iff, mem_preimage, (chartAt H c).left_inv (hf c Hx)] at hx
        /-
          case pos
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x c : M
          Hx : Membership.mem (tsupport ⇑(f c)) x
          hx : And (Membership.mem (chartAt H c).target (↑(chartAt H c) x)) (Membership. …
          ⊢ Membership.mem s x
        -/
        exact hx.2
        /-
          🎉 no goals
        -/
        /-
          case neg
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x : M
          hx : Not (Membership.mem s x)
          c : M
          Hx : Not (Membership.mem (tsupport ⇑(f c)) x)
          ⊢ Eq (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))) 0
        -/
      · have : x ∉ support (f c) := by contrapose! Hx; exact subset_tsupport _ Hx
        /-
          case neg
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x : M
          hx : Not (Membership.mem s x)
          c : M
          Hx : Not (Membership.mem (tsupport ⇑(f c)) x)
          this : Not (Membership.mem (Function.support ⇑(f c)) x)
          ⊢ Eq (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))) 0
        -/
        rw [nmem_support] at this
        /-
          case neg
          E : Type uE
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedSpace Real E
          H : Type uH
          inst✝⁶ : TopologicalSpace H
          I : ModelWithCorners Real E H
          M : Type uM
          inst✝⁵ : TopologicalSpace M
          inst✝⁴ : ChartedSpace H M
          inst✝³ : FiniteDimensional Real E
          inst✝² : SmoothManifoldWithCorners I M
          inst✝¹ : SigmaCompactSpace M
          inst✝ : T2Space M
          s : Set M
          hs : IsOpen s
          f : SmoothPartitionOfUnity M I M
          hf : f.IsSubordinate fun x => (chartAt H x).source
          g : M → H → Real
          g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
          g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
          hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
          h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
          h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
          x : M
          hx : Not (Membership.mem s x)
          c : M
          Hx : Not (Membership.mem (tsupport ⇑(f c)) x)
          this : Eq ((f c) x) 0
          ⊢ Eq (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x))) 0
        -/
        simp [this]
        /-
          🎉 no goals
        -/
    /-
      case intro.refine_2
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s : Set M
      hs : IsOpen s
      f : SmoothPartitionOfUnity M I M
      hf : f.IsSubordinate fun x => (chartAt H x).source
      g : M → H → Real
      g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
      g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
      hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
      h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
      h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
      ⊢ ContMDiff I (modelWithCornersSelf Real Real) Top.top fun x => finsum fun c = …
    -/
  · apply SmoothPartitionOfUnity.contMDiff_finsum_smul
    /-
      case intro.refine_2.hg
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s : Set M
      hs : IsOpen s
      f : SmoothPartitionOfUnity M I M
      hf : f.IsSubordinate fun x => (chartAt H x).source
      g : M → H → Real
      g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
      g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
      hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
      h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
      h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
      ⊢ ∀ (i x : M), Membership.mem (tsupport ⇑(f i)) x → ContMDiffAt I (modelWithCo …
    -/
    intro c x hx
    /-
      case intro.refine_2.hg
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s : Set M
      hs : IsOpen s
      f : SmoothPartitionOfUnity M I M
      hf : f.IsSubordinate fun x => (chartAt H x).source
      g : M → H → Real
      g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
      g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
      hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
      h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
      h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
      c x : M
      hx : Membership.mem (tsupport ⇑(f c)) x
      ⊢ ContMDiffAt I (modelWithCornersSelf Real Real) Top.top (fun x => g c (↑(char …
    -/
    apply (g_diff c (chartAt H c x)).comp
    exact contMDiffAt_of_mem_maximalAtlas (SmoothManifoldWithCorners.chart_mem_maximalAtlas _)
      (hf c hx)
    /-
      case intro.refine_3
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s : Set M
      hs : IsOpen s
      f : SmoothPartitionOfUnity M I M
      hf : f.IsSubordinate fun x => (chartAt H x).source
      g : M → H → Real
      g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
      g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
      hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
      h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
      h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
      ⊢ ∀ (x : M), LE.le 0 ((fun x => finsum fun c => HMul.hMul ((f c) x) (g c (↑(ch …
    -/
  · intro x
    /-
      case intro.refine_3
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s : Set M
      hs : IsOpen s
      f : SmoothPartitionOfUnity M I M
      hf : f.IsSubordinate fun x => (chartAt H x).source
      g : M → H → Real
      g_supp : ∀ (c : M), Eq (Function.support (g c)) (Inter.inter (chartAt H c).tar …
      g_diff : ∀ (c : M), ContMDiff I (modelWithCornersSelf Real Real) Top.top (g c)
      hg : ∀ (c : M), HasSubset.Subset (Set.range (g c)) (Set.Icc 0 1)
      h'g : ∀ (c : M) (x : H), LE.le 0 (g c x)
      h''g : ∀ (c x : M), LE.le 0 (HMul.hMul ((f c) x) (g c (↑(chartAt H c) x)))
      x : M
      ⊢ LE.le 0 ((fun x => finsum fun c => HMul.hMul ((f c) x) (g c (↑(chartAt H c)  …
    -/
    apply finsum_nonneg (fun c ↦ h''g c x)
    /-
      🎉 no goals
    -/


/-- Given an open set `s` containing a closed set `t` in a finite-dimensional real manifold, there
exists a smooth function with support equal to `s`, taking values in `[0,1]`, and equal to `1`
exactly on `t`. -/
theorem exists_msmooth_support_eq_eq_one_iff
    {s t : Set M} (hs : IsOpen s) (ht : IsClosed t) (h : t ⊆ s) :
    ∃ f : M → ℝ, ContMDiff I 𝓘(ℝ) ⊤ f ∧ range f ⊆ Icc 0 1 ∧ support f = s
      ∧ (∀ x, x ∈ t ↔ f x = 1) := by
  /- Take `f` with support equal to `s`, and `g` with support equal to `tᶜ`. Then `f / (f + g)`
  satisfies the conclusion of the theorem. -/
  /-
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s t : Set M
    hs : IsOpen s
    ht : IsClosed t
    h : HasSubset.Subset t s
    ⊢ Exists fun f => And (ContMDiff I (modelWithCornersSelf Real Real) Top.top f) …
  -/
  rcases hs.exists_msmooth_support_eq I with ⟨f, f_supp, f_diff, f_pos⟩
  /-
    case intro.intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s t : Set M
    hs : IsOpen s
    ht : IsClosed t
    h : HasSubset.Subset t s
    f : M → Real
    f_supp : Eq (Function.support f) s
    f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
    f_pos : ∀ (x : M), LE.le 0 (f x)
    ⊢ Exists fun f => And (ContMDiff I (modelWithCornersSelf Real Real) Top.top f) …
  -/
  rcases ht.isOpen_compl.exists_msmooth_support_eq I with ⟨g, g_supp, g_diff, g_pos⟩
  have A : ∀ x, 0 < f x + g x := by
    intro x
    by_cases xs : x ∈ support f
    · have : 0 < f x := lt_of_le_of_ne (f_pos x) (Ne.symm xs)
      linarith [g_pos x]
    · have : 0 < g x := by
        classical
        apply lt_of_le_of_ne (g_pos x) (Ne.symm ?_)
        rw [← mem_support, g_supp]
        contrapose! xs
        simp? at xs says simp only [mem_compl_iff, Decidable.not_not] at xs
        exact h.trans f_supp.symm.subset xs
      linarith [f_pos x]
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s t : Set M
    hs : IsOpen s
    ht : IsClosed t
    h : HasSubset.Subset t s
    f : M → Real
    f_supp : Eq (Function.support f) s
    f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
    f_pos : ∀ (x : M), LE.le 0 (f x)
    g : M → Real
    g_supp : Eq (Function.support g) (HasCompl.compl t)
    g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
    g_pos : ∀ (x : M), LE.le 0 (g x)
    A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
    ⊢ Exists fun f => And (ContMDiff I (modelWithCornersSelf Real Real) Top.top f) …
  -/
  refine ⟨fun x ↦ f x / (f x + g x), ?_, ?_, ?_, ?_⟩
  -- show that `f / (f + g)` is smooth
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      ⊢ ContMDiff I (modelWithCornersSelf Real Real) Top.top fun x => HDiv.hDiv (f x …
    -/
  · exact f_diff.div₀ (f_diff.add g_diff) (fun x ↦ ne_of_gt (A x))
    /-
      🎉 no goals
    -/
  -- show that the range is included in `[0, 1]`
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      ⊢ HasSubset.Subset (Set.range fun x => HDiv.hDiv (f x) (HAdd.hAdd (f x) (g x)) …
    -/
  · refine range_subset_iff.2 (fun x ↦ ⟨div_nonneg (f_pos x) (A x).le, ?_⟩)
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      x : M
      ⊢ LE.le (HDiv.hDiv (f x) (HAdd.hAdd (f x) (g x))) 1
    -/
    apply div_le_one_of_le₀ _ (A x).le
    /-
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      x : M
      ⊢ LE.le (f x) (HAdd.hAdd (f x) (g x))
    -/
    simpa only [le_add_iff_nonneg_right] using g_pos x
    /-
      🎉 no goals
    -/
  -- show that the support is `s`
    /-
      case intro.intro.intro.intro.intro.intro.refine_3
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      ⊢ Eq (Function.support fun x => HDiv.hDiv (f x) (HAdd.hAdd (f x) (g x))) s
    -/
  · have B : support (fun x ↦ f x + g x) = univ := eq_univ_of_forall (fun x ↦ (A x).ne')
    /-
      case intro.intro.intro.intro.intro.intro.refine_3
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      B : Eq (Function.support fun x => HAdd.hAdd (f x) (g x)) Set.univ
      ⊢ Eq (Function.support fun x => HDiv.hDiv (f x) (HAdd.hAdd (f x) (g x))) s
    -/
    simp only [support_div, f_supp, B, inter_univ]
    /-
      🎉 no goals
    -/
  -- show that the function equals one exactly on `t`
    /-
      case intro.intro.intro.intro.intro.intro.refine_4
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      ⊢ ∀ (x : M), Iff (Membership.mem t x) (Eq ((fun x => HDiv.hDiv (f x) (HAdd.hAd …
    -/
  · intro x
    /-
      case intro.intro.intro.intro.intro.intro.refine_4
      E : Type uE
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Real E
      H : Type uH
      inst✝⁶ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type uM
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      inst✝³ : FiniteDimensional Real E
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : SigmaCompactSpace M
      inst✝ : T2Space M
      s t : Set M
      hs : IsOpen s
      ht : IsClosed t
      h : HasSubset.Subset t s
      f : M → Real
      f_supp : Eq (Function.support f) s
      f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
      f_pos : ∀ (x : M), LE.le 0 (f x)
      g : M → Real
      g_supp : Eq (Function.support g) (HasCompl.compl t)
      g_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top g
      g_pos : ∀ (x : M), LE.le 0 (g x)
      A : ∀ (x : M), LT.lt 0 (HAdd.hAdd (f x) (g x))
      x : M
      ⊢ Iff (Membership.mem t x) (Eq ((fun x => HDiv.hDiv (f x) (HAdd.hAdd (f x) (g  …
    -/
    simp [div_eq_one_iff_eq (A x).ne', self_eq_add_right, ← nmem_support, g_supp]
    /-
      🎉 no goals
    -/


/-- Given two disjoint closed sets `s, t` in a Hausdorff σ-compact finite dimensional manifold,
there exists an infinitely smooth function that is equal to `0` exactly on `s` and to `1`
exactly on `t`. See also `exists_smooth_zero_one_of_isClosed` for a slightly weaker version. -/
theorem exists_msmooth_zero_iff_one_iff_of_isClosed {s t : Set M}
    (hs : IsClosed s) (ht : IsClosed t) (hd : Disjoint s t) :
    ∃ f : M → ℝ, ContMDiff I 𝓘(ℝ) ⊤ f ∧ range f ⊆ Icc 0 1 ∧ (∀ x, x ∈ s ↔ f x = 0)
      ∧ (∀ x, x ∈ t ↔ f x = 1) := by
  rcases exists_msmooth_support_eq_eq_one_iff I hs.isOpen_compl ht hd.subset_compl_left with
    ⟨f, f_diff, f_range, fs, ft⟩
  /-
    case intro.intro.intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    f : M → Real
    f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
    f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
    fs : Eq (Function.support f) (HasCompl.compl s)
    ft : ∀ (x : M), Iff (Membership.mem t x) (Eq (f x) 1)
    ⊢ Exists fun f => And (ContMDiff I (modelWithCornersSelf Real Real) Top.top f) …
  -/
  refine ⟨f, f_diff, f_range, ?_, ft⟩
  /-
    case intro.intro.intro.intro
    E : Type uE
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Real E
    H : Type uH
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type uM
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    inst✝³ : FiniteDimensional Real E
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : SigmaCompactSpace M
    inst✝ : T2Space M
    s t : Set M
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    f : M → Real
    f_diff : ContMDiff I (modelWithCornersSelf Real Real) Top.top f
    f_range : HasSubset.Subset (Set.range f) (Set.Icc 0 1)
    fs : Eq (Function.support f) (HasCompl.compl s)
    ft : ∀ (x : M), Iff (Membership.mem t x) (Eq (f x) 1)
    ⊢ ∀ (x : M), Iff (Membership.mem s x) (Eq (f x) 0)
  -/
  simp [← nmem_support, fs]
  /-
    🎉 no goals
  -/

